"""Compare declared substep resolutions with calibration-only daily baselines."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from dztr.calibration.periods import DatePeriod
from dztr.calibration.sampled import (_observations, SampledSplit, select_sampled_period,
    target_observations, validate_split)
from dztr.evaluation import evaluate_simulation
from dztr.evaluation.metrics import objective_metric
from dztr.parameterization import estimate_generalized_parameters
from dztr.schemas import ReservoirParameters
from dztr.simulation.root_policy import POLICIES, simulate_root_policy_reservoir
from dztr.simulation.sampled import SamplingSettings
from dztr.simulation.substeps import (SubstepSettings, SubstepSimulationError,
    critical_substep_bounds, simulate_substep_reservoir)
from .collection import _inside, _selected, _verified_inputs, load_collection
from .world_reservoirs import _code_hashes, _forcing, _new_output, _write_json, sha256


CONVENTIONS = {"inflow": "linear raw endpoint interpolation; exact source endpoints",
    "operating_month": "destination observation-row month held through its interval",
    "release_limits": "literal DZTR limits use internal substep duration",
    "initial_release": "configured initial release; thereafter carry physical release including spill",
    "overflow": "conservative trapezoidal endpoint correction at each substep",
    "release_output": "endpoint rate and integrated interval mean are separate",
    "observation_release_averaging": "not established; interval-mean fit is diagnostic",
    "refinement": "predeclared equal substeps; no adaptive retry or solver fallback",
    "interpretation": "sensitivity of a timestep-dependent scheme; no production selection"}


def _max(frame, column):
    values = frame[column].dropna()
    return float(values.abs().max()) if len(values) else 0.0


def _quality(result, tolerance):
    frame, parameters = result.simulation.frame, result.simulation.parameters
    physical = bool(frame.storage_m3.between(0, parameters.capacity_m3).all()
                    and np.isfinite(frame.release_m3s).all() and frame.release_m3s.ge(0).all())
    balance, substep, rule = (_max(frame, name) for name in ("balance_residual_m3",
        "max_substep_balance_residual_m3", "max_substep_release_rule_residual_m3s"))
    cumulative_excess = (frame.cumulative_balance_residual_m3.abs()
                         - frame.cumulative_roundoff_bound_m3).clip(lower=0)
    cumulative_valid = bool(cumulative_excess.le(tolerance).all())
    return {"numerically_valid": bool(physical and balance <= tolerance and substep <= tolerance
                                       and rule <= 1e-9 and cumulative_valid),
        "physical_states_valid": physical, "native_rows": len(frame),
        "accepted_substeps": int(frame.substep_count.sum()),
        "max_abs_interval_balance_residual_m3": balance,
        "max_abs_substep_balance_residual_m3": substep,
        "max_abs_preoverflow_release_rule_residual_m3s": rule,
        "max_abs_cumulative_balance_residual_m3": _max(frame, "cumulative_balance_residual_m3"),
        "cumulative_balance_within_roundoff_bound": cumulative_valid,
        "max_abs_native_endpoint_balance_residual_m3": _max(frame, "native_endpoint_balance_residual_m3"),
        "multiple_root_substeps": int(frame.multiple_root_substeps.sum()),
        "tie_break_substeps": int(frame.root_policy_tie_break_substeps.sum()),
        "overflow_substeps": int(frame.overflow_substeps.sum()),
        "clipping_correction_sum_m3": float(frame.overflow_volume_m3.sum()),
        "integrated_spill_release_volume_m3": float(frame.integrated_spill_volume_m3.sum()),
        "integrated_inflow_volume_m3": float(frame.integrated_inflow_volume_m3.sum()),
        "integrated_release_volume_m3": float(frame.integrated_release_volume_m3.sum())}


def _score(observations, result, period, sampling):
    observed = select_sampled_period(observations, period, sampling)
    simulation = replace(result.simulation, frame=result.simulation.frame.loc[period.start:period.end].copy())
    endpoint = evaluate_simulation(observed, simulation)
    mean = simulation.frame.interval_mean_release_m3s
    mean_metrics = {name: objective_metric(observed.frame.release_m3s, mean, name)
                    for name in ("nse", "kge", "pbias", "rmse")}
    return {"endpoint": endpoint, "interval_mean_release": mean_metrics,
            "interval_mean_interpretation": "Diagnostic: native release averaging metadata is unspecified."}


def _save(result, observations, split, output, partial=False):
    frame = result.simulation.frame.copy()
    for name in ("storage_m3", "release_m3s"):
        frame["observed_" + name] = observations.frame[name]
    frame["period"] = "spinup"
    frame.loc[split.calibration.start:split.calibration.end, "period"] = "calibration"
    frame.loc[split.validation.start:split.validation.end, "period"] = "validation"
    frame.to_csv(output / ("partial_simulation.csv" if partial else "simulation.csv"), index_label="date")
    if result.internal is not None:
        result.internal.to_csv(output / ("partial_internal_steps.csv" if partial else "internal_steps.csv"))


def _run_resolution(observations, template, sampling, split, count, policy, tolerance, output, record_internal):
    record = {"substeps_per_interval": count, "substep_seconds": 86400.0 / count,
        "root_policy": policy, "status": "blocked", "calibration_status": "blocked",
        "validation_status": "not_run", "metrics": {}, "one_substep_parity": None}
    complete = False
    try:
        result = simulate_substep_reservoir(observations, template, sampling=sampling,
            settings=SubstepSettings(count, policy), atol_storage_m3=tolerance, record_internal=record_internal)
        complete = True
    except SubstepSimulationError as error:
        result = error.partial
        record.update({"reason": str(error), "first_failure": error.detail})
    _save(result, observations, split, output, partial=not complete)
    record["trajectory_quality"] = _quality(result, tolerance)
    last = result.simulation.frame.index[-1]
    for label, period in (("calibration", split.calibration), ("validation", split.validation)):
        if last < period.end:
            failure = record.get("first_failure")
            attempted = last >= period.start or (failure is not None
                and pd.Timestamp(failure["date"]) >= period.start)
            record[label + "_status"] = "blocked" if label == "calibration" or attempted else "not_run"
            continue
        subset = replace(result, simulation=replace(result.simulation,
                         frame=result.simulation.frame.loc[period.start:period.end].copy()))
        quality = _quality(subset, tolerance)
        record[label + "_quality"] = quality
        record[label + "_status"] = "completed" if quality["numerically_valid"] else "numerically_invalid"
        try:
            record["metrics"][label] = _score(observations, result, period, sampling)
        except (TypeError, ValueError) as error:
            record[label + "_metric_error"] = str(error)
    if complete:
        record["status"] = "completed" if record["trajectory_quality"]["numerically_valid"] else "numerically_invalid"
    if count == 1 and len(result.simulation.frame) >= 2:
        prefix = _observations(observations.frame.loc[result.simulation.frame.index].copy(), observations.source)
        control = simulate_root_policy_reservoir(prefix, template, sampling=sampling,
            policy=policy, atol_storage_m3=tolerance)
        pd.testing.assert_frame_equal(result.simulation.frame[control.frame.columns], control.frame, check_exact=True)
        record["one_substep_parity"] = {"exact": True, "native_rows": len(control.frame),
                                         "common_columns": list(control.frame.columns)}
    _write_json(output / "review.json", record)
    return record, result


def _difference(coarse, fine, capacity, period):
    a = coarse.simulation.frame.loc[period.start:period.end]
    b = fine.simulation.frame.loc[period.start:period.end]
    pd.testing.assert_index_equal(a.index, b.index)
    result = {}
    for name in ("storage_m3", "release_m3s", "interval_mean_release_m3s"):
        delta = b[name].to_numpy(float) - a[name].to_numpy(float)
        result[name + "_max_abs_difference"] = float(np.max(np.abs(delta)))
        result[name + "_rmse_difference"] = float(np.sqrt(np.mean(delta ** 2)))
    result["storage_max_difference_fraction_capacity"] = result["storage_m3_max_abs_difference"] / capacity
    result["release_volume_difference_m3"] = float(b.integrated_release_volume_m3.sum() - a.integrated_release_volume_m3.sum())
    result["max_abs_cumulative_release_difference_m3"] = _max(
        pd.DataFrame({"delta": b.cumulative_release_volume_m3 - a.cumulative_release_volume_m3}), "delta")
    return result


def _matched_failure(observations, template, sampling, split, coarse, counts, policy, tolerance, output):
    detail = coarse.get("first_failure")
    if detail is None or detail["failure_code"] != "implicit_no_root":
        return []
    position = detail["interval_row"]
    pair = _observations(observations.frame.iloc[position - 1:position + 1].copy(), observations.source)
    initialized = replace(template, initial_storage_m3=detail["previous_storage_m3"],
                           initial_release_m3s=detail["previous_release_m3s"])
    folder = output / "matched_failure_interval"
    folder.mkdir()
    records = []
    for count in counts:
        subfolder = folder / f"substeps_{count}"
        subfolder.mkdir()
        record = {"substeps_per_interval": count, "root_policy": policy,
            "initialization": "Recorded one-substep first-failure previous state; isolated interval, not full trajectory.",
            "date": detail["date"], "status": "blocked"}
        try:
            result = simulate_substep_reservoir(pair, initialized, sampling=sampling,
                settings=SubstepSettings(count, policy), atol_storage_m3=tolerance, record_internal=True)
            quality = _quality(result, tolerance)
            record.update({"status": "completed" if quality["numerically_valid"] else "numerically_invalid",
                           "quality": quality})
        except SubstepSimulationError as error:
            result = error.partial
            record.update({"first_failure": error.detail, "reason": str(error)})
        _save(result, pair, split, subfolder, partial=record["status"] == "blocked")
        _write_json(subfolder / "review.json", record)
        records.append(record)
    return records


def run_substep_review(manifest_file, output_dir, *, reservoirs=("ex12_Trinity", "ex26_Ghost"),
                       substeps=(1, 24, 48, 96), root_policy="reject", record_internal=False):
    """Run a fixed daily sensitivity protocol; never choose settings on validation."""
    counts = tuple(substeps)
    if (not counts or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in counts)
            or len(set(counts)) != len(counts) or 1 not in counts):
        raise ValueError("Substeps must be distinct positive integers including the one-substep control.")
    counts = tuple(sorted(int(n) for n in counts))
    SubstepSettings(counts[0], root_policy)
    if not isinstance(record_internal, bool):
        raise TypeError("record_internal must be a boolean.")
    root, manifest, manifest_path, manifest_hash = load_collection(manifest_file)
    entries = _selected(manifest, reservoirs, False)
    code = _code_hashes()
    protected = [manifest_path]
    for entry in manifest["reservoirs"]:
        for key in ("config_file", "observations_file"):
            protected.append(_inside(root, entry[key], key))
        hashes = entry.get("files_sha256")
        if isinstance(hashes, dict):
            protected.extend(_inside(root, relative, "files_sha256 path") for relative in hashes)
    output = _new_output(output_dir, protected)
    verified, records, rows = {}, [], []
    for entry in entries:
        identifier = entry["reservoir_id"]
        report = {"reservoir_id": identifier, "status": "input_error", "resolutions": [],
                  "refinement_differences": [], "matched_failure_intervals": []}
        records.append(report)
        try:
            config_path, config, observations, hashes = _verified_inputs(root, entry)
            verified.update(hashes)
        except (OSError, TypeError, ValueError, KeyError) as error:
            report["reason"] = str(error)
            continue
        folder = output / identifier
        folder.mkdir()
        try:
            sampling = SamplingSettings(**config["sampling"])
            if sampling.frequency != "daily" or sampling.month_clock != "observed":
                raise ValueError("This first comparison workflow requires daily observed-calendar inputs.")
            if not config.get("simulation_supported", True):
                raise ValueError(config.get("simulation_unsupported_reason") or "External forcing is required.")
            periods = config["periods"]
            split = SampledSplit(periods["simulation_start"], DatePeriod(**periods["calibration"]),
                                  DatePeriod(**periods["validation"]), periods["warmup_rows"])
            validate_split(observations, split, sampling)
            fitting = target_observations(select_sampled_period(observations, split.calibration, sampling))
            estimates = estimate_generalized_parameters(fitting, config["target_percentiles"],
                channel_capacity_m3s=config["model"]["channel_capacity_m3s"], quantile_method=config["quantile_method"])
            template = ReservoirParameters(estimates.monthly_targets, **config["model"])
            estimates.monthly_targets.to_csv(folder / "calibration_targets_si.csv")
            bounds = critical_substep_bounds(template)
            bounds.to_csv(folder / "critical_boundary_substeps.csv")
            forced = _forcing(observations, config["inflow_preprocessing"])
            continuous = select_sampled_period(forced, split.simulation_period, sampling)
            tolerance = config["atol_storage_m3"]
            report.update({"status": "review_completed", "config_sha256": sha256(config_path),
                "periods": split.as_dict(), "fixed_model": config["model"],
                "sampling_settings": asdict(sampling), "inflow_preprocessing": config["inflow_preprocessing"],
                "target_percentiles": config["target_percentiles"], "quantile_method": config["quantile_method"],
                "critical_boundary_minimum_substeps_all_months": int(bounds.minimum_substeps.max()),
                "atol_storage_m3": tolerance})
            successes = []
            for count in counts:
                subfolder = folder / f"substeps_{count}"
                subfolder.mkdir()
                record, result = _run_resolution(continuous, template, sampling, split, count,
                    root_policy, tolerance, subfolder, record_internal)
                report["resolutions"].append(record)
                row = {"reservoir_id": identifier, **{key: record[key] for key in
                    ("substeps_per_interval", "substep_seconds", "root_policy", "status", "calibration_status", "validation_status")},
                    "reason": record.get("reason")}
                row.update(record["trajectory_quality"])
                for period, metrics in record["metrics"].items():
                    row.update({period + "_endpoint_" + k: v for k, v in metrics["endpoint"].items()})
                    row.update({period + "_interval_mean_release_" + k: v for k, v in metrics["interval_mean_release"].items()})
                rows.append(row)
                if record["status"] == "completed":
                    if successes:
                        previous_count, previous_result = successes[-1]
                        report["refinement_differences"].append({"coarse_substeps": previous_count,
                            "fine_substeps": count, "calibration": _difference(previous_result, result, template.capacity_m3, split.calibration),
                            "validation": _difference(previous_result, result, template.capacity_m3, split.validation),
                            "interpretation": "Sensitivity only; no setting selected from held-out results."})
                    successes = [(count, result)]
            report["matched_failure_intervals"] = _matched_failure(continuous, template, sampling, split,
                report["resolutions"][0], counts, root_policy, tolerance, folder)
        except (OSError, TypeError, ValueError, KeyError) as error:
            report.update({"status": "blocked", "reason": str(error)})
        if any(not path.is_file() or sha256(path) != value for path, value in hashes.items()):
            report.update({"status": "input_error", "reason": "Packaged inputs changed during review."})
        _write_json(folder / "review.json", report)
    guards = {"manifest_file_unchanged": manifest_path.is_file() and sha256(manifest_path) == manifest_hash,
              "input_files_unchanged": all(path.is_file() and sha256(path) == value for path, value in verified.items()),
              "code_files_unchanged": _code_hashes() == code}
    summary = {"schema_version": 1, "experiment": "declared_daily_substep_review",
        "status": "review_completed" if all(guards.values()) else "invalid_integrity",
        "reservoir_counts": dict(Counter(r["status"] for r in records)), "selected_reservoir_count": len(records),
        "resolution_counts": dict(Counter(row["status"] for row in rows)),
        "substeps": list(counts), "root_policy": root_policy, "settings_selected": None,
        "production_default_changed": False, "convergence_assessed": False,
        "selection_uses_validation": False, "search_evaluations": 0, "record_internal": record_internal,
        "conventions": CONVENTIONS, "manifest_file": str(manifest_path), "manifest_sha256": manifest_hash,
        "code_sha256": code, **guards, "reservoirs": records,
        "notes": ["Packaged collection Trinity differs from the standalone workbook example's initial state and inflow factor.",
            "Calibration-only estimates are fixed across all resolutions; validation does not choose settings.",
            "No-root diagnostic replays reset to the recorded coarse failure state and test one isolated interval.",
            "Endpoint-trapezoid residuals at native resolution need not vanish for an internally substepped trajectory."]}
    _write_json(output / "substep_review.json", summary)
    pd.DataFrame(rows).to_csv(output / "substep_review.csv", index=False)
    lines = ["# Declared daily DZTR substep review", "", "Fixed calibration-only targets; no production setting selected.", "",
        f"Root policy: `{root_policy}`. Search evaluations: 0. Internal durations enter the release limits.", "",
        "| Reservoir | Substeps / day | Status | Calibration | Validation | Accepted substeps | First failure |",
        "| --- | ---: | --- | --- | --- | ---: | --- |"]
    for row in rows:
        lines.append(f"| {row['reservoir_id']} | {row['substeps_per_interval']} | {row['status']} | "
            f"{row['calibration_status']} | {row['validation_status']} | {row['accepted_substeps']} | {row.get('reason') or ''} |")
    lines.extend(["", "Complete means numerical feasibility, not predictive skill or convergence.",
        "Integrated mean releases and endpoint releases are distinct; mean-release fit is diagnostic.",
        "Clipping correction sums and integrated spill-release volumes are distinct.", "",
        "## Isolated first no-root intervals", "",
        "| Reservoir | Substeps / day | Status | Failure |", "| --- | ---: | --- | --- |"])
    for record in records:
        for point in record["matched_failure_intervals"]:
            lines.append(f"| {record['reservoir_id']} | {point['substeps_per_interval']} | {point['status']} | {point.get('reason') or ''} |")
    for record in records:
        if record["status"] != "review_completed":
            lines.extend(["", f"{record['reservoir_id']}: {record.get('reason', record['status'])}"])
    (output / "substep_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/reservoirs/manifest.json")
    parser.add_argument("--reservoir", action="append")
    parser.add_argument("--substeps", type=int, action="append")
    parser.add_argument("--root-policy", choices=POLICIES, default="reject")
    parser.add_argument("--record-internal", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_substep_review(args.manifest, args.output,
            reservoirs=args.reservoir or ("ex12_Trinity", "ex26_Ghost"),
            substeps=args.substeps or (1, 24, 48, 96), root_policy=args.root_policy,
            record_internal=args.record_internal)
    except (OSError, TypeError, ValueError, KeyError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps({key: result[key] for key in ("status", "reservoir_counts", "resolution_counts",
                                                 "substeps", "root_policy", "settings_selected")}, indent=2))
    return 0 if result["status"] == "review_completed" and not any(
        r["status"] == "input_error" for r in result["reservoirs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
