"""Compare declared root conventions without changing permanent experiments."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from math import ulp
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.optimizer import NSGA2Settings, optimize_nsga2
from dztr.calibration.periods import DatePeriod
from dztr.calibration.root_policy import RootPolicyCalibrationObjective
from dztr.calibration.sampled import (SampledSplit, estimate_sampled_bounds,
    select_sampled_period, target_observations, validate_split)
from dztr.calibration.selection import select_pareto_representatives
from dztr.evaluation import evaluate_simulation
from dztr.evaluation.readiness import trace_implicit_failure
from dztr.model.balance import _fixed_point_solution
from dztr.model.release import _monthly_values, _release_for_values
from dztr.parameterization import estimate_generalized_parameters
from dztr.schemas import ReservoirParameters
from dztr.simulation.root_policy import POLICIES, simulate_root_policy_reservoir
from dztr.simulation.sampled import SamplingSettings, simulate_sampled_reservoir
from .collection import _inside, _selected, _verified_inputs, load_collection
from .world_reservoirs import _code_hashes, _forcing, _new_output, _write_json, sha256


REFERENCE = "matlab_fixed_point_reference"


def _quality(result, parameters, tolerance):
    frame = result.frame
    balance = float(frame.balance_residual_m3.abs().max())
    release_rule = float(frame.release_rule_residual_m3s.abs().max())
    physical = bool(frame.storage_m3.between(0, parameters.capacity_m3).all())
    return {"numerically_valid": bool(physical and balance <= tolerance and release_rule <= 1e-9),
        "max_abs_balance_residual_m3": balance, "max_abs_release_rule_residual_m3s": release_rule,
        "physical_storage_valid": physical,
        "multiple_root_steps": int(frame.root_policy_applied.sum()) if "root_policy_applied" in frame else None,
        "tie_break_steps": int(frame.root_policy_tie_break.sum()) if "root_policy_tie_break" in frame else None,
        "flood_zone_release_above_channel_steps": int(((frame.zone == 3)
            & (frame.overflow_volume_m3 == 0) & (frame.release_m3s > parameters.channel_capacity_m3s)).sum()),
        "overflow_steps": int((frame.overflow_volume_m3 > 0).sum())}


def _score(observations, result, period, sampling):
    observed = select_sampled_period(observations, period, sampling)
    simulated = replace(result, frame=result.frame.loc[period.start:period.end].copy())
    return evaluate_simulation(observed, simulated)


def _table(result, observations, split):
    frame = result.frame.copy()
    for column in ("storage_m3", "release_m3s"):
        frame["observed_" + column] = observations.frame[column]
    frame["period"] = "spinup"
    frame.loc[split.calibration.start:split.calibration.end, "period"] = "calibration"
    frame.loc[split.validation.start:split.validation.end, "period"] = "validation"
    return frame


def _baseline(observations, split, template, sampling, policy, tolerance, output):
    record = {"policy": policy, "status": "blocked", "calibration_status": "blocked",
              "validation_status": "not_run", "metrics": {}, "reference_only": policy == REFERENCE}

    def simulate(values):
        if policy == REFERENCE:
            return simulate_sampled_reservoir(values, template, sampling=sampling,
                solver="matlab_fixed_point", overflow_mode="conservative", max_iterations=1000,
                atol_storage_m3=tolerance)
        return simulate_root_policy_reservoir(values, template, sampling=sampling,
            policy=policy, atol_storage_m3=tolerance)

    prefix = select_sampled_period(observations, DatePeriod(split.simulation_start, split.calibration.end), sampling)
    try:
        calibration = simulate(prefix)
        record["calibration_quality"] = _quality(calibration, template, tolerance)
        record["metrics"]["calibration"] = _score(prefix, calibration, split.calibration, sampling)
        _table(calibration, prefix, split).to_csv(output / "calibration_simulation.csv")
        record["calibration_status"] = "completed" if record["calibration_quality"]["numerically_valid"] else "numerically_invalid"
        if policy != REFERENCE and record["calibration_status"] != "completed":
            record["reason"] = "Calibration trajectory fails storage/release/balance checks."
            return record
    except (TypeError, ValueError) as error:
        record["reason"] = str(error)
        return record
    try:
        continuous = select_sampled_period(observations, split.simulation_period, sampling)
        result = simulate(continuous)
        # Validation follows precisely the same initial state and prefix.
        pd.testing.assert_frame_equal(result.frame.loc[prefix.frame.index], calibration.frame, check_exact=True)
        record["validation_quality"] = _quality(
            replace(result, frame=result.frame.loc[split.validation.start:split.validation.end].copy()), template, tolerance)
        record["trajectory_quality"] = _quality(result, template, tolerance)
        record["metrics"]["validation"] = _score(continuous, result, split.validation, sampling)
        _table(result, continuous, split).to_csv(output / "simulation.csv")
        record["validation_status"] = "completed" if record["validation_quality"]["numerically_valid"] else "numerically_invalid"
        valid = record["trajectory_quality"]["numerically_valid"]
        record["status"] = "completed" if valid else "numerically_invalid_reference" if policy == REFERENCE else "blocked"
    except (TypeError, ValueError) as error:
        record["validation_status"], record["reason"] = "blocked", str(error)
    return record


def _first_failure(observations, template, sampling, split, tolerance):
    prefix = select_sampled_period(observations, DatePeriod(split.simulation_start, split.calibration.end), sampling)
    detail = trace_implicit_failure(prefix, template, sampling, tolerance=tolerance)
    if detail is None:
        return None
    predictor, dt, inflow = detail["predictor_m3"], detail["dt_seconds"], detail["inflow_m3s"]
    values = _monthly_values(detail["month"], template)
    q, states = _release_for_values(predictor, inflow, values, dt), []
    for _ in range(1000):
        s = predictor - dt / 2.0 * q
        q = _release_for_values(s, inflow, values, dt)
        states.append((float(s), float(q)))
    assert states[-1] == _fixed_point_solution(predictor, inflow, dt, values, 1000)
    residual = s - predictor + dt / 2.0 * q
    tol = max(tolerance, 64.0 * ulp(max(abs(predictor), 1.0)))
    tail = np.array(states[-64:])
    period = next((lag for lag in range(1, 9)
        if np.max(np.abs(tail[lag:, 0] - tail[:-lag, 0])) <= tol
        and dt / 2 * np.max(np.abs(tail[lag:, 1] - tail[:-lag, 1])) <= tol), None)
    detail["matlab_reference"] = {"iterations": 1000, "storage_m3": s, "release_m3s": q,
        "balance_residual_m3": residual, "balance_valid": abs(residual) <= tol,
        "tail_period": period, "converged": period == 1 and abs(residual) <= tol,
        "last_iterate_change_m3": abs(states[-1][0] - states[-2][0]),
        "note": "Same calibrated targets and clock at this step; no MATLAB executable was run."}
    return detail


def _calibration_pareto(records):
    candidates = []
    for record in records:
        if record["policy"] == REFERENCE or record["calibration_status"] != "completed":
            continue
        metrics = record["metrics"]["calibration"]
        costs = np.array([-metrics["release_nse"], -metrics["storage_nse"]])
        if np.isfinite(costs).all():
            candidates.append((record["policy"], costs))
    return [name for name, costs in candidates if not any(
        (other <= costs).all() and (other < costs).any() for _, other in candidates)]


class _RecordedObjective:
    def __init__(self, objective):
        self.objective, self.bounds, self.records, self.vectors = objective, objective.bounds, [], []

    def evaluate(self, vector):
        result = self.objective.evaluate(vector)
        self.records.append({"evaluation_id": len(self.records), "feasible": result.feasible,
            "release_cost": result.objectives[0], "storage_cost": result.objectives[1], "reason": result.reason})
        self.vectors.append(np.array(vector, copy=True))
        return result


def _pilots(observations, split, template, sampling, bounds, policy, tolerance, budget, seeds, output):
    objective = RootPolicyCalibrationObjective(observations, split, template, sampling=sampling,
        bounds=bounds, policy=policy, atol_storage_m3=tolerance)
    records = []
    for seed in seeds:
        folder = output / f"seed_{seed}"
        folder.mkdir()
        settings = NSGA2Settings(seed=seed, population_size=min(16, budget), evaluation_budget=budget)
        recorded = _RecordedObjective(objective)
        report = {"seed": seed, "policy": policy, "status": "failed", "settings": asdict(settings),
                  "convergence_assessed": False, "selection_uses_validation": False}
        try:
            result = optimize_nsga2(recorded, parameters_to_vector(template), settings, bounds=bounds)
            report.update({"status": "completed", "termination": result.termination,
                "front_size": len(result.front), "calibration_representatives": select_pareto_representatives(result.front)})
            pd.DataFrame([asdict(item) for item in result.history]).to_csv(folder / "history.csv", index=False)
            front = {point.evaluation_id: point for point in result.front}
            validation = []
            # Fix all representative IDs using calibration before any held-out call.
            for identifier in sorted(set(report["calibration_representatives"].values())):
                point = front[identifier]
                item = {"evaluation_id": identifier, "calibration_objectives": list(point.objectives)}
                try:
                    scored = objective.evaluate_validation(point.vector)
                    item.update({"feasible": scored.feasible, "reason": scored.reason, "metrics": scored.metrics})
                    if scored.simulation is not None:
                        item["trajectory_quality"] = _quality(scored.simulation, scored.parameters, tolerance)
                except (TypeError, ValueError) as error:
                    item.update({"feasible": False, "reason": str(error), "metrics": {}})
                validation.append(item)
            report["validation"] = validation
        except (TypeError, ValueError) as error:
            report["reason"] = str(error)
        report.update({"evaluations": len(recorded.records), "feasible_evaluations": sum(
            point["feasible"] for point in recorded.records)})
        pd.DataFrame(recorded.records, columns=["evaluation_id", "feasible", "release_cost", "storage_cost", "reason"]).to_csv(
            folder / "trials.csv", index=False)
        pd.DataFrame(recorded.vectors).to_csv(folder / "vectors.csv", index=False)
        _write_json(folder / "pilot.json", report)
        records.append(report)
    return records


def run_root_policy_review(manifest_file, output_dir, *, reservoirs=None, all_reservoirs=False,
        policies=POLICIES, include_matlab_reference=True, pilot_policy=None,
        optimization_budget=None, seeds=(42, 43, 44)):
    """Fit calibration targets once and compare fixed numerical conventions.

    A root-policy pilot requires both an explicit policy and a per-seed budget.
    Validation never selects a convention, candidate, seed or evaluation budget.
    Default collection configurations and historical results are not modified.
    """
    if isinstance(policies, str):
        policies = (policies,)
    policies = tuple(policies)
    if not policies or any(policy not in POLICIES for policy in policies) or len(set(policies)) != len(policies):
        raise ValueError("policies must be distinct supported policy names.")
    if not isinstance(include_matlab_reference, bool):
        raise TypeError("include_matlab_reference must be a boolean.")
    if (pilot_policy is None) != (optimization_budget is None):
        raise ValueError("A pilot requires both pilot_policy and optimization_budget explicitly.")
    if optimization_budget is not None and (pilot_policy not in policies or isinstance(optimization_budget, bool)
            or not isinstance(optimization_budget, Integral) or optimization_budget < 2):
        raise ValueError("Pilot policy must be in the comparison with an integer per-seed budget of at least two.")
    seeds = tuple(seeds)
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct nonnegative integers.")
    root, manifest, manifest_path, manifest_hash = load_collection(manifest_file)
    selected = _selected(manifest, reservoirs, all_reservoirs)
    protected = [manifest_path]
    for entry in manifest["reservoirs"]:
        protected.extend(_inside(root, entry[key], key) for key in ("config_file", "observations_file"))
        hashes = entry.get("files_sha256")
        if isinstance(hashes, dict):
            for path in hashes:
                try:
                    protected.append(_inside(root, path, "files_sha256 path"))
                except ValueError:
                    pass
    output = _new_output(output_dir, protected)
    code, records, verified, rows = _code_hashes(), [], {}, []
    comparisons = policies + ((REFERENCE,) if include_matlab_reference else ())
    for entry in selected:
        identifier = entry["reservoir_id"]
        report = {"reservoir_id": identifier, "sampling": entry["frequency"], "status": "input_error",
                  "policy_selected": None, "policies": [], "pilots": []}
        records.append(report)
        try:
            config_path, config, observations, hashes = _verified_inputs(root, entry)
            verified.update(hashes)
        except (OSError, ValueError, TypeError, KeyError) as error:
            report["reason"] = str(error)
            continue
        folder = output / identifier
        folder.mkdir()
        try:
            if not config.get("simulation_supported", True):
                raise ValueError(config.get("simulation_unsupported_reason") or "External forcing is required.")
            sampling = SamplingSettings(**config["sampling"])
            period = config["periods"]
            split = SampledSplit(period["simulation_start"], DatePeriod(**period["calibration"]),
                DatePeriod(**period["validation"]), period["warmup_rows"])
            validate_split(observations, split, sampling)
            fitting = target_observations(select_sampled_period(observations, split.calibration, sampling))
            estimates = estimate_generalized_parameters(fitting, config["target_percentiles"],
                channel_capacity_m3s=config["model"]["channel_capacity_m3s"], quantile_method=config["quantile_method"])
            estimates.monthly_targets.to_csv(folder / "calibration_targets_si.csv")
            template = ReservoirParameters(estimates.monthly_targets, **config["model"])
            forced = _forcing(observations, config["inflow_preprocessing"])
            prefix = select_sampled_period(forced, DatePeriod(split.simulation_start, split.calibration.end), sampling)
            inflow = prefix.frame.inflow_m3s.to_numpy(float)
            if sampling.allow_signed_inflow or not np.isfinite(inflow).all() or (inflow < 0).any():
                raise ValueError("Root selection requires finite nonnegative calibration forcing; signed values need a separate policy.")
            tolerance = config["atol_storage_m3"]
            report.update({"status": "review_completed", "config_sha256": sha256(config_path),
                "periods": split.as_dict(), "sampling_settings": asdict(sampling),
                "inflow_preprocessing": config["inflow_preprocessing"], "fixed_model": config["model"],
                "target_percentiles": config["target_percentiles"], "quantile_method": config["quantile_method"],
                "first_calibration_failure": _first_failure(forced, template, sampling, split, tolerance)})
            try:
                bounds = estimate_sampled_bounds(observations, split.calibration, template, sampling,
                                                quantile_method=config["quantile_method"])
                bounds.monthly_bounds.to_csv(folder / "calibration_bounds_si.csv")
                report["bounds_status"] = "ready"
            except (TypeError, ValueError) as error:
                bounds = None
                report.update({"bounds_status": "blocked", "bounds_reason": str(error)})
            for policy in comparisons:
                subfolder = folder / policy
                subfolder.mkdir()
                result = _baseline(forced, split, template, sampling, policy, tolerance, subfolder)
                _write_json(subfolder / "comparison.json", result)
                report["policies"].append(result)
                row = {"reservoir_id": identifier, "frequency": entry["frequency"], "policy": policy,
                       "status": result["status"], "calibration_status": result["calibration_status"],
                       "validation_status": result["validation_status"], "reason": result.get("reason")}
                for label, metrics in result["metrics"].items():
                    row.update({label + "_" + key: value for key, value in metrics.items()})
                row.update(result.get("trajectory_quality", result.get("calibration_quality", {})))
                rows.append(row)
            report["calibration_pareto_policies"] = _calibration_pareto(report["policies"])
            if optimization_budget is not None:
                if bounds is None:
                    report["pilot_reason"] = "Calibration bounds are not constructible."
                else:
                    pilots_folder = folder / "pilots"
                    pilots_folder.mkdir()
                    try:
                        report["pilots"] = _pilots(forced, split, template, sampling, bounds, pilot_policy,
                            tolerance, int(optimization_budget), seeds, pilots_folder)
                    except (TypeError, ValueError) as error:
                        report["pilot_reason"] = str(error)
        except (OSError, TypeError, ValueError, KeyError) as error:
            report.update({"status": "blocked", "reason": str(error)})
        if any(not path.is_file() or sha256(path) != value for path, value in hashes.items()):
            report.update({"status": "input_error", "reason": "Packaged inputs changed during the review."})
        _write_json(folder / "review.json", report)
    manifest_unchanged = manifest_path.is_file() and sha256(manifest_path) == manifest_hash
    inputs_unchanged = all(path.is_file() and sha256(path) == value for path, value in verified.items())
    code_unchanged = _code_hashes() == code
    integrity = manifest_unchanged and inputs_unchanged and code_unchanged
    result = {"schema_version": 1, "experiment": "explicit_root_policy_review",
        "status": "review_completed" if integrity else "invalid_integrity", "policy_selected": None,
        "default_policy_unchanged": "reject", "policies": list(comparisons), "selected_reservoir_count": len(records),
        "reservoir_counts": dict(Counter(record["status"] for record in records)),
        "policy_counts": {policy: dict(Counter(row["status"] for row in rows if row["policy"] == policy)) for policy in comparisons},
        "manifest_file": str(manifest_path), "manifest_sha256": manifest_hash,
        "manifest_file_unchanged": manifest_unchanged, "input_files_unchanged": inputs_unchanged,
        "code_files_unchanged": code_unchanged, "code_sha256": code,
        "pilot_policy": pilot_policy, "per_seed_evaluation_budget": optimization_budget, "seeds": list(seeds),
        "search_evaluations": sum(pilot["evaluations"] for record in records for pilot in record["pilots"]),
        "convergence_assessed": False, "selection_uses_validation": False,
        "excluded_reservoirs": manifest.get("excluded_reservoirs", []), "reservoirs": records,
        "notes": ["Target estimates and bounds use calibration observations only.",
            "Validation continues the same physical state and does not select policies or candidates.",
            "Reference uses the same calibrated targets and clocks with 1000 MATLAB-compatible iterations and conservative spill.",
            "Original source-setting replay and numerical-policy comparison are separate experiments.",
            "No policy repairs a missing root, invalid targets, unavailable forcing or unreviewed signed inflow.",
            "Channel capacity constrains the unchanged emergency branch; flood-zone exceedances are reported separately."]}
    _write_json(output / "root_policy_review.json", result)
    pd.DataFrame(rows).to_csv(output / "root_policy_review.csv", index=False)
    lines = ["# Explicit DZTR root policy review", "", "Default remains reject; no policy was selected automatically.", "",
        f"Reservoirs: {len(records)}. Search evaluations: {result['search_evaluations']}. Convergence assessed: false.", "",
        "| Reservoir | Policy | Status | Calibration NSE release / storage | Validation NSE release / storage | Multiple-root steps |", "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        def metrics(label):
            values = [row.get(f"{label}_{field}") for field in ("release_nse", "storage_nse")]
            return " / ".join(f"{value:.4f}" if value is not None and np.isfinite(value) else "unavailable" for value in values)
        lines.append(f"| {row['reservoir_id']} | {row['policy']} | {row['status']} | {metrics('calibration')} | {metrics('validation')} | {row.get('multiple_root_steps', 'unavailable')} |")
    for record in records:
        if record["status"] != "review_completed":
            lines.extend(["", f"{record['reservoir_id']}: {record.get('reason', record['status'])}"])
    lines.extend(["", "Calibration Pareto lists describe tradeoffs without choosing a policy. Held-out scores are reported separately.",
        "Numerically invalid fixed-iteration references are retained as evidence, not promoted to valid conservative solutions."])
    (output / "root_policy_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/reservoirs/manifest.json")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--reservoir", action="append")
    choice.add_argument("--all", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", action="append", choices=POLICIES)
    parser.add_argument("--skip-matlab-reference", action="store_true")
    parser.add_argument("--pilot-policy", choices=POLICIES)
    parser.add_argument("--optimize", type=int, help="Explicit search evaluations per reservoir per seed.")
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args(argv)
    try:
        result = run_root_policy_review(args.manifest, args.output, reservoirs=args.reservoir,
            all_reservoirs=args.all, policies=args.policy or POLICIES,
            include_matlab_reference=not args.skip_matlab_reference, pilot_policy=args.pilot_policy,
            optimization_budget=args.optimize, seeds=args.seed or (42, 43, 44))
    except (OSError, TypeError, ValueError, KeyError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps({key: result[key] for key in ("status", "selected_reservoir_count", "reservoir_counts",
        "policy_counts", "search_evaluations", "policy_selected")}, indent=2))
    return 0 if result["status"] == "review_completed" and not any(
        record["status"] == "input_error" for record in result["reservoirs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
