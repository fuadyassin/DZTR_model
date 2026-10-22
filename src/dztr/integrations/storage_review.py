"""Audit packaged reservoir storage and flow observations without fitting a model."""
from __future__ import annotations

import argparse
from collections import Counter
import json

import pandas as pd

from dztr.evaluation.storage_audit import audit_storage, INTERPRETATION
from dztr.simulation.sampled import validate_sampling_index
from .collection import _inside, _selected, _verified_inputs, load_collection
from .world_reservoirs import _code_hashes, _new_output, _write_json, sha256


def _date(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tz is not None or result != result.normalize():
        raise ValueError("Reporting periods require explicit timezone-naive midnight dates.")
    return result


def _periods(frame, config):
    periods = config["periods"]
    start = _date(periods["simulation_start"])
    cal = tuple(_date(periods["calibration"][key]) for key in ("start", "end"))
    val = tuple(_date(periods["validation"][key]) for key in ("start", "end"))
    if not start <= cal[0] <= cal[1] < val[0] <= val[1]:
        raise ValueError("Simulation, calibration and validation periods must be chronological and disjoint.")
    if any(date not in frame.index for date in (start, *cal, *val)):
        raise ValueError("Each reporting boundary must be present in the packaged observation calendar.")
    selected = {"warmup": frame.loc[(frame.index >= start) & (frame.index < cal[0])],
        "calibration": frame.loc[cal[0]:cal[1]], "validation": frame.loc[val[0]:val[1]]}
    return selected, start


def _metadata(config_path, config):
    def read(field):
        if field not in config:
            return {}
        value = json.loads((config_path.parent / config[field]).resolve().read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{field} must contain a JSON object.")
        return value
    provenance, source = read("provenance_file"), read("matlab_source_file")
    if not isinstance(source.get("model", {}), dict):
        raise ValueError("Source model metadata must be a JSON object.")
    evidence = source.get("evidence", [])
    if not isinstance(evidence, list) or any(not isinstance(item, dict) for item in evidence):
        raise ValueError("Source evidence metadata must be a list of JSON objects.")
    return {"observation_provenance": {key: provenance.get(key) for key in
        ("status", "aggregation", "calendar_source", "monthly_calendar_convention",
         "monthly_measurement_convention", "units", "unit_evidence", "issues")},
        "source_dead_storage_expression": source.get("model", {}).get("dead_storage_expression"),
        "source_initial_storage": source.get("initial_storage_source"),
        "source_initial_release": source.get("initial_release_source"),
        "source_dead_storage_evidence": [item for item in evidence if item.get("field") == "Ld"],
        "storage_endpoint_timing_confirmed": False, "flow_averaging_confirmed": False,
        "inflow_independent_measurement_confirmed": False,
        "metadata_note": "Unit/calendar conversion evidence does not establish measurement timing or independent inflow measurement."}


def _write_period(folder, result):
    folder.mkdir()
    result.intervals.to_csv(folder / "intervals.csv")
    result.monthly.to_csv(folder / "monthly.csv", index=False)
    result.extremes.to_csv(folder / "extremes.csv", index=False)
    result.flags.to_csv(folder / "flags.csv")
    _write_json(folder / "summary.json", result.summary)


def run_storage_review(manifest_file, output_dir, *, reservoirs=("ex26_Ghost", "ex12_Trinity"),
                       all_reservoirs=False, lag_diagnostics=True):
    """Review daily/monthly members independently, with period-local ledgers.

    To audit the entire active collection, pass reservoirs=None and
    all_reservoirs=True. External forcing and invalid operating targets do not
    prevent an observation audit, and successful arithmetic is not model readiness.
    """
    if not isinstance(all_reservoirs, bool) or not isinstance(lag_diagnostics, bool):
        raise TypeError("all_reservoirs and lag_diagnostics must be booleans.")
    root, manifest, manifest_path, manifest_hash = load_collection(manifest_file)
    entries = _selected(manifest, reservoirs, all_reservoirs)
    code = _code_hashes()
    protected = [manifest_path]
    for entry in manifest["reservoirs"]:
        protected.extend(_inside(root, entry[key], key) for key in ("config_file", "observations_file"))
        hashes = entry.get("files_sha256")
        if isinstance(hashes, dict):
            for relative in hashes:
                try:
                    protected.append(_inside(root, relative, "files_sha256 path"))
                except ValueError:
                    # Invalid metadata is isolated when its reservoir is loaded.
                    continue
    output = _new_output(output_dir, protected)
    verified, records, rows = {}, [], []
    for entry in entries:
        identifier = entry["reservoir_id"]
        record = {"reservoir_id": identifier, "name": entry.get("name"), "status": "input_error",
            "periods": {}, "model_ready_claimed": False}
        records.append(record)
        try:
            config_path, config, observations, hashes = _verified_inputs(root, entry)
            verified.update(hashes)
        except (OSError, TypeError, ValueError, KeyError) as error:
            record["reason"] = str(error)
            continue
        folder = output / identifier
        folder.mkdir()
        try:
            frame = observations.frame
            frequency = config["sampling"]["frequency"]
            validate_sampling_index(frame.index, frequency)
            periods, start = _periods(frame, config)
            model, forcing = config["model"], config["inflow_preprocessing"]
            first = frame.loc[start]
            record.update({"status": "review_completed", "config_sha256": sha256(config_path),
                "inputs_sha256": {path.relative_to(root).as_posix(): value for path, value in hashes.items()},
                "sampling": config["sampling"], "configured_periods": config["periods"],
                "fixed_model": model, "inflow_preprocessing": forcing, "metadata": _metadata(config_path, config),
                "source_simulation_supported": config.get("simulation_supported", True),
                "source_simulation_unsupported_reason": config.get("simulation_unsupported_reason"),
                "initial_state_comparison": {"date": str(start.date()),
                    "configured_storage_m3": model["initial_storage_m3"],
                    "observed_storage_m3": float(first.storage_m3),
                    "configured_minus_observed_storage_m3": float(model["initial_storage_m3"] - first.storage_m3),
                    "configured_release_m3s": model["initial_release_m3s"],
                    "observed_release_m3s": float(first.release_m3s),
                    "configured_minus_observed_release_m3s": float(model["initial_release_m3s"] - first.release_m3s),
                    "interpretation": "Initialization diagnostic; no automatic reset or claim of post-warmup bias."},
                "observation_rows_outside_reported_periods": len(frame) - sum(len(part) for part in periods.values())})
            for label, part in periods.items():
                if len(part) < 2:
                    record["periods"][label] = {"status": "insufficient_rows", "rows": len(part),
                        "reason": "At least two period-local storage states are needed for a balance interval."}
                    continue
                result = audit_storage(part, frequency=frequency, capacity_m3=model["capacity_m3"],
                    dead_storage_m3=model["capacity_m3"] * model["dead_storage_fraction"],
                    fixed_step_days=config["sampling"].get("fixed_step_days"),
                    inflow_factor=forcing["factor"], inflow_offset_m3s=forcing["offset_m3s"],
                    balance_inflow_correction=model["inflow_correction"], lag_diagnostics=lag_diagnostics)
                _write_period(folder / label, result)
                record["periods"][label] = {"status": "review_completed", **result.summary}
                for hypothesis in result.summary["hypotheses"]:
                    stats = {key: value for key, value in hypothesis["statistics"].items() if key != "segments"}
                    rows.append({"reservoir_id": identifier, "period": label,
                        **{key: hypothesis[key] for key in ("hypothesis", "forcing", "clock", "convention")},
                        **stats, **result.summary["observation_flags"]})
        except (OSError, TypeError, ValueError, KeyError) as error:
            record.update({"status": "blocked", "reason": str(error)})
        if any(not path.is_file() or sha256(path) != value for path, value in hashes.items()):
            record.update({"status": "input_error", "reason": "Packaged inputs changed during storage review."})
        _write_json(folder / "review.json", record)
    guards = {"manifest_file_unchanged": manifest_path.is_file() and sha256(manifest_path) == manifest_hash,
        "input_files_unchanged": all(path.is_file() and sha256(path) == value for path, value in verified.items()),
        "code_files_unchanged": _code_hashes() == code}
    report = {"schema_version": 1, "kind": "storage_diagnostic_review",
        "status": "review_completed" if all(guards.values()) else "invalid_integrity",
        "reservoir_counts": dict(Counter(record["status"] for record in records)),
        "selected_reservoir_count": len(records), "manifest_file": str(manifest_path),
        "manifest_sha256": manifest_hash, "code_sha256": code, **guards,
        "interpretation": INTERPRETATION, "lag_diagnostics": lag_diagnostics,
        "settings_selected": None, "selection_uses_validation": False,
        "production_default_changed": False, "model_ready_claimed": False,
        "search_evaluations": 0, "simulations_run": 0, "parameters_estimated": 0, "reservoirs": records}
    _write_json(output / "storage_review.json", report)
    pd.DataFrame(rows).to_csv(output / "storage_review.csv", index=False)
    lines = ["# Observed storage and water-balance diagnostic", "",
        "Residual = storage change − (inflow volume − observed release volume).",
        "Positive residual means more observed storage gain than recorded net input under the declared hypothesis.",
        "Timing, storage datum, upstream observation derivation and omitted contributions remain unresolved.", "",
        "Calibration and validation are sliced before arithmetic or timing shifts; cumulative balances reset per period and invalid segment.",
        "No parameters, forcing, initial state, solver profile or production defaults were changed. No convention or lag was selected.", "",
        "| Reservoir | Period | Forcing / clock / convention | Valid intervals | Residual RMSE (Mm³) | Signed sum (Mm³) | Below dead with positive Q | Above capacity |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        rmse, total = row["rmse_residual_m3"], row["sum_residual_m3"]
        lines.append(f"| {row['reservoir_id']} | {row['period']} | {row['hypothesis']} | "
            f"{row['valid_intervals']}/{row['total_intervals']} | "
            f"{rmse / 1e6 if rmse is not None else 'unavailable'} | "
            f"{total / 1e6 if total is not None else 'unavailable'} | "
            f"{row['positive_release_below_dead']} | {row['above_capacity']} |")
    lines.extend(["", "Signed sums over incomplete coverage are aggregates of valid intervals, not whole-period closure.",
        "Monthly labels identify native records: treating monthly mean storage as endpoint storage remains an assumption.",
        "The source fixed-duration clock, when different, is a separate compatibility hypothesis.",
        "Each period's summary.json contains timing comparisons on common valid support. A lead is descriptive, not a forecast.",
        "Native interval ledgers, seasonal statistics, largest residuals and per-record flags are saved in each period folder.", ""])
    for record in records:
        if record["status"] != "review_completed":
            lines.extend([f"{record['reservoir_id']}: {record.get('reason', record['status'])}", ""])
    (output / "storage_review.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/reservoirs/manifest.json")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--reservoir", action="append")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--no-lags", action="store_true", help="Omit descriptive one-sample flow timing comparisons.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_storage_review(args.manifest, args.output,
            reservoirs=None if args.all else args.reservoir or ("ex26_Ghost", "ex12_Trinity"),
            all_reservoirs=args.all, lag_diagnostics=not args.no_lags)
    except (OSError, TypeError, ValueError, KeyError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps({key: result[key] for key in ("status", "reservoir_counts", "selected_reservoir_count",
        "settings_selected", "simulations_run", "parameters_estimated")}, indent=2))
    return 0 if result["status"] == "review_completed" and all(
        record["status"] == "review_completed" for record in result["reservoirs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
