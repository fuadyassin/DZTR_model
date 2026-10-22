"""Portable reservoir samples from the paper's original MATLAB folders."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dztr.calibration.baseline import json_ready
from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.optimizer import NSGA2Settings, optimize_nsga2
from dztr.calibration.periods import DatePeriod
from dztr.calibration.sampled import (
    SampledCalibrationObjective, SampledSplit, _observations, default_sampled_split,
    estimate_sampled_bounds, run_sampled_baseline, select_sampled_period, target_observations,
)
from dztr.evaluation import evaluate_simulation
from dztr.parameterization import estimate_generalized_parameters
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters
from dztr.simulation.sampled import LegacyReservoirParameters, SamplingSettings, simulate_sampled_reservoir
from .world_observations import WorldObservationConflict, read_world_observations
from .world_sources import discover_world_sources


MODEL_FIELDS = ("capacity_m3", "initial_storage_m3", "initial_release_m3s", "channel_capacity_m3s",
                "dead_storage_fraction", "inflow_correction", "flood_mode")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    def convert(item):
        if isinstance(item, (Path, pd.Timestamp)):
            return str(item)
        if isinstance(item, np.integer):
            return int(item)
        raise TypeError(f"Cannot serialize {type(item).__name__}.")
    Path(path).write_text(json.dumps(json_ready(value), indent=2, allow_nan=False, default=convert) + "\n",
                          encoding="utf-8")


def _new_output(path, protected=()):
    output = Path(path).expanduser().resolve()
    for item in protected:
        item = Path(item).resolve()
        if output == item or output.is_relative_to(item) or item.is_relative_to(output):
            raise ValueError("Output must be separate from protected source inputs.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Use a new or empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _sampling(spec):
    return SamplingSettings(frequency=spec["sampling"]["frequency"],
                            fixed_step_days=spec["sampling"]["timestep_days"], month_clock="observed")


def _model(spec, observations):
    model = {name: spec["model"][name] for name in MODEL_FIELDS}
    if spec["initial_storage_source"] == "observed":
        model["initial_storage_m3"] = float(observations.frame.storage_m3.iloc[0])
    return model


def _forcing(observations, preprocessing):
    if (not isinstance(preprocessing, dict) or set(preprocessing) != {"factor", "offset_m3s"}
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
                   for value in preprocessing.values())):
        raise ValueError("An explicit finite inflow preprocessing factor and offset are required.")
    frame = observations.frame.copy(deep=True)
    frame["inflow_m3s"] = frame.inflow_m3s * preprocessing["factor"] + preprocessing["offset_m3s"]
    return _observations(frame, observations.source)


def _code_hashes():
    root = Path(__file__).resolve().parents[1]
    # New adapter modules are included; these experiments do not reuse old
    # daily adaptive-search checkpoint identities.
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*.py"))}


def _run(observations, config, output, *, optimization_budget=None, seed=42):
    report = {"reservoir_id": config["reservoir"]["id"], "sampling": config["sampling"],
              "parameterization_status": "blocked", "baseline_status": "blocked",
              "optimization_status": "blocked", "protocol": config["protocol"], "code_sha256": _code_hashes()}
    sampling = SamplingSettings(**config["sampling"])
    period = config["periods"]
    split = SampledSplit(period["simulation_start"], DatePeriod(**period["calibration"]),
                         DatePeriod(**period["validation"]), period["warmup_rows"])
    model = config["model"]
    # Parameter estimation uses measurements; inflow transformations only
    # enter the subsequent physical simulation, including the raw-inflow rule.
    try:
        fitting = target_observations(select_sampled_period(observations, split.calibration, sampling))
        estimates = estimate_generalized_parameters(fitting, config["target_percentiles"],
                    channel_capacity_m3s=model["channel_capacity_m3s"], quantile_method=config["quantile_method"])
        estimates.monthly_targets.to_csv(output / "calibration_targets_si.csv")
        report["parameterization_status"] = "estimated"
        report["estimation_issues"] = list(estimates.issues)
        template = ReservoirParameters(estimates.monthly_targets, **model)
        report["physical_parameters_status"] = "valid"
    except (ValueError, TypeError, KeyError) as error:
        report["parameterization_reason"] = str(error)
        report["physical_parameters_status"] = "blocked"
        _write_json(output / "experiment.json", report)
        return report
    if not config.get("simulation_supported", True):
        report["baseline_reason"] = report["optimization_reason"] = config["simulation_unsupported_reason"]
        _write_json(output / "experiment.json", report)
        return report
    try:
        forced = _forcing(observations, config["inflow_preprocessing"])
    except (TypeError, ValueError) as error:
        report["baseline_reason"] = report["optimization_reason"] = str(error)
        _write_json(output / "experiment.json", report)
        return report
    try:
        baseline = run_sampled_baseline(forced, split, sampling=sampling, model=model,
                     quantile_method=config["quantile_method"], target_percentiles=config["target_percentiles"],
                     atol_storage_m3=config["atol_storage_m3"])
        baseline.table().to_csv(output / "baseline_simulation.csv")
        _write_json(output / "baseline_summary.json", baseline.summary())
        report["baseline_status"] = "completed"
        report["baseline_metrics"] = baseline.metrics
        report["max_abs_balance_residual_m3"] = baseline.summary()["max_abs_balance_residual_m3"]
    except (TypeError, ValueError) as error:
        report["baseline_reason"] = str(error)
    try:
        bounds = estimate_sampled_bounds(observations, split.calibration, template, sampling,
                                         quantile_method=config["quantile_method"])
        bounds.monthly_bounds.to_csv(output / "calibration_bounds_si.csv")
        _write_json(output / "calibration_bounds.json", bounds.summary())
        objective = SampledCalibrationObjective(forced, split, template, sampling=sampling,
                      bounds=bounds, atol_storage_m3=config["atol_storage_m3"])
        vector = parameters_to_vector(template)
        check = objective.evaluate(vector)
        report["optimization_status"] = "ready"
        report["calibration_baseline_feasible"] = check.feasible
        report["calibration_baseline_reason"] = check.reason
        report["calibration_baseline_objectives"] = list(check.objectives)
        if optimization_budget is not None:
            settings = NSGA2Settings(seed=seed, population_size=min(16, optimization_budget),
                                      evaluation_budget=optimization_budget)
            result = optimize_nsga2(objective, vector, settings, bounds=bounds)
            trials = [{"evaluation_id": point.evaluation_id, "generation": point.generation,
                       "feasible": point.feasible, "release_cost": point.objectives[0],
                       "storage_cost": point.objectives[1], "reason": point.reason}
                      for point in result.evaluated]
            pd.DataFrame(trials).to_csv(output / "optimization_trials.csv", index=False)
            pd.DataFrame([point.vector for point in result.evaluated]).to_csv(output / "optimization_vectors.csv", index=False)
            # Export every calibration-nondominated pilot point. Validation is
            # reporting only and cannot influence the search or selection.
            validation = []
            for point in result.front:
                try:
                    scored = objective.evaluate_validation(point.vector)
                    validation.append({"evaluation_id": point.evaluation_id, "feasible": scored.feasible,
                                       "reason": scored.reason, "metrics": scored.metrics})
                except (TypeError, ValueError) as error:
                    validation.append({"evaluation_id": point.evaluation_id, "feasible": False,
                                       "reason": str(error), "metrics": {}})
            _write_json(output / "pilot_validation.json", validation)
            report["optimization_status"] = "pilot_completed"
            report["optimization"] = {"settings": asdict(settings), "evaluations": len(result.evaluated),
                                      "feasible": sum(point.feasible for point in result.evaluated),
                                      "front_size": len(result.front), "termination": result.termination}
    except (TypeError, ValueError) as error:
        report["optimization_status"] = "blocked" if optimization_budget is None else "pilot_failed"
        report["optimization_reason"] = str(error)
    _write_json(output / "experiment.json", report)
    return report


def run_world_sample(config_file, output_dir, *, optimization_budget=None, seed=42):
    """Rerun a normalized sample without the original E-drive source folder."""
    path = Path(config_file).expanduser().resolve(strict=True)
    config_bytes = path.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes.decode("utf-8"))
    if config.get("schema_version") != 1 or config.get("kind") != "world_reservoir_sample":
        raise ValueError("Expected a version 1 world_reservoir_sample configuration.")
    observations_path = (path.parent / config["observations_file"]).resolve(strict=True)
    observations_hash = sha256(observations_path)
    if observations_hash != config["observations_sha256"]:
        raise ValueError("Normalized observations SHA256 changed; reimport or explicitly review a new experiment.")
    if optimization_budget is not None and (isinstance(optimization_budget, bool)
            or not isinstance(optimization_budget, int) or optimization_budget < 2):
        raise ValueError("optimization_budget must be an integer of at least two.")
    output = _new_output(output_dir, (path, observations_path))
    frame = pd.read_csv(observations_path, index_col="date", parse_dates=["date"], float_precision="round_trip")
    observations = _observations(frame, str(observations_path))
    report = _run(observations, config, output, optimization_budget=optimization_budget, seed=seed)
    report["inputs_sha256"] = {"config": config_hash, "observations": observations_hash}
    report["input_files_unchanged"] = sha256(path) == config_hash and sha256(observations_path) == observations_hash
    _write_json(output / "experiment.json", report)
    if not report["input_files_unchanged"]:
        raise ValueError("Sample inputs changed during the experiment; results require a fresh run with reviewed inputs.")
    return report


def import_world_reservoirs(source_root, output_dir, *, run_baselines=True):
    """Import every discovered example, isolate failures, and preserve originals."""
    source_root = Path(source_root).expanduser().resolve(strict=True)
    output = _new_output(output_dir, (source_root,))
    code_hashes = _code_hashes()
    specs = discover_world_sources(source_root)
    _write_json(output / "source_inventory.json", specs)
    entries = []
    for spec in specs:
        reservoir = output / "reservoirs" / spec["reservoir_id"]
        reservoir.mkdir(parents=True)
        _write_json(reservoir / "matlab_source.json", spec)
        entry = {"reservoir_id": spec["reservoir_id"], "name": spec["name"],
                 "frequency": (spec.get("sampling") or {}).get("frequency"),
                 "import_status": "blocked", "baseline_status": "not_run", "optimization_status": "not_checked",
                 "legacy_status": "not_run"}
        entries.append(entry)
        if spec["status"] != "ready":
            entry["reason"] = spec["error"]
            continue
        pd.DataFrame(spec["raw_monthly_targets"], index=pd.Index(range(1, 13), name="month")).to_csv(reservoir / "legacy_targets_si.csv")
        pd.DataFrame(spec["raw_bounds"], index=pd.Index(range(1, 13), name="month")).to_csv(reservoir / "legacy_bounds_si.csv")
        try:
            observations, provenance = read_world_observations(spec)
            _write_json(reservoir / "observation_provenance.json", provenance)
            csv_file = reservoir / "observations_si.csv"
            observations.frame.to_csv(csv_file, float_format="%.17g")
            entry.update({"import_status": "imported", "rows": len(observations.frame),
                          "first_date": str(observations.frame.index[0].date()),
                          "last_date": str(observations.frame.index[-1].date()),
                          "review_issues": list(observations.quality.issues)})
            sampling = _sampling(spec)
            split = default_sampled_split(observations, sampling)
            config = {"schema_version": 1, "kind": "world_reservoir_sample",
                      "reservoir": {"id": spec["reservoir_id"], "name": spec["name"]},
                      "observations_file": "observations_si.csv", "observations_sha256": sha256(csv_file),
                      "units": {"storage": "m3", "flows": "m3/s"}, "sampling": asdict(sampling),
                      "periods": split.as_dict(), "model": _model(spec, observations),
                      "inflow_preprocessing": spec["inflow_preprocessing"],
                      "simulation_supported": spec["simulation_supported"],
                      "simulation_unsupported_reason": spec.get("simulation_unsupported_reason"),
                      "target_percentiles": [10, 45, 85], "quantile_method": "hazen",
                      "atol_storage_m3": 1e-4,
                      "legacy": {"sampling": spec["sampling"], "warmup_rows": spec["warmup_rows"],
                                 "overflow_mode": spec["overflow_mode"], "fixed_point_iterations": spec["fixed_point_iterations"],
                                 "release_shift_rows": spec["legacy_release_shift_rows"],
                                 "storage_shift_rows": spec["legacy_storage_shift_rows"]},
                      "protocol": {"split_policy": "first calendar year warmup; remaining years divided chronologically",
                                   "estimation_policy": "calibration observations only; validation excluded from fitting/search",
                                   "paper_reproduction": False,
                                   "fixed_settings": "source-selected MATLAB model constants, initial-state assignment and inflow preprocessing",
                                   "monthly_clock": "fixed source duration; operating targets use verified observation calendar months",
                                   "legacy_targets": "unmodified full-record MATLAB tables are retained separately, never used as calibration estimates"}}
            _write_json(reservoir / "sample.json", config)
            entry["config_file"] = str((reservoir / "sample.json").relative_to(output))
            if run_baselines:
                try:
                    # Original MATLAB accepts signed inflow directly. This is
                    # an explicit historical replay policy, never the modern
                    # conservative simulation's default forcing policy.
                    legacy, reported = legacy_replay(observations, spec, allow_signed_inflow=True)
                    reported.to_csv(reservoir / "legacy_simulation.csv")
                    warmup = spec["warmup_rows"]
                    scored_obs = _observations(observations.frame.iloc[warmup:], observations.source)
                    scored_sim = replace(legacy, frame=reported.iloc[warmup:].copy())
                    scored_sim.frame["storage_m3"] = scored_sim.frame.legacy_reported_storage_m3
                    scored_sim.frame["release_m3s"] = scored_sim.frame.legacy_reported_release_m3s
                    _write_json(reservoir / "legacy_summary.json", {
                        "experiment": "original_matlab_baseline_replay", "warmup_rows": warmup,
                        "metrics": evaluate_simulation(scored_obs, scored_sim),
                        "sampling": spec["sampling"], "inflow_preprocessing": spec["inflow_preprocessing"],
                        "overflow_mode": spec["overflow_mode"], "iterations": spec["fixed_point_iterations"],
                        "model": _model(spec, observations), "issues": list(legacy.issues),
                        "maximum_physical_balance_residual_m3": float(legacy.frame.balance_residual_m3.abs().max()),
                        "minimum_physical_storage_m3": float(legacy.frame.storage_m3.min()),
                        "notes": ["Unmodified full-record coefficient targets, historical synthetic month clock and source iteration order.",
                                  "This is the selected base script's baseline, not a reconstruction of optimized published scores.",
                                  "Original signed inflow is explicitly allowed for historical replay only.",
                                  "Physical trajectory and any shifted reporting columns are exported separately; shifted columns do not conserve water."]})
                    entry["legacy_status"] = "completed"
                except (TypeError, ValueError) as error:
                    entry["legacy_status"] = "blocked"
                    entry["legacy_reason"] = str(error)
                experiment = reservoir / "baseline"
                experiment.mkdir()
                report = _run(observations, config, experiment)
                entry.update({key: report[key] for key in ("parameterization_status", "baseline_status", "optimization_status")})
                for key in ("parameterization_reason", "baseline_reason", "optimization_reason", "physical_parameters_status",
                            "calibration_baseline_feasible", "max_abs_balance_residual_m3"):
                    if key in report:
                        entry[key] = report[key]
        except (WorldObservationConflict, ValueError, TypeError, KeyError, OSError) as error:
            entry["reason"] = str(error)
            if isinstance(error, WorldObservationConflict):
                entry["import_status"] = "source_conflict"
                _write_json(reservoir / "observation_provenance.json", error.provenance)
    changed = []
    for spec in specs:
        for path, digest in spec["source_sha256"].items():
            if not Path(path).is_file() or sha256(path) != digest:
                changed.append(path)
    summary = {"schema_version": 1, "source_root": str(source_root), "output_dir": str(output),
               "reservoir_count": len(entries), "source_files_unchanged": not changed, "changed_source_files": sorted(set(changed)),
               "code_sha256": code_hashes, "code_files_unchanged": _code_hashes() == code_hashes,
               "counts": {field: dict(Counter(entry[field] for entry in entries))
                          for field in ("frequency", "import_status", "legacy_status", "baseline_status", "optimization_status")},
               "reservoirs": entries,
               "notes": ["All matching ex<number> folders are included; numbering need not be contiguous.",
                         "These are new calibration-only baselines with explicit periods, not a claim to reproduce all published optimized scores.",
                         "Blocked samples retain evidence; no silent clipping, gap filling or guessed storage conversion.",
                         "No cohort optimization is launched by importing samples."]}
    _write_json(output / "cohort.json", summary)
    flat = [{key: value for key, value in entry.items() if not isinstance(value, (list, dict))} for entry in entries]
    pd.DataFrame(flat).to_csv(output / "cohort.csv", index=False)
    return summary


def legacy_replay(observations, spec, *, allow_signed_inflow=False):
    """Explicit historical clock, tables and spill policy; reporting shifts separated."""
    if not spec.get("simulation_supported", False):
        raise ValueError(spec.get("simulation_unsupported_reason", "Unsupported source simulation."))
    forced = _forcing(observations, spec["inflow_preprocessing"])
    targets = pd.DataFrame(spec["raw_monthly_targets"], index=pd.Index(range(1, 13), name="month"))
    parameters = LegacyReservoirParameters(targets, **_model(spec, observations))
    sampling = SamplingSettings(frequency=spec["sampling"]["frequency"], fixed_step_days=spec["sampling"]["timestep_days"],
                 month_clock="legacy", legacy_start=spec["sampling"]["legacy_start"], allow_signed_inflow=allow_signed_inflow)
    result = simulate_sampled_reservoir(forced, parameters, sampling=sampling, solver="matlab_fixed_point",
                 overflow_mode=spec["overflow_mode"], max_iterations=spec["fixed_point_iterations"])
    reported = result.frame.copy()
    for name, key in (("release_m3s", "legacy_release_shift_rows"), ("storage_m3", "legacy_storage_shift_rows")):
        shift = spec[key]
        values = reported[name].to_numpy(copy=True)
        if shift > 0:
            values = np.r_[values[:shift], values[:-shift]]
        elif shift < 0:
            values = np.r_[values[-shift:], values[shift:]]
        reported[f"legacy_reported_{name}"] = values
    return result, reported


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import", help="Discover, normalize and run cohort baselines")
    importer.add_argument("--source-root", required=True)
    importer.add_argument("--output", required=True)
    importer.add_argument("--skip-baselines", action="store_true")
    runner = commands.add_parser("run", help="Run one normalized sample; optionally run an explicit optimizer pilot")
    runner.add_argument("--config", required=True)
    runner.add_argument("--output", required=True)
    runner.add_argument("--optimize", type=int, metavar="EVALUATIONS")
    runner.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "import":
        report = import_world_reservoirs(args.source_root, args.output, run_baselines=not args.skip_baselines)
        print(json.dumps({"reservoir_count": report["reservoir_count"], "counts": report["counts"],
                          "source_files_unchanged": report["source_files_unchanged"]}, indent=2))
        return 0 if report["source_files_unchanged"] and report["code_files_unchanged"] else 1
    report = run_world_sample(args.config, args.output, optimization_budget=args.optimize, seed=args.seed)
    print(json.dumps({key: report.get(key) for key in ("reservoir_id", "parameterization_status", "baseline_status", "optimization_status")}, indent=2))
    return 0 if report["baseline_status"] == "completed" and report["optimization_status"] in ("ready", "pilot_completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
