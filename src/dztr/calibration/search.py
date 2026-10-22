"""Run a budgeted two-objective calibration search and report held-out scores."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from dztr import __version__
from dztr.data import read_reservoir_observations
from dztr.parameterization import estimate_generalized_parameters
from dztr.schemas import ReservoirParameters

from .baseline import json_ready
from .bounds import estimate_calibration_bounds
from .candidates import parameters_to_vector
from .objectives import CalibrationObjective
from .optimizer import NSGA2Settings, optimize_nsga2
from .periods import CalibrationValidationSplit, DatePeriod, select_period
from .selection import select_pareto_representatives


ARTIFACT_NAMES = (
    "evaluations.csv", "evaluated_vectors.csv", "pareto_front.csv", "pareto_vectors.csv", "history.csv",
    "selected_metrics.csv", "selected_targets_si.csv", "selected_daily_simulations.csv",
    "bounds_si.csv", "summary.json", "resolved_config.json",
)


def _trial_row(point) -> dict:
    return {"evaluation_id": point.evaluation_id, "generation": point.generation,
            "feasible": point.feasible, "reason": point.reason,
            "negative_release_nse": point.objectives[0] if point.feasible else None,
            "negative_storage_nse": point.objectives[1] if point.feasible else None,
            "release_nse": -point.objectives[0] if point.feasible else None,
            "storage_nse": -point.objectives[1] if point.feasible else None}


def _vectors_frame(points) -> pd.DataFrame:
    return pd.DataFrame([point.vector for point in points],
                        index=pd.Index([point.evaluation_id for point in points], name="evaluation_id"),
                        columns=[f"x{i:02d}" for i in range(72)])


def _score_description(evaluation) -> dict:
    description = {"feasible": evaluation.feasible, "reason": evaluation.reason,
            "objectives": list(evaluation.objectives), "metrics": evaluation.metrics,
            "paired_counts": evaluation.paired_counts, "period": evaluation.period.as_dict()}
    if evaluation.simulation is not None:
        frame = evaluation.simulation.frame.loc[evaluation.period.start:evaluation.period.end]
        corrected = frame.balance_residual_m3.copy()
        if evaluation.simulation.overflow_mode == "matlab":
            corrected += frame.overflow_volume_m3 / 2
        description["diagnostics"] = {
            "max_abs_balance_residual_m3": float(frame.balance_residual_m3.abs().max()),
            "max_abs_corrected_balance_residual_m3": float(corrected.abs().max()),
            "overflow_days": int(frame.overflow_volume_m3.gt(0).sum()),
            "minimum_storage_m3": float(frame.storage_m3.min()),
            "maximum_storage_m3": float(frame.storage_m3.max()),
        }
    return description


def run_configured_optimization(
    config_path: str | Path, repo_root: str | Path, output_dir: str | Path,
    *, optimization_overrides: dict | None = None, progress_callback=None,
    resume_state=None, checkpoint_callback=None, stopping_callback=None,
) -> dict:
    """Fit calibration-only bounds, search, freeze selection, then score validation.

    The evaluation budget counts all search attempts, including rejected vectors
    and the baseline. Selected-candidate replays and validation runs are separate
    reporting work, explicitly counted in the report.
    """
    config_path, root, output = Path(config_path), Path(repo_root), Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    search_config = {**config.get("optimization", {}), **(optimization_overrides or {})}
    settings = NSGA2Settings(**search_config)
    config["optimization"] = asdict(settings)
    selection = config["periods"]
    split = CalibrationValidationSplit(DatePeriod(**selection["calibration"]),
        DatePeriod(**selection["validation"]), warmup_steps=selection["warmup_steps"])
    source = root / config["observations_file"]
    if any((output / name).resolve() in {source.resolve(), config_path.resolve()} for name in ARTIFACT_NAMES):
        raise ValueError("The output folder would overwrite source observations or configuration; choose another folder.")
    observed = read_reservoir_observations(source)
    # Preflight reporting coverage without computing validation costs or using
    # validation storage/release to fit, rank, or choose search candidates.
    continuous = select_period(observed, split.simulation_period)
    if continuous.frame.inflow_m3s.isna().any():
        raise ValueError("Search/reporting requires finite inflow throughout the continuous simulation period.")
    validation_observations = select_period(observed, split.validation)
    for column in ("storage_m3", "release_m3s"):
        values = validation_observations.frame[column].dropna().to_numpy()
        if len(values) < 2 or np.ptp(values) == 0:
            raise ValueError(f"Validation NSE requires at least two nonmissing, nonconstant {column} observations.")
    physical = dict(config["model"])
    channel = physical.pop("channel_capacity_m3s", None)
    method = config.get("quantile_method", "hazen")
    estimates = estimate_generalized_parameters(select_period(observed, split.calibration),
        tuple(config.get("target_percentiles", (10, 45, 85))),
        channel_capacity_m3s=channel, quantile_method=method)
    template = ReservoirParameters.from_generalized(estimates, **physical)
    bounds = estimate_calibration_bounds(observed, split.calibration, template, quantile_method=method)
    problem = CalibrationObjective(observed, split.calibration, template, warmup_steps=split.warmup_steps,
                                   bounds=bounds, **config.get("simulation", {}))
    started = time.perf_counter()
    result = optimize_nsga2(problem, parameters_to_vector(template), settings, progress_callback=progress_callback,
                            resume_state=resume_state, checkpoint_callback=checkpoint_callback,
                            stopping_callback=stopping_callback)
    search_seconds = time.perf_counter() - started
    # Selection is completed from calibration costs before the first held-out
    # evaluation. Roles may refer to the same candidate on a small front.
    selected_ids = {"generalized_baseline": result.evaluated[0].evaluation_id,
                    **select_pareto_representatives(result.front)}
    front_costs = np.asarray([point.objectives for point in result.front], dtype=float)
    lookup = {point.evaluation_id: point for point in result.evaluated}
    scored, daily, targets = {}, [], []
    for evaluation_id in sorted(set(selected_ids.values())):
        point = lookup[evaluation_id]
        calibration = problem.evaluate(point.vector)
        if not calibration.feasible or not np.allclose(calibration.objectives, point.objectives, rtol=0, atol=1e-12):
            raise RuntimeError("Selected candidate did not reproduce its saved calibration objectives.")
        validation = problem.evaluate_validation(point.vector, split.validation)
        scored[evaluation_id] = {"calibration": _score_description(calibration),
                                 "validation": _score_description(validation)}
        table = calibration.parameters.monthly_targets.copy()
        table.insert(0, "evaluation_id", evaluation_id)
        targets.append(table.reset_index())
        if validation.simulation is not None:
            table = validation.simulation.frame.copy()
            for column in ("storage_m3", "release_m3s"):
                table[f"observed_{column}"] = continuous.frame[column]
            table["period"] = "unscored"
            table.loc[table.index < split.calibration.start, "period"] = "spinup"
            for name, period in (("calibration", split.calibration), ("validation", split.validation)):
                table.loc[period.start:period.end, "period"] = name
            table.insert(0, "evaluation_id", evaluation_id)
            daily.append(table.reset_index().rename(columns={table.index.name or "index": "date"}))
    metrics_rows = []
    for role, evaluation_id in selected_ids.items():
        for period_name, score in scored[evaluation_id].items():
            metrics_rows.append({"role": role, "evaluation_id": evaluation_id, "period": period_name,
                                 "feasible": score["feasible"], "reason": score["reason"],
                                 **score["metrics"],
                                 "release_pairs": score["paired_counts"]["release"],
                                 "storage_pairs": score["paired_counts"]["storage"]})
    rejected = [point for point in result.evaluated if not point.feasible]
    code_root = Path(__file__).resolve().parents[1]
    code_hashes = {path.relative_to(code_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(code_root.rglob("*.py"))}
    report = {
        "experiment": "nsga2_calibration_pilot", "algorithm": "NSGA-II (NumPy implementation)",
        "objective_order": ["negative_release_nse", "negative_storage_nse"], "objective_direction": "minimize",
        "reservoir": config.get("reservoir"), "periods": {
            "simulation": split.simulation_period.as_dict(), "calibration": split.calibration.as_dict(),
            "validation": split.validation.as_dict()},
        "warmup_steps": split.warmup_steps, "bounds": bounds.summary(), "optimization": asdict(settings),
        "termination": result.termination, "search_evaluations": result.evaluations,
        "feasible_evaluations": result.evaluations - len(rejected), "rejected_evaluations": len(rejected),
        "rejection_reasons": dict(Counter(point.reason for point in rejected)), "pareto_count": len(result.front),
        "final_population_ids": [point.evaluation_id for point in result.final_population],
        "search_elapsed_seconds": search_seconds, "reporting_calibration_evaluations": len(scored),
        "reporting_validation_evaluations": len(scored), "selected_candidate_ids": selected_ids,
        "unique_representative_count": len(set(selected_ids[role] for role in ("best_release", "best_storage", "balanced"))),
        "selection_policy": {
            "input": "Feasible nondominated archive; calibration objectives only.",
            "best_release": "Minimum negative release NSE, then negative storage NSE, then evaluation ID.",
            "best_storage": "Minimum negative storage NSE, then negative release NSE, then evaluation ID.",
            "balanced": "Minimum Euclidean distance to ideal using calibration Pareto objective ranges; constant ranges contribute zero; ties use release, storage, ID.",
            "validation": "Computed only after representative IDs are fixed; never used for search or selection.",
            "calibration_front_ideal": front_costs.min(axis=0).tolist(),
            "calibration_front_maximum": front_costs.max(axis=0).tolist()},
        "selected_scores": {role: {"evaluation_id": evaluation_id, **scored[evaluation_id]}
                            for role, evaluation_id in selected_ids.items()},
        "configuration": config,
        "source_sha256": {"observations": hashlib.sha256(source.read_bytes()).hexdigest(),
                          "configuration": hashlib.sha256(config_path.read_bytes()).hexdigest()},
        "code_sha256": code_hashes,
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "dztr": __version__},
        "random_generator": "numpy.random.Generator / PCG64",
        "notes": [
            "This short seeded search is a workflow pilot, not evidence of optimizer convergence or a global Pareto front.",
            "NSGA-II is a new adapter; it does not reproduce the historical AMALGAM search.",
            "Every search evaluation, including rejected candidates and the starting baseline, counts against the budget.",
            "Selected-candidate replay and validation evaluations are reporting work outside the search budget.",
            "Only monthly targets are optimized; physical constants, initial state and inflow factor remain fixed configuration.",
            "Validation carries the continuous simulated state from spin-up; no validation observations initialize it.",
            "The inherited Trinity inflow-factor provenance remains qualified in docs/calibration_baseline.md.",
            "Undefined metrics and rejected costs are JSON null; feasibility and reasons distinguish rejected evaluations.",
            "Exact reproduction requires the saved settings, data/code hashes and software versions; elapsed time varies.",
        ],
    }
    report = json_ready(report)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_trial_row(point) for point in result.evaluated]).to_csv(output / "evaluations.csv", index=False)
    _vectors_frame(result.evaluated).to_csv(output / "evaluated_vectors.csv")
    pd.DataFrame([_trial_row(point) for point in result.front]).to_csv(output / "pareto_front.csv", index=False)
    _vectors_frame(result.front).to_csv(output / "pareto_vectors.csv")
    pd.DataFrame([asdict(record) for record in result.history]).to_csv(output / "history.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(output / "selected_metrics.csv", index=False)
    pd.concat(targets, ignore_index=True).to_csv(output / "selected_targets_si.csv", index=False)
    if daily:
        pd.concat(daily, ignore_index=True).to_csv(output / "selected_daily_simulations.csv", index=False)
    else:
        pd.DataFrame(columns=["evaluation_id", "date", "period"]).to_csv(output / "selected_daily_simulations.csv", index=False)
    bounds.monthly_bounds.to_csv(output / "bounds_si.csv")
    for name, value in (("summary.json", report), ("resolved_config.json", config)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/trinity_optimization"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--evaluation-budget", type=int)
    args = parser.parse_args()
    overrides = {name: getattr(args, name) for name in ("seed", "population_size", "evaluation_budget")
                 if getattr(args, name) is not None}

    def progress(record):
        print(f"Generation {record.generation}: {record.evaluations} evaluations, "
              f"{record.feasible_count} feasible, {record.front_count} nondominated", flush=True)

    report = run_configured_optimization(args.config or args.repo / "configs/trinity_optimization.json",
        args.repo, args.output, optimization_overrides=overrides, progress_callback=progress)
    columns = ["role", "period", "release_nse", "storage_nse"]
    print(pd.read_csv(args.output / "selected_metrics.csv")[columns].round(4).to_string(index=False))
    print(f"Search: {report['search_evaluations']} evaluations; {report['pareto_count']} Pareto candidates; {report['termination']}")
    print(f"Artifacts: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
