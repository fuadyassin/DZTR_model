"""Compare nested calibration budgets and seeds without validation selection."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from dztr import __version__
from dztr.data import read_reservoir_observations
from dztr.parameterization import estimate_generalized_parameters
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters

from .baseline import json_ready
from .bounds import estimate_calibration_bounds
from .candidates import parameters_from_vector
from .convergence import checkpoint_metrics, hypervolume_2d, normalized_parameter_spread, prefix_pareto_front
from .objectives import CalibrationObjective
from .optimizer import EvaluatedCandidate, NSGA2Settings
from .periods import CalibrationValidationSplit, DatePeriod, select_period
from .search import ARTIFACT_NAMES as SEARCH_ARTIFACT_NAMES, _score_description, run_configured_optimization
from .selection import select_pareto_representatives


STUDY_ARTIFACT_NAMES = (
    "checkpoint_metrics.csv", "budget_summary.csv", "checkpoint_fronts.csv",
    "pooled_front.csv", "pooled_vectors.csv", "representative_vectors.csv", "representative_targets_si.csv",
    "parameter_stability.csv", "selected_metrics.csv", "selected_targets_si.csv", "selected_daily_simulations.csv",
    "bounds_si.csv", "summary.json", "resolved_config.json",
)
ROLES = ("best_release", "best_storage", "balanced")


@dataclass(frozen=True)
class StudySettings:
    seeds: tuple[int, ...] = (42, 43, 44)
    evaluation_checkpoints: tuple[int, ...] = (96, 384, 1024)
    hypervolume_reference: tuple[float, float] = (1.0, 1.0)
    workers: int = 1

    def __post_init__(self):
        for name in ("seeds", "evaluation_checkpoints"):
            try:
                values = tuple(getattr(self, name))
            except TypeError as exc:
                raise ValueError(f"{name} must be a sequence of integers.") from exc
            if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
                raise ValueError(f"{name} must contain integers.")
            values = tuple(int(value) for value in values)
            if len(values) < 2 or len(set(values)) != len(values):
                raise ValueError(f"{name} requires at least two distinct values.")
            if name == "seeds":
                if min(values) < 0:
                    raise ValueError("Seeds must be nonnegative.")
                values = tuple(sorted(values))
            elif min(values) < 1 or values != tuple(sorted(values)):
                raise ValueError("Evaluation checkpoints must be positive and increasing.")
            object.__setattr__(self, name, values)
        if isinstance(self.workers, bool) or not isinstance(self.workers, Integral) or self.workers < 1:
            raise ValueError("workers must be a positive integer.")
        object.__setattr__(self, "workers", int(self.workers))
        # The indicator validates real finite reference coordinates even when
        # there are no points. Store a detached, serializable pair.
        hypervolume_2d(np.empty((0, 2)), self.hypervolume_reference)
        object.__setattr__(self, "hypervolume_reference", tuple(float(value) for value in self.hypervolume_reference))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "dztr": __version__}


def _analysis_hashes() -> dict:
    code_root = Path(__file__).resolve().parents[1]
    return {path.relative_to(code_root).as_posix(): _sha(path) for path in sorted(code_root.rglob("*.py"))}


def _search_identity(config: dict, root: Path, seed: int, budget: int) -> dict:
    resolved = dict(config)
    resolved.pop("study", None)
    resolved["optimization"] = asdict(NSGA2Settings(**{**config.get("optimization", {}), "seed": seed, "evaluation_budget": budget}))
    code_root = Path(__file__).resolve().parents[1]
    # Analysis-only changes cannot change previously evaluated trial costs.
    analysis_only = {"calibration/study.py", "calibration/convergence.py", "calibration/stopping.py",
                     "calibration/budget_review.py", "calibration/fleet.py"}
    code = {path.relative_to(code_root).as_posix(): _sha(path) for path in sorted(code_root.rglob("*.py"))
            if path.relative_to(code_root).as_posix() not in analysis_only}
    return {"resolved_search_configuration": resolved, "observations_sha256": _sha(root / config["observations_file"]),
            "search_code_sha256": code, "software_versions": _versions()}


def _run_seed(config_path: str, root_path: str, output_path: str, seed: int, budget: int,
              checkpoints: tuple[int, ...], analyze_only: bool) -> dict:
    config_path, root, output = Path(config_path), Path(root_path), Path(output_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    identity = _search_identity(config, root, seed, budget)
    manifest_path = output / "seed_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError(f"Saved seed {seed} differs in inputs, search settings, code or versions; choose another output folder.")
        if any(not (output / name).is_file() or _sha(output / name) != digest
               for name, digest in manifest["artifact_sha256"].items()):
            raise ValueError(f"Saved seed {seed} artifacts failed their SHA-256 integrity check.")
        if set(manifest["artifact_sha256"]) != set(SEARCH_ARTIFACT_NAMES):
            raise ValueError(f"Saved seed {seed} manifest does not contain the complete search artifact set.")
        print(f"Seed {seed}: reusing verified {budget}-evaluation artifacts", flush=True)
        return {"seed": seed, "output": str(output), "reused": True}
    if analyze_only:
        raise ValueError(f"Seed {seed} has no verified completed run for analysis.")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Seed {seed} output is incomplete or lacks an integrity manifest; choose another output folder.")

    def progress(record):
        if record.generation == 0 or record.evaluations in checkpoints or record.generation % 8 == 0:
            print(f"Seed {seed}: {record.evaluations}/{budget} evaluations; "
                  f"{record.feasible_count} feasible; {record.front_count} nondominated", flush=True)

    report = run_configured_optimization(config_path, root, output,
        optimization_overrides={"seed": seed, "evaluation_budget": budget}, progress_callback=progress)
    if _search_identity(config, root, seed, budget) != identity:
        raise RuntimeError("Search inputs or code changed during a seed run; results require review.")
    manifest = {"identity": identity, "search_evaluations": report["search_evaluations"],
                "artifact_sha256": {name: _sha(output / name) for name in SEARCH_ARTIFACT_NAMES}}
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"seed": seed, "output": str(output), "reused": False}


def _read_trials(folder: Path) -> tuple[EvaluatedCandidate, ...]:
    table = pd.read_csv(folder / "evaluations.csv", float_precision="round_trip")
    vectors = pd.read_csv(folder / "evaluated_vectors.csv", index_col="evaluation_id", float_precision="round_trip")
    if vectors.index.has_duplicates or len(vectors) != len(table) or list(vectors.columns) != [f"x{i:02d}" for i in range(72)]:
        raise ValueError("Saved trial vectors do not match the evaluation table.")
    points = []
    for row in table.itertuples(index=False):
        if not isinstance(row.feasible, (bool, np.bool_)):
            raise ValueError("Saved trial feasibility must be boolean.")
        vector = vectors.loc[row.evaluation_id].to_numpy(dtype=float).copy()
        vector.setflags(write=False)
        objectives = ((float(row.negative_release_nse), float(row.negative_storage_nse)) if row.feasible
                      else (float("inf"), float("inf")))
        points.append(EvaluatedCandidate(int(row.evaluation_id), int(row.generation), vector, objectives,
                                        bool(row.feasible), None if pd.isna(row.reason) else str(row.reason)))
    return tuple(points)


def _point_row(point, *, seed=None, budget=None, pooled_id=None) -> dict:
    row = {"evaluation_id": point.evaluation_id, "negative_release_nse": point.objectives[0],
           "negative_storage_nse": point.objectives[1], "release_nse": -point.objectives[0], "storage_nse": -point.objectives[1]}
    if seed is not None:
        row["seed"] = seed
    if budget is not None:
        row["budget"] = budget
    if pooled_id is not None:
        row["pooled_id"] = pooled_id
    return row


def run_calibration_study(config_path: str | Path, repo_root: str | Path, output_dir: str | Path,
                          *, workers: int | None = None, analyze_only: bool = False) -> dict:
    """Run one maximum-budget search per seed, then analyze nested prefixes.

    Existing complete seed outputs are reused only after data/settings/code/
    version and artifact-hash verification. Pooled representatives are chosen
    solely from calibration costs. Validation is reported after that selection.
    """
    config_path, root, output = Path(config_path).resolve(), Path(repo_root).resolve(), Path(output_dir).resolve()
    analysis_hashes = _analysis_hashes()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = StudySettings(**{**config.get("study", {}), **({"workers": workers} if workers is not None else {})})
    budget = settings.evaluation_checkpoints[-1]
    optimizer_settings = NSGA2Settings(**{**config.get("optimization", {}), "evaluation_budget": budget})
    if settings.evaluation_checkpoints[0] < optimizer_settings.population_size:
        raise ValueError("The first checkpoint must include the entire initial population.")
    if any(count % optimizer_settings.population_size for count in settings.evaluation_checkpoints[:-1]):
        raise ValueError("Intermediate checkpoints must end at population-sized generations for nested-run parity.")
    config["study"] = asdict(settings)
    config["optimization"] = asdict(optimizer_settings)
    source = root / config["observations_file"]
    protected = {source.resolve(), config_path}
    destinations = [output / name for name in STUDY_ARTIFACT_NAMES]
    destinations += [output / "runs" / f"seed_{seed}" / name for seed in settings.seeds
                     for name in (*SEARCH_ARTIFACT_NAMES, "seed_manifest.json")]
    if any(path.resolve() in protected for path in destinations):
        raise ValueError("Study outputs would overwrite source observations or configuration; choose another folder.")
    if output.exists() and not output.is_dir():
        raise ValueError("Study output must be a directory.")
    input_hashes = {"observations": _sha(source), "configuration": _sha(config_path)}
    selection = config["periods"]
    split = CalibrationValidationSplit(DatePeriod(**selection["calibration"]), DatePeriod(**selection["validation"]),
                                       warmup_steps=selection["warmup_steps"])
    observed = read_reservoir_observations(source)
    physical = dict(config["model"])
    channel = physical.pop("channel_capacity_m3s", None)
    method = config.get("quantile_method", "hazen")
    estimates = estimate_generalized_parameters(select_period(observed, split.calibration),
        tuple(config.get("target_percentiles", (10, 45, 85))), channel_capacity_m3s=channel, quantile_method=method)
    template = ReservoirParameters.from_generalized(estimates, **physical)
    bounds = estimate_calibration_bounds(observed, split.calibration, template, quantile_method=method)
    problem = CalibrationObjective(observed, split.calibration, template, warmup_steps=split.warmup_steps,
                                   bounds=bounds, **config.get("simulation", {}))
    started = time.perf_counter()
    tasks = [(str(config_path), str(root), str(output / "runs" / f"seed_{seed}"), seed, budget,
              settings.evaluation_checkpoints, analyze_only) for seed in settings.seeds]
    runs = {}
    if settings.workers == 1:
        for task in tasks:
            run = _run_seed(*task)
            runs[run["seed"]] = run
    else:
        with ProcessPoolExecutor(max_workers=min(settings.workers, len(tasks))) as executor:
            futures = [executor.submit(_run_seed, *task) for task in tasks]
            for future in as_completed(futures):
                run = future.result()
                runs[run["seed"]] = run
    if input_hashes != {"observations": _sha(source), "configuration": _sha(config_path)}:
        raise RuntimeError("Study inputs changed during execution; results require review.")
    points_by_seed = {seed: _read_trials(Path(runs[seed]["output"])) for seed in settings.seeds}
    metric_rows, front_rows, representative_rows, target_rows = [], [], [], []
    representatives = {}
    for seed, points in points_by_seed.items():
        previous_hv = None
        for count in settings.evaluation_checkpoints:
            metrics = checkpoint_metrics(points, count, reference=settings.hypervolume_reference)
            front = prefix_pareto_front(points, count)
            chosen = metrics.pop("representative_ids")
            metrics.pop("reference")
            metrics.pop("evaluations")
            row = {"seed": seed, "budget": count, **metrics,
                   "hypervolume_gain_previous": None if previous_hv is None else metrics["hypervolume"] - previous_hv,
                   "hypervolume_relative_gain_previous": None if previous_hv is None or previous_hv == 0 else
                       (metrics["hypervolume"] - previous_hv) / previous_hv}
            if previous_hv is not None and metrics["hypervolume"] + 1e-12 < previous_hv:
                raise RuntimeError("An accumulated Pareto archive unexpectedly lost hypervolume.")
            metric_rows.append(row)
            previous_hv = metrics["hypervolume"]
            front_rows.extend(_point_row(point, seed=seed, budget=count) for point in front)
            if chosen is None:
                raise ValueError(f"Seed {seed} checkpoint {count} has no feasible representative.")
            lookup = {point.evaluation_id: point for point in points}
            for role, evaluation_id in chosen.items():
                point = lookup[evaluation_id]
                representatives[(seed, count, role)] = point
                representative_rows.append({"seed": seed, "budget": count, "role": role, "evaluation_id": evaluation_id,
                                            **{f"x{i:02d}": value for i, value in enumerate(point.vector)}})
                table = parameters_from_vector(point.vector, template).monthly_targets.copy().reset_index()
                for name, value in (("evaluation_id", evaluation_id), ("role", role), ("budget", count), ("seed", seed)):
                    table.insert(0, name, value)
                target_rows.append(table)
    metrics_table = pd.DataFrame(metric_rows)
    summary_fields = ("hypervolume", "max_release_nse", "max_storage_nse", "balanced_release_nse", "balanced_storage_nse",
                      "hypervolume_gain_previous", "hypervolume_relative_gain_previous")
    budget_table = metrics_table.groupby("budget", sort=True)[list(summary_fields)].agg(["min", "median", "max"])
    budget_table.columns = [f"{metric}_{statistic}" for metric, statistic in budget_table.columns]
    stability_rows = []
    stability_summaries = {}
    capacity_scales = np.r_[np.full(36, template.capacity_m3), np.ones(36)]
    lower_si, upper_si = bounds.lower_vector * capacity_scales, bounds.upper_vector * capacity_scales
    for count in settings.evaluation_checkpoints:
        for role in ROLES:
            vectors = np.asarray([representatives[(seed, count, role)].vector for seed in settings.seeds])
            spread = normalized_parameter_spread(vectors, bounds)
            native_si = vectors * capacity_scales
            free_spread = spread[bounds.upper_vector > bounds.lower_vector]
            stability_summaries[f"{count}/{role}"] = {
                "median_normalized_spread": float(np.median(free_spread)) if len(free_spread) else 0.0,
                "max_normalized_spread": float(np.max(free_spread)) if len(free_spread) else 0.0,
                "free_dimensions": int(len(free_spread)), "seed_count": len(settings.seeds)}
            for i in range(72):
                stability_rows.append({"budget": count, "role": role, "vector_index": i,
                    "target_column": MONTHLY_TARGET_COLUMNS[i // 12], "month": i % 12 + 1,
                    "unit": "m3" if i < 36 else "m3/s", "lower_si": lower_si[i], "upper_si": upper_si[i],
                    "min_si": float(native_si[:, i].min()), "median_si": float(np.median(native_si[:, i])),
                    "max_si": float(native_si[:, i].max()), "normalized_spread": spread[i]})
    pooled, origins = [], {}
    for seed, points in points_by_seed.items():
        for point in points:
            pooled_id = len(pooled)
            origins[pooled_id] = (seed, point.evaluation_id)
            pooled.append(EvaluatedCandidate(pooled_id, point.generation, point.vector, point.objectives, point.feasible, point.reason))
    pooled_front = prefix_pareto_front(pooled, len(pooled))
    # Freeze pooled calibration choices. Per-seed reports exist, but this
    # analysis never reads their validation metrics to choose candidates.
    selected = {"generalized_baseline": 0, **select_pareto_representatives(pooled_front)}
    for seed in settings.seeds:
        for role in ROLES:
            local_id = representatives[(seed, budget, role)].evaluation_id
            selected[f"seed_{seed}_{role}"] = next(i for i, origin in origins.items() if origin == (seed, local_id))
    scored, selected_targets, daily = {}, [], []
    for pooled_id in sorted(set(selected.values())):
        point = pooled[pooled_id]
        calibration = problem.evaluate(point.vector)
        if not calibration.feasible or not np.allclose(calibration.objectives, point.objectives, rtol=0, atol=1e-12):
            raise RuntimeError("A pooled candidate did not reproduce its recorded calibration costs.")
        validation = problem.evaluate_validation(point.vector, split.validation)
        scored[pooled_id] = {"calibration": _score_description(calibration), "validation": _score_description(validation)}
        table = calibration.parameters.monthly_targets.copy().reset_index()
        table.insert(0, "pooled_id", pooled_id)
        selected_targets.append(table)
        if validation.simulation is not None:
            table = validation.simulation.frame.copy()
            for column in ("storage_m3", "release_m3s"):
                table[f"observed_{column}"] = observed.frame[column]
            table["period"] = "unscored"
            table.loc[table.index < split.calibration.start, "period"] = "spinup"
            for name, period in (("calibration", split.calibration), ("validation", split.validation)):
                table.loc[period.start:period.end, "period"] = name
            table.insert(0, "pooled_id", pooled_id)
            daily.append(table.reset_index().rename(columns={table.index.name or "index": "date"}))
    selected_rows = []
    for role, pooled_id in selected.items():
        seed, local_id = origins[pooled_id]
        for name, score in scored[pooled_id].items():
            selected_rows.append({"role": role, "pooled_id": pooled_id, "seed": seed, "evaluation_id": local_id,
                "period": name, "feasible": score["feasible"], "reason": score["reason"], **score["metrics"],
                "release_pairs": score["paired_counts"]["release"], "storage_pairs": score["paired_counts"]["storage"]})
    report = {
        "experiment": "nsga2_seed_budget_study", "reservoir": config.get("reservoir"),
        "execution_mode": "analyze_only" if analyze_only else "search_and_analysis",
        "study": asdict(settings), "optimization": asdict(optimizer_settings),
        "periods": {"calibration": split.calibration.as_dict(), "validation": split.validation.as_dict()},
        "bounds": bounds.summary(), "independent_seed_count": len(settings.seeds),
        "total_search_evaluations": len(pooled), "feasible_evaluations": sum(point.feasible for point in pooled),
        "new_search_evaluations_this_invocation": sum(len(points_by_seed[seed]) for seed in settings.seeds if not runs[seed]["reused"]),
        "reused_search_evaluations": sum(len(points_by_seed[seed]) for seed in settings.seeds if runs[seed]["reused"]),
        "rejected_evaluations": sum(not point.feasible for point in pooled), "pooled_front_count": len(pooled_front),
        "pooled_hypervolume": hypervolume_2d([point.objectives for point in pooled_front], settings.hypervolume_reference),
        "budget_summary": json_ready(budget_table.reset_index().to_dict(orient="records")),
        "parameter_stability": stability_summaries,
        "selected_candidates": {role: {"pooled_id": i, "seed": origins[i][0], "evaluation_id": origins[i][1]}
                                for role, i in selected.items()},
        "selected_scores": {role: {"pooled_id": i, **scored[i]} for role, i in selected.items()},
        "selection_policy": "Calibration-only pooled nondominated front; objective extrema and normalized distance-to-ideal roles; ties by costs then pooled ID (sorted seed and local evaluation order).",
        "reporting_calibration_evaluations": len(scored), "reporting_validation_evaluations": len(scored),
        "seed_reporting_evaluations": {str(seed): json.loads((Path(runs[seed]["output"]) / "summary.json").read_text(encoding="utf-8"))["reporting_validation_evaluations"] for seed in settings.seeds},
        "seed_search_elapsed_seconds": {str(seed): json.loads((Path(runs[seed]["output"]) / "summary.json").read_text(encoding="utf-8"))["search_elapsed_seconds"] for seed in settings.seeds},
        "seed_runs": {str(seed): {"relative_folder": f"runs/seed_{seed}", "reused": runs[seed]["reused"],
                                  "manifest_sha256": _sha(Path(runs[seed]["output"]) / "seed_manifest.json")} for seed in settings.seeds},
        "source_sha256": input_hashes, "software_versions": _versions(), "configuration": config,
        "analysis_code_sha256": analysis_hashes,
        "study_elapsed_seconds": time.perf_counter() - started,
        "convergence_established": False,
        "notes": [
            "Multiple seeds and nested budgets measure search progress; no rule declares global convergence.",
            "Hypervolume uses one predeclared reference in negative-NSE coordinates; scores and reference are never normalized per run.",
            f"The hypervolume reference {list(settings.hypervolume_reference)} corresponds to NSE {[-value for value in settings.hypervolume_reference]}; only points weakly dominating the configured reference contribute area.",
            "Budget checkpoints are prefixes of each seed's single maximum-budget run, not additional independent experiments.",
            "Intermediate checkpoints end at complete generations, preserving parity with smaller standalone searches.",
            "All rejected attempts count toward search budgets; reporting replays are counted separately.",
            "Parameter spreads are descriptive seed variation among matching representative roles, not confidence intervals or posterior uncertainty.",
            "Balanced choices use each checkpoint's own calibration front ranges; moving normalization can change that role.",
            "Per-seed validation has already been reported; pooled selection reads only calibration trial costs, not validation metrics.",
            "Validation has been inspected during development and is not an untouched final test set.",
            "Fixed physical constants and the inherited 0.92 inflow-factor provenance remain as documented for the baseline.",
            "Cached seed runs are reused only after search inputs/code/versions and all exported artifact hashes are verified.",
        ],
    }
    if _analysis_hashes() != analysis_hashes:
        raise RuntimeError("Analysis code changed during the study; regenerate reporting with the reviewed code.")
    report = json_ready(report)
    output.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(output / "checkpoint_metrics.csv", index=False)
    budget_table.to_csv(output / "budget_summary.csv")
    pd.DataFrame(front_rows).to_csv(output / "checkpoint_fronts.csv", index=False)
    pooled_rows, pooled_vectors = [], []
    for point in pooled_front:
        seed, local_id = origins[point.evaluation_id]
        row = _point_row(point, seed=seed, pooled_id=point.evaluation_id)
        row["evaluation_id"] = local_id
        pooled_rows.append(row)
        pooled_vectors.append({"pooled_id": point.evaluation_id, "seed": seed, "evaluation_id": local_id,
                               **{f"x{i:02d}": value for i, value in enumerate(point.vector)}})
    pd.DataFrame(pooled_rows).to_csv(output / "pooled_front.csv", index=False)
    pd.DataFrame(pooled_vectors).to_csv(output / "pooled_vectors.csv", index=False)
    pd.DataFrame(representative_rows).to_csv(output / "representative_vectors.csv", index=False)
    pd.concat(target_rows, ignore_index=True).to_csv(output / "representative_targets_si.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(output / "parameter_stability.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(output / "selected_metrics.csv", index=False)
    pd.concat(selected_targets, ignore_index=True).to_csv(output / "selected_targets_si.csv", index=False)
    if daily:
        pd.concat(daily, ignore_index=True).to_csv(output / "selected_daily_simulations.csv", index=False)
    else:
        pd.DataFrame(columns=["pooled_id", "date", "period"]).to_csv(output / "selected_daily_simulations.csv", index=False)
    bounds.monthly_bounds.to_csv(output / "bounds_si.csv")
    for name, value in (("summary.json", report), ("resolved_config.json", config)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/trinity_convergence"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--analyze-only", action="store_true", help="Require verified seed artifacts and recompute only study reporting.")
    args = parser.parse_args()
    report = run_calibration_study(args.config or args.repo / "configs/trinity_convergence.json", args.repo,
                                   args.output, workers=args.workers, analyze_only=args.analyze_only)
    table = pd.read_csv(args.output / "selected_metrics.csv")
    print(table.loc[table.role.isin(("generalized_baseline", *ROLES)),
                    ["role", "seed", "period", "release_nse", "storage_nse"]].round(4).to_string(index=False))
    print(f"Study: {report['total_search_evaluations']} evaluations; {report['pooled_front_count']} pooled Pareto candidates")
    print(f"Artifacts: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
