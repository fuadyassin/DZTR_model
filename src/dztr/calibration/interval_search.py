"""Fixed-budget, calibration-only search under a reviewed daily input policy.

Validation is deliberately absent from ``run_interval_seed``. Callers freeze
calibration representatives first, then explicitly use the loaded objective's
validation method in a separate reporting stage. Stopping diagnostics describe
the saved trial stream; they never shorten this pilot's declared budget.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from numbers import Integral
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from dztr import __version__
from dztr.data.intervals import DailyIntervalPolicy, prepare_daily_intervals
from dztr.schemas import ReservoirParameters
from dztr.simulation.intervals import IntervalSimulationSettings

from .baseline import json_ready
from .candidates import parameters_to_vector
from .checkpoint import save_optimizer_checkpoint
from .convergence import checkpoint_metrics
from .intervals import IntervalCalibrationObjective, estimate_interval_bounds, fit_interval_targets
from .optimizer import EvaluatedCandidate, NSGA2Settings, optimize_nsga2
from .periods import DatePeriod
from .selection import select_pareto_representatives
from .stopping import StoppingPolicy, assess_stopping


ARTIFACT_NAMES = (
    "evaluations.csv", "evaluated_vectors.csv", "pareto_front.csv", "pareto_vectors.csv",
    "history.csv", "checkpoint_metrics.csv", "stopping_history.csv",
    "target_sample_support.csv", "monthly_targets_si.csv", "monthly_bounds_si.csv", "summary.json",
    "optimizer_checkpoint.npz",
)
DEFAULT_CHECKPOINTS = (16, 32, 64, 96, 128)
HYPERVOLUME_REFERENCE = (1.0, 1.0)
HYPERVOLUME_NORMALIZATION_AREA = 4.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_hashes() -> dict[str, str]:
    code_root = Path(__file__).resolve().parents[1]
    return {path.relative_to(code_root).as_posix(): _sha256(path)
            for path in sorted(code_root.rglob("*.py"))}


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    return value


def _number_is(value, expected) -> bool:
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float))
            and np.isfinite(value) and value == expected)


def load_interval_problem(policy_path: str | Path):
    """Load one explicit reviewed SI policy and fit calibration-only targets.

    Return ``(problem, baseline_vector, validation_period, context)``. Context
    supplies ``fit``, ``preparation``, ``bounds``, source/config/policy paths,
    configuration, consumed-file hashes and declared periods for independent
    reporting. No validation score is computed, and all initialization offsets
    are fixed at zero. Model constants and the source-quality policy are never
    search coordinates.
    """
    policy_path = Path(policy_path).resolve()
    policy_json = _read_object(policy_path)
    if policy_json.get("schema_version") != 1 or policy_json.get("kind") != "reservoir_interval_review_policy":
        raise ValueError("Expected a version-1 reservoir_interval_review_policy.")
    if not isinstance(policy_json.get("policy_id"), str) or not policy_json["policy_id"].strip():
        raise ValueError("A reviewed policy requires a nonempty policy_id.")
    temporal = policy_json["interval_inputs"]
    if (temporal.get("flow_alignment") != "interval_mean_end"
            or not _number_is(temporal.get("nominal_step_seconds"), 86400)):
        raise ValueError("Only declared end-labelled nominal 86400-second daily means are supported.")
    if not _number_is(temporal.get("inflow_offset_m3s", 0), 0):
        raise ValueError("Policy inflow offsets require a separate reviewed adapter.")
    if temporal.get("additional_loss_model") is not None:
        raise ValueError("Additional loss models are not supported by this interval pilot.")
    config_path = (policy_path.parent / policy_json["reservoir_config_file"]).resolve()
    source_path = (policy_path.parent / policy_json["observations_file"]).resolve()
    config = _read_object(config_path)
    if config["sampling"].get("frequency") != "daily":
        raise ValueError("A reviewed daily reservoir configuration is required.")
    if config.get("simulation_supported", True) is not True:
        raise ValueError("Unsupported source integration requires a separate reviewed adapter.")
    if (not _number_is(config["sampling"].get("fixed_step_days", 1), 1)
            or config["sampling"].get("month_clock", "observed") != "observed"):
        raise ValueError("Only nominal daily durations and observed calendar months are supported.")
    if config.get("units") != {"storage": "m3", "flows": "m3/s"}:
        raise ValueError("Packaged inputs must explicitly declare storage m3 and flows m3/s.")
    configured_source = (config_path.parent / config["observations_file"]).resolve()
    if configured_source != source_path:
        raise ValueError("Reviewed policy and reservoir configuration refer to different observations.")
    preprocessing = config.get("inflow_preprocessing", {})
    if (set(preprocessing) - {"factor", "offset_m3s"}
            or not _number_is(preprocessing.get("factor", 1), 1)
            or not _number_is(preprocessing.get("offset_m3s", 0), 0)):
        raise ValueError("Unsupported forcing preprocessing; review a separate adapter before searching.")
    if config.get("additional_loss_model") is not None or config.get("loss_model") is not None:
        raise ValueError("Additional loss models are not supported by this interval pilot.")
    model = config["model"]
    if set(model) - {"capacity_m3", "initial_storage_m3", "initial_release_m3s", "channel_capacity_m3s",
                     "dead_storage_fraction", "inflow_correction", "flood_mode"}:
        raise ValueError("Unsupported model constants or extra loss terms require a separately reviewed adapter.")
    if temporal.get("flow_semantics") == "computed_net_inflow":
        if (model.get("flood_mode", "storage") != "storage"
                or not _number_is(model.get("inflow_correction", 1), 1)
                or not _number_is(temporal.get("inflow_correction", 1), 1)):
            raise ValueError("Computed net inflow requires storage-only flood rules and unit inflow correction.")
    elif not _number_is(temporal.get("inflow_correction", model.get("inflow_correction", 1)),
                        model.get("inflow_correction", 1)):
        raise ValueError("Policy and model inflow corrections disagree.")
    future = policy_json.get("future_simulation", {})
    if future.get("flood_mode", model.get("flood_mode", "storage")) != model.get("flood_mode", "storage"):
        raise ValueError("Policy and model flood modes disagree.")
    if future.get("carry_state_across_protocol_period_boundaries", True) is not True:
        raise ValueError("Clean phase boundaries must carry simulated state.")
    if not _number_is(future.get("additional_restart_spinup_days", 0), 0):
        raise ValueError("Additional restart spinup requires a separately implemented policy.")
    source_hash = _sha256(source_path)
    if source_hash != policy_json["observations_sha256"]:
        raise ValueError("Observation SHA256 does not match the reviewed policy.")
    if config.get("observations_sha256", source_hash) != source_hash:
        raise ValueError("Observation SHA256 does not match the reservoir configuration.")
    source = pd.read_csv(source_path, parse_dates=["date"], index_col="date", float_precision="round_trip")
    calibration = DatePeriod(**config["periods"]["calibration"])
    validation = DatePeriod(**config["periods"]["validation"])
    if validation.start <= calibration.end:
        raise ValueError("Validation must follow calibration without overlap.")
    simulation_start = pd.Timestamp(config["periods"].get("simulation_start", source.index[0]))
    DatePeriod(simulation_start, calibration.end)
    if simulation_start > calibration.start:
        raise ValueError("Simulation start must precede or equal calibration start.")
    if source.index.min() > simulation_start or source.index.max() < validation.end:
        raise ValueError("Packaged calendar must cover declared simulation and validation boundaries.")
    labels = pd.Series("unscored", index=source.index)
    labels.loc[(labels.index >= simulation_start) & (labels.index < calibration.start)] = "warmup"
    labels.loc[calibration.start:calibration.end] = "calibration"
    labels.loc[validation.start:validation.end] = "validation"
    policy = DailyIntervalPolicy(flow_semantics=temporal["flow_semantics"],
        quarantined_interval_end_dates=policy_json["quarantine"]["interval_end_dates"])
    preparation = prepare_daily_intervals(source, policy=policy, period_labels=labels)
    method = config.get("quantile_method", "hazen")
    fit = fit_interval_targets(preparation, calibration,
        channel_capacity_m3s=model.get("channel_capacity_m3s"),
        target_percentiles=tuple(config.get("target_percentiles", (10, 45, 85))), quantile_method=method)
    physical = {name: model[name] for name in (
        "capacity_m3", "initial_storage_m3", "initial_release_m3s", "dead_storage_fraction",
        "inflow_correction", "flood_mode") if name in model}
    template = ReservoirParameters.from_generalized(fit.estimates, **physical)
    bounds = estimate_interval_bounds(preparation, calibration, template, quantile_method=method)
    settings = IntervalSimulationSettings(root_policy="reject", initial_storage_offset_capacity_fraction=0.0)
    problem = IntervalCalibrationObjective(preparation, calibration, template,
        simulation_start=simulation_start, settings=settings, bounds=bounds,
        atol_storage_m3=config.get("atol_storage_m3", 1e-4))
    context = {
        "fit": fit, "preparation": preparation, "bounds": bounds, "source_path": source_path,
        "config_path": config_path, "policy_path": policy_path, "config": config, "policy": policy_json,
        "consumed_files_sha256": {str(path): _sha256(path) for path in (source_path, config_path, policy_path)},
        "periods": {"simulation_start": simulation_start.date().isoformat(),
                    "calibration": calibration.as_dict(), "validation": validation.as_dict()},
    }
    return problem, parameters_to_vector(template), validation, context


def _trial_row(point: EvaluatedCandidate) -> dict:
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


def _checkpoints(values, budget: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("checkpoints must be positive unique integer trial counts.")
    values = tuple(values)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           or not 1 <= value <= budget for value in values):
        raise ValueError("Every checkpoint must be a positive integer within the declared budget.")
    if len(values) != len(set(values)):
        raise ValueError("Checkpoint trial counts must be unique.")
    return tuple(sorted(int(value) for value in values))


class _RecordedObjective:
    """Preserve failed trial receipts even if the underlying archive is empty."""

    def __init__(self, problem, population_size):
        self.problem = problem
        self.bounds = problem.bounds
        self.population_size = population_size
        self.records = []

    def evaluate(self, vector):
        evaluation = self.problem.evaluate(vector)
        costs = np.asarray(evaluation.objectives)
        if costs.shape != (2,) or np.iscomplexobj(costs):
            return evaluation  # The optimizer reports malformed objective contracts.
        feasible = bool(evaluation.feasible) and bool(np.isfinite(costs).all())
        index = len(self.records)
        saved = np.asarray(vector, dtype=float).copy()
        saved.setflags(write=False)
        self.records.append(EvaluatedCandidate(index, index // self.population_size, saved,
            tuple(float(value) for value in costs) if feasible else (np.inf, np.inf),
            feasible, None if feasible else str(evaluation.reason or "Nonfinite or infeasible objectives.")))
        return evaluation


def run_interval_seed(
    policy_path: str | Path, output_dir: str | Path, *, settings: NSGA2Settings | None = None,
    checkpoints=DEFAULT_CHECKPOINTS, progress_callback=None, stopping_policy: StoppingPolicy | None = None,
) -> dict:
    """Search and save one seed without ever calling held-out validation.

    The baseline is evaluation zero. Every evaluated unique vector, including
    rejection, consumes its declared budget. Duplicate exhaustion is reported
    explicitly. Output directories must not already exist. Diagnostic stopping
    suggestions do not alter the trial stream, and never prove convergence.
    """
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh output directory; existing results are preserved.")
    settings = NSGA2Settings(evaluation_budget=128) if settings is None else settings
    if not isinstance(settings, NSGA2Settings):
        raise TypeError("settings must be NSGA2Settings.")
    checkpoints = _checkpoints(checkpoints, settings.evaluation_budget)
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")
    if stopping_policy is None and settings.population_size == 16 and settings.evaluation_budget == 128:
        stopping_policy = StoppingPolicy(population_size=16, check_interval=16, window_evaluations=64,
            minimum_evaluations=96, patience=2, max_evaluations=128,
            normalized_hv_tolerance=.0025, front_tolerance=.01, extrema_tolerance=.005)
    if stopping_policy is not None:
        if not isinstance(stopping_policy, StoppingPolicy):
            raise TypeError("stopping_policy must be StoppingPolicy or None.")
        if stopping_policy.population_size != settings.population_size:
            raise ValueError("Stopping policy and search population sizes must match.")
        if stopping_policy.max_evaluations != settings.evaluation_budget:
            raise ValueError("Stopping diagnostic cap must match the declared search budget.")
        if stopping_policy.hypervolume_reference != HYPERVOLUME_REFERENCE:
            raise ValueError("This pilot uses the predeclared hypervolume reference (1, 1).")
    problem, baseline, _, context = load_interval_problem(policy_path)
    code_hashes = _module_hashes()
    recorded = _RecordedObjective(problem, settings.population_size)
    history = []
    complete_states = []
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "dztr": __version__}
    nonbudget_settings = {name: value for name, value in asdict(settings).items()
                          if name != "evaluation_budget"}
    checkpoint_identity = {
        "policy_id": context["policy"]["policy_id"],
        "consumed_files_sha256": context["consumed_files_sha256"], "code_sha256": code_hashes,
        "software_versions": versions, "nonbudget_settings": nonbudget_settings,
        "root_policy": "reject", "initial_storage_offset_capacity_fraction": 0.0,
        "periods": context["periods"], "bounds": context["bounds"].summary(),
    }

    def progress(record):
        history.append(record)
        if progress_callback is not None:
            progress_callback(record)

    started = time.perf_counter()
    try:
        result = optimize_nsga2(recorded, baseline, settings, progress_callback=progress,
                               checkpoint_callback=lambda state: complete_states.__setitem__(slice(None), [state]))
    except ValueError as exc:
        if (not str(exc).startswith("No feasible candidate was found")
                or not recorded.records or any(point.feasible for point in recorded.records)):
            raise
        evaluated, front, final_population = tuple(recorded.records), (), ()
        termination = "no_feasible_candidate"
    else:
        evaluated, front, final_population = result.evaluated, result.front, result.final_population
        termination = result.termination
    elapsed = time.perf_counter() - started
    for name, expected in context["consumed_files_sha256"].items():
        if _sha256(Path(name)) != expected:
            raise RuntimeError(f"Consumed input changed during search: {name}")
    if _module_hashes() != code_hashes:
        raise RuntimeError("Model module hashes changed during search; this trial stream is not reproducible.")
    if len(evaluated) != len(recorded.records):
        raise RuntimeError("Search budget accounting does not match recorded objective attempts.")
    completed_checkpoints = tuple(value for value in checkpoints if value <= len(evaluated))
    diagnostics = []
    for checkpoint in completed_checkpoints:
        row = checkpoint_metrics(evaluated, checkpoint, reference=HYPERVOLUME_REFERENCE)
        row["normalized_hypervolume"] = row["hypervolume"] / HYPERVOLUME_NORMALIZATION_AREA
        diagnostics.append(row)
    assessment = assess_stopping(evaluated, stopping_policy) if stopping_policy is not None else None
    rejected = [point for point in evaluated if not point.feasible]
    representatives = select_pareto_representatives(front) if front else {}
    report = json_ready({
        "experiment": "reviewed_interval_nsga2_seed_pilot", "policy_id": context["policy"]["policy_id"],
        "reservoir": context["config"].get("reservoir"), "conditional_hindcast": True,
        "validation_label": "segmented conditional hindcast", "optimization": asdict(settings),
        "objective_order": ["negative_release_nse", "negative_storage_nse"], "objective_direction": "minimize",
        "root_policy": "reject", "initial_storage_offset_capacity_fraction": 0.0,
        "periods": context["periods"], "bounds": context["bounds"].summary(),
        "target_percentiles": list(context["fit"].estimates.target_percentiles),
        "quantile_method": context["fit"].estimates.quantile_method,
        "search_evaluations": len(evaluated), "baseline_evaluation_id": 0, "baseline_included_in_budget": True,
        "feasible_evaluations": len(evaluated) - len(rejected), "rejected_evaluations": len(rejected),
        "rejection_reasons": dict(Counter(point.reason for point in rejected)),
        "pareto_count": len(front), "selected_candidate_ids": {"generalized_baseline": 0, **representatives},
        "final_population_ids": [point.evaluation_id for point in final_population], "termination": termination,
        "search_elapsed_seconds": elapsed, "reporting_calibration_evaluations": 0, "reporting_validation_evaluations": 0,
        "requested_checkpoints": list(checkpoints), "completed_checkpoints": list(completed_checkpoints),
        "hypervolume_reference": list(HYPERVOLUME_REFERENCE),
        "hypervolume_normalization_area": HYPERVOLUME_NORMALIZATION_AREA,
        "stopping_policy": asdict(stopping_policy) if stopping_policy is not None else None,
        "stopping_assessment": asdict(assessment) if assessment is not None else None,
        "stopping_controls_search": False, "optimizer_convergence_established": False,
        "selection_policy": "Representatives are fixed from the complete feasible calibration Pareto archive before any validation call.",
        "consumed_files_sha256": context["consumed_files_sha256"], "code_sha256": code_hashes,
        "software_versions": versions,
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_evaluations": len(complete_states[0].evaluated) if complete_states else None,
        "checkpoint_saved": bool(complete_states),
        "checkpoint_note": "The last complete generation is saved after the seed finishes; partial trailing trials remain in CSV only.",
        "random_generator": "numpy.random.Generator / PCG64",
        "notes": [
            "Targets and empirical bounds use independently eligible calibration storage and mean-release observations only.",
            "All attempted vectors, including rejected roots and trial-zero baseline, consume the search budget.",
            "Only monthly targets are optimized; input policy, physical constants and provisional nominal anchors are fixed.",
            "Computed net inflow and observed storage/release are statistically dependent; this is a conditional hindcast.",
            "Stopping thresholds are provisional diagnostics, not a validated reservoir evaluation-budget recommendation.",
            "Budget exhaustion or a small change in recent objectives does not establish global convergence or parameter identifiability.",
            "No selected-candidate replay or validation simulation is performed by this seed function.",
            "The complete packaged daily frame is read and structurally preflighted; held-out values do not enter fitting, objectives, ranking or selection.",
        ],
    })
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame([_trial_row(point) for point in evaluated]).to_csv(output / "evaluations.csv", index=False)
    _vectors_frame(evaluated).to_csv(output / "evaluated_vectors.csv")
    pd.DataFrame([_trial_row(point) for point in front], columns=list(_trial_row(evaluated[0]))).to_csv(output / "pareto_front.csv", index=False)
    _vectors_frame(front).to_csv(output / "pareto_vectors.csv")
    pd.DataFrame([asdict(record) for record in history]).to_csv(output / "history.csv", index=False)
    checkpoint_columns = ["evaluations", "feasible_count", "rejected_count", "front_size", "hypervolume",
        "reference", "front_points_outside_reference", "max_release_nse", "max_storage_nse",
        "representative_ids", "balanced_release_nse", "balanced_storage_nse", "normalized_hypervolume"]
    pd.DataFrame(diagnostics, columns=checkpoint_columns).to_csv(output / "checkpoint_metrics.csv", index=False)
    pd.DataFrame(list(assessment.history) if assessment is not None else [],
                 columns=None if assessment is not None and assessment.history else ["evaluations", "status"]).to_csv(output / "stopping_history.csv", index=False)
    context["fit"].support.to_csv(output / "target_sample_support.csv", index_label="month")
    context["fit"].estimates.monthly_targets.to_csv(output / "monthly_targets_si.csv")
    context["bounds"].monthly_bounds.to_csv(output / "monthly_bounds_si.csv")
    if complete_states:
        save_optimizer_checkpoint(output / "optimizer_checkpoint.npz", complete_states[0], checkpoint_identity)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
