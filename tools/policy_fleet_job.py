"""One policy-aware, checkpointed reservoir job; no work at import time."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dztr.calibration.bounds import CalibrationBounds
from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.checkpoint import load_optimizer_checkpoint, save_optimizer_checkpoint
from dztr.calibration.optimizer import NSGA2Settings, optimize_nsga2
from dztr.calibration.periods import DatePeriod
from dztr.calibration.root_policy import RootPolicyCalibrationObjective
from dztr.calibration.sampled import SampledSplit, select_sampled_period, target_observations
from dztr.calibration.selection import select_pareto_representatives
from dztr.integrations.collection import _verified_inputs, load_collection
from dztr.integrations.world_reservoirs import _forcing
from dztr.parameterization import estimate_generalized_parameters
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters
from dztr.simulation.sampled import SamplingSettings


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def initialize(objective, candidates, bounds, template):
    """At most two declared checks, no random search or held-out evaluation."""
    ledger = []
    for label, targets in candidates:
        try:
            parameters = ReservoirParameters(targets, **{k: getattr(template, k) for k in (
                "capacity_m3", "initial_storage_m3", "initial_release_m3s", "channel_capacity_m3s", "dead_storage_fraction", "inflow_correction", "flood_mode")})
            vector = parameters_to_vector(parameters)
            bounds.validate(vector, template)
        except ValueError as error:
            ledger.append({"candidate": label, "model_called": False, "feasible": False, "reason": str(error)})
            continue
        assessed = objective.evaluate(vector)
        ledger.append({"candidate": label, "model_called": True, "feasible": assessed.feasible, "reason": assessed.reason})
        if assessed.feasible:
            return vector, assessed, ledger
    return None, None, ledger


def identity_for(protocol, job, config, hashes):
    return {"kind": "policy_fleet_fixed_budget", "reservoir_id": job["reservoir_id"],
        "seed": job["seed"], "root_policy": job["root_policy"],
        "root_tie_rule": "minimum_storage", "overflow": "conservative",
        "initialization_protocol": "generalized_then_calibration_box_critical_gap_witness_v1",
        "input_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in hashes.items()},
        "code_sha256": protocol["code_sha256"], "runner_sha256": sha(Path(__file__)),
        "protocol_identity": protocol, "sampling": config["sampling"], "periods": config["periods"],
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "settings": {k: job[k] for k in ("seed", "population_size", "evaluation_budget")}}


def safe_validation(objective, vector):
    # Forcing/score support can fail before the model produces a candidate.
    try:
        result = objective.evaluate_validation(vector)
        return result, {"feasible": result.feasible, "reason": result.reason, "metrics": result.metrics}
    except ValueError as error:
        return None, {"feasible": False, "reason": str(error), "metrics": {}}


def trajectory(path, result, observations, split):
    if result is None or not result.feasible:
        return
    frame = result.simulation.frame.copy()
    for name in ("storage_m3", "release_m3s"):
        frame["observed_" + name] = observations.frame[name]
    frame["period"] = "warmup"
    frame.loc[split.calibration.start:split.calibration.end, "period"] = "calibration"
    frame.loc[split.validation.start:split.validation.end, "period"] = "validation"
    frame.to_csv(path, float_format="%.17g")


def run(out, job, *, resume=False):
    from policy_fleet import checked
    protocol = checked(out)
    folder = out / "pilots" / job["reservoir_id"] / f"seed_{job['seed']}"
    if resume:
        if not (folder / "checkpoint.npz").exists() or (folder / "fleet_completion.json").exists():
            raise ValueError("Resume requires an incomplete job with a saved checkpoint.")
    else:
        folder.mkdir(parents=True, exist_ok=False)
    root, manifest, _, _ = load_collection(ROOT / "data/reservoirs/manifest.json")
    entry = next(r for r in manifest["reservoirs"] if r["reservoir_id"] == job["reservoir_id"])
    _, config, observations, hashes = _verified_inputs(root, entry)
    identity = identity_for(protocol, job, config, hashes)
    # Identity mismatch is rejected before replay or other model calls.
    state = load_optimizer_checkpoint(folder / "checkpoint.npz", identity) if resume else None
    admission = read(out / "admission.json")
    record = next(r for r in admission["reservoirs"] if r["reservoir_id"] == job["reservoir_id"])
    audit = record["bounds_audit"]
    sampling = SamplingSettings(**config["sampling"])
    dates = config["periods"]
    split = SampledSplit(dates["simulation_start"], DatePeriod(**dates["calibration"]), DatePeriod(**dates["validation"]), dates["warmup_rows"])
    fitting = target_observations(select_sampled_period(observations, split.calibration, sampling))
    estimates = estimate_generalized_parameters(fitting, config["target_percentiles"], channel_capacity_m3s=config["model"]["channel_capacity_m3s"], quantile_method=config["quantile_method"])
    candidates = [("generalized", estimates.monthly_targets)]
    witness = audit["critical_gap_box"]
    if witness["feasible"]:
        candidates.append(("critical_gap_witness", pd.DataFrame(witness["monthly_targets_si"], index=pd.Index(range(1, 13), name="month"), columns=MONTHLY_TARGET_COLUMNS)))
    # Template validity is independent of whether the generalized estimate is
    # a feasible trajectory. A separate ordered witness supplies its structure.
    template = ReservoirParameters(pd.DataFrame(audit["ordered_box"]["monthly_targets_si"], index=pd.Index(range(1, 13), name="month"), columns=MONTHLY_TARGET_COLUMNS), **config["model"])
    bound_frame = pd.DataFrame(np.c_[audit["lower_si"], audit["upper_si"]], index=pd.Index(range(1, 13), name="month"), columns=[side + "_" + name for side in ("lower", "upper") for name in MONTHLY_TARGET_COLUMNS])
    bounds = CalibrationBounds(bound_frame, template.capacity_m3, template.dead_storage_fraction, split.calibration, config["quantile_method"])
    objective = RootPolicyCalibrationObjective(_forcing(observations, config["inflow_preprocessing"]), split, template, sampling=sampling, policy=job["root_policy"], bounds=bounds, atol_storage_m3=config["atol_storage_m3"])
    if resume:
        vector = state.baseline_vector.copy()
        baseline = objective.evaluate(vector)
        if not baseline.feasible:
            raise ValueError("Saved initialization no longer replays feasibly.")
        initialization = read(folder / "initialization.json")
        resume_checks = 1
    else:
        vector, baseline, ledger = initialize(objective, candidates, bounds, template)
        initialization = {"checks": ledger, "simulation_calls": sum(r["model_called"] for r in ledger), "selected_candidate": next((r["candidate"] for r in ledger if r["feasible"]), None)}
        write(folder / "initialization.json", initialization)
        if vector is None:
            raise ValueError("Declared initialization checks found no feasible trajectory; no optimization started.")
        resume_checks = 0
    write(folder / "checkpoint_identity.json", identity)
    write(folder / "resolved_config.json", {**config, "declared_root_policy": job["root_policy"]})
    estimates.monthly_targets.to_csv(folder / "calibration_targets_si.csv")
    bound_frame.to_csv(folder / "calibration_bounds_si.csv")
    settings = NSGA2Settings(**{k: job[k] for k in ("seed", "population_size", "evaluation_budget")})
    result = optimize_nsga2(objective, vector, settings, bounds=bounds, resume_state=state,
        checkpoint_callback=lambda current: save_optimizer_checkpoint(folder / "checkpoint.npz", current, identity))
    trials = [{"evaluation_id": p.evaluation_id, "generation": p.generation, "feasible": p.feasible, "release_cost": p.objectives[0], "storage_cost": p.objectives[1], "reason": p.reason} for p in result.evaluated]
    pd.DataFrame(trials).to_csv(folder / "trials.csv", index=False)
    pd.DataFrame([p.vector for p in result.evaluated], index=pd.Index([p.evaluation_id for p in result.evaluated], name="evaluation_id")).to_csv(folder / "vectors.csv", float_format="%.17g")
    pd.DataFrame([asdict(g) for g in result.history]).to_csv(folder / "generation_history.csv", index=False)
    pd.DataFrame([{**trials[p.evaluation_id], "release_nse": -p.objectives[0], "storage_nse": -p.objectives[1]} for p in result.front]).to_csv(folder / "pareto_front.csv", index=False)
    representatives = select_pareto_representatives(result.front)
    write(folder / "selection.json", {"representatives": representatives, "validation_used": False, "written_before_validation": True})
    assessed, baseline_validation = safe_validation(objective, vector)
    trajectory(folder / "baseline_simulation.csv", assessed if assessed is not None and assessed.feasible else baseline, observations, split)
    validation_rows, selected_rows, unique = [], [], {}
    for index in sorted(set(representatives.values())):
        calibrated = objective.evaluate(result.evaluated[index].vector)
        if not calibrated.feasible:
            raise ValueError("Selected calibration replay failed.")
        np.testing.assert_array_equal(calibrated.objectives, result.evaluated[index].objectives)
        assessed, report = safe_validation(objective, result.evaluated[index].vector)
        trajectory(folder / f"selected_{index}_simulation.csv", assessed if assessed is not None and assessed.feasible else calibrated, observations, split)
        unique[index] = (calibrated, report)
        validation_rows.append({"evaluation_id": index, **report})
    for role, index in representatives.items():
        calibrated, report = unique[index]
        selected_rows.append({"role": role, "evaluation_id": index, "calibration_release_nse": calibrated.metrics["release_nse"], "calibration_storage_nse": calibrated.metrics["storage_nse"], "validation_feasible": report["feasible"], "validation_reason": report["reason"], "validation_release_nse": report["metrics"].get("release_nse"), "validation_storage_nse": report["metrics"].get("storage_nse")})
    write(folder / "selected_validation.json", validation_rows)
    checked(out)
    record = {"reservoir_id": job["reservoir_id"], "seed": job["seed"], "root_policy": job["root_policy"], "status": "completed", "settings": asdict(settings), "search_evaluations": len(result.evaluated), "new_search_evaluations_this_invocation": len(result.evaluated) - (len(state.evaluated) if state is not None else 0), "representatives": selected_rows, "baseline_calibration": baseline.metrics, "baseline_validation": baseline_validation, "initialization": initialization, "resume_calibration_checks": resume_checks, "representative_calibration_replays": len(unique), "validation_attempts": 1 + len(unique), "selection_uses_validation": False, "convergence_established": False}
    write(folder / "result.json", record)
    write(folder / "fleet_completion.json", {"status": "passed", "artifact_sha256": {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name not in ("fleet_failure.json", "fleet_completion.json")}})
    return record
