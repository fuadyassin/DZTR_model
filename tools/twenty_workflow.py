"""Controlled512-to1024 checkpoint continuation, preserving the completed fleet."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from policy_fleet import checked as check_parent
from policy_fleet_job import ROOT, read, sha, write, trajectory, safe_validation
from dztr.calibration.bounds import CalibrationBounds
from dztr.calibration.checkpoint import load_optimizer_checkpoint, save_optimizer_checkpoint
from dztr.calibration.optimizer import optimize_nsga2
from dztr.calibration.periods import DatePeriod
from dztr.calibration.root_policy import RootPolicyCalibrationObjective
from dztr.calibration.sampled import SampledSplit
from dztr.calibration.selection import select_pareto_representatives
from dztr.integrations.collection import _verified_inputs, load_collection
from dztr.integrations.world_reservoirs import _forcing
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters
from dztr.simulation.sampled import SamplingSettings

PARENT = ROOT / "outputs/policy_fleet_20261012"
OUT = ROOT / "outputs/twenty_reservoir_workflow_20261012"


def objective_for(folder, state):
    config = read(folder / "resolved_config.json")
    root, manifest, _, _ = load_collection(ROOT / "data/reservoirs/manifest.json")
    entry = next(r for r in manifest["reservoirs"] if r["reservoir_id"] == config["reservoir"]["id"])
    _, verified, observations, _ = _verified_inputs(root, entry)
    assert all(config[k] == verified[k] for k in ("model", "sampling", "periods", "inflow_preprocessing"))
    targets = state.baseline_vector.reshape(6, 12).T.copy()
    targets[:, :3] *= config["model"]["capacity_m3"]
    template = ReservoirParameters(pd.DataFrame(targets, columns=MONTHLY_TARGET_COLUMNS, index=pd.Index(range(1, 13), name="month")), **config["model"])
    dates = config["periods"]
    split = SampledSplit(dates["simulation_start"], DatePeriod(**dates["calibration"]), DatePeriod(**dates["validation"]), dates["warmup_rows"])
    bounds = CalibrationBounds(pd.read_csv(folder / "calibration_bounds_si.csv", index_col="month", float_precision="round_trip"), template.capacity_m3, template.dead_storage_fraction, split.calibration, config["quantile_method"])
    np.testing.assert_array_equal(bounds.lower_vector, state.lower_vector)
    np.testing.assert_array_equal(bounds.upper_vector, state.upper_vector)
    objective = RootPolicyCalibrationObjective(_forcing(observations, config["inflow_preprocessing"]), split, template, sampling=SamplingSettings(**config["sampling"]), policy=config["declared_root_policy"], bounds=bounds, atol_storage_m3=config["atol_storage_m3"])
    return objective, bounds, observations, split


def prepare():
    check_parent(PARENT)
    if OUT.exists():
        raise ValueError("Existing workflow is preserved.")
    final = read(PARENT / "execution_verification.json")
    if final["status"] != "passed":
        raise ValueError("Parent execution must be verified.")
    OUT.mkdir()
    shutil.copyfile(PARENT / "reservoir_results.csv", OUT / "other_reservoirs_and_prior_results.csv")
    write(OUT / "selection_rule.json", {"rule": "Pooled three-seed calibration nondominated front; equal-weight range-normalized Euclidean distance from calibration ideal.", "tie_break": ["release_cost", "storage_cost", "seed", "evaluation_id"], "validation_used": False, "scope": "twenty admitted reservoirs only", "freeze_before_continuation": True})
    jobs = read(PARENT / "job_plan.json")["jobs"]
    hashes = {}
    for job in jobs:
        folder = PARENT / "pilots" / job["reservoir_id"] / f"seed_{job['seed']}"
        receipt = read(folder / "fleet_completion.json")
        for name, digest in receipt["artifact_sha256"].items():
            if sha(folder / name) != digest:
                raise ValueError("Parent job changed.")
        for name in ("checkpoint.npz", "checkpoint_identity.json", "resolved_config.json", "calibration_bounds_si.csv", "result.json"):
            hashes[(folder / name).relative_to(ROOT).as_posix()] = sha(folder / name)
    write(OUT / "protocol.json", {"jobs": jobs, "parent_artifact_sha256": hashes, "parent_execution_sha256": sha(PARENT / "execution_verification.json"), "runner_sha256": sha(Path(__file__)), "selection_rule_sha256": sha(OUT / "selection_rule.json"), "target_budget": 1024, "new_search_budget": 30720, "workers": 3, "code_sha256": read(PARENT / "protocol.json")["code_sha256"], "population_seeds_bounds_policies_unchanged": True})
    write(OUT / "preparation_identity.json", {"protocol_sha256": sha(OUT / "protocol.json")})


def checked():
    check_parent(PARENT)
    protocol = read(OUT / "protocol.json")
    if sha(OUT / "protocol.json") != read(OUT / "preparation_identity.json")["protocol_sha256"] or sha(Path(__file__)) != protocol["runner_sha256"] or sha(OUT / "selection_rule.json") != protocol["selection_rule_sha256"]:
        raise ValueError("Continuation protocol or runner changed.")
    return protocol


def run_one(identifier, seed, resume=False):
    protocol = checked()
    job = next((j for j in protocol["jobs"] if j["reservoir_id"] == identifier and j["seed"] == seed), None)
    if job is None:
        raise ValueError("Reservoir/seed outside frozen plan.")
    source = PARENT / "pilots" / identifier / f"seed_{seed}"
    for name in ("checkpoint.npz", "checkpoint_identity.json", "resolved_config.json", "calibration_bounds_si.csv", "result.json"):
        if sha(source / name) != protocol["parent_artifact_sha256"][(source / name).relative_to(ROOT).as_posix()]:
            raise ValueError("Parent checkpoint/input changed.")
    parent_identity = read(source / "checkpoint_identity.json")
    parent_state = load_optimizer_checkpoint(source / "checkpoint.npz", parent_identity)
    assert len(parent_state.evaluated) == 512 and parent_state.settings.seed == seed
    folder = OUT / "pilots" / identifier / f"seed_{seed}"
    identity = {"kind": "verified_policy_fleet_continuation", "parent_identity": parent_identity, "parent_checkpoint_sha256": sha(source / "checkpoint.npz"), "continuation_protocol_sha256": sha(OUT / "protocol.json"), "root_policy": job["root_policy"], "seed": seed, "target_budget": 1024}
    if resume:
        if (folder / "fleet_completion.json").exists():
            raise ValueError("Completed continuation preserved.")
        state = load_optimizer_checkpoint(folder / "checkpoint.npz", identity)
    else:
        folder.mkdir(parents=True, exist_ok=False)
        state = parent_state
        for name in ("resolved_config.json", "calibration_bounds_si.csv", "calibration_targets_si.csv", "baseline_simulation.csv", "initialization.json"):
            shutil.copyfile(source / name, folder / name)
    objective, bounds, observations, split = objective_for(folder, parent_state)
    write(folder / "checkpoint_identity.json", identity)
    result = optimize_nsga2(objective, parent_state.baseline_vector, replace(parent_state.settings, evaluation_budget=1024), bounds=bounds, resume_state=state, checkpoint_callback=lambda current: save_optimizer_checkpoint(folder / "checkpoint.npz", current, identity))
    np.testing.assert_array_equal([p.vector for p in result.evaluated[:512]], [p.vector for p in parent_state.evaluated])
    np.testing.assert_array_equal([p.objectives for p in result.evaluated[:512]], [p.objectives for p in parent_state.evaluated])
    if len(result.evaluated) != 1024:
        raise ValueError("Continuation did not reach the declared checkpoint.")
    trials = [{"evaluation_id": p.evaluation_id, "generation": p.generation, "feasible": p.feasible, "release_cost": p.objectives[0], "storage_cost": p.objectives[1], "reason": p.reason} for p in result.evaluated]
    pd.DataFrame(trials).to_csv(folder / "trials.csv", index=False)
    pd.DataFrame([p.vector for p in result.evaluated], index=pd.Index([p.evaluation_id for p in result.evaluated], name="evaluation_id")).to_csv(folder / "vectors.csv", float_format="%.17g")
    pd.DataFrame([asdict(g) for g in result.history]).to_csv(folder / "generation_history.csv", index=False)
    pd.DataFrame([{**trials[p.evaluation_id], "release_nse": -p.objectives[0], "storage_nse": -p.objectives[1]} for p in result.front]).to_csv(folder / "pareto_front.csv", index=False)
    roles = select_pareto_representatives(result.front)
    write(folder / "selection.json", {"representatives": roles, "validation_used": False, "written_before_validation": True})
    assessed = {}
    for index in sorted(set(roles.values())):
        point = result.evaluated[index]
        calibration = objective.evaluate(point.vector)
        if not calibration.feasible:
            raise ValueError("Calibration replay failed.")
        np.testing.assert_array_equal(calibration.objectives, point.objectives)
        heldout, report = safe_validation(objective, point.vector)
        trajectory(folder / f"selected_{index}_simulation.csv", heldout if heldout is not None and heldout.feasible else calibration, observations, split)
        assessed[index] = (calibration.metrics, report)
    rows = []
    for role, index in roles.items():
        cal, val = assessed[index]
        rows.append({"role": role, "evaluation_id": index, "calibration_release_nse": cal["release_nse"], "calibration_storage_nse": cal["storage_nse"], "validation_feasible": val["feasible"], "validation_reason": val["reason"], "validation_release_nse": val["metrics"].get("release_nse"), "validation_storage_nse": val["metrics"].get("storage_nse")})
    parent = read(source / "result.json")
    write(folder / "result.json", {"reservoir_id": identifier, "seed": seed, "root_policy": job["root_policy"], "settings": asdict(result.settings), "status": "completed", "search_evaluations": 1024, "new_search_evaluations": 1024-len(state.evaluated), "parent_search_evaluations": 512, "parent_prefix_preserved": True, "baseline_calibration": parent["baseline_calibration"], "baseline_validation": parent["baseline_validation"], "representatives": rows, "initialization": parent["initialization"], "new_initialization_calls": 0, "representative_calibration_replays": len(assessed), "validation_attempts": len(assessed), "baseline_metrics_reused": True, "selection_uses_validation": False})
    checked()
    write(folder / "fleet_completion.json", {"status": "passed", "artifact_sha256": {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != "fleet_completion.json"}})


def run_all():
    protocol = checked()
    logs = OUT / "logs"
    logs.mkdir(exist_ok=False)
    statuses = []
    def execute(job):
        log = logs / f"{job['reservoir_id']}_seed_{job['seed']}.log"
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, "-B", str(Path(__file__)), "--job", job["reservoir_id"], str(job["seed"])], stdout=stream, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return {**job, "status": "completed" if result.returncode == 0 else "failed", "exit_code": result.returncode}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(execute, job) for job in protocol["jobs"]]):
            statuses.append(future.result())
            write(OUT / "job_status.json", statuses)
            print(statuses[-1], flush=True)
    write(OUT / "execution_summary.json", {"status": "completed" if all(j["exit_code"] == 0 for j in statuses) else "completed_with_failures", "jobs": statuses})
    if any(j["exit_code"] for j in statuses):
        raise SystemExit("Some continuations failed; saved checkpoints/logs are retained.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run-authorized", action="store_true")
    mode.add_argument("--job", nargs=2)
    mode.add_argument("--resume-job", nargs=2)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.job or args.resume_job:
        identifier, seed = args.job or args.resume_job
        run_one(identifier, int(seed), bool(args.resume_job))
    else:
        run_all()
