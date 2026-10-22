"""Run four exploratory checkpointed reservoir jobs in one process pool.

Trinity seeds 42/43/44 use cumulative stages 512, 1024 and 2048. Gardiner seed
42 uses 512 and 1024. Early online stopping is followed by an explicitly
labelled audit extension to the final requested budget. This is not a diverse
cohort validation or a production budget recommendation.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dztr.calibration.adaptive import (
    CHECKPOINT_NAME, IDENTITY_NAME, MANIFEST_NAME,
    run_adaptive_optimization,
)
from dztr.calibration.baseline import json_ready
from dztr.calibration.checkpoint import _replace_checkpoint, load_optimizer_checkpoint
from dztr.calibration.convergence import checkpoint_metrics, prefix_pareto_front
from dztr.calibration.search import ARTIFACT_NAMES as HISTORICAL_ARTIFACT_NAMES
from dztr.calibration.stopping import StoppingPolicy, assess_stopping


HISTORY_NAME = "stage_summary.json"
JOBS = (
    {"reservoir": "Trinity", "seed": 42, "config": "configs/trinity_adaptive.json",
     "folder": "outputs/trinity_adaptive/seed_42", "budgets": (512, 1024, 2048)},
    {"reservoir": "Trinity", "seed": 43, "config": "configs/trinity_adaptive.json",
     "folder": "outputs/trinity_adaptive/seed_43", "budgets": (512, 1024, 2048)},
    {"reservoir": "Trinity", "seed": 44, "config": "configs/trinity_adaptive.json",
     "folder": "outputs/trinity_adaptive/seed_44", "budgets": (512, 1024, 2048)},
    {"reservoir": "Gardiner", "seed": 42, "config": "configs/gardiner_adaptive.json",
     "folder": "outputs/gardiner_adaptive/seed_42", "budgets": (512, 1024)},
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    payload = json.dumps(json_ready(value), indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(payload, encoding="utf-8")
    _replace_checkpoint(temporary, path)


def _verify_artifacts(folder, manifest_name, expected_names):
    manifest_path = folder / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digests = manifest.get("artifact_sha256", {})
    if set(digests) != set(expected_names):
        raise ValueError(f"Incomplete artifact manifest: {manifest_path}")
    for name, digest in digests.items():
        path = folder / name
        if not path.is_file() or _sha(path) != digest:
            raise ValueError(f"Artifact integrity check failed: {path}")
    return manifest


def _historical_parity(root, job, state, identity):
    folder = root / "outputs/trinity_convergence/runs" / f"seed_{job['seed']}"
    manifest = _verify_artifacts(folder, "seed_manifest.json", HISTORICAL_ARTIFACT_NAMES)
    old_identity = manifest["identity"]
    if old_identity["observations_sha256"] != identity["observations_sha256"]:
        raise ValueError("Historical Trinity observations differ from the adaptive input.")
    if manifest["search_evaluations"] < 1024 or state.evaluations < 1024:
        raise ValueError("Trinity parity requires two complete 1024-evaluation prefixes.")
    table = pd.read_csv(folder / "evaluations.csv", float_precision="round_trip")
    vectors = pd.read_csv(folder / "evaluated_vectors.csv", float_precision="round_trip")
    expected_ids = np.arange(len(table))
    if not np.array_equal(table.evaluation_id.to_numpy(), expected_ids):
        raise ValueError("Historical evaluation IDs are not contiguous in file order.")
    if not np.array_equal(vectors.evaluation_id.to_numpy(), expected_ids):
        raise ValueError("Historical vector IDs do not match their trial table.")
    if list(vectors.columns) != ["evaluation_id", *[f"x{i:02d}" for i in range(72)]]:
        raise ValueError("Historical vectors do not use the expected 72-coordinate order.")
    if not pd.api.types.is_bool_dtype(table.feasible.dtype):
        raise ValueError("Historical feasibility flags must be boolean.")
    trials = state.evaluated[:1024]
    actual_ids = np.asarray([point.evaluation_id for point in trials])
    actual_feasible = np.asarray([point.feasible for point in trials], dtype=bool)
    actual_vectors = np.asarray([point.vector for point in trials])
    actual_costs = np.asarray([point.objectives for point in trials])
    old_feasible = table.feasible.to_numpy()[:1024]
    old_costs = table[["negative_release_nse", "negative_storage_nse"]].to_numpy()[:1024]
    if not np.isfinite(old_costs[old_feasible]).all():
        raise ValueError("Historical feasible objectives must be finite.")
    # CSV exports represent rejected +infinity costs as missing cells.
    old_costs[~old_feasible] = np.inf
    checks = {
        "evaluation_ids": np.array_equal(actual_ids, expected_ids[:1024]),
        "feasibility_flags": np.array_equal(actual_feasible, old_feasible),
        "parameter_vectors": np.array_equal(actual_vectors, vectors.iloc[:1024, 1:].to_numpy()),
        "objective_costs": np.array_equal(actual_costs, old_costs),
    }
    if not all(checks.values()):
        raise ValueError(f"Adaptive/historical Trinity prefix parity failed: {checks}")
    return {"evaluations": 1024, "bit_exact": True, "checks": checks,
            "historical_folder": folder.relative_to(root).as_posix(),
            "historical_manifest_sha256": _sha(folder / "seed_manifest.json"),
            "observations_sha256": identity["observations_sha256"],
            "historical_code_identity_matches_current":
                old_identity["search_code_sha256"] == identity["search_code_sha256"],
            "qualification": "Historical code identity may differ; every saved artifact and the input hash were verified before empirical prefix comparison."}


def _stage_record(job, report, phase):
    return {"reservoir": job["reservoir"], "seed": job["seed"], "phase": phase,
            **{key: report[key] for key in (
                "requested_stage_budget", "search_evaluations", "new_search_evaluations",
                "reused_search_evaluations", "termination_reason", "honor_stopping",
                "stopping_status", "earliest_stopping_candidate_evaluations", "candidate_invalidated",
                "search_elapsed_seconds", "reporting_calibration_evaluations", "reporting_validation_evaluations")}}


def _validate_history(history, job, config_path, available_count):
    if history.get("schema_version") != 1 or history.get("job") != json_ready(job):
        raise ValueError("Saved stage history does not match this job plan.")
    if history.get("configuration_sha256") != _sha(config_path):
        raise ValueError("Job configuration changed since stage history was saved.")
    rows = history.get("stages")
    if not isinstance(rows, list):
        raise ValueError("Saved stage history must contain a stage list.")
    previous_count = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("reservoir") != job["reservoir"] or row.get("seed") != job["seed"]:
            raise ValueError("A saved stage belongs to a different reservoir/seed.")
        if row.get("phase") not in {"online", "audit_extension", "resume_verification", "resume_recovery"}:
            raise ValueError("A saved stage has an unknown phase.")
        counts = [row.get(name) for name in ("search_evaluations", "new_search_evaluations", "reused_search_evaluations")]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("Saved stage evaluation counts must be nonnegative integers.")
        count, new, reused = counts
        target = row.get("requested_stage_budget")
        if target not in job["budgets"] or new + reused != count or not previous_count <= count <= min(target, available_count):
            raise ValueError("Saved stage counts or targets are inconsistent with the checkpoint/job plan.")
        if reused < previous_count:
            raise ValueError("A saved stage discarded previously recorded trials.")
        previous_count = count


def _extension_evidence(state, stopped, policy):
    if stopped is None or state.evaluations <= stopped:
        return None
    first = prefix_pareto_front(state.evaluated, stopped)
    last = prefix_pareto_front(state.evaluated, state.evaluations)
    old = np.unique(np.asarray([point.objectives for point in first]), axis=0)
    new = np.unique(np.asarray([point.objectives for point in last]), axis=0)
    if not len(old) or not len(new):
        raise ValueError("Extension evidence requires feasible fronts at both endpoints.")
    distances = np.max(np.abs(old[:, None, :] - new[None, :, :]), axis=2)
    movement = float(max(np.max(np.min(distances, axis=0)), np.max(np.min(distances, axis=1))))
    before = checkpoint_metrics(state.evaluated, stopped, reference=policy.hypervolume_reference)
    after = checkpoint_metrics(state.evaluated, state.evaluations, reference=policy.hypervolume_reference)
    gain = max(0.0, after["hypervolume"] - before["hypervolume"])
    normalized = gain / policy.hypervolume_normalization_area
    extrema = float(np.max(np.abs(old.min(axis=0) - new.min(axis=0))))
    assessment = assess_stopping(state.evaluated, policy)
    return {"online_stop_evaluations": stopped, "final_evaluations": state.evaluations,
            "additional_evaluations": state.evaluations - stopped,
            "hypervolume_gain": gain, "normalized_hypervolume_gain": normalized,
            "front_movement": movement, "extrema_drift": extrema,
            "cumulative_extension_exceeds_policy_tolerances":
                normalized >= policy.normalized_hv_tolerance or movement > policy.front_tolerance or extrema > policy.extrema_tolerance,
            "later_window_invalidated_candidate": assessment.candidate_invalidated,
            "qualification": "Calibration-only cumulative stop-to-final change; this spans the audit extension, not one policy comparison window."}


def _cross_seed_front_agreement(root, outcomes):
    completed = [row for row in outcomes
                 if row["job"]["reservoir"] == "Trinity" and row["status"] == "complete"]
    if len(completed) != 3:
        return {"status": "insufficient_completed_seeds", "completed_seed_count": len(completed),
                "global_convergence_established": False}
    fronts = {}
    for row in completed:
        job = row["job"]
        folder = root / job["folder"]
        identity = json.loads((folder / IDENTITY_NAME).read_text(encoding="utf-8"))
        state = load_optimizer_checkpoint(folder / CHECKPOINT_NAME, identity)
        front = prefix_pareto_front(state.evaluated, state.evaluations)
        fronts[job["seed"]] = np.unique(np.asarray([point.objectives for point in front]), axis=0)
    comparisons = []
    seeds = sorted(fronts)
    tolerance = 0.01
    for i, seed in enumerate(seeds):
        for other in seeds[i + 1:]:
            distances = np.max(np.abs(fronts[seed][:, None, :] - fronts[other][None, :, :]), axis=2)
            movement = float(max(np.max(np.min(distances, axis=0)), np.max(np.min(distances, axis=1))))
            comparisons.append({"first_seed": seed, "second_seed": other,
                                "front_distance": movement, "within_tolerance": movement <= tolerance})
    return {"status": "reported", "reservoir": "Trinity", "evaluations_per_seed": 2048,
            "metric": "symmetric full-front L-infinity Hausdorff distance in raw negative-NSE space",
            "provisional_tolerance": tolerance, "comparisons": comparisons,
            "all_pairs_within_tolerance": all(row["within_tolerance"] for row in comparisons),
            "global_convergence_established": False,
            "qualification": "Calibration-only independent-seed comparison; the agreement tolerance is illustrative and agreement does not prove global convergence."}


def _run_job(root_text, job, resume):
    root = Path(root_text)
    folder = root / job["folder"]
    config_path = root / job["config"]
    history_path = folder / HISTORY_NAME
    history = {"schema_version": 1, "job": json_ready(job), "configuration_sha256": _sha(config_path),
               "stages": [], "online_stop_evaluations": None, "historical_parity": None,
               "source_qualification": "Gardiner uses qualified local catalog metadata and a first-warmup-observation initial state; this is an exploratory second case." if job["reservoir"] == "Gardiner"
               else "Trinity retains the verified MATLAB-compatible physical configuration; existing archive parity is checked empirically."}
    invocation_stages = []
    initial_reused = 0
    report = None

    def progress(record):
        if record.generation == 0 or record.evaluations % 128 == 0:
            print(f"{job['reservoir']} seed {job['seed']}: {record.evaluations} evaluations; "
                  f"{record.feasible_count} feasible; {record.front_count} nondominated", flush=True)

    def record_report(result, phase):
        nonlocal report
        report = result
        row = _stage_record(job, result, phase)
        history["stages"].append(row)
        invocation_stages.append(row)
        if result["termination_reason"] == "stopping_candidate" and phase != "resume_verification":
            if history["online_stop_evaluations"] is None:
                history["online_stop_evaluations"] = result["search_evaluations"]
        state = load_optimizer_checkpoint(folder / CHECKPOINT_NAME, result["checkpoint_identity"])
        if job["reservoir"] == "Trinity" and state.evaluations >= 1024 and history["historical_parity"] is None:
            history["historical_parity"] = _historical_parity(root, job, state, result["checkpoint_identity"])
            print(f"Trinity seed {job['seed']}: historical 1024-prefix bit-exact parity PASS", flush=True)
        history["last_manifest_sha256"] = _sha(folder / MANIFEST_NAME)
        history["last_search_evaluations"] = result["search_evaluations"]
        history["complete"] = result["search_evaluations"] >= job["budgets"][-1]
        _write_json(history_path, history)

    try:
        existing = folder.exists() and any(folder.iterdir())
        if existing and not resume:
            raise ValueError(f"Existing job output requires explicit --resume: {folder}")
        if resume and existing:
            if not (folder / IDENTITY_NAME).is_file() or not (folder / CHECKPOINT_NAME).is_file():
                raise ValueError("Existing output has no owned adaptive identity/checkpoint.")
            identity = json.loads((folder / IDENTITY_NAME).read_text(encoding="utf-8"))
            saved_state = load_optimizer_checkpoint(folder / CHECKPOINT_NAME, identity)
            initial_reused = saved_state.evaluations
            if saved_state.settings.evaluation_budget not in job["budgets"]:
                raise ValueError("Saved checkpoint target is outside this orchestrator's staged job plan.")
            if history_path.is_file():
                history = json.loads(history_path.read_text(encoding="utf-8"))
                _validate_history(history, job, config_path, saved_state.evaluations)
            # The adaptive API reconstructs current input/code/software identity.
            # It also owns reporting-manifest checks and recovery when a newer
            # checkpoint survived an interrupted reporting export.
            # An interrupted active stage can finish here; a completed one makes
            # no new search calls and only refreshes selected reporting.
            target = saved_state.settings.evaluation_budget
            phase = "resume_verification" if saved_state.evaluations == target else "resume_recovery"
            honor = history["online_stop_evaluations"] is None
            result = run_adaptive_optimization(config_path, root, folder, evaluation_budget=target,
                seed=job["seed"], resume=True, honor_stopping=honor, progress_callback=progress)
            if phase == "resume_verification" and result["new_search_evaluations"] != 0:
                raise RuntimeError("Completed-stage resume unexpectedly evaluated new trials.")
            record_report(result, phase)

        final_budget = job["budgets"][-1]
        for budget in job["budgets"]:
            count = 0 if report is None else report["search_evaluations"]
            if count >= budget:
                continue
            if history["online_stop_evaluations"] is not None:
                result = run_adaptive_optimization(config_path, root, folder, evaluation_budget=final_budget,
                    seed=job["seed"], resume=True, honor_stopping=False, progress_callback=progress)
                record_report(result, "audit_extension")
                break
            result = run_adaptive_optimization(config_path, root, folder, evaluation_budget=budget,
                seed=job["seed"], resume=report is not None, honor_stopping=True, progress_callback=progress)
            record_report(result, "online")
            if result["termination_reason"] == "duplicate_exhausted":
                raise RuntimeError("Search exhausted unique offspring before the requested final budget.")
        if report is not None and report["search_evaluations"] < final_budget and history["online_stop_evaluations"] is not None:
            result = run_adaptive_optimization(config_path, root, folder, evaluation_budget=final_budget,
                seed=job["seed"], resume=True, honor_stopping=False, progress_callback=progress)
            record_report(result, "audit_extension")
        if report is None or report["search_evaluations"] != final_budget:
            raise RuntimeError("Job did not complete its final planned or audit budget.")
        state = load_optimizer_checkpoint(folder / CHECKPOINT_NAME, report["checkpoint_identity"])
        policy = StoppingPolicy(**report["stopping_policy"])
        metrics = checkpoint_metrics(state.evaluated, state.evaluations, reference=policy.hypervolume_reference)
        selected = metrics.pop("representative_ids")
        metrics.pop("reference")
        metrics = {"reservoir": job["reservoir"], "seed": job["seed"], **metrics,
                   **{f"{role}_evaluation_id": value for role, value in selected.items()}}
        history["extension_evidence"] = _extension_evidence(state, history["online_stop_evaluations"], policy)
        history["final_calibration_metrics"] = metrics
        history["complete"] = True
        history["last_manifest_sha256"] = _sha(folder / MANIFEST_NAME)
        _write_json(history_path, history)
        return {"status": "complete", "job": json_ready(job), "stage_history": history,
                "invocation_stages": invocation_stages, "initial_reused_search_evaluations": initial_reused,
                "new_search_evaluations_this_invocation": sum(row["new_search_evaluations"] for row in invocation_stages),
                "final_calibration_metrics": metrics}
    except Exception as error:
        return {"status": "error", "job": json_ready(job), "error": str(error),
                "traceback": traceback.format_exc(), "stage_history": history,
                "invocation_stages": invocation_stages, "initial_reused_search_evaluations": initial_reused,
                "new_search_evaluations_this_invocation": sum(row["new_search_evaluations"] for row in invocation_stages)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive.")
    root = args.repo.resolve()
    output = root / "outputs/checkpoint_reservoir_study"
    if not output.resolve().is_relative_to(root):
        parser.error("Study output must remain within --repo.")
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error("Existing study output requires explicit --resume.")
    output.mkdir(parents=True, exist_ok=True)
    outcomes = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_job, str(root), job, args.resume): job for job in JOBS}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {"status": "error", "job": json_ready(job), "error": str(error),
                          "traceback": traceback.format_exc(), "invocation_stages": []}
            outcomes.append(result)
            print(f"{job['reservoir']} seed {job['seed']}: {result['status']}"
                  + (f"; {result['error']}" if result["status"] == "error" else ""), flush=True)
    outcomes.sort(key=lambda row: (row["job"]["reservoir"], row["job"]["seed"]))
    stages = [row for result in outcomes for row in result.get("stage_history", {}).get("stages", [])]
    final = [result["final_calibration_metrics"] for result in outcomes if result["status"] == "complete"]
    failures = [result for result in outcomes if result["status"] == "error"]
    report = {"experiment": "exploratory_checkpoint_reservoir_study", "workers": args.workers,
              "resume_requested": args.resume, "jobs": outcomes, "failed_job_count": len(failures),
              "new_search_evaluations_this_invocation": sum(row.get("new_search_evaluations_this_invocation", 0) for row in outcomes),
              "initial_reused_search_evaluations": sum(row.get("initial_reused_search_evaluations", 0) for row in outcomes),
              "cross_seed_front_agreement": _cross_seed_front_agreement(root, outcomes),
              "global_convergence_established": False,
              "qualification": "Trinity plus one qualified Gardiner exploratory case is not an independent diverse-cohort validation. Stopping thresholds remain illustrative."}
    _write_json(output / "summary.json", report)
    pd.DataFrame(stages).to_csv(output / "stages.csv", index=False)
    pd.DataFrame(final).to_csv(output / "final_calibration_metrics.csv", index=False)
    print(f"Study artifacts: {output}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} job(s) failed; independent jobs finished and failures are recorded in summary.json.")


if __name__ == "__main__":
    main()
