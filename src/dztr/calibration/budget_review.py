"""Retrospective budget diagnostics from verified calibration study archives.

This review launches no searches or daily simulations. A provisional stopping
candidate is evidence of recent stagnation, never a production budget or proof
of global convergence. Validation scores do not enter any calculation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from itertools import combinations
import json
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import json_ready
from .convergence import checkpoint_metrics, prefix_pareto_front
from .search import ARTIFACT_NAMES as SEARCH_ARTIFACT_NAMES
from .stopping import StoppingPolicy, _front_distance, assess_stopping
from .study import _read_trials


ARTIFACT_NAMES = ("seed_status.csv", "checkpoint_diagnostics.csv", "cross_seed_fronts.csv",
                  "summary.json", "resolved_policy.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pairs(records, count):
    front = prefix_pareto_front(records, count)
    return np.asarray([point.objectives for point in front], dtype=float).reshape(-1, 2)


def _shared_identity(identity: dict) -> dict:
    detached = json.loads(json.dumps(identity))
    settings = detached["resolved_search_configuration"]["optimization"]
    settings.pop("seed", None)
    settings.pop("evaluation_budget", None)
    return detached


def _load_verified_runs(study: Path):
    folders = sorted((study / "runs").glob("seed_*"))
    if not folders:
        raise ValueError("The study has no completed seed archives in runs/seed_*.")
    records, manifests, manifest_paths = {}, {}, {}
    shared = None
    for folder in folders:
        path = folder / "seed_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        hashes = manifest["artifact_sha256"]
        if set(hashes) != set(SEARCH_ARTIFACT_NAMES):
            raise ValueError(f"{folder.name}: the manifest lacks the complete search artifact set.")
        if any(not (folder / name).is_file() or _sha(folder / name) != digest
               for name, digest in hashes.items()):
            raise ValueError(f"{folder.name}: search artifacts failed their SHA-256 integrity check.")
        identity = manifest["identity"]
        seed = identity["resolved_search_configuration"]["optimization"]["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed in records:
            raise ValueError("Seed archives require distinct nonnegative integer seeds.")
        candidate_shared = _shared_identity(identity)
        if shared is not None and shared != candidate_shared:
            raise ValueError("Seed archives differ in observations, model, search settings, code or software versions.")
        shared = candidate_shared
        trials = _read_trials(folder)
        if len(trials) != manifest["search_evaluations"] or not trials:
            raise ValueError(f"{folder.name}: the recorded search count does not match the trial archive.")
        records[seed], manifests[seed], manifest_paths[seed] = trials, manifest, path
    return records, manifests, manifest_paths


def run_budget_review(study_dir: str | Path, output_dir: str | Path, *,
                      policy: StoppingPolicy | None = None, cross_seed_tolerance: float = 0.01) -> dict:
    """Audit saved prefixes and seed agreement without inspecting validation.

    Artifact integrity and consistent archived search identities are required.
    Source observations are not reread: their original hashes and search code
    identities are retained as historical provenance, rather than asserted to
    match the user's current input files. Output must be outside the study.
    """
    study, output = Path(study_dir).resolve(), Path(output_dir).resolve()
    if output == study or study in output.parents:
        raise ValueError("Budget review output must be outside the original study directory.")
    if any((output / name).exists() for name in ARTIFACT_NAMES):
        try:
            previous = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        if (not isinstance(previous, dict)
                or previous.get("experiment") != "retrospective_calibration_budget_review"
                or previous.get("study_directory") != str(study)):
            raise ValueError("Budget review outputs would overwrite existing files; choose a new output folder.")
    if (isinstance(cross_seed_tolerance, (bool, np.bool_)) or not isinstance(cross_seed_tolerance, Real)
            or not np.isfinite(cross_seed_tolerance) or cross_seed_tolerance < 0):
        raise ValueError("cross_seed_tolerance must be a finite nonnegative NSE distance.")
    cross_seed_tolerance = float(cross_seed_tolerance)
    code_root = Path(__file__).resolve().parents[1]
    code_names = ("calibration/budget_review.py", "calibration/stopping.py",
                  "calibration/convergence.py", "calibration/study.py")
    code_hashes = {name: _sha(code_root / name) for name in code_names}
    records, manifests, paths = _load_verified_runs(study)
    seeds = sorted(records)
    population = manifests[seeds[0]]["identity"]["resolved_search_configuration"]["optimization"]["population_size"]
    if policy is None:
        policy = StoppingPolicy(population_size=population)
    if not isinstance(policy, StoppingPolicy) or policy.population_size != population:
        raise ValueError("Stopping policy population_size must match the archived searches.")
    original_hashes = {seed: _sha(path) for seed, path in paths.items()}
    assessments = {seed: assess_stopping(records[seed], policy) for seed in seeds}
    seed_rows, history = [], []
    for seed in seeds:
        assessment = assessments[seed]
        count = len(records[seed])
        final = checkpoint_metrics(records[seed], count, reference=policy.hypervolume_reference)
        first = assessment.suggested_stop_evaluations
        gain = movement = drift = None
        extension_exceeds = None
        if first is not None:
            initial = checkpoint_metrics(records[seed], first, reference=policy.hypervolume_reference)
            gain = max(0.0, final["hypervolume"] - initial["hypervolume"]) / policy.hypervolume_normalization_area
            start_pairs, end_pairs = _pairs(records[seed], first), _pairs(records[seed], count)
            if len(start_pairs) and len(end_pairs):
                movement = _front_distance(start_pairs, end_pairs)
                drift = float(np.max(np.abs(np.min(start_pairs, axis=0) - np.min(end_pairs, axis=0))))
            extension_exceeds = (gain >= policy.normalized_hv_tolerance
                                 or movement is None or movement > policy.front_tolerance
                                 or drift is None or drift > policy.extrema_tolerance)
        seed_rows.append({
            "seed": seed, "evaluations": count, "status": assessment.status, "reason": assessment.reason,
            "earliest_stopping_candidate_evaluations": first,
            "later_window_invalidated_candidate": assessment.candidate_invalidated,
            "available_extension_evaluations": None if first is None else count - first,
            "extension_normalized_hv_gain": gain, "extension_front_movement": movement,
            "extension_extrema_drift": drift, "extension_exceeds_policy_tolerances": extension_exceeds,
            "hypervolume": final["hypervolume"], "max_release_nse": final["max_release_nse"],
            "max_storage_nse": final["max_storage_nse"], "feasible_count": final["feasible_count"],
            "rejected_count": final["rejected_count"], "front_size": final["front_size"],
        })
        history.extend({"seed": seed, **row} for row in assessment.history)

    common_count = min(len(records[seed]) for seed in seeds)
    agreement_rows = []
    counts = set(range(policy.minimum_evaluations, common_count + 1, policy.check_interval))
    counts.add(common_count)
    for count in sorted(counts):
        fronts = {seed: _pairs(records[seed], count) for seed in seeds}
        for first, second in combinations(seeds, 2):
            left, right = fronts[first], fronts[second]
            coverage = (bool(len(left)) and bool(len(right))
                        and bool(np.all(left <= policy.hypervolume_reference))
                        and bool(np.all(right <= policy.hypervolume_reference)))
            distance = _front_distance(left, right) if len(left) and len(right) else None
            agreement_rows.append({"evaluations": count, "first_seed": first, "second_seed": second,
                                   "front_distance": distance, "complete_reference_coverage": coverage,
                                   "within_tolerance": coverage and distance <= cross_seed_tolerance})
    latest = [row for row in agreement_rows if row["evaluations"] == common_count]
    agreement = len(seeds) >= 3 and bool(latest) and all(row["within_tolerance"] for row in latest)
    equal_counts = len({len(value) for value in records.values()}) == 1
    sustained = all(value.status == "stopping_candidate" for value in assessments.values())
    extension_failed = any(row["extension_exceeds_policy_tolerances"] or row["later_window_invalidated_candidate"]
                           for row in seed_rows)
    candidate = agreement and equal_counts and sustained and not extension_failed
    at_cap = any(len(value) >= policy.max_evaluations for value in records.values())
    decision = ("stopping_candidate_requires_independent_audit" if candidate
                else "hard_cap_reached_review_required" if at_cap else "continue_search")
    next_budget = None if candidate or at_cap else min(policy.max_evaluations,
        max(((common_count // policy.check_interval) + 1) * policy.check_interval,
            int(np.ceil(common_count * 2 / policy.population_size)) * policy.population_size))
    report = {
        "experiment": "retrospective_calibration_budget_review", "study_directory": str(study),
        "new_search_evaluations": 0, "reporting_simulations": 0, "validation_used": False,
        "policy_status": "illustrative_unvalidated", "policy": asdict(policy),
        "cross_seed_tolerance": cross_seed_tolerance, "independent_seed_count": len(seeds),
        "seed_status": seed_rows, "common_tested_evaluations_per_seed": common_count,
        "cross_seed_front_agreement": agreement, "budget_decision": decision,
        "suggested_next_experiment_budget_per_seed": next_budget,
        "production_budget_recommendation": None, "global_convergence_established": False,
        "analysis_code_sha256": code_hashes,
        "archived_search_identity": manifests[seeds[0]]["identity"],
        "seed_manifest_sha256": {str(seed): digest for seed, digest in original_hashes.items()},
        "limitations": [
            "Thresholds are illustrative and require independent extension and reservoir-cohort audits.",
            "Agreement is tested in fixed NSE units, not separately normalized fronts or hypervolume alone.",
            "Parameter identification and acceptable fit quality are separate from search stagnation.",
            "Saved archives support analysis; they do not contain resumable optimizer state.",
            "Source observations were not reread; original search identities are historical provenance.",
        ],
    }
    if original_hashes != {seed: _sha(path) for seed, path in paths.items()}:
        raise RuntimeError("Seed manifests changed during the review; results require inspection.")
    if any(_sha(paths[seed].parent / name) != digest
           for seed in seeds for name, digest in manifests[seed]["artifact_sha256"].items()):
        raise RuntimeError("Search artifacts changed during the review; results require inspection.")
    if code_hashes != {name: _sha(code_root / name) for name in code_names}:
        raise RuntimeError("Analysis code changed during the review; results require inspection.")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(seed_rows).to_csv(output / "seed_status.csv", index=False)
    table = pd.DataFrame(history)
    if "reasons" in table:
        table["reasons"] = table.reasons.map(lambda value: ";".join(value))
    table.to_csv(output / "checkpoint_diagnostics.csv", index=False)
    pd.DataFrame(agreement_rows, columns=["evaluations", "first_seed", "second_seed", "front_distance",
                                        "complete_reference_coverage", "within_tolerance"]).to_csv(
        output / "cross_seed_fronts.csv", index=False)
    (output / "resolved_policy.json").write_text(json.dumps(asdict(policy), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    ready = json_ready(report)
    (output / "summary.json").write_text(json.dumps(ready, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return ready


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="outputs/trinity_convergence")
    parser.add_argument("--output", default="outputs/trinity_budget_review")
    parser.add_argument("--policy", help="JSON StoppingPolicy settings; thresholds remain provisional.")
    parser.add_argument("--cross-seed-tolerance", type=float, default=0.01)
    args = parser.parse_args(argv)
    policy = None if args.policy is None else StoppingPolicy(**json.loads(Path(args.policy).read_text(encoding="utf-8")))
    report = run_budget_review(args.study, args.output, policy=policy, cross_seed_tolerance=args.cross_seed_tolerance)
    print(f"Budget decision: {report['budget_decision']}; no new search evaluations")
    for row in report["seed_status"]:
        print(f"Seed {row['seed']}: {row['status']} at {row['evaluations']} evaluations")
    print(f"Report: {Path(args.output).resolve() / 'summary.json'}")


if __name__ == "__main__":
    main()
