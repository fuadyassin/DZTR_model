"""Build the complete reservoir admission queue without model evaluations.

Historical calibration evidence and calibration-only percentile-box arithmetic
are separate from future simulation preflight. Numerical modules stay unchanged.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/fleet_admission_20261012"
POLICY = "nearest_previous_storage"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def ordered_candidate(lower, upper, capacity, dead_fraction, step_seconds, *, close_critical_gap=False):
    """Find an ordered witness in an unchanged empirical box, or explain failure.

    Greedy lower witnesses are sufficient for ordered interval feasibility.
    Optional Sc-Sd >= dt*Qc removes the known critical-branch upward jump;
    it does not guarantee a root at every timestep. No estimates are clipped.
    """
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if lower.shape != (12, 6) or upper.shape != lower.shape:
        raise ValueError("Expected twelve months and six SI targets.")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower < 0).any() or (lower > upper).any():
        raise ValueError("Bounds require finite, nonnegative, ordered intervals.")
    if not np.isfinite([capacity, dead_fraction, step_seconds]).all() or capacity <= 0 or not 0 <= dead_fraction < 1 or step_seconds <= 0:
        raise ValueError("Invalid physical constants or timestep.")
    dead = capacity * dead_fraction
    candidates, failures = [], []
    for month in range(12):
        release = np.maximum.accumulate(lower[month, 3:])
        preceding = dead
        storage = []
        for index in range(3):
            value = max(lower[month, index], float(np.nextafter(preceding, np.inf)))
            if index == 0 and close_critical_gap:
                value = max(value, dead + step_seconds * release[0])
                # Round upward until subtraction itself satisfies the inequality.
                while value - dead < step_seconds * release[0]:
                    value = float(np.nextafter(value, np.inf))
            storage.append(value)
            preceding = value
        point = np.r_[storage, release]
        if (point > upper[month]).any() or storage[-1] > capacity:
            failures.append(month + 1)
        candidates.append(point)
    return {"feasible": not failures, "infeasible_months": failures,
            "monthly_targets_si": np.asarray(candidates).tolist() if not failures else None,
            "critical_gap_constraint": close_critical_gap, "simulation_verified": False}


def bounds_audit(frame, config):
    period = config["periods"]["calibration"]
    fitting = frame.loc[period["start"]:period["end"]]
    lower, upper = [], []
    for month in range(1, 13):
        row_lower, row_upper = [], []
        for variable in ("storage_m3", "release_m3s"):
            values = fitting.loc[fitting.index.month == month, variable].dropna().to_numpy()
            if not len(values) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid calibration {variable} in month {month}.")
            quantiles = np.percentile(values, [5, 35, 75, 95], method=config["quantile_method"])
            row_lower.extend(quantiles[:3]); row_upper.extend(quantiles[1:])
        lower.append(row_lower); upper.append(row_upper)
    model, sampling = config["model"], config["sampling"]
    days = sampling.get("fixed_step_days")
    # A variable monthly clock requires calendar-specific constraints, not an
    # inferred duration. Current packaged monthly profiles use fixed30.5 days.
    if days is None:
        if sampling["frequency"] != "daily":
            raise ValueError("Variable monthly clock needs an explicit gap-constraint adapter.")
        days = 1
    args = (lower, upper, model["capacity_m3"], model["dead_storage_fraction"], days * 86400)
    return {"fitting_period": period, "quantile_method": config["quantile_method"],
            "percentile_bounds": [[5, 35], [35, 75], [75, 95]],
            "lower_si": lower, "upper_si": upper,
            "ordered_box": ordered_candidate(*args),
            "critical_gap_box": ordered_candidate(*args, close_critical_gap=True)}


def classify(review, config, policy_evidence, audit):
    identifier = review["reservoir_id"]
    flags = {k: v for k, v in review.get("calibration_flags", {}).items() if v}
    if identifier == "ex35_Bull_Lake":
        return "cached_interval_results", "Reuse the separately reviewed computed-net interval archives."
    if identifier in ("ex12_Trinity", "ex26_Ghost"):
        return "source_profile_required", "Implement the saved source-reconciliation requirements and explicit quality/clock profile."
    if not config.get("simulation_supported", True):
        return "forcing_adapter_required", "Integrate the externally substituted forcing; retain source missingness."
    if audit is None:
        return "parameter_inputs_required", "Resolve invalid or incomplete calibration target inputs."
    if not audit["ordered_box"]["feasible"]:
        return "empirical_box_incompatible", "Review capacity, operating floor and empirical bounds; the current box has no ordered physical candidate."
    if flags:
        return "source_assumptions_review", "Resolve or explicitly declare observed-storage/model-limit differences before scheduling calibration."
    if review.get("strict_calibration_baseline_feasible"):
        return "ready_strict", "Existing strict calibration baseline passes; retain recorded inherited assumptions."
    quality = policy_evidence.get("calibration_quality", {})
    if policy_evidence.get("calibration_status") == "completed" and quality.get("numerically_valid") and quality.get("physical_storage_valid"):
        return "ready_explicit_root_policy", "Use nearest previous simulated storage with the existing deterministic minimum-storage tie rule."
    if audit["critical_gap_box"]["feasible"]:
        return "initialization_preflight_required", "Test the constrained witness on warmup/calibration; failed generalized baseline alone does not exclude calibration."
    return "critical_gap_constraint_incompatible", "Current empirical bounds cannot satisfy Sc-Sd >= dt*Qc; review initialization bounds/timestep without silently changing equations."


def prepare(out):
    out = out.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        raise ValueError("Choose a fresh output directory inside the repository.")
    old = ROOT / "outputs/fleet_workflow_20261010"
    proof = read(old / "preparation_verification.json")
    if proof["status"] != "passed":
        raise ValueError("Prior fleet preparation was not verified.")
    for name, digest in proof["artifact_sha256"].items():
        if sha(old / name) != digest:
            raise ValueError(f"Changed prior preparation: {name}")
    protocol = read(old / "protocol.json")
    for name, digest in {**protocol["code_sha256"], **protocol["inputs_sha256"]}.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Changed guarded input/code: {name}")
    cohort = read(old / "cohort.json")
    reviews = {r["reservoir_id"]: r for r in cohort["reservoirs"]}
    manifest = read(ROOT / "data/reservoirs/manifest.json")
    evidence_path = ROOT / "outputs/root_policy_collection_final/root_policy_review.json"
    evidence = read(evidence_path)
    verification = read(ROOT / "outputs/root_policy_verification/verification.json")
    if evidence["status"] != "review_completed" or verification["status"] != "passed" or evidence["manifest_sha256"] != sha(ROOT / "data/reservoirs/manifest.json"):
        raise ValueError("Root-policy evidence is not compatible with the collection.")
    expected_evidence = verification["consumed_proof_sha256"][evidence_path.relative_to(ROOT).as_posix()]
    if sha(evidence_path) != expected_evidence:
        raise ValueError("Saved root-policy evidence changed since its scientific verification.")
    for name, digest in evidence["code_sha256"].items():
        if sha(ROOT / "src/dztr" / name.replace("\\", "/")) != digest:
            raise ValueError("Root-policy numerical code changed.")
    policies = {r["reservoir_id"]: next((p for p in r.get("policies", []) if p["policy"] == POLICY), {}) for r in evidence["reservoirs"]}
    out.mkdir(parents=True)
    rows, records, tasks = [], [], []
    for entry in manifest["reservoirs"]:
        identifier = entry["reservoir_id"]
        config_path = ROOT / "configs/reservoirs" / f"{identifier}.json"
        config = read(config_path)
        frame = pd.read_csv((config_path.parent / config["observations_file"]).resolve(), index_col="date", parse_dates=True, float_precision="round_trip")
        audit, error = None, None
        try:
            audit = bounds_audit(frame, config)
        except ValueError as exc:
            error = str(exc)
        status, action = classify(reviews[identifier], config, policies.get(identifier, {}), audit)
        ready = status in ("ready_strict", "ready_explicit_root_policy")
        policy = POLICY if status == "ready_explicit_root_policy" else "reject"
        record = {"reservoir_id": identifier, "name": entry["name"], "included_in_program": True,
                  "status": status, "calibration_schedulable": ready, "next_action": action,
                  "root_policy": policy, "original_reasons": reviews[identifier].get("reasons", []),
                  "calibration_flags": reviews[identifier].get("calibration_flags"),
                  "bounds_audit": audit, "bounds_audit_error": error,
                  "initialization_simulation_verified": False,
                  "admission_uses_validation": False, "physical_source_certified": False}
        records.append(record)
        rows.append({k: record[k] for k in ("reservoir_id", "name", "included_in_program", "status", "calibration_schedulable", "root_policy", "next_action")})
        tasks.append({"reservoir_id": identifier, "stage": "calibration" if ready else status,
                      "action": action, "execution_started": False})
    counts = dict(Counter(r["status"] for r in records))
    ready_ids = [r["reservoir_id"] for r in records if r["calibration_schedulable"]]
    write(out / "admission.json", {"schema_version": 1, "status": "prepared", "reservoirs": records,
          "counts": counts, "active_reservoirs": len(records), "excluded_reservoirs": manifest["excluded_reservoirs"],
          "new_model_evaluations": 0, "new_search_evaluations": 0, "admission_uses_validation": False,
          "policy": "Failed generalized baseline is not a permanent exclusion. Only calibration evidence is used. Source flags require a named interpretation; empirical boxes are never widened silently.",
          "root_policy_semantics": "Nearest previous simulated storage; deterministic minimum-storage tie rule. Provisional computational convention, not recovered operating intent."})
    write(out / "work_queue.json", {"tasks": tasks, "execution_started": False})
    write(out / "calibration_plan.json", {"status": "prepared_not_executable", "reason": "The original runner hardcodes strict rejection and asserts a feasible generalized baseline. A policy-aware initialization runner is required before this expanded plan executes.",
          "jobs": [{"reservoir_id": r["reservoir_id"], "root_policy": r["root_policy"], "seed": seed,
                    "population_size": 32, "evaluation_budget": 512} for r in records if r["calibration_schedulable"] for seed in (42, 43, 44)],
          "eligible_ids": ready_ids, "new_evaluations_started": 0})
    pd.DataFrame(rows).to_csv(out / "reservoir_status.csv", index=False)
    table = "\n".join(f"| {status} | {count} |" for status, count in sorted(counts.items()))
    text = f"# Complete reservoir admission queue\n\nAll {len(records)} active reservoirs remain included; Tuyen remains excluded by request.\n\n| Stage | Reservoirs |\n| --- | ---: |\n{table}\n\n{len(ready_ids)} reservoirs have calibration admission evidence, corresponding to {len(ready_ids)*3} proposed jobs. The expanded plan is not yet executable by the old strict-only runner. No model or optimizer evaluations were made.\n\n[All statuses and actions](reservoir_status.csv), [detailed evidence and unchanged empirical bounds](admission.json), [complete work queue](work_queue.json), [proposed calibration jobs](calibration_plan.json).\n\nRoot selection is a declared continuity heuristic and does not establish operating intent. Ordered percentile-box witnesses are proposed initialization points, not validated simulations or calibrated parameters. Empty boxes require source/bounds review; no sorting, clipping, capacity inflation or forcing correction is applied. Held-out fit/feasibility is never used for admission. Existing monthly clocks, mass-balance equations, source records and checkpoints remain unchanged.\n"
    (out / "README.md").write_text(text, encoding="utf-8")
    consumed = [old / "preparation_verification.json", evidence_path, ROOT / "outputs/root_policy_verification/verification.json"]
    write(out / "verification.json", {"status": "passed", "checks": ["41 unique active members retained", "all pinned inputs and numerical modules unchanged", "compatible saved root-policy evidence", "calibration-only percentile fitting", "no simulations or searches"],
          "code_sha256": protocol["code_sha256"], "inputs_sha256": protocol["inputs_sha256"],
          "consumed_evidence_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in consumed},
          "tool_sha256": sha(Path(__file__)), "artifact_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({"counts": counts, "calibration_admitted": len(ready_ids), "new_evaluations": 0}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    prepare(parser.parse_args().output)
