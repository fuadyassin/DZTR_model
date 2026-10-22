"""Verify admission against public percentile estimates and physical schemas."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dztr.calibration.sampled import _observations
from dztr.parameterization import estimate_percentiles
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(out):
    proof = read(out / "verification.json")
    for name, digest in proof["artifact_sha256"].items():
        assert sha(out / name) == digest, name
    for mapping in (proof["code_sha256"], proof["inputs_sha256"], proof["consumed_evidence_sha256"]):
        for name, digest in mapping.items():
            assert sha(ROOT / name) == digest, name
    assert sha(ROOT / "tools/fleet_admission.py") == proof["tool_sha256"]
    registry = read(out / "admission.json")
    records = registry["reservoirs"]
    manifest = read(ROOT / "data/reservoirs/manifest.json")
    assert len(records) == 41 and {r["reservoir_id"] for r in records} == {r["reservoir_id"] for r in manifest["reservoirs"]}
    assert all(r["included_in_program"] and not r["admission_uses_validation"] for r in records)
    plan = read(out / "calibration_plan.json")
    assert len(plan["jobs"]) == 60 and plan["new_evaluations_started"] == 0
    assert len(read(out / "work_queue.json")["tasks"]) == 41
    policy = read(ROOT / "outputs/root_policy_collection_final/root_policy_review.json")
    policy_rows = {r["reservoir_id"]: r for r in policy["reservoirs"]}
    percentiles_checked = witnesses_checked = root_admissions_checked = 0
    for record in records:
        identifier = record["reservoir_id"]
        config_path = ROOT / "configs/reservoirs" / (identifier + ".json")
        config = read(config_path)
        if record["status"] == "ready_explicit_root_policy":
            item = next(p for p in policy_rows[identifier]["policies"] if p["policy"] == record["root_policy"])
            assert item["calibration_status"] == "completed"
            assert item["calibration_quality"]["numerically_valid"] and item["calibration_quality"]["physical_storage_valid"]
            root_admissions_checked += 1
        audit = record["bounds_audit"]
        if audit is None:
            continue
        frame = pd.read_csv((config_path.parent / config["observations_file"]).resolve(), index_col="date", parse_dates=True, float_precision="round_trip")
        period = config["periods"]["calibration"]
        fitting = frame.loc[period["start"]:period["end"]].copy()
        fitting["inflow_m3s"] = np.nan
        quantiles = estimate_percentiles(_observations(fitting), (5, 35, 75, 95), quantile_method=config["quantile_method"]).frame
        for side, probabilities in (("lower", (5, 35, 75)), ("upper", (35, 75, 95))):
            values = []
            for month in range(1, 13):
                values.append([float(quantiles.loc[(quantiles.variable == variable) & (quantiles.month == month) & (quantiles.percentile == p), "value"].iloc[0]) for variable in ("storage_m3", "release_m3s") for p in probabilities])
            np.testing.assert_array_equal(values, audit[side + "_si"])
        percentiles_checked += 1
        for label in ("ordered_box", "critical_gap_box"):
            candidate = audit[label]
            if not candidate["feasible"]:
                assert candidate["monthly_targets_si"] is None and candidate["infeasible_months"]
                continue
            targets = np.asarray(candidate["monthly_targets_si"])
            assert (targets >= audit["lower_si"]).all() and (targets <= audit["upper_si"]).all()
            ReservoirParameters(pd.DataFrame(targets, columns=MONTHLY_TARGET_COLUMNS, index=pd.Index(range(1, 13), name="month")), **config["model"])
            if label == "critical_gap_box":
                dead = config["model"]["capacity_m3"] * config["model"]["dead_storage_fraction"]
                step = (config["sampling"].get("fixed_step_days") or 1) * 86400
                assert (targets[:, 0] - dead >= step * targets[:, 3]).all()
            witnesses_checked += 1
    assert root_admissions_checked == 11
    result = {"status": "passed", "active_members": 41, "proposed_jobs": 60,
              "public_percentile_estimates_matched": percentiles_checked,
              "physical_schema_witnesses_checked": witnesses_checked,
              "saved_calibration_root_admissions_checked": root_admissions_checked,
              "new_model_evaluations": 0, "numerical_modules_unchanged": len(proof["code_sha256"]),
              "registry_sha256": sha(out / "admission.json"), "verifier_sha256": sha(Path(__file__))}
    destination = out / "independent_verification.json"
    if destination.exists():
        raise ValueError("Do not overwrite an existing independent receipt.")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fleet_admission_verified_20261012")
    verify(parser.parse_args().output.resolve())
