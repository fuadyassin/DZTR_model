"""Arithmetic and preservation audit of the four-reservoir archive export."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    proof = read(out / "verification.json")
    assert proof["status"] == "passed" and proof["new_model_evaluations"] == proof["validation_calls"] == 0
    for name, digest in proof["artifact_sha256"].items():
        assert sha(out / name) == digest, name
    for name, digest in proof["consumed_files_sha256"].items():
        assert sha(root / name) == digest, name
    index = read(out / "result_index.json")["results"]
    expected = {"ex35_Bull_Lake": (3, 32, 512), "ex14_Gardiner": (2, 16, 128),
                "ex6_Oahe": (2, 16, 128), "ex18_High_aswan": (2, 16, 128)}
    assert len(index) == 9 and {r["reservoir_id"] for r in index} == set(expected)
    jobs = pd.read_csv(out / "job_comparison.csv", float_precision="round_trip").set_index("job_id")
    inventory = pd.read_csv(out / "reservoir_comparison.csv").set_index("reservoir_id")
    assert len(inventory) == 41 and inventory.archived_jobs_in_this_catalog.sum() == 9
    assert not inventory.new_job_authorized.any()
    catalog = read(out / "input_catalog.json")
    assert [r["reservoir_id"] for r in catalog["excluded_reservoirs"]] == ["ex43_TuyenQuang"]
    assert catalog["reservoir_count"] == 41
    tables = 0
    for item in index:
        folder = (out / item["result_file"]).parent
        result = read(folder / "result.json")
        identifier = item["reservoir_id"]
        count, population, budget = expected[identifier]
        assert inventory.at[identifier, "archived_jobs_in_this_catalog"] == count
        assert result["optimizer"]["population_size"] == population and result["search_evaluations"] == budget
        assert not result["convergence_established"] and result["status"] == "budget_reached_convergence_unconfirmed"
        assert not result["validation_scored_by_export"] and result["new_model_evaluations"] == 0
        archive = root / result["archive_directory"]
        assert sha(root / result["checkpoint_file"]) == result["checkpoint_sha256"]
        assert (folder / "pareto_front.csv").read_bytes() == (archive / "pareto_front.csv").read_bytes()
        sampled = identifier != "ex35_Bull_Lake"
        vectors = pd.read_csv(archive / ("vectors.csv" if sampled else "evaluated_vectors.csv"), index_col="evaluation_id", float_precision="round_trip")
        if sampled:
            original = read(archive / "result.json")
            assert result["checkpoint_identity"] == original["identity"]
            assert result["sampling"] == original["identity"]["sampling"]
            for name in ("vectors.csv", "trials.csv", "generation_history.csv", "representatives.csv", "selected_validation.json", "selection.json"):
                assert (folder / name).read_bytes() == (archive / name).read_bytes()
            with np.load(archive / "checkpoint.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["vectors"], vectors.values)
            archived_roles = {r["role"]: r for r in original["representatives"]}
            validation = result["selected_parameters"]["generalized_baseline"]["historical_validation"]
            assert validation["storage_nse"] == original["baseline_validation"]["metrics"]["storage_nse"]
            assert jobs.at[item["job_id"], "validation_status"] == "historical_reporting_imported"
        else:
            previous = root / "outputs/archive_workflow_20261010" / item["result_file"]
            assert (folder / "result.json").read_bytes() == previous.read_bytes()
            assert result["validation_metrics"] is None and jobs.at[item["job_id"], "validation_status"] == "not_scored"
        capacity = result["physical_constants"]["capacity_m3"]
        for name, role in result["selected_parameters"].items():
            targets = pd.read_csv(folder / role["monthly_targets_file"], index_col="month", float_precision="round_trip")
            assert targets.shape == (12, 6) and list(targets.index) == list(range(1, 13))
            array = targets.values.copy()
            array[:, :3] /= capacity
            np.testing.assert_allclose(array.T.ravel(), vectors.loc[role["evaluation_id"]].values, rtol=0, atol=2e-15)
            if sampled and name != "generalized_baseline":
                old = archived_roles[name]
                assert role["evaluation_id"] == old["evaluation_id"]
                assert role["calibration_release_nse"] == old["calibration_release_nse"]
                assert role["calibration_storage_nse"] == old["calibration_storage_nse"]
                assert role["historical_validation"]["release_nse"] == old["validation_release_nse"]
                assert role["historical_validation"]["storage_nse"] == old["validation_storage_nse"]
                assert result["validation_metrics"][name] == role["historical_validation"]
            tables += 1
        balanced = result["selected_parameters"]["balanced"]
        np.testing.assert_array_equal(jobs.loc[item["job_id"], ["balanced_calibration_release_nse", "balanced_calibration_storage_nse"]].values.astype(float),
                                      [balanced["calibration_release_nse"], balanced["calibration_storage_nse"]])
        if sampled:
            np.testing.assert_array_equal(jobs.loc[item["job_id"], ["balanced_historical_validation_release_nse", "balanced_historical_validation_storage_nse"]].values.astype(float),
                                          [balanced["historical_validation"]["release_nse"], balanced["historical_validation"]["storage_nse"]])
        if identifier == "ex6_Oahe":
            assert balanced["historical_validation"]["storage_nse"] < 0 < validation["storage_nse"]
    assert tables == 36
    for identifier in ("ex6_Oahe", "ex18_High_aswan"):
        assert jobs.loc[jobs.reservoir_id.eq(identifier), "fixed_step_days"].eq(30.5).all()
    old = root / "outputs/archive_workflow_20261010"
    old_proof = read(old / "workflow_verification.json")
    for name, digest in old_proof["artifact_sha256"].items():
        assert sha(old / name) == digest
    for name, digest in old_proof["new_root_files_sha256"].items():
        assert sha(root / name) == digest
    reference = read(root / "outputs/bull_lake_interval_population_20261009/preflight.json")["code_sha256"]
    actual = {p.relative_to(root / "src/dztr").as_posix(): sha(p) for p in (root / "src/dztr").rglob("*.py")}
    assert actual == reference and len(actual) == 59
    historical_balance = read(root / "outputs/ready_cohort_20261008/pilot_verification.json")
    return {"status": "passed", "active_reservoirs": 41, "reservoirs_with_results": 4,
            "exported_jobs": 9, "monthly_role_tables": tables, "numerical_modules_unchanged": 59,
            "prior_standardized_artifacts_unchanged": len(old_proof["artifact_sha256"]),
            "historical_validation_scores_preserved": True, "new_model_evaluations": 0, "validation_calls": 0,
            "oahe_storage_generalization_limitation_preserved": True,
            "historical_scientific_verification_reused": historical_balance, "scientific_simulations_rerun": 0}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(root, args.output), indent=2))
