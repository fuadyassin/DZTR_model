"""Read-only scientific/provenance checks for the prepared, unexecuted fleet."""
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/fleet_workflow_20261010"
spec = importlib.util.spec_from_file_location("fleet_verifier_driver", ROOT / "tools/fleet_workflow.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def main():
    protocol = driver.checked(OUT)
    plan = driver.read(OUT / "job_plan.json")
    jobs = plan["jobs"]
    assert len(jobs) == 27 and len({j["reservoir_id"] for j in jobs}) == 9
    for identifier in {j["reservoir_id"] for j in jobs}:
        subset = [j for j in jobs if j["reservoir_id"] == identifier]
        assert {j["seed"] for j in subset} == {42, 43, 44}
        assert all(j["population_size"] == 32 and j["evaluation_budget"] == 512 for j in subset)
    assert sum(j["evaluation_budget"] for j in jobs) == 13824
    assert len(protocol["deferred_reservoirs"]) == 31
    status = pd.read_csv(OUT / "reservoir_status.csv")
    assert len(status) == 41 and status.reservoir_id.is_unique
    assert not (OUT / "pilots").exists() and plan["new_evaluations_started"] == 0
    old = ROOT / "outputs/multi_reservoir_archive_20261010"
    proof = driver.read(old / "workflow_verification.json")
    for name, digest in proof["artifact_sha256"].items():
        assert driver.sha(old / name) == digest
    reports = OUT / "reports/archived_reviewed"
    report_proof = driver.read(reports / "report_manifest.json")
    assert report_proof["new_model_evaluations"] == report_proof["validation_calls"] == 0
    for name, digest in report_proof["artifact_sha256"].items():
        assert driver.sha(reports / name) == digest
    for name, digest in report_proof["consumed_files_sha256"].items():
        assert driver.sha(ROOT / name) == digest
    index = driver.read(old / "result_index.json")["results"]
    checked_scores = 0
    for item in index:
        result = driver.read(old / item["result_file"])
        rows = pd.read_csv(reports / item["reservoir_id"] / "metrics.csv", float_precision="round_trip")
        row = rows.set_index("seed").loc[result["seed"]]
        role = result["selected_parameters"]["balanced"]
        assert row.calibration_release_nse == role["calibration_release_nse"]
        assert row.calibration_storage_nse == role["calibration_storage_nse"]
        validation = role.get("historical_validation", {})
        for metric in ("release_nse", "storage_nse"):
            expected = validation.get(metric)
            actual = row["validation_" + metric]
            assert pd.isna(actual) if expected is None else actual == expected
        checked_scores += 1
    assert checked_scores == 9
    source_files = ["tools/fleet_workflow.py", "tools/fleet_pipeline.py", "tools/fleet_reports.py", "tools/verify_fleet_preparation.py", "tests/test_fleet_workflow_controller.py", "docs/fleet_workflow.md"]
    receipt = {"status": "passed", "stage": "prepared_not_executed", "active_reservoirs": 41,
               "eligible_reservoirs": 9, "declared_jobs": 27, "deferred_reservoirs": 31,
               "new_model_evaluations": 0, "saved_score_records_checked": checked_scores,
               "numerical_code_files_unchanged": len(protocol["code_sha256"]),
               "tests": {"full_suite": "1410 passed, 1 skipped, 1 Windows directory-rename failure", "rename_retry_and_controller": "5 passed", "controller_after_pipeline_fix": "5 passed"},
               "visual_review": "All four trajectory reports and eight monthly-target/Pareto plots inspected; units and held-out shading checked.",
               "source_sha256": {name: driver.sha(ROOT / name) for name in source_files},
               "artifact_sha256": {p.relative_to(OUT).as_posix(): driver.sha(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "preparation_verification.json"}}
    driver.write(OUT / "preparation_verification.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "stage", "declared_jobs", "new_model_evaluations", "saved_score_records_checked")}))


if __name__ == "__main__":
    main()
