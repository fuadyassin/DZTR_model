"""Read-only audit of standardized exports; no model or optimizer imports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    proof = read(out / "verification.json")
    assert proof["status"] == "passed" and proof["new_model_evaluations"] == proof["validation_calls"] == 0
    for name, digest in proof["artifact_sha256"].items():
        assert sha(out / name) == digest, name
    for name, digest in proof["consumed_files_sha256"].items():
        assert sha(root / name) == digest, name
    catalog = read(out / "input_catalog.json")
    assert catalog["reservoir_count"] == 41 and len(catalog["reservoirs"]) == 41
    assert len({r["reservoir"]["id"] for r in catalog["reservoirs"]}) == 41
    assert [r["reservoir_id"] for r in catalog["excluded_reservoirs"]] == ["ex43_TuyenQuang"]
    assert not any(r["new_job_authorized"] for r in catalog["reservoirs"])
    counts = {f: sum(r["sampling"]["frequency"] == f for r in catalog["reservoirs"]) for f in ("daily", "monthly")}
    assert counts == {"daily": 27, "monthly": 14}
    index = read(out / "result_index.json")["results"]
    assert len(index) == 3 and {r["seed"] for r in index} == {42, 43, 44}
    roles = 0
    for entry in index:
        folder = (out / entry["result_file"]).parent
        result = read(folder / "result.json")
        assert result["status"] == "budget_reached_convergence_unconfirmed" and not result["convergence_established"]
        assert result["validation_metrics"] is None and not result["validation_scored_by_export"]
        assert result["new_model_evaluations"] == 0 and result["search_evaluations"] == 512
        archive = root / result["archive_directory"]
        original = read(archive / "summary.json")
        assert result["checkpoint_identity"] == original["checkpoint_identity"]
        assert sha(root / result["checkpoint_file"]) == result["checkpoint_sha256"]
        for name in ("pareto_front.csv", "pareto_vectors.csv", "checkpoint_metrics.csv", "stopping_history.csv"):
            assert (folder / name).read_bytes() == (archive / name).read_bytes()
        vectors = pd.read_csv(archive / "evaluated_vectors.csv", index_col="evaluation_id", float_precision="round_trip")
        trials = pd.read_csv(archive / "evaluations.csv", index_col="evaluation_id", float_precision="round_trip")
        assert {k: v["evaluation_id"] for k, v in result["selected_parameters"].items()} == original["selected_candidate_ids"]
        capacity = result["physical_constants"]["capacity_m3"]
        for role in result["selected_parameters"].values():
            monthly = pd.read_csv(folder / role["monthly_targets_file"], index_col="month", float_precision="round_trip")
            assert list(monthly.index) == list(range(1, 13)) and monthly.shape == (12, 6)
            reconstructed = monthly.to_numpy().copy()
            reconstructed[:, :3] /= capacity
            np.testing.assert_allclose(reconstructed.T.ravel(), vectors.loc[role["evaluation_id"]].values, rtol=0, atol=2e-15)
            np.testing.assert_array_equal([role["calibration_release_nse"], role["calibration_storage_nse"]], trials.loc[role["evaluation_id"], ["release_nse", "storage_nse"]].values.astype(float))
            roles += 1
    # Authenticate the previous complete study, including its numerical code
    # identity. Reporting additions do not enter this hashed source tree.
    old = root / "outputs/bull_lake_interval_population_20261009"
    prior = read(old / "verification.json")
    for name, digest in prior["artifact_sha256"].items():
        assert sha(old / name) == digest, name
    for name, digest in prior["new_root_files_sha256"].items():
        assert sha(root / name) == digest, name
    expected_code = read(old / "preflight.json")["code_sha256"]
    actual_code = {p.relative_to(root / "src/dztr").as_posix(): sha(p) for p in (root / "src/dztr").rglob("*.py")}
    assert actual_code == expected_code and len(actual_code) == 59
    return {"status": "passed", "active_reservoirs": 41, "native_sampling": counts,
            "exported_jobs": 3, "monthly_role_tables": roles, "prior_population_artifacts_unchanged": len(prior["artifact_sha256"]),
            "numerical_modules_unchanged": 59, "new_model_evaluations": 0, "validation_calls": 0,
            "native_vectors_reconstructed_from_si_tables": True, "all_seed_fronts_preserved": True}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(root, args.output), indent=2))
