"""Archive-only export: units, identities, exclusions, and refusal of drift."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("archive_export", Path(__file__).parents[1] / "tools/standardize_archives.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root):
    entries = []
    for identifier, frequency in (("daily", "daily"), ("monthly", "monthly")):
        observations = root / f"data/{identifier}.csv"
        observations.parent.mkdir(exist_ok=True)
        observations.write_text("date,inflow_m3s,release_m3s,storage_m3\n2001-01-01,-1,2,500\n", encoding="utf-8")
        config_path = root / f"configs/{identifier}.json"
        config = {"kind": "world_reservoir_sample", "reservoir": {"id": identifier, "name": identifier},
                  "observations_file": f"../data/{identifier}.csv", "units": {"storage": "m3", "flows": "m3/s"},
                  "sampling": {"frequency": frequency, "fixed_step_days": 1 if frequency == "daily" else 30.5, "allow_signed_inflow": True},
                  "periods": {"calibration": {"start": "2001-01-01", "end": "2001-12-31"}},
                  "model": {"capacity_m3": 1000, "dead_storage_fraction": .1, "channel_capacity_m3s": 10}}
        put(config_path, config)
        files = {p.relative_to(root).as_posix(): module.sha(p) for p in (observations, config_path)}
        entries.append({"reservoir_id": identifier, "frequency": frequency, "config_file": f"configs/{identifier}.json",
                        "observations_file": f"data/{identifier}.csv", "config_sha256": module.sha(config_path),
                        "observations_sha256": module.sha(observations), "files_sha256": files,
                        "readiness": {"optimization_status": "ready"}})
    collection = {"kind": "reservoir_collection", "schema_version": 1, "reservoir_count": 2,
                  "reservoirs": entries, "excluded_reservoirs": [{"reservoir_id": "Tuyen"}]}
    put(root / "collection.json", collection)
    policy = {"reservoir_id": "daily", "policy_id": "signed_daily_v1", "observations_sha256": entries[0]["observations_sha256"]}
    put(root / "policy.json", policy)
    archive = root / "archives/seed_42"
    archive.mkdir(parents=True)
    vector = np.repeat([.2, .4, .8, 1, 2, 3], 12)
    pd.DataFrame(np.tile(vector, (512, 1)), columns=[f"x{i:02}" for i in range(72)], index=pd.Index(range(512), name="evaluation_id")).to_csv(archive / "evaluated_vectors.csv")
    pd.DataFrame({"evaluation_id": range(512), "feasible": True, "release_nse": .5, "storage_nse": .6}).to_csv(archive / "evaluations.csv", index=False)
    summary = {"reservoir": {"id": "daily", "name": "daily"}, "optimization": {"population_size": 32, "seed": 42, "evaluation_budget": 512},
               "search_evaluations": 512, "checkpoint_evaluations": 512, "objective_order": ["negative_release_nse", "negative_storage_nse"],
               "reporting_validation_evaluations": 0, "policy_id": "signed_daily_v1",
               "consumed_files_sha256": {str(p.resolve()): module.sha(p) for p in (root / "configs/daily.json", root / "data/daily.csv", root / "policy.json")},
               "periods": {"calibration": {"start": "2001-01-01", "end": "2001-12-31"}},
               "selected_candidate_ids": {"generalized_baseline": 0, "best_release": 1, "best_storage": 2, "balanced": 3},
               "checkpoint_identity": {}, "bounds": {}, "root_policy": "reject", "conditional_hindcast": True,
               "objective_direction": "minimize", "feasible_evaluations": 512, "rejected_evaluations": 0, "pareto_count": 1, "stopping_policy": {}}
    put(archive / "summary.json", summary)
    for name in module.ARCHIVE_NAMES:
        if not (archive / name).exists():
            (archive / name).write_text("preserved archived bytes\n", encoding="utf-8")
    proof = {"status": "passed", "artifact_sha256": {f"seed_42/{name}": module.sha(archive / name) for name in module.ARCHIVE_NAMES}}
    put(root / "archives/verification.json", proof)
    worklist = {"schema_version": 1, "kind": "reservoir_archive_worklist", "execution_mode": "archive_export_only",
                "collection_file": "collection.json", "collection_sha256": module.sha(root / "collection.json"),
                "decision": {"population_size": 32, "evaluation_cap": 512, "seeds": [42], "convergence_established": False,
                             "additional_model_evaluations_authorized": False, "evidence_file": "archives/verification.json",
                             "evidence_sha256": module.sha(root / "archives/verification.json")},
                "archives": [{"reservoir_id": "daily", "seed": 42, "archive_directory": "archives/seed_42", "policy_file": "policy.json",
                              "policy_sha256": module.sha(root / "policy.json"), "verification_file": "archives/verification.json",
                              "verification_sha256": module.sha(root / "archives/verification.json")}]}
    put(root / "worklist.json", worklist)
    return worklist


def test_offline_export_preserves_native_inputs_and_archives(tmp_path):
    fixture(tmp_path)
    proof = module.build(tmp_path, tmp_path / "worklist.json", tmp_path / "export")
    assert proof["new_model_evaluations"] == proof["validation_calls"] == 0
    catalog = module.read(tmp_path / "export/input_catalog.json")
    assert catalog["reservoir_count"] == 2 and catalog["excluded_reservoirs"] == [{"reservoir_id": "Tuyen"}]
    assert catalog["reservoirs"][1]["sampling"]["fixed_step_days"] == 30.5
    assert not any(r["new_job_authorized"] for r in catalog["reservoirs"])
    folder = tmp_path / "export/results/daily__seed_42__pop_32__budget_512"
    result = module.read(folder / "result.json")
    assert result["validation_metrics"] is None and not result["convergence_established"]
    monthly = pd.read_csv(folder / "balanced_monthly_targets_si.csv", index_col="month")
    np.testing.assert_allclose(monthly.values, np.tile([200, 400, 800, 1, 2, 3], (12, 1)))
    for name in ("pareto_front.csv", "pareto_vectors.csv"):
        assert (folder / name).read_bytes() == (tmp_path / "archives/seed_42" / name).read_bytes()
    for name, digest in proof["artifact_sha256"].items():
        assert module.sha(tmp_path / "export" / name) == digest


@pytest.mark.parametrize("file", ["data/daily.csv", "archives/seed_42/evaluated_vectors.csv", "policy.json"])
def test_changed_input_or_archive_refused_before_output(tmp_path, file):
    fixture(tmp_path)
    path = tmp_path / file
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        module.build(tmp_path, tmp_path / "worklist.json", tmp_path / "export")
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("path", ["../escape", "C:/outside", "/absolute", "folder/../../escape"])
def test_repository_path_escape_refused(tmp_path, path):
    with pytest.raises(ValueError):
        module.inside(tmp_path, path)


@pytest.mark.parametrize("bad", [np.repeat([.4, .2, .8, 1, 2, 3], 12), np.repeat([.2, .4, 1.1, 1, 2, 3], 12), np.repeat([.2, .4, .8, 3, 2, 1], 12)])
def test_invalid_targets_are_not_repaired(bad):
    with pytest.raises(ValueError, match="ordering"):
        module.decode_monthly(bad, {"capacity_m3": 1000})


def test_existing_output_preserved(tmp_path):
    fixture(tmp_path)
    output = tmp_path / "export"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        module.build(tmp_path, tmp_path / "worklist.json", output)
    assert (output / "keep.txt").read_text() == "keep"


def test_excluded_membership_refused(tmp_path):
    worklist = fixture(tmp_path)
    collection = module.read(tmp_path / "collection.json")
    collection["excluded_reservoirs"].append({"reservoir_id": "daily"})
    put(tmp_path / "collection.json", collection)
    worklist["collection_sha256"] = module.sha(tmp_path / "collection.json")
    put(tmp_path / "worklist.json", worklist)
    with pytest.raises(ValueError, match="Excluded"):
        module.build(tmp_path, tmp_path / "worklist.json", tmp_path / "export")


def test_execution_request_refused(tmp_path):
    worklist = fixture(tmp_path)
    worklist["execution_mode"] = "optimize"
    put(tmp_path / "worklist.json", worklist)
    with pytest.raises(ValueError, match="archive_export_only"):
        module.build(tmp_path, tmp_path / "worklist.json", tmp_path / "export")
