"""Byte-preserving collection packaging with portable, miniature inputs."""

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from dztr.integrations import packaging
from dztr.integrations.packaging import package_world_samples


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def collection(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    catalog_rows, cohort_rows, active, originals = [], [], [], {}
    for reservoir_id, name, frequency, dates in (
        ("ex12_Trinity", "Trinity", "daily", ["2000-01-01", "2000-01-02", "2000-01-03"]),
        ("ex31_Brazeau", "Brazeau", "monthly", ["2000-01-01", "2000-02-01", "2000-03-01"]),
    ):
        folder = root / "outputs" / "historical" / "reservoirs" / reservoir_id
        folder.mkdir(parents=True)
        observations = folder / "observations_si.csv"
        observations.write_bytes(("date,storage_m3,inflow_m3s,release_m3s\n" +
                "\n".join(f"{day},300000000.12345678,-0.25,3.125" for day in dates) + "\n").encode())
        config = {"schema_version": 1, "kind": "world_reservoir_sample", "reservoir": {"id": reservoir_id, "name": name},
            "units": {"storage": "m3", "flows": "m3/s"}, "observations_file": "observations_si.csv",
            "observations_sha256": digest(observations),
            "model": {"capacity_m3": 1e9, "initial_storage_m3": 3e8, "initial_release_m3s": 1,
                      "channel_capacity_m3s": 60, "dead_storage_fraction": 0.1, "inflow_correction": 0.92, "flood_mode": "inflow"},
            "sampling": {"frequency": frequency, "fixed_step_days": 1.0 if frequency == "daily" else 30.5,
                         "month_clock": "observed", "legacy_start": None, "allow_signed_inflow": False},
            "periods": {"simulation_start": dates[0], "calibration": {"start": dates[0], "end": dates[1]},
                        "validation": {"start": dates[2], "end": dates[2]}, "warmup_rows": 0},
            "inflow_preprocessing": {"factor": 1.1, "offset_m3s": 8},
            "simulation_supported": reservoir_id != "ex31_Brazeau", "target_percentiles": [10, 45, 85],
            "quantile_method": "hazen", "atol_storage_m3": 1e-4,
            "legacy": {"fixed_point_iterations": 1000, "release_shift_rows": 4, "storage_shift_rows": -3},
            "protocol": {"paper_reproduction": False, "note": "Preserve all original settings."}}
        write_json(folder / "sample.json", config)
        write_json(folder / "observation_provenance.json", {"reservoir_id": reservoir_id,
             "source_path": "E:\\archived\\res_info_out.mat", "signed_values_preserved": True})
        write_json(folder / "matlab_source.json", {"reservoir_id": reservoir_id, "capacity_warning": "Preserved"})
        (folder / "legacy_targets_si.csv").write_bytes(b"month,storage_critical_m3\r\n1,300000000.12345678\r\n")
        (folder / "legacy_bounds_si.csv").write_bytes(b"month,lower_storage_critical_m3\n1,299000000.0\n")
        source_config = str((folder / "sample.json").relative_to(root)).replace("\\", "/")
        stages = {"baseline_status": "blocked", "optimization_status": "ready" if frequency == "daily" else "blocked",
                  "legacy_status": "completed" if frequency == "daily" else "blocked"}
        record = {"reservoir_id": reservoir_id, "name": name, "frequency": frequency, "import_status": "imported",
                  "sample_config": source_config, "model": config["model"], "sampling": config["sampling"], "periods": config["periods"], **stages}
        catalog_rows.append(record)
        cohort_rows.append({"reservoir_id": reservoir_id, "frequency": frequency, "import_status": "imported",
                            "rows": 3, "first_date": dates[0], "last_date": dates[-1], **stages,
                            "baseline_reason": "Explicit forcing review required.", "optimization_reason": "External forcing required." if frequency == "monthly" else None})
        active.append({"reservoir_id": reservoir_id, "sample_config": source_config})
        originals[reservoir_id] = config
    excluded = {"reservoir_id": "ex43_TuyenQuang", "name": "TuyenQuang", "status": "intentionally_excluded",
                "reason": "Excluded at user request.", "audit_file": "outputs/historical/reservoirs/ex43_TuyenQuang/observation_provenance.json"}
    catalog_rows.append({"reservoir_id": excluded["reservoir_id"], "import_status": "source_conflict"})
    cohort_rows.append({"reservoir_id": excluded["reservoir_id"], "import_status": "source_conflict"})
    cohort = {"schema_version": 1, "source_files_unchanged": True, "code_files_unchanged": True, "reservoirs": cohort_rows}
    write_json(root / "outputs/historical/cohort.json", cohort)
    write_json(root / "outputs/historical/verification.json", {"status": "passed", "cohort_sha256": digest(root / "outputs/historical/cohort.json")})
    catalog = {"schema_version": 1, "kind": "world_reservoir_sample_catalog", "cohort_report": "outputs/historical/cohort.json", "reservoirs": catalog_rows}
    write_json(root / "configs/world_reservoir_collection.json", catalog)
    scope = {"schema_version": 1, "kind": "world_reservoir_scope", "paths_relative_to": "repository_root",
             "catalog_file": "configs/world_reservoir_collection.json", "catalog_sha256": digest(root / "configs/world_reservoir_collection.json"),
             "source_reservoir_count": 3, "active_reservoir_count": 2, "active_reservoirs": active, "excluded_reservoirs": [excluded]}
    write_json(root / "configs/world_reservoir_scope.json", scope)
    return root, originals


def all_hashes(path):
    return {str(item.relative_to(path)): digest(item) for item in path.rglob("*") if item.is_file()}


def test_packages_only_declared_scope_with_byte_exact_data_and_preserved_settings(collection):
    root, originals = collection
    old_outputs = all_hashes(root / "outputs")
    old_scope, old_catalog = digest(root / "configs/world_reservoir_scope.json"), digest(root / "configs/world_reservoir_collection.json")
    manifest = package_world_samples(root)
    assert manifest["kind"] == "reservoir_collection" and manifest["schema_version"] == 1
    assert manifest["repository_root"] == "../.." and manifest["reservoir_count"] == 2
    assert manifest["units"] == {"storage": "m3", "flows": "m3/s"}
    assert manifest["counts"]["frequency"] == {"daily": 1, "monthly": 1}
    assert manifest["counts"]["optimization_status"] == {"ready": 1, "blocked": 1}
    assert manifest["excluded_reservoirs"][0]["reservoir_id"] == "ex43_TuyenQuang"
    assert not (root / "data/reservoirs/ex43_TuyenQuang").exists()
    for record in manifest["reservoirs"]:
        reservoir_id = record["reservoir_id"]
        config_path = root / record["config_file"]
        actual = json.loads(config_path.read_text())
        expected = originals[reservoir_id]
        assert actual["observations_file"] == f"../../data/reservoirs/{reservoir_id}/observations_si.csv"
        assert {key: value for key, value in actual.items() if key not in packaging._REFERENCES and key != "observations_file"} == {
                key: value for key, value in expected.items() if key != "observations_file"}
        for key, filename in packaging._REFERENCES.items():
            assert (config_path.parent / actual[key]).resolve() == root / "data/reservoirs" / reservoir_id / filename
        for relative_path, expected_hash in record["files_sha256"].items():
            assert digest(root / relative_path) == expected_hash
        for filename in packaging._DATA_FILES:
            assert (root / "data/reservoirs" / reservoir_id / filename).read_bytes() == (
                root / "outputs/historical/reservoirs" / reservoir_id / filename).read_bytes()
        assert len(record["files_sha256"]) == 6 and record["rows"] == 3
        assert "Explicit forcing review required." in record["readiness"]["reasons"]
    receipt = json.loads((root / "data/reservoirs/package_receipt.json").read_text())
    assert receipt["manifest_sha256"] == digest(root / "data/reservoirs/manifest.json")
    assert receipt["source_files_sha256"]["configs/world_reservoir_scope.json"] == old_scope
    assert receipt["source_files_sha256"]["configs/world_reservoir_collection.json"] == old_catalog
    assert old_outputs == all_hashes(root / "outputs")
    assert digest(root / "configs/world_reservoir_scope.json") == old_scope


def test_exact_second_packaging_is_idempotent_without_replacing_files(collection):
    root, _ = collection
    first = package_world_samples(root)
    hashes = all_hashes(root / "data")
    times = {str(path): path.stat().st_mtime_ns for path in (root / "data").rglob("*") if path.is_file()}
    assert package_world_samples(root) == first
    assert all_hashes(root / "data") == hashes
    assert {str(path): path.stat().st_mtime_ns for path in (root / "data").rglob("*") if path.is_file()} == times


def test_packaged_files_are_portable_without_historical_outputs(collection, tmp_path):
    root, _ = collection
    manifest = package_world_samples(root)
    moved = tmp_path / "relocated"
    shutil.copytree(root / "data", moved / "data")
    shutil.copytree(root / "configs/reservoirs", moved / "configs/reservoirs")
    manifest_path = moved / "data/reservoirs/manifest.json"
    repository = (manifest_path.parent / manifest["repository_root"]).resolve()
    assert repository == moved.resolve() and not (moved / "outputs").exists()
    for record in manifest["reservoirs"]:
        config_file = repository / record["config_file"]
        config = json.loads(config_file.read_text())
        assert (config_file.parent / config["observations_file"]).is_file()
        for field in packaging._REFERENCES:
            assert (config_file.parent / config[field]).is_file()
        assert all(digest(repository / path) == sha for path, sha in record["files_sha256"].items())


@pytest.mark.parametrize("tampered", ["observations_si.csv", "catalog", "configuration", "cohort"])
def test_source_tampering_fails_before_creating_any_destination(collection, tampered):
    root, _ = collection
    if tampered == "observations_si.csv":
        path = root / "outputs/historical/reservoirs/ex31_Brazeau/observations_si.csv"
        path.write_bytes(path.read_bytes() + b"\n")
    elif tampered == "catalog":
        path = root / "configs/world_reservoir_collection.json"
        path.write_text(path.read_text() + "\n")
    elif tampered == "configuration":
        path = root / "outputs/historical/reservoirs/ex31_Brazeau/sample.json"
        config = json.loads(path.read_text())
        config["model"]["inflow_correction"] = 0.5
        write_json(path, config)
    else:
        path = root / "outputs/historical/cohort.json"
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="(SHA256|settings differ)"):
        package_world_samples(root)
    assert not (root / "data").exists() and not (root / "configs/reservoirs").exists()


def test_later_destination_conflict_does_not_leave_earlier_new_copies(collection):
    root, _ = collection
    conflict = root / "data/reservoirs/ex31_Brazeau/legacy_bounds_si.csv"
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"user changes")
    with pytest.raises(ValueError, match="will not be overwritten"):
        package_world_samples(root)
    assert conflict.read_bytes() == b"user changes"
    assert not (root / "data/reservoirs/ex12_Trinity").exists()
    assert not (root / "configs/reservoirs").exists()


def test_missing_evidence_blocks_entire_plan(collection):
    root, _ = collection
    (root / "outputs/historical/reservoirs/ex31_Brazeau/matlab_source.json").unlink()
    with pytest.raises(ValueError, match="Required packaging source is missing"):
        package_world_samples(root)
    assert not (root / "data").exists()


@pytest.mark.parametrize("mutation,match", [
    (lambda scope: scope["active_reservoirs"].append(scope["active_reservoirs"][0]), "Duplicate"),
    (lambda scope: scope.update(active_reservoir_count=99), "counts disagree"),
    (lambda scope: scope["active_reservoirs"][0].update(reservoir_id="../unsafe"), "Unsafe"),
    (lambda scope: scope["active_reservoirs"][0].update(sample_config="../outside.json"), "traversal"),
    (lambda scope: scope["active_reservoirs"][0].update(sample_config="E:/outside.json"), "repository-relative"),
    (lambda scope: scope.update(excluded_reservoirs=[]), "IDs must agree"),
])
def test_malformed_scope_is_rejected_without_outputs(collection, mutation, match):
    root, _ = collection
    path = root / "configs/world_reservoir_scope.json"
    scope = json.loads(path.read_text())
    mutation(scope)
    write_json(path, scope)
    with pytest.raises(ValueError, match=match):
        package_world_samples(root)
    assert not (root / "data").exists()


def test_source_changes_after_preflight_trigger_rollback_without_editing_archive(collection, monkeypatch):
    root, _ = collection
    original_guard = packaging._PackagePlan.guard_sources
    calls = []
    altered = root / "outputs/historical/reservoirs/ex12_Trinity/matlab_source.json"

    def concurrent_edit(self):
        calls.append(1)
        if len(calls) == 2:
            altered.write_bytes(b"concurrent user edit")
        original_guard(self)

    monkeypatch.setattr(packaging._PackagePlan, "guard_sources", concurrent_edit)
    with pytest.raises(ValueError, match="source changed"):
        package_world_samples(root)
    assert altered.read_bytes() == b"concurrent user edit"
    assert not (root / "data/reservoirs/manifest.json").exists()
    assert not (root / "configs/reservoirs").exists()


def test_publish_failure_rolls_back_only_new_files_and_preserves_existing_exact_copy(collection, monkeypatch):
    root, _ = collection
    existing = root / "data/reservoirs/ex12_Trinity/observations_si.csv"
    existing.parent.mkdir(parents=True)
    shutil.copyfile(root / "outputs/historical/reservoirs/ex12_Trinity/observations_si.csv", existing)
    before = existing.read_bytes()
    original_link = packaging.os.link
    calls = []

    def fail_link(source, destination):
        calls.append(destination)
        if len(calls) == 3:
            raise OSError("Injected publication failure")
        original_link(source, destination)

    monkeypatch.setattr(packaging.os, "link", fail_link)
    with pytest.raises(OSError, match="publication failure"):
        package_world_samples(root)
    assert existing.read_bytes() == before
    assert sorted(path.name for path in existing.parent.iterdir()) == ["observations_si.csv"]
    assert not list(root.rglob(".package-*.tmp"))
    assert not (root / "data/reservoirs/manifest.json").exists()
