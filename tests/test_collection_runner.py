"""Collection membership, portable inputs, job isolation and hash protection."""

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from dztr.integrations import collection as module


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def collection_fixture(root, count=1):
    entries = []
    for number in range(1, count + 1):
        identifier = f"ex{number}_Example"
        dates = pd.date_range("2000-01-01", "2007-12-01", freq="MS", name="date")
        years = dates.year.to_numpy() - 2000
        months = dates.month.to_numpy()
        frame = pd.DataFrame({"storage_m3": 3e8 + 1e6 * years + 2e6 * months,
                              "inflow_m3s": 2 + 0.04 * months + 0.01 * years,
                              "release_m3s": 1 + 0.03 * months + 0.05 * years}, index=dates)
        csv = root / "data/reservoirs" / identifier / "observations_si.csv"
        csv.parent.mkdir(parents=True)
        frame.to_csv(csv, float_format="%.17g")
        config_path = root / "configs/reservoirs" / f"{identifier}.json"
        config = {"schema_version": 1, "kind": "world_reservoir_sample", "reservoir": {"id": identifier, "name": f"Example {number}"},
                  "observations_file": f"../../data/reservoirs/{identifier}/observations_si.csv", "observations_sha256": module.sha256(csv),
                  "units": {"storage": "m3", "flows": "m3/s"},
                  "sampling": {"frequency": "monthly", "fixed_step_days": 30.5, "month_clock": "observed", "legacy_start": None,
                               "allow_signed_inflow": False},
                  "periods": {"simulation_start": "2000-01-01", "calibration": {"start": "2001-01-01", "end": "2003-12-01"},
                              "validation": {"start": "2004-01-01", "end": "2007-12-01"}, "warmup_rows": 12},
                  "model": {"capacity_m3": 1e9, "initial_storage_m3": float(frame.storage_m3.iloc[0]), "initial_release_m3s": 1,
                            "channel_capacity_m3s": 20, "dead_storage_fraction": 0.1, "inflow_correction": 1, "flood_mode": "storage"},
                  "inflow_preprocessing": {"factor": 1, "offset_m3s": 0}, "simulation_supported": True,
                  "target_percentiles": [10, 45, 85], "quantile_method": "hazen", "atol_storage_m3": 1e-4,
                  "protocol": {"estimation_policy": "calibration only"}}
        _json(config_path, config)
        files = {str(path.relative_to(root)).replace("\\", "/"): module.sha256(path) for path in (csv, config_path)}
        entries.append({"reservoir_id": identifier, "name": f"Example {number}", "frequency": "monthly", "rows": len(frame),
                        "first_date": "2000-01-01", "last_date": "2007-12-01",
                        "config_file": str(config_path.relative_to(root)).replace("\\", "/"), "config_sha256": module.sha256(config_path),
                        "observations_file": str(csv.relative_to(root)).replace("\\", "/"), "observations_sha256": module.sha256(csv),
                        "files_sha256": files})
    manifest = {"schema_version": 1, "kind": "reservoir_collection", "repository_root": "../..", "reservoirs": entries,
                "excluded_reservoirs": [{"reservoir_id": "ex43_TuyenQuang", "reason": "Excluded by user"}]}
    path = root / "data/reservoirs/manifest.json"
    _json(path, manifest)
    return path, manifest


def test_run_selected_name_uses_permanent_inputs_and_exports_readiness(tmp_path):
    root = tmp_path / "repo"
    path, _ = collection_fixture(root)
    result = module.run_collection(path, tmp_path / "result", reservoirs=["Example 1"])
    assert result["status"] == "completed" and result["counts"] == {"completed": 1}
    assert result["input_files_unchanged"] and result["manifest_file_unchanged"] and result["code_files_unchanged"]
    assert (tmp_path / "result/ex1_Example/experiment/baseline_simulation.csv").is_file()
    assert (tmp_path / "result/ex1_Example/readiness.json").is_file()


def test_portable_collection_after_repository_move(tmp_path):
    root = tmp_path / "original"
    path, _ = collection_fixture(root)
    before = module.run_collection(path, tmp_path / "before", all_reservoirs=True)
    moved = tmp_path / "moved"
    shutil.move(str(root), moved)
    after = module.run_collection(moved / "data/reservoirs/manifest.json", tmp_path / "after", all_reservoirs=True)
    assert before["counts"] == after["counts"]
    for name in ("baseline_simulation.csv", "calibration_targets_si.csv", "calibration_bounds_si.csv"):
        old = pd.read_csv(tmp_path / "before/ex1_Example/experiment" / name, float_precision="round_trip")
        new = pd.read_csv(tmp_path / "after/ex1_Example/experiment" / name, float_precision="round_trip")
        pd.testing.assert_frame_equal(old, new, check_exact=True)


def test_tampered_member_is_isolated_without_simulation(tmp_path):
    root = tmp_path / "repo"
    path, manifest = collection_fixture(root, count=2)
    damaged = root / manifest["reservoirs"][0]["observations_file"]
    damaged.write_text(damaged.read_text() + "tamper\n")
    result = module.run_collection(path, tmp_path / "run", all_reservoirs=True)
    assert result["counts"] == {"input_error": 1, "completed": 1}
    assert "SHA256" in result["reservoirs"][0]["reason"]
    assert not (tmp_path / "run/ex1_Example/experiment").exists()


def test_declared_exclusion_cannot_be_selected(tmp_path):
    path, _ = collection_fixture(tmp_path / "repo")
    with pytest.raises(ValueError, match="active member"):
        module.run_collection(path, tmp_path / "run", reservoirs="ex43_TuyenQuang")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("mutation", ["duplicate", "excluded_member", "unsafe_id", "path_escape", "windows_absolute"])
def test_manifest_rejects_ambiguous_or_unsafe_membership(tmp_path, mutation):
    path, manifest = collection_fixture(tmp_path / "repo")
    entry = manifest["reservoirs"][0]
    if mutation == "duplicate":
        manifest["reservoirs"].append(entry.copy())
    elif mutation == "excluded_member":
        manifest["excluded_reservoirs"].append({"reservoir_id": entry["reservoir_id"]})
    elif mutation == "unsafe_id":
        entry["reservoir_id"] = "../somewhere"
    elif mutation == "path_escape":
        entry["config_file"] = "../outside.json"
    else:
        entry["config_file"] = "E:\\outside.json"
    _json(path, manifest)
    with pytest.raises(ValueError):
        module.load_collection(path)


@pytest.mark.parametrize("selection", [[], ["Example 1", "Example 1"], ["unknown"]])
def test_selection_errors_are_detected_before_outputs(tmp_path, selection):
    path, _ = collection_fixture(tmp_path / "repo")
    with pytest.raises(ValueError):
        module.run_collection(path, tmp_path / "run", reservoirs=selection)
    assert not (tmp_path / "run").exists()


def test_output_cannot_overwrite_another_members_inputs(tmp_path):
    root = tmp_path / "repo"
    path, _ = collection_fixture(root, count=2)
    with pytest.raises(ValueError, match="protected source inputs"):
        module.run_collection(path, root / "data/reservoirs/ex2_Example", reservoirs=["Example 1"])


def test_manifest_mutation_during_execution_invalidates_run(tmp_path, monkeypatch):
    path, _ = collection_fixture(tmp_path / "repo")
    original = module.run_world_sample

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        path.write_text(path.read_text() + " ")
        return result

    monkeypatch.setattr(module, "run_world_sample", changed)
    report = module.run_collection(path, tmp_path / "run", all_reservoirs=True)
    assert report["status"] == "invalid_integrity" and not report["manifest_file_unchanged"]


def test_observation_mutation_during_execution_invalidates_run(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    path, manifest = collection_fixture(root)
    csv = root / manifest["reservoirs"][0]["observations_file"]
    original = module.run_world_sample

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        csv.write_text(csv.read_text() + " ")
        return result

    monkeypatch.setattr(module, "run_world_sample", changed)
    report = module.run_collection(path, tmp_path / "run", all_reservoirs=True)
    assert report["status"] == "invalid_integrity" and not report["input_files_unchanged"]
    assert report["reservoirs"][0]["status"] == "input_error"


def test_collection_pilot_is_an_explicit_budget(tmp_path):
    path, _ = collection_fixture(tmp_path / "repo")
    result = module.run_collection(path, tmp_path / "run", all_reservoirs=True, optimization_budget=2, seed=42)
    assert result["status"] == "completed"
    report = json.loads((tmp_path / "run/ex1_Example/experiment/experiment.json").read_text())
    assert report["optimization_status"] == "pilot_completed" and report["optimization"]["evaluations"] == 2


def test_legacy_cannot_accidentally_launch_an_optimizer(tmp_path):
    path, _ = collection_fixture(tmp_path / "repo")
    with pytest.raises(ValueError, match="baseline mode"):
        module.run_collection(path, tmp_path / "run", all_reservoirs=True, mode="legacy", optimization_budget=2)
