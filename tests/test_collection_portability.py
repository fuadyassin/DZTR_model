"""Offline historical replay from only the permanent data/configuration tree."""

from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from dztr.integrations import collection as module
from dztr.schemas import MONTHLY_TARGET_COLUMNS


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def offline_fixture(root):
    entries, sources, observations = [], {}, {}
    for identifier, frequency in (("ex1_Daily", "daily"), ("ex2_Monthly", "monthly")):
        days = 1 if frequency == "daily" else 30.5
        index = pd.date_range("2000-01-01", periods=16, freq="D" if frequency == "daily" else "MS", name="date")
        inflow = 1 + np.arange(16) * 0.02
        inflow[1] = -0.4  # Historical replay explicitly preserves signed forcing.
        frame = pd.DataFrame({"storage_m3": 3e8 + np.arange(16) * 2e5,
                              "inflow_m3s": inflow, "release_m3s": 0.7 + np.arange(16) * 0.03}, index=index)
        folder = root / "data/reservoirs" / identifier
        folder.mkdir(parents=True)
        csv = folder / "observations_si.csv"
        frame.to_csv(csv, float_format="%.17g")
        model = {"capacity_m3": 1e9, "initial_storage_m3": 2.5e8, "initial_release_m3s": 0.75,
                 "channel_capacity_m3s": 30, "dead_storage_fraction": 0.1,
                 "inflow_correction": 0.9, "flood_mode": "inflow"}
        months = np.arange(1, 13)
        targets = dict(zip(MONTHLY_TARGET_COLUMNS, [list(row) for row in (
            2e8 + months * 1e6, np.full(12, 5e8), np.full(12, 9e8),
            0.5 + months * 0.1, 1 + months * 0.2, 2 + months * 0.3)]))
        spec = {"reservoir_id": identifier, "status": "ready", "simulation_supported": True,
            "folder": "E:\\unavailable_originals\\" + identifier,
            "base_script": "E:\\unavailable_originals\\SimRISM14.m",
            "coefficient_file": "E:\\unavailable_originals\\coefficients.txt",
            "model": model, "raw_monthly_targets": targets,
            "initial_storage_source": "observed", "initial_release_source": "coefficient",
            "inflow_preprocessing": {"factor": 1.1, "offset_m3s": 0.25},
            "sampling": {"frequency": frequency, "timestep_days": days, "legacy_start": "2000-01-01"},
            "warmup_rows": 2, "fixed_point_iterations": 1000, "overflow_mode": "matlab",
            "legacy_release_shift_rows": 4 if frequency == "monthly" else 0,
            "legacy_storage_shift_rows": -3 if frequency == "monthly" else 0}
        write_json(folder / "matlab_source.json", spec)
        write_json(folder / "observation_provenance.json", {"source": "E:\\unavailable_originals\\res_info_out.mat",
                    "old_outputs": "outputs/unavailable_archives", "status": "verified"})
        pd.DataFrame(targets, index=pd.Index(range(1, 13), name="month")).to_csv(folder / "legacy_targets_si.csv")
        (folder / "legacy_bounds_si.csv").write_text("month,lower_storage_critical_m3\n1,100000000\n")
        config_path = root / "configs/reservoirs" / (identifier + ".json")
        config = {"schema_version": 1, "kind": "world_reservoir_sample",
            "reservoir": {"id": identifier, "name": frequency.title()}, "units": {"storage": "m3", "flows": "m3/s"},
            "observations_file": f"../../data/reservoirs/{identifier}/observations_si.csv", "observations_sha256": module.sha256(csv),
            "model": {**model, "initial_storage_m3": float(frame.storage_m3.iloc[0])},
            "sampling": {"frequency": frequency, "fixed_step_days": days, "month_clock": "observed",
                         "legacy_start": None, "allow_signed_inflow": False},
            "periods": {"simulation_start": "2000-01-01", "calibration": {"start": str(index[2].date()), "end": str(index[12].date())},
                        "validation": {"start": str(index[13].date()), "end": str(index[15].date())}, "warmup_rows": 2},
            "legacy": {"sampling": spec["sampling"], "warmup_rows": 2, "fixed_point_iterations": 1000,
                       "release_shift_rows": spec["legacy_release_shift_rows"], "storage_shift_rows": spec["legacy_storage_shift_rows"]},
            "inflow_preprocessing": spec["inflow_preprocessing"], "simulation_supported": True,
            "target_percentiles": [10, 45, 85], "quantile_method": "hazen", "atol_storage_m3": 1e-4,
            "protocol": {"paper_reproduction": False}}
        for field, filename in (("provenance_file", "observation_provenance.json"), ("matlab_source_file", "matlab_source.json"),
                                ("legacy_targets_file", "legacy_targets_si.csv"), ("legacy_bounds_file", "legacy_bounds_si.csv")):
            config[field] = f"../../data/reservoirs/{identifier}/{filename}"
        write_json(config_path, config)
        hashes = {str(path.relative_to(root)).replace("\\", "/"): module.sha256(path) for path in [*folder.iterdir(), config_path]}
        entries.append({"reservoir_id": identifier, "name": frequency.title(), "frequency": frequency,
            "config_file": str(config_path.relative_to(root)).replace("\\", "/"), "config_sha256": module.sha256(config_path),
            "observations_file": str(csv.relative_to(root)).replace("\\", "/"), "observations_sha256": module.sha256(csv),
            "files_sha256": hashes, "rows": len(index), "first_date": str(index[0].date()), "last_date": str(index[-1].date())})
        sources[identifier], observations[identifier] = spec, frame
    manifest = {"schema_version": 1, "kind": "reservoir_collection", "repository_root": "../..", "reservoir_count": 2,
        "units": {"storage": "m3", "flows": "m3/s"}, "reservoirs": entries,
        "excluded_reservoirs": [{"reservoir_id": "ex43_TuyenQuang", "audit_file": "outputs/unavailable_archives/tuyen.json"}],
        "provenance": {"cohort_file": "outputs/unavailable_archives/cohort.json"}}
    manifest_path = root / "data/reservoirs/manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path, manifest, sources, observations


def independent_legacy(frame, spec):
    """Literal scalar MATLAB recurrence, without using the Python solver."""
    model, targets = spec["model"], spec["raw_monthly_targets"]
    capacity, dt = model["capacity_m3"], spec["sampling"]["timestep_days"] * 86400
    forcing = frame.inflow_m3s.to_numpy() * spec["inflow_preprocessing"]["factor"] + spec["inflow_preprocessing"]["offset_m3s"]
    start = datetime.fromisoformat(spec["sampling"]["legacy_start"])
    months = [(start + timedelta(days=spec["sampling"]["timestep_days"] * row)).month for row in range(len(frame))]
    storage, release = [float(frame.storage_m3.iloc[0])], [model["initial_release_m3s"]]

    def rule(value, raw_inflow, month):
        fraction = value / capacity
        critical, normal, flood = [targets[field][month - 1] / capacity for field in MONTHLY_TARGET_COLUMNS[:3]]
        qcritical, qnormal, qflood = [targets[field][month - 1] for field in MONTHLY_TARGET_COLUMNS[3:]]
        dead = model["dead_storage_fraction"]
        if fraction <= dead:
            return 0.0
        if fraction <= critical:
            return min(qcritical, (fraction - dead) * capacity / dt)
        if fraction <= normal:
            return qcritical + (qnormal - qcritical) * (fraction - critical) / (normal - critical)
        if fraction <= flood:
            rise = max(raw_inflow - qnormal, qflood - qnormal) if model["flood_mode"] == "inflow" else qflood - qnormal
            return qnormal + (fraction - normal) / (flood - normal) * rise
        return min(max((fraction - flood) * capacity / dt, qflood), model["channel_capacity_m3s"])

    for row in range(1, len(frame)):
        predictor = storage[-1] + dt / 2 * (forcing[row] * model["inflow_correction"] + forcing[row - 1] * model["inflow_correction"] - release[-1])
        q = rule(predictor, forcing[row], months[row])
        for _ in range(spec["fixed_point_iterations"]):
            s = predictor - dt / 2 * q
            q = rule(s, forcing[row], months[row])
        if s > capacity:
            q += (s - capacity) / dt
            s = capacity
        storage.append(s)
        release.append(q)
    return np.asarray(storage), np.asarray(release), forcing, np.asarray(months)


def test_daily_and_monthly_historical_replay_after_only_data_configs_are_copied(tmp_path, monkeypatch):
    original = tmp_path / "original"
    _, manifest, specs, observations = offline_fixture(original)
    moved = tmp_path / "offline_copy"
    shutil.copytree(original / "data", moved / "data")
    shutil.copytree(original / "configs", moved / "configs")
    assert not (moved / "outputs").exists()
    real_bytes, real_text, real_open = Path.read_bytes, Path.read_text, Path.open
    reads = []

    def guard(path):
        text = str(path)
        assert not text.casefold().startswith("e:") and "unavailable_archives" not in text
        assert not path.resolve().is_relative_to(original.resolve())
        reads.append(path)

    def read_bytes(path, *args, **kwargs):
        guard(path)
        return real_bytes(path, *args, **kwargs)

    def read_text(path, *args, **kwargs):
        guard(path)
        return real_text(path, *args, **kwargs)

    def open_path(path, mode="r", *args, **kwargs):
        if "r" in mode:
            guard(path)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "open", open_path)
    output = tmp_path / "offline_results"
    result = module.run_collection(moved / "data/reservoirs/manifest.json", output, all_reservoirs=True, mode="legacy")
    assert result["status"] == "completed" and result["counts"] == {"completed": 2}
    assert result["input_files_unchanged"] and result["code_files_unchanged"]
    for record in manifest["reservoirs"]:
        identifier = record["reservoir_id"]
        simulated = pd.read_csv(output / identifier / "legacy_simulation.csv", index_col="date", parse_dates=["date"], float_precision="round_trip")
        expected_s, expected_q, forcing, months = independent_legacy(observations[identifier], specs[identifier])
        # MATLAB uses storage fractions while Python uses storage volumes;
        # accept only a few floating-point ULPs at this capacity.
        np.testing.assert_allclose(simulated.storage_m3, expected_s, rtol=0, atol=1e-6)
        np.testing.assert_allclose(simulated.release_m3s, expected_q, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(simulated.inflow_m3s, forcing)
        np.testing.assert_array_equal(simulated.rule_month, months)
        if identifier == "ex2_Monthly":
            assert simulated.rule_month.iloc[1] == 1 and simulated.index[1].month == 2
            np.testing.assert_array_equal(simulated.legacy_reported_release_m3s.iloc[4:].to_numpy(), simulated.release_m3s.iloc[:-4].to_numpy())
            np.testing.assert_array_equal(simulated.legacy_reported_storage_m3.iloc[:-3].to_numpy(), simulated.storage_m3.iloc[3:].to_numpy())
        summary = json.loads((output / identifier / "legacy_summary.json").read_text())
        assert summary["warmup_rows"] == 2 and summary["legacy_status"] == "completed"
    assert reads


@pytest.mark.parametrize("bad_hashes", [None, 7, {"../outside.csv": "0" * 64}])
def test_invalid_member_hash_metadata_does_not_stop_another_legacy_job(tmp_path, bad_hashes):
    path, manifest, _, _ = offline_fixture(tmp_path / "repository")
    manifest["reservoirs"][0]["files_sha256"] = bad_hashes
    write_json(path, manifest)
    output = tmp_path / "results"
    result = module.run_collection(path, output, all_reservoirs=True, mode="legacy")
    assert result["counts"] == {"input_error": 1, "completed": 1}
    assert not (output / "ex1_Daily").exists()
    assert (output / "ex2_Monthly/legacy_simulation.csv").is_file()


@pytest.mark.parametrize("field,value", [("reservoir", None), ("reservoir", []), ("sampling", None), ("sampling", [])])
def test_invalid_nested_config_metadata_does_not_stop_another_legacy_job(tmp_path, field, value):
    root = tmp_path / "repository"
    path, manifest, _, _ = offline_fixture(root)
    entry = manifest["reservoirs"][0]
    config_path = root / entry["config_file"]
    config = json.loads(config_path.read_text())
    config[field] = value
    write_json(config_path, config)
    entry["config_sha256"] = module.sha256(config_path)
    entry["files_sha256"][entry["config_file"]] = entry["config_sha256"]
    write_json(path, manifest)
    output = tmp_path / "results"
    result = module.run_collection(path, output, all_reservoirs=True, mode="legacy")
    assert result["counts"] == {"input_error": 1, "completed": 1}
    assert not (output / "ex1_Daily").exists()
    assert (output / "ex2_Monthly/legacy_simulation.csv").is_file()


def test_concurrent_bad_source_metadata_stays_isolated_and_invalidates_integrity(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    path, manifest, _, _ = offline_fixture(root)
    source_path = root / "data/reservoirs/ex1_Daily/matlab_source.json"
    original = module._run_legacy

    def mutate_after_simulation(config_path, config, observations, output):
        report = original(config_path, config, observations, output)
        if config["reservoir"]["id"] == "ex1_Daily":
            metadata = json.loads(source_path.read_text())
            metadata["model"] = None
            write_json(source_path, metadata)
        return report

    monkeypatch.setattr(module, "_run_legacy", mutate_after_simulation)
    result = module.run_collection(path, tmp_path / "results", all_reservoirs=True, mode="legacy")
    assert result["status"] == "invalid_integrity" and not result["input_files_unchanged"]
    assert [row["status"] for row in result["reservoirs"]] == ["input_error", "completed"]
    assert "diagnostic_reason" in result["reservoirs"][0]


def test_legacy_cli_runs_one_selected_offline_member(tmp_path, capsys):
    path, _, _, _ = offline_fixture(tmp_path / "repository")
    output = tmp_path / "cli_results"
    assert module.main(["--manifest", str(path), "--reservoir", "Monthly", "--mode", "legacy", "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["selected_reservoir_count"] == 1 and printed["mode"] == "legacy"
    assert not (output / "ex1_Daily").exists()
    assert (output / "ex2_Monthly/legacy_simulation.csv").is_file()
