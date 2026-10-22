"""Portable cohort outputs, forcing policies and calibration-only pilots."""

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from dztr.calibration.sampled import _observations
from dztr.integrations import world_reservoirs as module
from dztr.integrations.world_observations import WorldObservationConflict
from dztr.schemas import MONTHLY_TARGET_COLUMNS, SimulationResult


def monthly_observations():
    dates = pd.date_range("2000-01-01", "2007-12-01", freq="MS", name="date")
    year = dates.year.to_numpy() - 2000
    month = dates.month.to_numpy()
    return _observations(pd.DataFrame({
        "storage_m3": 3e8 + 1e6 * year + 2e6 * month,
        "inflow_m3s": 2 + 0.04 * month + 0.01 * year,
        "release_m3s": 1 + 0.03 * month + 0.05 * year,
    }, index=dates), "portable-fixture")


def source_spec(source_root, reservoir_id="ex12_Example", *, supported=True):
    folder = source_root / reservoir_id
    folder.mkdir()
    pinned = folder / "source.m"
    pinned.write_text("% portable input proof\n")
    monthly = dict(zip(MONTHLY_TARGET_COLUMNS, [[value] * 12 for value in (2e8, 3e8, 4e8, 1, 2, 3)]))
    bounds = {side + "_" + key: [value] * 12 for side, value in (("lower", 0), ("upper", 1e9))
              for key in MONTHLY_TARGET_COLUMNS}
    return {"reservoir_id": reservoir_id, "folder": str(folder), "name": "Example",
        "status": "ready", "source_sha256": {str(pinned): module.sha256(pinned)},
        "sampling": {"frequency": "monthly", "timestep_days": 30.5, "legacy_start": "2000-01-01"},
        "model": {"capacity_m3": 1e9, "initial_storage_m3": 3e8,
                  "initial_release_m3s": 1, "channel_capacity_m3s": 20,
                  "dead_storage_fraction": 0.1, "inflow_correction": 0.92, "flood_mode": "inflow"},
        "initial_storage_source": "observed", "initial_release_source": "coefficient",
        "inflow_preprocessing": {"factor": 1.45, "offset_m3s": 8},
        "simulation_supported": supported, "simulation_unsupported_reason": None if supported else "External inflow adapter required.",
        "raw_monthly_targets": monthly, "raw_bounds": bounds,
        "warmup_rows": 12, "fixed_point_iterations": 2, "overflow_mode": "matlab",
        "legacy_release_shift_rows": 0, "legacy_storage_shift_rows": 0, "warnings": [], "evidence": []}


def imported_fixture(tmp_path, monkeypatch, *, supported=True, run_baselines=False):
    source = tmp_path / "original_source"
    source.mkdir()
    spec = source_spec(source, supported=supported)
    observations = monthly_observations()
    monkeypatch.setattr(module, "discover_world_sources", lambda root: [spec])
    monkeypatch.setattr(module, "read_world_observations", lambda row: (observations,
                            {"status": "verified", "source_sha256": row["source_sha256"]}))
    output = tmp_path / "imported"
    summary = module.import_world_reservoirs(source, output, run_baselines=run_baselines)
    config_path = output / "reservoirs" / spec["reservoir_id"] / "sample.json"
    return source, output, spec, observations, summary, config_path


def test_import_batch_isolates_source_and_observation_conflicts(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    good = source_spec(source, "ex1_Good")
    ambiguous = source_spec(source, "ex2_Ambiguous")
    ambiguous.update(status="error", error="Ambiguous base scripts.")
    conflict = source_spec(source, "ex3_Conflict")
    external = source_spec(source, "ex4_External", supported=False)
    specs = [good, ambiguous, conflict, external]
    calls = []

    def reader(spec):
        calls.append(spec["reservoir_id"])
        if spec["reservoir_id"] == "ex3_Conflict":
            raise WorldObservationConflict("Storage unit conflict.", {"measurement_differences": ["storage"]})
        return monthly_observations(), {"status": "verified", "source_sha256": spec["source_sha256"]}

    monkeypatch.setattr(module, "discover_world_sources", lambda root: specs)
    monkeypatch.setattr(module, "read_world_observations", reader)
    summary = module.import_world_reservoirs(source, tmp_path / "cohort", run_baselines=False)
    assert summary["reservoir_count"] == 4 and summary["source_files_unchanged"]
    rows = {row["reservoir_id"]: row for row in summary["reservoirs"]}
    assert rows["ex1_Good"]["import_status"] == "imported"
    assert rows["ex2_Ambiguous"]["import_status"] == "blocked"
    assert rows["ex3_Conflict"]["import_status"] == "source_conflict"
    assert rows["ex4_External"]["import_status"] == "imported"
    assert calls == ["ex1_Good", "ex3_Conflict", "ex4_External"]
    proof = json.loads((tmp_path / "cohort/reservoirs/ex3_Conflict/observation_provenance.json").read_text())
    assert proof["status"] == "conflict" and proof["measurement_differences"] == ["storage"]
    config = json.loads((tmp_path / "cohort/reservoirs/ex4_External/sample.json").read_text())
    assert not config["simulation_supported"]
    assert not (tmp_path / "cohort/reservoirs/ex2_Ambiguous/observations_si.csv").exists()


def test_import_outputs_explicit_protocol_and_unchanged_source_measurements(tmp_path, monkeypatch):
    source, output, spec, observations, summary, config_path = imported_fixture(tmp_path, monkeypatch)
    config = json.loads(config_path.read_text())
    assert config["observations_file"] == "observations_si.csv"
    assert config["observations_sha256"] == module.sha256(config_path.with_name("observations_si.csv"))
    assert config["model"]["initial_storage_m3"] == observations.frame.storage_m3.iloc[0]
    assert config["model"]["inflow_correction"] == 0.92
    assert config["inflow_preprocessing"] == {"factor": 1.45, "offset_m3s": 8}
    assert config["periods"]["warmup_rows"] == 12
    assert config["periods"]["calibration"] == {"start": "2001-01-01", "end": "2003-12-01"}
    assert config["periods"]["validation"] == {"start": "2004-01-01", "end": "2007-12-01"}
    assert config["protocol"]["paper_reproduction"] is False
    assert config["sampling"]["month_clock"] == "observed"
    assert config["legacy"]["sampling"]["legacy_start"] == "2000-01-01"
    raw = pd.read_csv(config_path.with_name("observations_si.csv"), index_col="date", parse_dates=["date"])
    np.testing.assert_allclose(raw.inflow_m3s, observations.frame.inflow_m3s, rtol=1e-15)
    assert summary["counts"]["baseline_status"] == {"not_run": 1}
    assert module.sha256(next(iter(spec["source_sha256"]))) == next(iter(spec["source_sha256"].values()))


def test_normalized_csv_runs_after_original_source_and_import_paths_move(tmp_path, monkeypatch):
    source, output, spec, observations, summary, config_path = imported_fixture(tmp_path, monkeypatch)
    portable = tmp_path / "portable"
    portable.mkdir()
    for filename in ("sample.json", "observations_si.csv"):
        shutil.copyfile(config_path.with_name(filename), portable / filename)
    source.rename(tmp_path / "source_archived")
    output.rename(tmp_path / "import_archived")
    report = module.run_world_sample(portable / "sample.json", tmp_path / "portable_result")
    assert report["baseline_status"] == "completed"
    assert report["parameterization_status"] == "estimated"
    assert report["optimization_status"] == "ready"
    assert report["inputs_sha256"]["config"] == module.sha256(portable / "sample.json")
    simulation = pd.read_csv(tmp_path / "portable_result/baseline_simulation.csv", float_precision="round_trip")
    expected = observations.frame.inflow_m3s * 1.45 + 8
    np.testing.assert_array_equal(simulation.inflow_m3s, expected)
    np.testing.assert_array_equal(simulation.effective_inflow_m3s, expected * 0.92)


def test_normalized_input_hash_guard_precedes_creating_output(tmp_path, monkeypatch):
    _, _, _, _, _, config_path = imported_fixture(tmp_path, monkeypatch)
    csv = config_path.with_name("observations_si.csv")
    csv.write_text(csv.read_text() + "\n")
    output = tmp_path / "should_not_exist"
    with pytest.raises(ValueError, match="SHA256 changed"):
        module.run_world_sample(config_path, output)
    assert not output.exists()


@pytest.mark.parametrize("mutated_file", ["config", "observations"])
def test_portable_inputs_are_guarded_again_after_execution(tmp_path, monkeypatch, mutated_file):
    _, _, _, _, _, config_path = imported_fixture(tmp_path, monkeypatch)

    def changing_run(observations, config, output, **options):
        path = config_path if mutated_file == "config" else config_path.with_name("observations_si.csv")
        path.write_text(path.read_text() + "\n")
        return {"reservoir_id": config["reservoir"]["id"], "baseline_status": "completed"}

    monkeypatch.setattr(module, "_run", changing_run)
    with pytest.raises(ValueError, match="(?i)(sha256|input).*(changed|change)"):
        module.run_world_sample(config_path, tmp_path / "changed_run")


@pytest.mark.parametrize("target", ["source", "source/inside"])
def test_import_refuses_outputs_in_source_tree(tmp_path, monkeypatch, target):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(module, "discover_world_sources", lambda root: pytest.fail("Discovery must not run"))
    with pytest.raises(ValueError, match="protected source inputs"):
        module.import_world_reservoirs(source, tmp_path / target, run_baselines=False)


def test_portable_output_cannot_replace_an_input_or_its_parent(tmp_path, monkeypatch):
    _, _, _, _, _, config_path = imported_fixture(tmp_path, monkeypatch)
    for output in (config_path, config_path.with_name("observations_si.csv"), config_path.parent):
        with pytest.raises(ValueError, match="protected source inputs"):
            module.run_world_sample(config_path, output)


def test_existing_outputs_are_never_overwritten(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "user_notes.txt"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="new or empty"):
        module.import_world_reservoirs(source, output, run_baselines=False)
    assert sentinel.read_text() == "preserve"


def test_forcing_transforms_raw_inflow_without_mutating_observations():
    observations = monthly_observations()
    original = observations.frame.copy(deep=True)
    forced = module._forcing(observations, {"factor": 1.1, "offset_m3s": 8})
    pd.testing.assert_frame_equal(original, observations.frame)
    np.testing.assert_array_equal(forced.frame.inflow_m3s, original.inflow_m3s * 1.1 + 8)
    np.testing.assert_array_equal(forced.frame.storage_m3, original.storage_m3)
    np.testing.assert_array_equal(forced.frame.release_m3s, original.release_m3s)


@pytest.mark.parametrize("preprocessing", [None, {"factor": 1}, {"factor": True, "offset_m3s": 1}, {"factor": 1, "offset_m3s": np.inf}])
def test_unknown_forcing_transform_is_not_guessed(preprocessing):
    with pytest.raises(ValueError, match="explicit finite"):
        module._forcing(monthly_observations(), preprocessing)


def test_external_source_import_blocks_baseline_and_search_with_reason(tmp_path, monkeypatch):
    _, _, _, _, _, config_path = imported_fixture(tmp_path, monkeypatch, supported=False)
    report = module.run_world_sample(config_path, tmp_path / "external_result", optimization_budget=2)
    assert report["baseline_status"] == report["optimization_status"] == "blocked"
    assert "External inflow adapter" in report["baseline_reason"]
    assert not (tmp_path / "external_result/optimization_trials.csv").exists()


def test_small_optimization_pilot_exports_calibration_trials_and_separate_validation(tmp_path, monkeypatch):
    _, _, _, _, _, config_path = imported_fixture(tmp_path, monkeypatch)
    output = tmp_path / "pilot"
    report = module.run_world_sample(config_path, output, optimization_budget=2, seed=17)
    assert report["optimization_status"] == "pilot_completed"
    assert report["optimization"]["evaluations"] == 2
    assert report["optimization"]["settings"]["seed"] == 17
    assert len(pd.read_csv(output / "optimization_trials.csv")) == 2
    validation = json.loads((output / "pilot_validation.json").read_text())
    assert len(validation) == report["optimization"]["front_size"]


def test_validation_scoring_failure_does_not_relabel_finished_calibration_search(tmp_path, monkeypatch):
    _, _, _, _, _, config_path = imported_fixture(tmp_path, monkeypatch)
    original = module.SampledCalibrationObjective

    class UndefinedValidation(original):
        def evaluate_validation(self, vector):
            raise ValueError("Validation NSE is undefined.")

    monkeypatch.setattr(module, "SampledCalibrationObjective", UndefinedValidation)
    output = tmp_path / "undefined_validation"
    report = module.run_world_sample(config_path, output, optimization_budget=2)
    assert report["optimization_status"] == "pilot_completed"
    validation = json.loads((output / "pilot_validation.json").read_text())
    assert validation and all(row["feasible"] is False for row in validation)
    assert all("undefined" in row["reason"] for row in validation)


def test_validation_measurements_cannot_change_calibration_targets_or_search(tmp_path, monkeypatch):
    _, _, _, _, _, config_path = imported_fixture(tmp_path, monkeypatch)
    first = tmp_path / "original_pilot"
    module.run_world_sample(config_path, first, optimization_budget=2, seed=17)
    csv = config_path.with_name("observations_si.csv")
    lines = csv.read_bytes().decode("utf-8").splitlines(keepends=True)
    columns = lines[0].strip().split(",")
    original_prefix = "".join(line for line in lines[1:] if line.split(",")[0] < "2004-01-01")
    for index, line in enumerate(lines[1:], 1):
        fields = line.rstrip("\r\n").split(",")
        if fields[0] >= "2004-01-01":
            for column, scale in (("storage_m3", 10), ("release_m3s", 100)):
                position = columns.index(column)
                fields[position] = str(float(fields[position]) * scale)
            ending = line[len(line.rstrip("\r\n")):]
            lines[index] = ",".join(fields) + ending
    assert original_prefix == "".join(line for line in lines[1:] if line.split(",")[0] < "2004-01-01")
    csv.write_bytes("".join(lines).encode("utf-8"))
    config = json.loads(config_path.read_text())
    config["observations_sha256"] = module.sha256(csv)
    config_path.write_text(json.dumps(config))
    second = tmp_path / "altered_validation_pilot"
    module.run_world_sample(config_path, second, optimization_budget=2, seed=17)
    for filename in ("calibration_targets_si.csv", "calibration_bounds_si.csv", "optimization_trials.csv", "optimization_vectors.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(first / filename), pd.read_csv(second / filename), check_exact=True)


def test_legacy_reporting_shifts_are_separate_from_physical_states(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    spec = source_spec(source)
    spec["legacy_release_shift_rows"], spec["legacy_storage_shift_rows"] = 4, -3
    observations = monthly_observations()
    physical = pd.DataFrame({"storage_m3": np.arange(96, dtype=float), "release_m3s": 100 + np.arange(96, dtype=float)}, index=observations.frame.index)

    def fake_simulation(forced, parameters, **settings):
        assert settings["solver"] == "matlab_fixed_point" and settings["overflow_mode"] == "matlab"
        assert settings["sampling"].month_clock == "legacy"
        return SimulationResult(physical.copy(deep=True), parameters, settings["solver"], settings["overflow_mode"])

    monkeypatch.setattr(module, "simulate_sampled_reservoir", fake_simulation)
    result, reported = module.legacy_replay(observations, spec)
    pd.testing.assert_frame_equal(result.frame, physical)
    np.testing.assert_array_equal(reported.legacy_reported_release_m3s[:8], [100, 101, 102, 103, 100, 101, 102, 103])
    np.testing.assert_array_equal(reported.legacy_reported_storage_m3[-6:], [93, 94, 95, 93, 94, 95])
