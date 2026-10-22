"""Independent per-reservoir preparation, input preservation and isolation."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dztr.calibration import fleet
from dztr.calibration.stopping import StoppingPolicy
from dztr.data import read_reservoir_observations


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def observed_table():
    dates = pd.date_range("2000-12-31", "2002-01-10", freq="D")
    variation = dates.day.to_numpy(dtype=float)
    return pd.DataFrame({"date": dates, "Storage": 40 + variation,
                         "Inflow": 3 + variation / 20, "Outflow": 1 + variation / 10})


def configuration(source="data/observed.csv"):
    return {
        "reservoir": {"name": "Synthetic reservoir"}, "observations_file": source,
        "periods": {"calibration": {"start": "2001-01-01", "end": "2001-12-31"},
                    "validation": {"start": "2002-01-01", "end": "2002-01-10"}, "warmup_steps": 1},
        "model": {"capacity_m3": 100_000_000, "initial_storage_m3": 50_000_000,
                  "initial_release_m3s": 2, "inflow_correction": 0.92},
        "quantile_method": "hazen", "target_percentiles": [10, 45, 85],
        "simulation": {"solver": "implicit", "overflow_mode": "conservative", "atol_storage_m3": 1e-4},
        "optimization": {"population_size": 4, "evaluation_budget": 12, "seed": 17},
    }


def job(identifier="reservoir_a", config="configs/a.json"):
    return {"reservoir_id": identifier, "configuration_file": config,
            "observation_units": {"storage": "mcm", "flow": "m3/s"}}


@pytest.fixture
def prepared_input(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "data/observed.csv"
    source.parent.mkdir()
    observed_table().to_csv(source, index=False)
    config = write_json(root / "configs/a.json", configuration())
    manifest = write_json(root / "configs/manifest.json", {"schema_version": 1, "jobs": [job()]})
    return root, source, config, manifest, tmp_path / "outputs"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forbid_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Fleet preflight must not run simulation, candidate evaluation or optimization.")

    monkeypatch.setattr("dztr.calibration.optimizer.optimize_nsga2", forbidden)
    monkeypatch.setattr("dztr.calibration.objectives.CalibrationObjective.evaluate", forbidden)
    monkeypatch.setattr("dztr.calibration.objectives.CalibrationObjective.evaluate_validation", forbidden)
    monkeypatch.setattr("dztr.calibration.objectives.simulate_reservoir", forbidden)
    monkeypatch.setattr("dztr.simulation.simulate_reservoir", forbidden)


def test_ready_job_normalizes_units_preserves_sources_and_prepares_search_configuration(prepared_input, monkeypatch):
    root, source, config, manifest, output = prepared_input
    original = {path: path.read_bytes() for path in (source, config, manifest)}
    forbid_execution(monkeypatch)
    report = fleet.run_fleet_preflight(manifest, root, output)
    assert report["status_counts"] == {"ready": 1, "skipped_placeholder": 0, "skipped_unavailable": 0, "error": 0}
    assert report["simulation_runs"] == report["optimization_evaluations"] == 0
    ready = report["jobs"][0]
    assert ready["status"] == "ready"
    destination = output / "reservoir_a"
    normalized = read_reservoir_observations(destination / "observations_si.csv")
    np.testing.assert_allclose(normalized.frame.storage_m3, observed_table().Storage * 1e6)
    np.testing.assert_allclose(normalized.frame.inflow_m3s, observed_table().Inflow)
    np.testing.assert_allclose(normalized.frame.release_m3s, observed_table().Outflow)
    resolved = json.loads((destination / "resolved_config.json").read_text(encoding="utf-8"))
    provenance = json.loads((destination / "preflight_provenance.json").read_text(encoding="utf-8"))
    assert resolved["observations_file"] == str((destination / "observations_si.csv").resolve())
    assert (root / resolved["observations_file"]).resolve().is_file()
    assert resolved["model"]["inflow_correction"] == 0.92
    assert resolved["model"]["capacity_m3"] == 100_000_000
    assert resolved["optimization"]["seed"] == 17
    assert resolved["optimization"]["evaluation_budget"] == 12
    assert resolved["simulation"] == configuration()["simulation"]
    assert provenance["source_sha256"] == digest(source)
    assert provenance["configuration_sha256"] == digest(config)
    assert provenance["manifest_sha256"] == digest(manifest)
    assert provenance["input_observation_units"] == {"storage": "mcm", "flow": "m3/s"}
    assert provenance["output_observation_units"] == {"storage": "m3", "flow": "m3/s"}
    assert provenance["counts"]["spinup_rows"] == 1
    assert provenance["counts"]["calibration_rows"] == 365
    assert provenance["counts"]["validation_rows"] == 10
    assert provenance["counts"]["unscored_gap_rows"] == 0
    assert provenance["model_source_statuses"]["inflow_correction"] == "explicit_configuration"
    assert provenance["model_source_statuses"]["channel_capacity_m3s"] == "calibration_release_p99"
    assert provenance["model_source_statuses"]["dead_storage_fraction"] == "generic_default"
    assert provenance["bounds"]["estimation_period"] == configuration()["periods"]["calibration"]
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == report
    assert pd.read_csv(output / "fleet_status.csv").status.tolist() == ["ready"]
    for path, data in original.items():
        assert path.read_bytes() == data


def test_missing_bad_placeholder_and_unavailable_jobs_do_not_block_ready_jobs(prepared_input):
    root, source, config, manifest, output = prepared_input
    zeros = observed_table()
    zeros.loc[:, ["Storage", "Inflow", "Outflow"]] = 0
    zeros.to_csv(root / "data/zero.csv", index=False)
    missing = observed_table()
    missing.loc[:, ["Storage", "Inflow", "Outflow"]] = np.nan
    missing.to_csv(root / "data/unavailable.csv", index=False)
    write_json(root / "configs/zero.json", {"observations_file": "data/zero.csv"})
    write_json(root / "configs/unavailable.json", {"observations_file": "data/unavailable.csv"})
    write_json(root / "configs/missing.json", configuration("data/missing.csv"))
    (root / "configs/invalid.json").write_text("not JSON", encoding="utf-8")
    write_json(manifest, {"schema_version": 1, "jobs": [job(), job("placeholder", "configs/zero.json"),
        job("unavailable", "configs/unavailable.json"), job("source_missing", "configs/missing.json"),
        job("bad_config", "configs/invalid.json"), job("config_missing", "configs/not_here.json")]})
    report = fleet.run_fleet_preflight(manifest, root, output)
    assert [row["status"] for row in report["jobs"]] == ["ready", "skipped_placeholder", "skipped_unavailable", "error", "error", "error"]
    assert report["status_counts"] == {"ready": 1, "skipped_placeholder": 1, "skipped_unavailable": 1, "error": 3}
    assert [path.name for path in output.iterdir() if path.is_dir()] == ["reservoir_a"]
    assert report["jobs"][3]["failure_stage"] == "observations"
    assert report["jobs"][3]["reason"].startswith("FileNotFoundError:")
    assert report["jobs"][4]["reason"].startswith("JSONDecodeError:")


@pytest.mark.parametrize("identifier", ["../escape", "a/b", "a\\b", "..", ".", "CON", "nul", "com1", "LPT9", "a b", "a.", "", "x" * 65])
def test_path_unsafe_or_reserved_ids_are_rejected_before_output(prepared_input, identifier):
    root, source, config, manifest, output = prepared_input
    write_json(manifest, {"schema_version": 1, "jobs": [job(identifier)]})
    with pytest.raises(ValueError, match="reservoir_id"):
        fleet.run_fleet_preflight(manifest, root, output)
    assert not output.exists()


@pytest.mark.parametrize("identifiers", [("same", "same"), ("Trinity", "trinity")])
def test_duplicate_ids_are_rejected_case_insensitively(prepared_input, identifiers):
    root, source, config, manifest, output = prepared_input
    write_json(manifest, {"schema_version": 1, "jobs": [job(name) for name in identifiers]})
    with pytest.raises(ValueError, match="unique"):
        fleet.run_fleet_preflight(manifest, root, output)
    assert not output.exists()


@pytest.mark.parametrize("version", [0, 2, "1", True, None])
def test_manifest_schema_version_is_explicit_integer_one(prepared_input, version):
    root, source, config, manifest, output = prepared_input
    write_json(manifest, {"schema_version": version, "jobs": [job()]})
    with pytest.raises(ValueError, match="schema_version"):
        fleet.run_fleet_preflight(manifest, root, output)
    assert not output.exists()


@pytest.mark.parametrize("missing_field", ["capacity_m3", "initial_storage_m3", "initial_release_m3s", "inflow_correction"])
def test_available_jobs_require_explicit_reservoir_physical_values(prepared_input, missing_field):
    root, source, config, manifest, output = prepared_input
    original = configuration()
    del original["model"][missing_field]
    write_json(config, original)
    report = fleet.run_fleet_preflight(manifest, root, output)
    assert report["jobs"][0]["status"] == "error"
    assert missing_field in report["jobs"][0]["reason"]
    assert "explicit" in report["jobs"][0]["reason"]
    assert not (output / "reservoir_a").exists()


@pytest.mark.parametrize("problem", ["missing_date", "missing_inflow", "infinite_inflow", "constant_validation_release", "constant_calibration_storage"])
def test_invalid_daily_forcing_or_undefined_nse_is_reported(prepared_input, problem):
    root, source, config, manifest, output = prepared_input
    data = observed_table()
    if problem == "missing_date":
        data = data.drop(index=100)
    elif problem == "missing_inflow":
        data.loc[100, "Inflow"] = np.nan
    elif problem == "infinite_inflow":
        data.loc[100, "Inflow"] = np.inf
    elif problem == "constant_validation_release":
        data.loc[data.date >= pd.Timestamp("2002-01-01"), "Outflow"] = 5
    else:
        data.loc[data.date.between("2001-01-01", "2001-12-31"), "Storage"] = 50
    data.to_csv(source, index=False)
    report = fleet.run_fleet_preflight(manifest, root, output)
    assert report["jobs"][0]["status"] == "error"
    assert not (output / "reservoir_a").exists()
    if problem == "missing_date":
        assert "complete consecutive daily coverage" in report["jobs"][0]["reason"]
    elif problem == "missing_inflow":
        assert "finite inflow" in report["jobs"][0]["reason"]
    elif problem == "infinite_inflow":
        assert "infinite" in report["jobs"][0]["reason"]
    else:
        assert "nonconstant" in report["jobs"][0]["reason"]


def test_read_options_are_forwarded_and_sources_remain_unchanged(prepared_input):
    root, source, config, manifest, output = prepared_input
    data = observed_table().to_csv(index=False)
    source.write_text("metadata line to skip\n" + data, encoding="utf-8")
    entry = job()
    entry["read_options"] = {"skiprows": 1}
    write_json(manifest, {"schema_version": 1, "jobs": [entry]})
    before = source.read_bytes()
    report = fleet.run_fleet_preflight(Path("configs/manifest.json"), root, output)
    assert report["jobs"][0]["status"] == "ready"
    provenance = json.loads((output / "reservoir_a/preflight_provenance.json").read_text(encoding="utf-8"))
    assert provenance["read_options"] == {"skiprows": 1}
    assert source.read_bytes() == before


def test_calibration_only_channel_and_bounds_ignore_spinup_and_validation_values(prepared_input, monkeypatch):
    root, source, config, manifest, output = prepared_input
    fitted_boxes = []
    original = fleet.estimate_calibration_bounds

    def fit_bounds(*args, **kwargs):
        bounds = original(*args, **kwargs)
        fitted_boxes.append(bounds.monthly_bounds.copy(deep=True))
        return bounds

    monkeypatch.setattr(fleet, "estimate_calibration_bounds", fit_bounds)
    first = fleet.run_fleet_preflight(manifest, root, output)
    data = observed_table()
    outside_calibration = (data.date < "2001-01-01") | (data.date >= "2002-01-01")
    data.loc[outside_calibration, "Outflow"] *= 10000
    data.loc[outside_calibration, "Storage"] *= 10000
    data.to_csv(source, index=False)
    second = fleet.run_fleet_preflight(manifest, root, output.parent / "second")
    assert first["jobs"][0]["status"] == second["jobs"][0]["status"] == "ready"
    assert first["jobs"][0]["model"] == second["jobs"][0]["model"]
    assert first["jobs"][0]["model"]["channel_capacity_m3s"] < 10
    a = json.loads((output / "reservoir_a/preflight_provenance.json").read_text(encoding="utf-8"))
    b = json.loads((output.parent / "second/reservoir_a/preflight_provenance.json").read_text(encoding="utf-8"))
    assert a["bounds"] == b["bounds"]
    pd.testing.assert_frame_equal(fitted_boxes[0], fitted_boxes[1])


def test_explicit_channel_and_model_settings_are_preserved(prepared_input):
    root, source, config, manifest, output = prepared_input
    configured = configuration()
    configured["model"].update({"channel_capacity_m3s": 77, "dead_storage_fraction": 0.2,
                                "flood_mode": "inflow", "inflow_correction": 1.1})
    configured["simulation"] = {"solver": "matlab_fixed_point", "overflow_mode": "matlab"}
    write_json(config, configured)
    report = fleet.run_fleet_preflight(manifest, root, output)
    assert report["jobs"][0]["status"] == "ready"
    resolved = json.loads((output / "reservoir_a/resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["model"] == configured["model"]
    assert resolved["simulation"] == configured["simulation"]
    assert report["jobs"][0]["model_source_statuses"]["channel_capacity_m3s"] == "explicit_configuration"


def test_shared_policy_is_validated_and_does_not_replace_search_budget(prepared_input):
    root, source, config, manifest, output = prepared_input
    policy = StoppingPolicy(population_size=4, check_interval=4, window_evaluations=4,
                            minimum_evaluations=8, patience=2, max_evaluations=40)
    write_json(root / "configs/policy.json", asdict(policy))
    write_json(manifest, {"schema_version": 1, "stopping_policy_file": "configs/policy.json", "jobs": [job()]})
    report = fleet.run_fleet_preflight(manifest, root, output)
    assert report["stopping_policy"]["max_evaluations"] == 40
    assert report["stopping_policy_source"]["illustrative"] is True
    resolved = json.loads((output / "reservoir_a/resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["optimization"]["evaluation_budget"] == 12
    assert "illustrative" in report["stopping_policy_role"]


def test_policy_population_mismatch_is_isolated_to_the_job(prepared_input):
    root, source, config, manifest, output = prepared_input
    write_json(root / "configs/policy.json", asdict(StoppingPolicy()))
    write_json(manifest, {"schema_version": 1, "stopping_policy_file": "configs/policy.json", "jobs": [job()]})
    report = fleet.run_fleet_preflight(manifest, root, output)
    assert report["jobs"][0]["status"] == "error"
    assert "population_size" in report["jobs"][0]["reason"]


def test_invalid_common_policy_raises_before_writes(prepared_input):
    root, source, config, manifest, output = prepared_input
    write_json(root / "configs/policy.json", {"population_size": 4, "check_interval": 3})
    write_json(manifest, {"schema_version": 1, "stopping_policy_file": "configs/policy.json", "jobs": [job()]})
    with pytest.raises(ValueError):
        fleet.run_fleet_preflight(manifest, root, output)
    assert not output.exists()


@pytest.mark.parametrize("config_path", ["../outside.json", "/absolute/config.json"])
def test_configuration_paths_must_be_inside_the_repository(prepared_input, config_path):
    root, source, config, manifest, output = prepared_input
    write_json(manifest, {"schema_version": 1, "jobs": [job("unsafe", config_path), job()]})
    report = fleet.run_fleet_preflight(manifest, root, output)
    assert [row["status"] for row in report["jobs"]] == ["error", "ready"]
    assert "repo_root" in report["jobs"][0]["reason"]


def test_absolute_observation_path_is_supported(prepared_input):
    root, source, config, manifest, output = prepared_input
    write_json(config, configuration(str(source.resolve())))
    assert fleet.run_fleet_preflight(manifest, root, output)["jobs"][0]["status"] == "ready"


def test_nonempty_output_cannot_reuse_or_overwrite_artifacts(prepared_input):
    root, source, config, manifest, output = prepared_input
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_text("existing artifact", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        fleet.run_fleet_preflight(manifest, root, output)
    assert sentinel.read_text(encoding="utf-8") == "existing artifact"


def test_existing_empty_output_is_allowed(prepared_input):
    root, source, config, manifest, output = prepared_input
    output.mkdir()
    assert fleet.run_fleet_preflight(manifest, root, output)["jobs"][0]["status"] == "ready"


def test_missing_source_target_inside_output_is_protected_from_artifact_writes(prepared_input):
    root, source, config, manifest, output = prepared_input
    write_json(config, configuration(str(output / "fleet_status.csv")))
    with pytest.raises(ValueError, match="contain.*input"):
        fleet.run_fleet_preflight(manifest, root, output)
    assert not output.exists()


def test_all_job_reads_complete_before_any_outputs_are_created(prepared_input, monkeypatch):
    root, source, config, manifest, output = prepared_input
    write_json(manifest, {"schema_version": 1, "jobs": [job("first"), job("second")]})
    original_reader = fleet.read_reservoir_observations
    calls = []

    def reader(*args, **kwargs):
        assert not output.exists()
        calls.append(args[0])
        return original_reader(*args, **kwargs)

    monkeypatch.setattr(fleet, "read_reservoir_observations", reader)
    report = fleet.run_fleet_preflight(manifest, root, output)
    assert len(calls) == 2
    assert report["status_counts"]["ready"] == 2


def test_invalid_optimizer_and_simulation_configuration_are_job_errors(prepared_input):
    root, source, config, manifest, output = prepared_input
    a = configuration()
    a["optimization"]["seed"] = True
    write_json(config, a)
    b = configuration()
    b["simulation"]["solver"] = "unknown"
    write_json(root / "configs/b.json", b)
    write_json(manifest, {"schema_version": 1, "jobs": [job(), job("b", "configs/b.json")]})
    report = fleet.run_fleet_preflight(manifest, root, output)
    assert all(row["status"] == "error" for row in report["jobs"])
    assert "seed" in report["jobs"][0]["reason"]
    assert "solver" in report["jobs"][1]["reason"]


def test_backend_parse_error_is_isolated(prepared_input, monkeypatch):
    root, source, config, manifest, output = prepared_input
    write_json(manifest, {"schema_version": 1, "jobs": [job("bad"), job("good")]})
    original = fleet.read_reservoir_observations
    count = 0

    class SyntheticBackendError(Exception):
        pass

    def reader(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            raise SyntheticBackendError("Malformed workbook")
        return original(*args, **kwargs)

    monkeypatch.setattr(fleet, "read_reservoir_observations", reader)
    report = fleet.run_fleet_preflight(manifest, root, output)
    assert [row["status"] for row in report["jobs"]] == ["error", "ready"]
    assert "SyntheticBackendError" in report["jobs"][0]["reason"]
