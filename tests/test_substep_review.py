"""Portable daily sensitivity reports, held-out isolation and input guards."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dztr.calibration.sampled import _observations
from dztr.integrations import substep_review as module
from dztr.simulation.root_policy import simulate_root_policy_reservoir
from test_collection_runner import _json, collection_fixture
from test_root_policy_review import csv, pin_config, pin_frame


GUARDS = ("manifest_file_unchanged", "input_files_unchanged", "code_files_unchanged")


def daily_fixture(root, count=1):
    path, manifest = collection_fixture(root, count=count)
    index = pd.date_range("2019-12-31", "2021-01-31", name="date")
    for entry in manifest["reservoirs"]:
        frame = pd.DataFrame({
            "storage_m3": 3e8 + index.month * 2e6 + index.day * 1e5 + (index.year - 2019) * 1e6,
            "inflow_m3s": 2. + index.month * .03 + index.day * .001,
            "release_m3s": 1. + index.month * .03 + index.day * .002,
        }, index=index)
        entry.update({"frequency": "daily", "rows": len(index), "first_date": str(index[0].date()),
                      "last_date": str(index[-1].date())})
        pin_frame(root, path, manifest, entry, frame)
        config = json.loads((root / entry["config_file"]).read_text(encoding="utf-8"))
        config["sampling"].update({"frequency": "daily", "fixed_step_days": None})
        config["periods"] = {"simulation_start": "2019-12-31", "calibration": {"start": "2020-01-01", "end": "2020-12-31"},
                             "validation": {"start": "2021-01-01", "end": "2021-01-31"}, "warmup_rows": 1}
        config["model"]["initial_storage_m3"] = float(frame.storage_m3.iloc[0])
        pin_config(root, path, manifest, entry, config)
    return path, manifest


def run(path, output, *, manifest=None, **kwargs):
    if "reservoirs" not in kwargs:
        if manifest is None:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        kwargs["reservoirs"] = tuple(entry["reservoir_id"] for entry in manifest["reservoirs"])
    kwargs.setdefault("substeps", (1, 2, 4))
    return module.run_substep_review(path, output, **kwargs)


def resolutions(report):
    return {record["substeps_per_interval"]: record for record in report["resolutions"]}


def input_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_daily_review_n1_matches_existing_control_and_fits_once_for_all_resolutions(tmp_path, monkeypatch):
    root, output = tmp_path / "repository", tmp_path / "review"
    path, manifest = daily_fixture(root)
    entry = manifest["reservoirs"][0]
    config = json.loads((root / entry["config_file"]).read_text(encoding="utf-8"))
    original_inputs = input_bytes(root)
    estimator = module.estimate_generalized_parameters
    fitting_calls = []

    def captured(observations, *args, **kwargs):
        fitting_calls.append(observations.frame.copy())
        return estimator(observations, *args, **kwargs)

    monkeypatch.setattr(module, "estimate_generalized_parameters", captured)
    result = run(path, output, substeps=(4, 1, 2), record_internal=True)
    assert result["status"] == "review_completed"
    assert result["reservoir_counts"] == {"review_completed": 1}
    assert result["resolution_counts"] == {"completed": 3}
    assert result["substeps"] == [1, 2, 4]
    assert result["settings_selected"] is None
    assert not result["selection_uses_validation"] and not result["convergence_assessed"]
    assert not result["production_default_changed"] and result["search_evaluations"] == 0
    assert all(result[key] for key in GUARDS)
    assert input_bytes(root) == original_inputs
    assert len(fitting_calls) == 1
    assert fitting_calls[0].index[0] == pd.Timestamp("2020-01-01")
    assert fitting_calls[0].index[-1] == pd.Timestamp("2020-12-31")
    assert fitting_calls[0].inflow_m3s.isna().all()
    targets = pd.read_csv(output / entry["reservoir_id"] / "calibration_targets_si.csv", index_col="month", float_precision="round_trip")
    parameters = module.ReservoirParameters(targets, **config["model"])
    observed = _observations(csv(root / entry["observations_file"]))
    control = simulate_root_policy_reservoir(observed, parameters, sampling=module.SamplingSettings(**config["sampling"]))
    stored = csv(output / entry["reservoir_id"] / "substeps_1/simulation.csv")
    pd.testing.assert_frame_equal(stored[control.frame.columns], control.frame, check_exact=True, check_freq=False, check_dtype=False)
    report = result["reservoirs"][0]
    assert resolutions(report)[1]["one_substep_parity"]["exact"]
    assert len(report["refinement_differences"]) == 2
    for count, record in resolutions(report).items():
        assert record["calibration_status"] == record["validation_status"] == "completed"
        assert record["trajectory_quality"]["numerically_valid"]
        assert set(record["metrics"]) == {"calibration", "validation"}
        assert set(record["metrics"]["calibration"]) == {"endpoint", "interval_mean_release", "interval_mean_interpretation"}
        assert "Diagnostic" in record["metrics"]["validation"]["interval_mean_interpretation"]
        assert (output / entry["reservoir_id"] / f"substeps_{count}/internal_steps.csv").is_file()
    persisted = json.loads((output / "substep_review.json").read_text(encoding="utf-8"))
    assert persisted["manifest_sha256"] == module.sha256(path)
    assert persisted["code_sha256"] == module._code_hashes()
    assert (output / "substep_review.csv").is_file() and (output / "substep_review.md").is_file()


def test_heldout_storage_release_and_forcing_change_no_targets_calibration_results_or_selection(tmp_path):
    root = tmp_path / "repository"
    path, manifest = daily_fixture(root)
    entry = manifest["reservoirs"][0]
    before_output, after_output = tmp_path / "before", tmp_path / "after"
    before = run(path, before_output)
    frame = csv(root / entry["observations_file"])
    heldout = frame.index >= "2021-01-01"
    frame.loc[heldout, "storage_m3"] *= 1.7
    frame.loc[heldout, "release_m3s"] = frame.loc[heldout, "release_m3s"] * 1.5 + 1.
    frame.loc[heldout, "inflow_m3s"] *= 3.
    pin_frame(root, path, manifest, entry, frame)
    after = run(path, after_output)
    assert before["settings_selected"] is after["settings_selected"] is None
    assert not before["selection_uses_validation"] and not after["selection_uses_validation"]
    assert all(after[key] for key in GUARDS)
    for filename in ("calibration_targets_si.csv", "critical_boundary_substeps.csv"):
        assert (before_output / entry["reservoir_id"] / filename).read_bytes() == (after_output / entry["reservoir_id"] / filename).read_bytes()
    first, second = resolutions(before["reservoirs"][0]), resolutions(after["reservoirs"][0])
    for count in first:
        assert first[count]["metrics"]["calibration"] == second[count]["metrics"]["calibration"]
        assert first[count]["calibration_quality"] == second[count]["calibration_quality"]
        a = csv(before_output / entry["reservoir_id"] / f"substeps_{count}/simulation.csv")
        b = csv(after_output / entry["reservoir_id"] / f"substeps_{count}/simulation.csv")
        pd.testing.assert_frame_equal(a.loc[:"2020-12-31"], b.loc[:"2020-12-31"], check_exact=True)
        assert first[count]["metrics"]["validation"] != second[count]["metrics"]["validation"]
    for first_difference, second_difference in zip(before["reservoirs"][0]["refinement_differences"], after["reservoirs"][0]["refinement_differences"]):
        assert first_difference["calibration"] == second_difference["calibration"]


def test_negative_heldout_endpoint_preserves_calibration_and_honest_partial_prefix(tmp_path):
    root = tmp_path / "repository"
    path, manifest = daily_fixture(root)
    entry = manifest["reservoirs"][0]
    before_output, after_output = tmp_path / "before", tmp_path / "after"
    before = run(path, before_output, record_internal=True)
    frame = csv(root / entry["observations_file"])
    frame.loc[frame.index >= "2021-01-01", "inflow_m3s"] = -1.
    pin_frame(root, path, manifest, entry, frame)
    after = run(path, after_output, record_internal=True)
    assert after["status"] == "review_completed" and all(after[key] for key in GUARDS)
    first, second = resolutions(before["reservoirs"][0]), resolutions(after["reservoirs"][0])
    for count, record in second.items():
        assert record["status"] == "blocked"
        assert record["calibration_status"] == "completed" and record["validation_status"] == "blocked"
        assert record["metrics"]["calibration"] == first[count]["metrics"]["calibration"]
        assert "validation" not in record["metrics"]
        assert record["first_failure"]["failure_code"] == "invalid_forcing"
        assert record["first_failure"]["date"] == "2021-01-01"
        assert record["first_failure"]["substep"] == 1
        old = csv(before_output / entry["reservoir_id"] / f"substeps_{count}/simulation.csv")
        partial = csv(after_output / entry["reservoir_id"] / f"substeps_{count}/partial_simulation.csv")
        pd.testing.assert_frame_equal(partial, old.loc[:"2020-12-31"], check_exact=True)
        assert not (after_output / entry["reservoir_id"] / f"substeps_{count}/simulation.csv").exists()
        internals = pd.read_csv(after_output / entry["reservoir_id"] / f"substeps_{count}/partial_internal_steps.csv")
        assert len(internals) == count * (len(partial) - 1)
        assert pd.to_datetime(internals.date).max() == partial.index[-1]
    assert after["settings_selected"] is None and after["search_evaluations"] == 0


def test_actual_analytical_gap_has_honest_isolated_two_row_resolution_diagnostic(tmp_path):
    root, output = tmp_path / "repository", tmp_path / "review"
    path, manifest = daily_fixture(root)
    entry = manifest["reservoirs"][0]
    frame = csv(root / entry["observations_file"])
    frame.storage_m3 = 1e8 + 86400. + (frame.index.day - 1) * 1e5
    frame.release_m3s = 10. + (frame.index.day - 1) * .001
    frame.inflow_m3s = 1.
    pin_frame(root, path, manifest, entry, frame)
    config = json.loads((root / entry["config_file"]).read_text(encoding="utf-8"))
    config["model"].update({"initial_storage_m3": 100500000., "initial_release_m3s": 0.})
    pin_config(root, path, manifest, entry, config)
    result = run(path, output, substeps=(1, 4))
    report = result["reservoirs"][0]
    coarse, fine = resolutions(report)[1], resolutions(report)[4]
    assert coarse["status"] == "blocked" and coarse["validation_status"] == "not_run"
    assert coarse["first_failure"]["failure_code"] == "implicit_no_root"
    assert coarse["first_failure"]["date"] == "2020-01-01"
    assert coarse["one_substep_parity"] is None
    assert fine["status"] == "completed" and fine["trajectory_quality"]["numerically_valid"]
    targets = pd.read_csv(output / entry["reservoir_id"] / "calibration_targets_si.csv", index_col="month", float_precision="round_trip")
    critical, qcritical = targets.loc[1, "storage_critical_m3"], targets.loc[1, "release_critical_m3s"]
    left = critical + (critical - 1e8) / 2.
    right = critical + 86400. * qcritical / 2.
    assert left < coarse["first_failure"]["predictor_m3"] < right
    assert coarse["first_failure"]["previous_storage_m3"] == config["model"]["initial_storage_m3"]
    matched = {record["substeps_per_interval"]: record for record in report["matched_failure_intervals"]}
    assert matched[1]["status"] == "blocked" and matched[4]["status"] == "completed"
    assert "isolated interval, not full trajectory" in matched[4]["initialization"]
    isolated = csv(output / entry["reservoir_id"] / "matched_failure_interval/substeps_4/simulation.csv")
    full = csv(output / entry["reservoir_id"] / "substeps_4/simulation.csv")
    assert len(isolated) == 2
    pd.testing.assert_frame_equal(isolated, full.iloc[:2], check_exact=True)
    partial = csv(output / entry["reservoir_id"] / "matched_failure_interval/substeps_1/partial_simulation.csv")
    assert len(partial) == 1
    assert (output / entry["reservoir_id"] / "matched_failure_interval/substeps_4/internal_steps.csv").is_file()
    assert result["settings_selected"] is None and not result["convergence_assessed"]


def test_monthly_collection_is_explicitly_blocked_before_simulation(tmp_path, monkeypatch):
    path, _ = collection_fixture(tmp_path / "repository")

    def forbidden(*args, **kwargs):
        raise AssertionError("A monthly input must not be silently converted to daily substeps.")

    monkeypatch.setattr(module, "simulate_substep_reservoir", forbidden)
    result = run(path, tmp_path / "review")
    assert result["reservoir_counts"] == {"blocked": 1}
    report = result["reservoirs"][0]
    assert "daily observed-calendar" in report["reason"]
    assert report["resolutions"] == []
    assert all(result[key] for key in GUARDS)


@pytest.mark.parametrize("damage", ["observations", "config", "hash_manifest"])
def test_changed_member_is_isolated_and_other_pinned_member_completes(tmp_path, damage):
    root, output = tmp_path / "repository", tmp_path / "review"
    path, manifest = daily_fixture(root, count=2)
    damaged = manifest["reservoirs"][0]
    if damage == "hash_manifest":
        damaged["files_sha256"] = None
        _json(path, manifest)
    else:
        target = root / damaged["observations_file" if damage == "observations" else "config_file"]
        target.write_text(target.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
    result = run(path, output, substeps=(1,))
    assert result["reservoir_counts"] == {"input_error": 1, "review_completed": 1}
    assert result["reservoirs"][1]["resolutions"][0]["status"] == "completed"
    assert not (output / damaged["reservoir_id"]).exists()
    assert (output / "ex2_Example/review.json").is_file()


@pytest.mark.parametrize("target_name", ["repository", "manifest", "member_data", "member_config", "auxiliary"])
def test_output_guard_protects_all_members_even_unselected_auxiliary_inputs(tmp_path, target_name):
    root = tmp_path / "repository"
    path, manifest = daily_fixture(root, count=2)
    other = manifest["reservoirs"][1]
    auxiliary = root / "data/reservoirs/ex2_Example/metadata/source_note.txt"
    auxiliary.parent.mkdir()
    auxiliary.write_text("immutable source note", encoding="utf-8")
    other["files_sha256"][str(auxiliary.relative_to(root)).replace("\\", "/")] = module.sha256(auxiliary)
    _json(path, manifest)
    outputs = {"repository": root, "manifest": path.parent, "member_data": (root / other["observations_file"]).parent,
               "member_config": root / other["config_file"], "auxiliary": auxiliary.parent}
    before = input_bytes(root)
    with pytest.raises(ValueError, match="protected source inputs"):
        run(path, outputs[target_name], reservoirs=("ex1_Example",))
    assert input_bytes(root) == before


def test_nonempty_output_is_refused_without_overwriting_receipts(tmp_path):
    path, _ = daily_fixture(tmp_path / "repository")
    output = tmp_path / "already_exists"
    output.mkdir()
    receipt = output / "sentinel.json"
    receipt.write_bytes(b'{"preserve":true}\n')
    with pytest.raises(ValueError, match="new or empty output"):
        run(path, output)
    assert receipt.read_bytes() == b'{"preserve":true}\n'
    assert list(output.iterdir()) == [receipt]


@pytest.mark.parametrize("kwargs", [
    {"substeps": ()}, {"substeps": (2, 4)}, {"substeps": (1, 1)}, {"substeps": (True, 1)},
    {"substeps": (1, 0)}, {"substeps": (1, -1)}, {"substeps": (1, 1.)}, {"substeps": (1, "2")},
    {"root_policy": "automatic"}, {"record_internal": 1}, {"reservoirs": ()}, {"reservoirs": ("unknown",)},
])
def test_invalid_settings_are_rejected_before_any_output_creation(tmp_path, kwargs):
    path, _ = daily_fixture(tmp_path / "repository")
    output = tmp_path / "review"
    with pytest.raises((ValueError, TypeError)):
        run(path, output, **kwargs)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["manifest", "observations", "config", "code"])
def test_final_hash_guards_expose_controlled_changes_without_touching_production(tmp_path, monkeypatch, mutation):
    root, output = tmp_path / "repository", tmp_path / "review"
    path, manifest = daily_fixture(root)
    if mutation == "code":
        original = module._code_hashes
        calls = [0]

        def changed_code():
            calls[0] += 1
            hashes = original()
            return hashes if calls[0] == 1 else {**hashes, "synthetic_changed_module.py": "0" * 64}

        monkeypatch.setattr(module, "_code_hashes", changed_code)
    else:
        original = module._run_resolution

        def changed_resolution(*args, **kwargs):
            result = original(*args, **kwargs)
            relative = "observations_file" if mutation == "observations" else "config_file"
            target = path if mutation == "manifest" else root / manifest["reservoirs"][0][relative]
            target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
            return result

        monkeypatch.setattr(module, "_run_resolution", changed_resolution)
    result = run(path, output, substeps=(1,))
    assert result["status"] == "invalid_integrity"
    field = {"manifest": "manifest_file_unchanged", "observations": "input_files_unchanged",
             "config": "input_files_unchanged", "code": "code_files_unchanged"}[mutation]
    assert not result[field]
    if mutation in ("observations", "config"):
        assert result["reservoirs"][0]["status"] == "input_error"
    persisted = json.loads((output / "substep_review.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "invalid_integrity" and not persisted[field]


def test_cli_defaults_are_explicit_daily_case_study_without_optimization(tmp_path, monkeypatch):
    captured = {}

    def fake(manifest, output, **kwargs):
        captured.update({"manifest": manifest, "output": output, **kwargs})
        return {"status": "review_completed", "reservoir_counts": {}, "resolution_counts": {},
                "substeps": list(kwargs["substeps"]), "root_policy": kwargs["root_policy"],
                "settings_selected": None, "reservoirs": []}

    monkeypatch.setattr(module, "run_substep_review", fake)
    assert module.main(["--output", str(tmp_path / "cli")]) == 0
    assert captured["manifest"] == "data/reservoirs/manifest.json"
    assert captured["reservoirs"] == ("ex12_Trinity", "ex26_Ghost")
    assert captured["substeps"] == (1, 24, 48, 96)
    assert captured["root_policy"] == "reject" and captured["record_internal"] is False
