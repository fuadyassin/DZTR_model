"""Portable period-local observation reviews, without simulation or fitting."""

import builtins
import io
import json
import os

import numpy as np
import pandas as pd
import pytest

from dztr.integrations import storage_review as module
from test_collection_runner import _json, collection_fixture
from test_root_policy_review import csv, pin_config, pin_frame


GUARDS = ("manifest_file_unchanged", "input_files_unchanged", "code_files_unchanged")
PERIOD_FILES = ("intervals.csv", "monthly.csv", "extremes.csv", "flags.csv", "summary.json")


def daily_fixture(root, count=1):
    path, manifest = collection_fixture(root, count=count)
    index = pd.date_range("2019-12-28", "2020-01-16", name="date")
    for entry in manifest["reservoirs"]:
        frame = pd.DataFrame({"storage_m3": 3e8 + np.arange(len(index)) * 2 * 86400,
                              "inflow_m3s": 3., "release_m3s": 1.}, index=index)
        entry.update({"frequency": "daily", "rows": len(index),
                      "first_date": str(index[0].date()), "last_date": str(index[-1].date())})
        pin_frame(root, path, manifest, entry, frame)
        config = json.loads((root / entry["config_file"]).read_text(encoding="utf-8"))
        config["sampling"].update({"frequency": "daily", "fixed_step_days": 1.})
        config["periods"] = {"simulation_start": "2019-12-28",
            "calibration": {"start": "2020-01-01", "end": "2020-01-08"},
            "validation": {"start": "2020-01-09", "end": "2020-01-16"}, "warmup_rows": 4}
        pin_config(root, path, manifest, entry, config)
    return path, manifest


def run(path, output, **kwargs):
    kwargs.setdefault("reservoirs", None)
    kwargs.setdefault("all_reservoirs", True)
    return module.run_storage_review(path, output, **kwargs)


def input_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def pin_metadata(root, path, manifest, entry, *, source=None, provenance=None):
    folder = root / "data/reservoirs" / entry["reservoir_id"]
    config = json.loads((root / entry["config_file"]).read_text(encoding="utf-8"))
    for field, name, value in (("matlab_source_file", "matlab_source.json", source),
                                ("provenance_file", "observation_provenance.json", provenance)):
        if value is not None:
            target = folder / name
            _json(target, value)
            relative = target.relative_to(root).as_posix()
            entry["files_sha256"][relative] = module.sha256(target)
            config[field] = f"../../{relative}"
    pin_config(root, path, manifest, entry, config)


def test_daily_review_has_only_within_period_intervals_and_leaves_inputs_unchanged(tmp_path):
    root, output = tmp_path / "repo", tmp_path / "review"
    path, manifest = daily_fixture(root)
    original = input_bytes(root)
    report = run(path, output)
    assert report["schema_version"] == 1 and report["kind"] == "storage_diagnostic_review"
    assert report["status"] == "review_completed"
    assert report["reservoir_counts"] == {"review_completed": 1}
    assert report["selected_reservoir_count"] == 1
    assert all(report[key] for key in GUARDS)
    assert report["settings_selected"] is None and not report["selection_uses_validation"]
    assert not report["production_default_changed"] and not report["model_ready_claimed"]
    assert report["search_evaluations"] == report["simulations_run"] == report["parameters_estimated"] == 0
    assert input_bytes(root) == original
    member = report["reservoirs"][0]
    assert not member["model_ready_claimed"]
    assert member["observation_rows_outside_reported_periods"] == 0
    expected = {"warmup": ("2019-12-28", "2019-12-31", 4),
                "calibration": ("2020-01-01", "2020-01-08", 8),
                "validation": ("2020-01-09", "2020-01-16", 8)}
    for label, (start, end, rows) in expected.items():
        record = member["periods"][label]
        assert record["status"] == "review_completed"
        assert record["first_date"] == start and record["last_date"] == end
        assert record["rows"] == rows and record["intervals"] == rows - 1
        for filename in PERIOD_FILES:
            assert (output / "ex1_Example" / label / filename).is_file()
        table = pd.read_csv(output / "ex1_Example" / label / "intervals.csv", parse_dates=["date", "start_date"])
        assert table.start_date.min() == pd.Timestamp(start)
        assert table.date.min() == pd.Timestamp(start) + pd.Timedelta(days=1)
        assert table.date.max() == pd.Timestamp(end)
        assert table.residual_m3.eq(0).all() and table.cumulative_residual_m3.eq(0).all()
        assert table.valid.all()
        assert len(table) == 3 * (rows - 1)
    assert json.loads((output / "storage_review.json").read_text()) == report
    aggregate = pd.read_csv(output / "storage_review.csv")
    assert len(aggregate) == 9 and aggregate.complete_closure.all()
    assert aggregate.total_intervals.sum() == 3 * 17


def test_monthly_review_keeps_calendar_and_source_duration_separate(tmp_path):
    path, _ = collection_fixture(tmp_path / "repo")
    report = run(path, tmp_path / "review")
    member = report["reservoirs"][0]
    assert member["status"] == "review_completed"
    for label in ("warmup", "calibration", "validation"):
        period = member["periods"][label]
        assert period["frequency"] == "monthly"
        assert len(period["hypotheses"]) == 6
        table = pd.read_csv(tmp_path / "review/ex1_Example" / label / "intervals.csv")
        calendar = table.loc[table.hypothesis.str.contains("__calendar__")]
        fixed = table.loc[table.hypothesis.str.contains("__source_fixed__")]
        assert fixed.step_seconds.eq(30.5 * 86400).all()
        assert calendar.step_seconds.isin(np.array([28, 29, 30, 31]) * 86400).all()
        assert not calendar.step_seconds.eq(30.5 * 86400).any()
        assert period["model_ready_claimed"] is False
        assert "Monthly mean storage" in period["interpretation"]["monthly"]


def test_calibration_outputs_and_lags_are_exactly_invariant_to_held_out_edits(tmp_path):
    root = tmp_path / "repo"
    path, manifest = daily_fixture(root)
    before = run(path, tmp_path / "before")
    entry = manifest["reservoirs"][0]
    frame = csv(root / entry["observations_file"])
    frame.loc["2020-01-09":, "storage_m3"] = np.nan
    frame.loc["2020-01-09":, "inflow_m3s"] = -1e5
    frame.loc["2020-01-09":, "release_m3s"] = 1e6
    pin_frame(root, path, manifest, entry, frame)
    after = run(path, tmp_path / "after")
    assert before["reservoirs"][0]["periods"]["calibration"] == after["reservoirs"][0]["periods"]["calibration"]
    for period in ("warmup", "calibration"):
        for filename in PERIOD_FILES:
            old = tmp_path / "before/ex1_Example" / period / filename
            new = tmp_path / "after/ex1_Example" / period / filename
            assert old.read_bytes() == new.read_bytes()
    assert after["reservoirs"][0]["periods"]["validation"]["observation_flags"]["negative_inflow"] == 8
    assert all(record["statistics"]["valid_intervals"] == 0
               for record in after["reservoirs"][0]["periods"]["validation"]["hypotheses"])
    # Even an all-invalid period must retain an importable empty extremes schema.
    extremes = pd.read_csv(tmp_path / "after/ex1_Example/validation/extremes.csv")
    assert extremes.empty and {"hypothesis", "date", "residual_m3"} <= set(extremes.columns)


def test_external_signed_and_invalid_operating_settings_do_not_prevent_observation_audit(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    path, manifest = daily_fixture(root)
    entry = manifest["reservoirs"][0]
    frame = csv(root / entry["observations_file"])
    frame.inflow_m3s = -3.
    frame.release_m3s = -1.
    pin_frame(root, path, manifest, entry, frame)
    config = json.loads((root / entry["config_file"]).read_text())
    config.update({"simulation_supported": False, "simulation_unsupported_reason": "External MESH forcing is unresolved."})
    config["model"].update({"flood_mode": "unreviewed", "dead_storage_fraction": 1.1})
    config["inflow_preprocessing"] = {"factor": 2., "offset_m3s": 1.}
    pin_config(root, path, manifest, entry, config)

    def forbidden(*args, **kwargs):
        raise AssertionError("Observation review must not simulate, estimate or optimize.")

    import dztr.calibration.sampled as calibration
    import dztr.simulation.runner as daily
    import dztr.simulation.substeps as substeps
    monkeypatch.setattr(calibration, "estimate_generalized_parameters", forbidden)
    monkeypatch.setattr(daily, "simulate_reservoir", forbidden)
    monkeypatch.setattr(substeps, "simulate_substep_reservoir", forbidden)
    report = run(path, tmp_path / "review", lag_diagnostics=False)
    member = report["reservoirs"][0]
    assert member["status"] == "review_completed"
    assert not member["source_simulation_supported"] and not member["model_ready_claimed"]
    assert member["fixed_model"] == config["model"]
    for period in member["periods"].values():
        assert period["configured_dead_exceeds_capacity"]
        assert period["observation_flags"]["negative_inflow"] == period["rows"]
        assert period["observation_flags"]["negative_release"] == period["rows"]
        assert len(period["hypotheses"]) == 6
        assert all(entry["lag_comparisons"] == [] for entry in period["hypotheses"])
    assert report["simulations_run"] == report["parameters_estimated"] == report["search_evaluations"] == 0


def test_metadata_e_drive_strings_are_evidence_and_never_dereferenced(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    path, manifest = daily_fixture(root)
    entry = manifest["reservoirs"][0]
    source = {"folder": "E:\\missing\\example", "model": {"dead_storage_expression": "min(Lc)"},
              "initial_storage_source": "coefficient", "initial_release_source": "coefficient",
              "evidence": [{"file": "E:\\missing\\SimRISM14.m", "field": "Ld", "code": "Ld=min(Lc);"}]}
    provenance = {"status": "verified", "workbook_file": "E:\\missing\\res_info.xlsx", "issues": [],
                  "units": {"storage": "m3", "flows": "m3/s"}, "aggregation": "none"}
    pin_metadata(root, path, manifest, entry, source=source, provenance=provenance)
    original_builtin, original_io = builtins.open, io.open

    def guarded(original):
        def open_file(file, *args, **kwargs):
            if not isinstance(file, int) and os.fspath(file).upper().startswith("E:"):
                raise AssertionError("Evidence paths on E: must not be opened.")
            return original(file, *args, **kwargs)
        return open_file

    monkeypatch.setattr(builtins, "open", guarded(original_builtin))
    monkeypatch.setattr(io, "open", guarded(original_io))
    report = run(path, tmp_path / "review")
    metadata = report["reservoirs"][0]["metadata"]
    assert metadata["source_dead_storage_expression"] == "min(Lc)"
    assert metadata["source_dead_storage_evidence"] == source["evidence"]
    assert not metadata["storage_endpoint_timing_confirmed"]
    assert not metadata["flow_averaging_confirmed"]
    assert not metadata["inflow_independent_measurement_confirmed"]
    assert all(report[key] for key in GUARDS)


@pytest.mark.parametrize("failure", ["observations_sha", "config_sha", "invalid_period", "metadata_list", "metadata_model_list"])
def test_bad_member_is_isolated_and_other_member_still_reviewed(tmp_path, failure):
    root = tmp_path / "repo"
    path, manifest = daily_fixture(root, count=2)
    entry = manifest["reservoirs"][0]
    expected = "input_error"
    if failure == "observations_sha":
        target = root / entry["observations_file"]
        target.write_text(target.read_text() + "tamper\n")
    elif failure == "config_sha":
        target = root / entry["config_file"]
        target.write_text(target.read_text() + " ")
    elif failure == "invalid_period":
        config = json.loads((root / entry["config_file"]).read_text())
        config["periods"]["validation"]["start"] = "2020-01-08"
        pin_config(root, path, manifest, entry, config)
        expected = "blocked"
    else:
        pin_metadata(root, path, manifest, entry,
                     source=[] if failure == "metadata_list" else {"model": []})
        expected = "blocked"
    report = run(path, tmp_path / "review")
    assert report["reservoir_counts"] == {expected: 1, "review_completed": 1}
    assert report["reservoirs"][0]["status"] == expected
    assert report["reservoirs"][0]["reason"]
    assert report["reservoirs"][1]["status"] == "review_completed"
    assert (tmp_path / "review/ex2_Example/calibration/summary.json").is_file()


@pytest.mark.parametrize("target", ["repository", "manifest_parent", "unselected_observations", "unselected_config", "unselected_metadata"])
def test_output_cannot_overlap_any_pinned_member_input(tmp_path, target):
    root = tmp_path / "repo"
    path, manifest = daily_fixture(root, count=2)
    other = manifest["reservoirs"][1]
    pin_metadata(root, path, manifest, other, source={"model": {}, "evidence": []})
    original = input_bytes(root)
    destinations = {"repository": root, "manifest_parent": path.parent,
                    "unselected_observations": root / other["observations_file"],
                    "unselected_config": root / other["config_file"],
                    "unselected_metadata": root / "data/reservoirs/ex2_Example/matlab_source.json"}
    with pytest.raises(ValueError, match="protected source inputs"):
        module.run_storage_review(path, destinations[target], reservoirs=["Example 1"])
    assert input_bytes(root) == original


def test_nonempty_output_is_never_overwritten(tmp_path):
    path, _ = daily_fixture(tmp_path / "repo")
    output = tmp_path / "review"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("existing scientific evidence")
    with pytest.raises(ValueError):
        run(path, output)
    assert sentinel.read_text() == "existing scientific evidence"
    assert list(output.iterdir()) == [sentinel]


@pytest.mark.parametrize("kwargs", [
    {"lag_diagnostics": 1}, {"all_reservoirs": 1},
    {"reservoirs": [], "all_reservoirs": False},
    {"reservoirs": ["unknown"], "all_reservoirs": False},
    {"reservoirs": ["Example 1", "Example 1"], "all_reservoirs": False},
])
def test_invalid_selection_or_options_are_rejected_before_creating_output(tmp_path, kwargs):
    path, _ = daily_fixture(tmp_path / "repo")
    output = tmp_path / "review"
    with pytest.raises((TypeError, ValueError)):
        run(path, output, **kwargs)
    assert not output.exists()


@pytest.mark.parametrize("changed", ["manifest", "observations", "config", "metadata", "code"])
def test_final_guards_detect_changes_during_review(tmp_path, monkeypatch, changed):
    root = tmp_path / "repo"
    path, manifest = daily_fixture(root)
    entry = manifest["reservoirs"][0]
    pin_metadata(root, path, manifest, entry, source={"model": {}, "evidence": []})
    if changed == "code":
        original = module._code_hashes
        calls = 0

        def changed_hashes():
            nonlocal calls
            calls += 1
            result = original()
            if calls > 1:
                result["evaluation/storage_audit.py"] = "0" * 64
            return result

        monkeypatch.setattr(module, "_code_hashes", changed_hashes)
        guard = "code_files_unchanged"
    else:
        destinations = {"manifest": path, "observations": root / entry["observations_file"],
                        "config": root / entry["config_file"],
                        "metadata": root / "data/reservoirs/ex1_Example/matlab_source.json"}
        original = module.audit_storage
        modified = False

        def changed_audit(*args, **kwargs):
            nonlocal modified
            result = original(*args, **kwargs)
            if not modified:
                destination = destinations[changed]
                destination.write_text(destination.read_text() + " ")
                modified = True
            return result

        monkeypatch.setattr(module, "audit_storage", changed_audit)
        guard = "manifest_file_unchanged" if changed == "manifest" else "input_files_unchanged"
    report = run(path, tmp_path / "review")
    assert report["status"] == "invalid_integrity" and not report[guard]
    assert report["model_ready_claimed"] is False
    if changed in ("observations", "config", "metadata"):
        assert report["reservoirs"][0]["status"] == "input_error"
    persisted = json.loads((tmp_path / "review/storage_review.json").read_text())
    assert persisted["status"] == "invalid_integrity" and not persisted[guard]


def test_one_row_warmup_is_reported_without_manufacturing_interval(tmp_path):
    root = tmp_path / "repo"
    path, manifest = daily_fixture(root)
    entry = manifest["reservoirs"][0]
    config = json.loads((root / entry["config_file"]).read_text())
    config["periods"].update({"simulation_start": "2019-12-31", "warmup_rows": 1})
    pin_config(root, path, manifest, entry, config)
    report = run(path, tmp_path / "review")
    member = report["reservoirs"][0]
    assert member["periods"]["warmup"]["status"] == "insufficient_rows"
    assert member["periods"]["warmup"]["rows"] == 1
    assert not (tmp_path / "review/ex1_Example/warmup").exists()
    assert member["periods"]["calibration"]["intervals"] == 7
    assert member["observation_rows_outside_reported_periods"] == 3


@pytest.mark.parametrize("arguments,expected_all,expected_lags", [([], False, True), (["--all", "--no-lags"], True, False)])
def test_cli_default_and_all_selection_never_launch_other_work(tmp_path, monkeypatch, arguments, expected_all, expected_lags):
    captured = {}

    def fake(manifest, output, **kwargs):
        captured.update({"manifest": manifest, "output": output, **kwargs})
        return {"status": "review_completed", "reservoir_counts": {}, "selected_reservoir_count": 0,
                "settings_selected": None, "simulations_run": 0, "parameters_estimated": 0, "reservoirs": []}

    monkeypatch.setattr(module, "run_storage_review", fake)
    assert module.main(["--output", str(tmp_path / "cli"), *arguments]) == 0
    assert captured["manifest"] == "data/reservoirs/manifest.json"
    assert captured["reservoirs"] == (None if expected_all else ("ex26_Ghost", "ex12_Trinity"))
    assert captured["all_reservoirs"] is expected_all
    assert captured["lag_diagnostics"] is expected_lags
