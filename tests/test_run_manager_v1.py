"""Gateway dependencies, station arithmetic, input pinning and real stage reuse."""
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from strategies.representative_years import YearSelection
from workflows import run_manager_v1 as manager
from workflows import station_inputs_v1 as stations


@pytest.fixture
def workspace():
    parent = manager.ROOT / "outputs"
    folder = Path(tempfile.mkdtemp(prefix="test_run_manager_", dir=parent)).resolve()
    yield folder
    assert folder.is_relative_to(parent.resolve()) and folder.name.startswith("test_run_manager_")
    shutil.rmtree(folder)


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def feature(station, day, discharge=None, level=None):
    return {"properties": {"STATION_NUMBER": station, "DATE": day, "DISCHARGE": discharge, "LEVEL": level}}


def station_spec():
    return {"start": "2000-01-01", "end": "2000-01-02", "inflow_stations": ["A", "B"],
            "outflow_stations": ["C"], "level_stations": ["L"], "evidence": "Synthetic independent tributaries and outlet; selected lake gauge"}


def mock_provider(monkeypatch):
    payloads = {"A": [feature("A", "2000-01-01", 10), feature("A", "2000-01-02", 12)],
                "B": [feature("B", "2000-01-02", 3)],
                "C": [feature("C", "2000-01-01", 0), feature("C", "2000-01-02", 7)],
                "L": [feature("L", "2000-01-01", level=-2), feature("L", "2000-01-02", level=-1)]}
    calls = []
    class Response:
        def __init__(self, station): self.station = station
        def raise_for_status(self): pass
        def json(self): return {"features": payloads[self.station]}
    def get(url, params, timeout):
        calls.append(dict(params))
        return Response(params["STATION_NUMBER"])
    monkeypatch.setattr(stations.requests, "get", get)
    return calls


def test_station_totals_missing_days_level_and_cached_raw(workspace, monkeypatch):
    calls = mock_provider(monkeypatch)
    out = workspace / "inputs"
    result = stations.acquire(station_spec(), out)
    frame = pd.read_csv(result["observations_csv"])
    assert pd.isna(frame.inflow_m3s.iloc[0]) and frame.inflow_m3s.iloc[1] == 15
    assert frame.release_m3s.tolist() == [0, 7]
    assert frame.storage_m3.isna().all()
    assert pd.read_csv(result["levels_csv"]).level_m.tolist() == [-2, -1]
    assert len(calls) == 4
    stations.acquire(station_spec(), out)
    assert len(calls) == 4
    page = next((out / "raw").rglob("page_*.json"))
    page.write_text("{}")
    with pytest.raises(ValueError, match="Cached hydrometric response changed"):
        stations.acquire(station_spec(), out)


@pytest.mark.parametrize("change", [{"inflow_stations": ["A", "A"]}, {"level_stations": ["L", "M"]},
                                   {"outflow_stations": ["A"]}, {"start": "2000-01-03"}, {"evidence": ""}])
def test_station_ambiguous_input_rejected(change):
    with pytest.raises(ValueError):
        stations.validate({**station_spec(), **change})


def test_station_conflicting_duplicates_wrong_station_and_signed(workspace):
    page = workspace / "page.json"
    index = pd.date_range("2000-01-01", periods=2)
    for values, match in [([feature("A", "2000-01-01", 1), feature("A", "2000-01-01", 2)], "Conflicting"),
                          ([feature("B", "2000-01-01", 1)], "different station"),
                          ([feature("A", "2000-01-01", -1)], "Signed")]:
        write(page, {"features": values})
        with pytest.raises(ValueError, match=match):
            stations._series([page], "A", "DISCHARGE", index)


def test_station_pagination_preserves_all_pages_and_quality_flags(workspace, monkeypatch):
    index = pd.date_range("2000-01-01", periods=1001)
    rows = [feature("A", d.date().isoformat(), i) for i, d in enumerate(index)]
    rows[0]["properties"]["DISCHARGE_SYMBOL"] = "E"
    offsets = []
    class Response:
        def __init__(self, offset): self.offset = offset
        def raise_for_status(self): pass
        def json(self): return {"features": rows[self.offset:self.offset + 1000]}
    def get(url, params, timeout):
        offsets.append(params["offset"])
        return Response(params["offset"])
    monkeypatch.setattr(stations.requests, "get", get)
    pages, proof = stations._download("A", "2000-01-01", index[-1].date().isoformat(), workspace)
    assert offsets == [0, 1000] and len(proof["page_sha256"]) == 2
    np.testing.assert_array_equal(stations._series(pages, "A", "DISCHARGE", index), np.arange(1001))
    assert manager.w.read(pages[0])["features"][0]["properties"]["DISCHARGE_SYMBOL"] == "E"


def fixture_pipeline(folder):
    index = pd.date_range("2000-01-01", "2005-12-01", freq="MS", name="date")
    year = index.year - 2000
    frame = pd.DataFrame({"inflow_m3s": 10., "storage_m3": 48e6 + year * 1e6 + np.sin(index.month) * 2e6,
                          "release_m3s": 10 + year * .2 + np.sin(index.month)}, index=index)
    frame.to_csv(folder / "observations.csv", index_label="date")
    roles = tuple("warmup" if y == 2000 else "assessment" if y == 2005 else "calibration" for y in index.year)
    selection = YearSelection(tuple(d.date().isoformat() for d in index), roles, tuple(int(y) for y in index.year),
                              tuple("unclassified" if r == "warmup" else ("dry", "normal", "wet")[int(y) % 3]
                                    for r, y in zip(roles, index.year)), .65, 42, 1, 1)
    write(folder / "selection.json", selection.as_dict())
    model = {"model": {"capacity_m3": 100e6, "initial_storage_m3": 51e6, "initial_release_m3s": 10.,
                       "channel_capacity_m3s": 100., "dead_storage_fraction": .1, "inflow_correction": 1., "flood_mode": "storage"},
             "sampling": {"frequency": "monthly", "fixed_step_days": 30.5, "month_clock": "observed"},
             "root_policy": "nearest_previous_storage", "atol_storage_m3": 1e-4}
    write(folder / "model.json", model)
    spec = {"kind": "dztr_data_availability_v1", "schema_version": 1, "methods": ["observed_monthly"],
            "reservoirs": [{"reservoir_id": "sample", "name": "Sample", "GRAND_ID": None,
                            "observations_csv": "observations.csv", "selection_json": "selection.json",
                            "model_config": "model.json", "forcing_stage": "prepared", "source_gates": []}]}
    cfg = {"kind": "dztr_run_manager_v1", "schema_version": 1,
           "stages": [{"id": "targets", "type": "parameterize", "config": spec},
                      {"id": "baseline", "type": "baseline", "config": {"campaign": "@targets.campaign"}}]}
    return write(folder / "manager.json", cfg)


def test_real_pipeline_until_resume_and_artifact_tampering(workspace, monkeypatch):
    path = fixture_pipeline(workspace)
    out = workspace / "run"
    original = pd.read_csv(workspace / "observations.csv")
    first = manager.run(path, out, until="targets")
    assert first["status"] == "stopped_at_requested_stage"
    state = manager.run(path, out, resume=True)
    assert state["status"] == "completed" and state["stages"][0]["status"] == "reused"
    summary = manager.w.read(out / "stages/baseline/baseline_execution.json")
    assert summary["new_optimizer_evaluations"] == 0 and summary["new_fixed_profile_attempts"] == 1
    def no_execution(*args, **kwargs): raise AssertionError("Completed stage was re-executed")
    monkeypatch.setattr(manager, "execute", no_execution)
    assert all(s["status"] == "reused" for s in manager.run(path, out, resume=True)["stages"])
    pd.testing.assert_frame_equal(original, pd.read_csv(workspace / "observations.csv"))
    artifact = out / "stages/targets/campaign/prepared/sample/observations_si.csv"
    artifact.write_text(artifact.read_text() + "\n")
    with pytest.raises(ValueError, match="Completed stage artifact changed"):
        manager.run(path, out, resume=True)
    assert not (out / "manager.lock").exists()
    assert manager.w.read(out / "status.json")["status"] == "failed"


def test_plan_no_computation_and_changed_input_blocks_resume(workspace, monkeypatch):
    path = fixture_pipeline(workspace)
    monkeypatch.setattr(manager, "execute", lambda *args: pytest.fail("Plan executes no jobs"))
    before = manager.plan(path)
    assert before["stages"][1]["depends_on"] == ["targets"]
    assert not (workspace / "run").exists()
    run = workspace / "run"; run.mkdir(); write(run / "plan.json", before)
    write(workspace / "model.json", {"changed": True})
    with pytest.raises(ValueError, match="inputs, decisions or code changed"):
        manager.run(path, run, resume=True)


@pytest.mark.parametrize("reference", ["@later.campaign", "@targets.unknown", "prefix@targets.campaign", "@targets.campaign/suffix"])
def test_invalid_dependency_or_placeholder_rejected(workspace, reference):
    path = fixture_pipeline(workspace)
    cfg = manager.w.read(path)
    cfg["stages"][1]["config"]["campaign"] = reference
    write(path, cfg)
    with pytest.raises((ValueError, FileNotFoundError)):
        manager.plan(path)


def test_acquisition_stage_is_reused_without_redownload(workspace, monkeypatch):
    calls = mock_provider(monkeypatch)
    path = write(workspace / "manager.json", {"kind": "dztr_run_manager_v1", "schema_version": 1,
                 "stages": [{"id": "stations", "type": "acquire", "config": station_spec()}]})
    manager.run(path, workspace / "run")
    manager.run(path, workspace / "run", resume=True)
    assert len(calls) == 4


def test_download_outputs_feed_geometry_targets_without_manual_paths(workspace, monkeypatch):
    dates = pd.date_range("2000-01-01", "2005-12-31", name="date")
    raw = [feature("A", d.date().isoformat(), 10 + np.sin(d.month), 95 + .1 * (d.year - 2000) + np.sin(d.month))
           for d in dates]
    class Response:
        def __init__(self, offset): self.offset = offset
        def raise_for_status(self): pass
        def json(self): return {"features": raw[self.offset:self.offset + 1000]}
    monkeypatch.setattr(stations.requests, "get", lambda url, params, timeout: Response(params["offset"]))
    roles = tuple("warmup" if d.year == 2000 else "assessment" if d.year == 2005 else "calibration" for d in dates)
    selection = YearSelection(tuple(d.date().isoformat() for d in dates), roles, tuple(int(d.year) for d in dates),
         tuple("unclassified" if r == "warmup" else ("dry", "normal", "wet")[d.year % 3] for r, d in zip(roles, dates)), .65, 42, 1, 1)
    fixture_pipeline(workspace)  # Reuse its physical model, then replace the monthly mask.
    write(workspace / "selection.json", selection.as_dict())
    model = manager.w.read(workspace / "model.json")
    model["sampling"] = {"frequency": "daily", "fixed_step_days": 1., "month_clock": "observed"}
    write(workspace / "model.json", model)
    pd.DataFrame({"depth_m": [0, 10], "storage_m3": [0, 100e6]}).to_csv(workspace / "curve.csv", index=False)
    spec = {"kind": "dztr_data_availability_v1", "schema_version": 1, "methods": ["level_geometry_monthly"],
            "reservoirs": [{"reservoir_id": "sample", "name": "Sample", "GRAND_ID": None,
                            "observations_csv": "@download.observations_csv", "selection_json": "selection.json",
                            "model_config": "model.json", "forcing_stage": "canonical", "source_gates": ["Review timing"]}],
            "supplemental_inputs": {"sample": {"levels_csv": "@download.levels_csv", "level_time_basis": "daily_mean_level",
                "geometry_csv": "curve.csv", "geometry_format": "surveyed_si", "geometry_evidence": "Synthetic linear geometry",
                "reference": {"level_m": 100., "depth_m": 10., "basis": "source_reference", "evidence": "Synthetic datum"},
                "allow_mean_level_proxy": True}}}
    config = {"kind": "dztr_run_manager_v1", "schema_version": 1, "stages": [
        {"id": "download", "type": "acquire", "config": {"start": "2000-01-01", "end": "2005-12-31",
         "outflow_stations": ["A"], "level_stations": ["A"], "evidence": "Synthetic test: release and lake level at one gauge"}},
        {"id": "targets", "type": "parameterize", "config": spec}]}
    path = write(workspace / "manager.json", config)
    state = manager.run(path, workspace / "run")
    assert state["status"] == "completed"
    campaign = Path(state["stages"][1]["outputs"]["campaign"])
    conversion = pd.read_csv(campaign / "prepared/sample/level_conversion_si.csv")
    np.testing.assert_allclose(conversion.storage_m3, [(.1 * (d.year - 2000) + np.sin(d.month) + 5) * 1e7 for d in dates])
    profile = manager.w.read(campaign / "profile_registry.json")["profiles"][0]
    assert not profile["simulation_ready"] and not profile["calibration_ready"]
    assert "Review timing" in profile["reasons"]


def test_native_calibration_and_simulation_dispatch_is_explicit(workspace, monkeypatch):
    calls = []
    monkeypatch.setattr(manager, "_command", lambda args, folder: calls.append(list(map(str, args))))
    calibrate = {"id": "cal", "type": "calibrate", "backend": "v1",
                 "config": {"schema_version": 1, "optimization": {"evaluation_budget": 512}}}
    folder = workspace / "cal"; folder.mkdir()
    outputs = manager.execute(calibrate, folder, {})
    assert calls[0][0].endswith("reservoir_workflow.py") and calls[0][1] == "run"
    assert outputs["registry"].endswith("releases/budget_512/registry.json") or outputs["registry"].endswith("releases\\budget_512\\registry.json")
    folder = workspace / "sim"; folder.mkdir()
    simulate = {"id": "sim", "type": "simulate", "backend": "v2",
                "config": {"registry": "@cal.registry", "forcing_csv": "forcing.csv", "grand_id": 253}}
    manager.execute(simulate, folder, {"cal": outputs})
    assert calls[1][0].endswith("reservoir_workflow_v2.py") and "--grand-id" in calls[1]


def test_failure_records_stage_and_does_not_run_dependents(workspace, monkeypatch):
    path = fixture_pipeline(workspace)
    def failed(*args): raise RuntimeError("Interrupted preparation")
    monkeypatch.setattr(manager, "execute", failed)
    with pytest.raises(RuntimeError, match="Interrupted"):
        manager.run(path, workspace / "run")
    state = manager.w.read(workspace / "run/status.json")
    assert [s["status"] for s in state["stages"]] == ["failed", "pending"]
    assert not (workspace / "run/manager.lock").exists()


def test_inputs_cannot_change_between_plan_and_receipt(workspace, monkeypatch):
    path = fixture_pipeline(workspace)
    def changed(stage, folder, results):
        write(workspace / "model.json", {"changed_during_stage": True})
        return {}
    monkeypatch.setattr(manager, "execute", changed)
    with pytest.raises(ValueError, match="changed during execution"):
        manager.run(path, workspace / "run")
    assert not (workspace / "run/stages/targets/receipt.json").exists()
