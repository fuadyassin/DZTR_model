"""Exact handoff, causal fitting, real checkpoint continuation and manager replay."""
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from dztr.calibration.candidates import parameters_to_vector
from strategies.representative_years import YearSelection
from workflows import data_availability_v1 as availability
from workflows import prepared_calibration_v1 as adapter
from workflows import reservoir_v2 as native
from workflows import run_manager_v2 as manager


@pytest.fixture
def workspace():
    parent = native.ROOT / "outputs"
    folder = Path(tempfile.mkdtemp(prefix="test_prepared_calibration_", dir=parent)).resolve()
    yield folder
    assert folder.is_relative_to(parent.resolve()) and folder.name.startswith("test_prepared_calibration_")
    shutil.rmtree(folder)


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def make_inputs(folder, *, methods=None, gates=None, changed_assessment=False):
    dates = pd.date_range("2000-01-01", "2009-12-01", freq="MS", name="date")
    x = np.arange(len(dates)); phase = 2 * np.pi * x / 12
    frame = pd.DataFrame({"inflow_m3s": 80 + 15 * np.sin(phase) + 8 * np.sin(x / 19),
                         "storage_m3": 500e6 + 80e6 * np.sin(phase - .8) + 2e6 * (dates.year - 2000),
                         "release_m3s": 80 + 12 * np.sin(phase - .3) + 3 * np.sin(x / 15)}, index=dates)
    roles = tuple("warmup" if y == 2000 else "assessment" if y in (2003, 2007) else "calibration" for y in dates.year)
    selection = YearSelection(tuple(d.date().isoformat() for d in dates), roles, tuple(int(y) for y in dates.year),
       tuple("unclassified" if r == "warmup" else ("dry", "normal", "wet")[int(y) % 3] for r, y in zip(roles, dates.year)), .65, 42, 1, 1)
    if changed_assessment:
        frame.loc[selection.mask(frame.index, "assessment"), ["storage_m3", "release_m3s"]] *= 1.2
    frame.to_csv(folder / "observations.csv", index_label="date", float_format="%.17g")
    write(folder / "selection.json", selection.as_dict())
    model = {"model": {"capacity_m3": 1e9, "initial_storage_m3": 500e6, "initial_release_m3s": 80.,
                        "channel_capacity_m3s": 500., "dead_storage_fraction": .05, "inflow_correction": 1., "flood_mode": "storage"},
             "sampling": {"frequency": "monthly", "fixed_step_days": 30.5, "month_clock": "observed"},
             "root_policy": "nearest_previous_storage", "atol_storage_m3": 1e-4}
    write(folder / "model.json", model)
    cfg = {"kind": "dztr_data_availability_v1", "schema_version": 1, "methods": methods or ["observed_monthly"],
       "reservoirs": [{"reservoir_id": "sample", "name": "Sample", "GRAND_ID": 123,
        "observations_csv": "observations.csv", "model_config": "model.json", "selection_json": "selection.json",
        "forcing_stage": "prepared", "source_gates": gates or []}]}
    return write(folder / "targets.json", cfg)


def handoff_config(folder, campaign, *, scoring="pooled"):
    return write(folder / "calibrate.json", {"kind": "dztr_prepared_calibration_v1", "schema_version": 1,
      "campaign": str(campaign), "profiles": [{"reservoir_id": "sample", "method": "observed_monthly"}],
      "scoring": scoring, "optimization": {"seeds": [7, 8], "population": 4, "evaluation_budget": 8, "workers": 2}})


def test_handoff_keeps_exact_targets_calendar_constants_and_never_reselects(workspace, monkeypatch):
    targets = make_inputs(workspace)
    campaign = availability.prepare(targets, workspace / "targets")
    path = handoff_config(workspace, campaign)
    def forbidden(*args, **kwargs): raise AssertionError("Year selection must not run during handoff")
    monkeypatch.setattr(native, "select_representative_years", forbidden)
    out = adapter.prepare(path, workspace / "calibration")
    adapter.verify(out)
    profile = native.read(campaign / "profile_registry.json")["profiles"][0]
    source = campaign / profile["prepared_folder"]
    objective, template, bounds, plan, frame, model = native.objective_for(out, "sample")
    assert native.sha(source / "monthly_targets_si.csv") == native.sha(out / "prepared/sample/initial_targets_si.csv")
    assert native.sha(source.parent / "observations_si.csv") == native.sha(out / "prepared/sample/observations_si.csv")
    assert native.sha(source.parent / "selection.json") == native.sha(out / "prepared/sample/selection.json")
    assert plan.signature == profile["identity"]["selection_signature"] == bounds.selection_signature
    assert model["physical_constants"] == profile["identity"]["physical_constants"]
    assert objective.cost_mode == "pooled_nse"
    assert native.read(out / "preparation.json")["selection_recomputed"] is False
    assert native.read(campaign / "profile_registry.json")["profiles"][0]["calibration_ready"] is False
    assert native.read(out / "preparation.json")["reservoirs"][0]["calibration_ready"]
    stored = pd.read_csv(out / "prepared/sample/bounds_si.csv", index_col="month", float_precision="round_trip")
    pd.testing.assert_frame_equal(stored, bounds.empirical.monthly_bounds)


@pytest.mark.parametrize("scoring", ["pooled", "equal_regime"])
def test_held_out_values_cannot_change_fitting_targets_bounds_or_mask(workspace, scoring):
    for name, altered in (("original", False), ("changed", True)):
        folder = workspace / name; folder.mkdir()
        campaign = availability.prepare(make_inputs(folder, changed_assessment=altered), folder / "targets")
        adapter.prepare(handoff_config(folder, campaign, scoring=scoring), folder / "calibration")
    a = native.objective_for(workspace / "original/calibration", "sample")
    b = native.objective_for(workspace / "changed/calibration", "sample")
    assert a[3].signature == b[3].signature
    np.testing.assert_array_equal(parameters_to_vector(a[1]), parameters_to_vector(b[1]))
    np.testing.assert_array_equal(a[2].lower_vector, b[2].lower_vector)
    np.testing.assert_array_equal(a[2].upper_vector, b[2].upper_vector)


def test_real_continuation_matches_uninterrupted_and_published_baseline_reuses_cache(workspace, monkeypatch):
    campaign = availability.prepare(make_inputs(workspace), workspace / "targets")
    availability.run_baselines(campaign)
    cfg = handoff_config(workspace, campaign)
    out = adapter.prepare(cfg, workspace / "continued")
    for seed in (7, 8): native.calibrate(out, "sample", seed, 4)
    first = adapter.finish(out, 4); first_hash = native.sha(first.parent / "manifest.json")
    for seed in (7, 8): native.calibrate(out, "sample", seed, 8)
    later = adapter.finish(out, 8)
    assert native.sha(first.parent / "manifest.json") == first_hash
    direct = adapter.prepare(cfg, workspace / "direct")
    native.calibrate(direct, "sample", 7, 8)
    for name in ("trials.csv", "vectors.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(out / f"jobs/sample/seed_7/budget_8/{name}", float_precision="round_trip"),
                                      pd.read_csv(direct / f"jobs/sample/seed_7/budget_8/{name}", float_precision="round_trip"))
    vectors = pd.read_csv(out / "jobs/sample/seed_7/budget_8/vectors.csv", index_col="evaluation_id", float_precision="round_trip")
    np.testing.assert_array_equal(vectors.loc[0], parameters_to_vector(native.objective_for(out, "sample")[1]))
    profile, parameters = native.load_profile(later, grand_id=123)
    assert profile["parameterization"]["selection_recomputed"] is False
    assert profile["selection_uses_assessment"] is False and not profile["convergence_established"]
    comparison = pd.read_csv(later.parent / "baseline_comparison.csv")
    assert len(comparison) == 6 and set(comparison.role) == {"seed", "selected"}
    assert comparison.baseline_status.eq("completed").all()
    np.testing.assert_allclose(comparison.improvement_storage_nse, comparison.calibrated_storage_nse - comparison.baseline_storage_nse)
    assert native.read(later.parent / "baseline_reuse.json")["new_baseline_simulations"] == 0
    table = pd.read_csv(later.parent / "reservoir_parameterization_by_GRAND_ID.csv")
    assert table.GRAND_ID.iloc[0] == 123 and table.capacity_m3.iloc[0] == 1e9
    def forbidden(*args, **kwargs): raise AssertionError("Completed release was recalculated")
    monkeypatch.setattr(native, "finish", forbidden)
    assert adapter.finish(out, 8) == later
    with pytest.raises(ValueError, match="cannot shrink"): native.calibrate(out, "sample", 7, 4)


def test_unsupported_methods_source_gates_and_decision_tampering_are_rejected(workspace):
    cfg = make_inputs(workspace, methods=["observed_monthly", "flow_only"], gates=["Unresolved source datum"])
    campaign = availability.prepare(cfg, workspace / "targets")
    path = handoff_config(workspace, campaign)
    with pytest.raises(ValueError, match="unresolved source"): adapter.prepare(path, workspace / "blocked")
    value = native.read(path); value["profiles"][0]["method"] = "flow_only"; write(path, value)
    with pytest.raises(ValueError, match="observed_monthly"): adapter.prepare(path, workspace / "unsupported")
    fresh = workspace / "fresh"; fresh.mkdir()
    campaign = availability.prepare(make_inputs(fresh), fresh / "targets")
    path = handoff_config(fresh, campaign)
    out = adapter.prepare(path, fresh / "calibration")
    (out / "handoff_config.json").write_text("{}")
    with pytest.raises(ValueError, match="adapter decisions changed"): adapter.verify(out)


def test_manager_runs_exact_handoff_export_and_selected_profile_replay(workspace):
    targets = make_inputs(workspace)
    calibration = {"kind": "dztr_prepared_calibration_v1", "schema_version": 1, "campaign": "@targets.campaign",
      "profiles": [{"reservoir_id": "sample", "method": "observed_monthly"}],
      "optimization": {"seeds": [7], "population": 4, "evaluation_budget": 8, "workers": 1}}
    cfg = {"kind": "dztr_run_manager_v2", "schema_version": 2, "stages": [
       {"id": "targets", "type": "parameterize", "config": str(targets)},
       {"id": "baseline", "type": "baseline", "config": {"campaign": "@targets.campaign"}},
       {"id": "calibration", "type": "calibrate", "backend": "prepared_v1", "config": calibration},
       {"id": "replay", "type": "simulate", "backend": "v2", "config": {
        "registry": "@calibration.registry", "forcing_csv": "@calibration.forcing_csv", "grand_id": 123}}]}
    path = write(workspace / "manager.json", cfg)
    plan = manager.plan(path)
    assert plan["stages"][2]["depends_on"] == ["targets"]
    out = workspace / "managed"
    state = manager.run(path, out)
    assert state["status"] == "completed"
    registry = Path(state["stages"][2]["outputs"]["registry"])
    native.verify_registry(registry)
    assert native.read(out / "stages/replay/campaign/receipt.json")["maximum_balance_residual_m3"] <= 1e-4
    assert all(s["status"] == "reused" for s in manager.run(path, out, resume=True)["stages"])


def test_batch_rejects_ambiguous_grand_aliases_before_preparation(workspace):
    path = make_inputs(workspace)
    cfg = native.read(path)
    other = dict(cfg["reservoirs"][0], reservoir_id="other", name="Other")
    cfg["reservoirs"].append(other)
    write(path, cfg)
    campaign = availability.prepare(path, workspace / "targets")
    path = handoff_config(workspace, campaign)
    cfg = native.read(path)
    cfg["profiles"].append({"reservoir_id": "other", "method": "observed_monthly"})
    write(path, cfg)
    with pytest.raises(ValueError, match="Duplicate verified GRAND_ID"):
        adapter.prepare(path, workspace / "ambiguous")
    assert not (workspace / "ambiguous").exists()
