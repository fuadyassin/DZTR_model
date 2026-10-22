"""Scientific preparation, leakage, failure support and persistent-job checks."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from dztr.calibration.sampled import _observations
from dztr.parameterization import estimate_generalized_parameters
from strategies.data_availability_v1 import NaturalizedOutflow, build_targets, capabilities
from strategies.representative_years import YearSelection
from workflows import data_availability_v1 as fleet
from workflows.geometry_v1 import (GeometryCurve, LevelReference, calibration_maximum_reference,
                                   convert_levels, read_regeom)
from workflows.observation_operators_v1 import score_native


@pytest.fixture
def series_data():
    index = pd.date_range("2000-01-01", "2005-12-01", freq="MS", name="date")
    years = index.year - 2000
    frame = pd.DataFrame({"inflow_m3s": 10.,
        "storage_m3": 48e6 + years * 1e6 + np.sin(index.month) * 2e6,
        "release_m3s": 10 + years * .2 + np.sin(index.month)}, index=index)
    roles = tuple("warmup" if y == 2000 else "assessment" if y == 2005 else "calibration" for y in index.year)
    selection = YearSelection(tuple(d.date().isoformat() for d in index), roles, tuple(int(y) for y in index.year),
        tuple("unclassified" if r == "warmup" else ("dry", "normal", "wet")[int(y) % 3] for r, y in zip(roles, index.year)),
        .65, 42, 1, 1)
    return frame, pd.Series(roles, index=index).eq("calibration"), selection


@pytest.fixture
def curve():
    return GeometryCurve((0, 5, 10), (0, 25e6, 100e6), (0, 1e6, 2e6), "fixture", "0" * 64)


def test_conversion_is_vectorized_si_and_never_clamps(curve):
    dates = pd.date_range("2000-01-01", periods=5)
    levels = pd.Series([100., 97.5, 90., 101., np.nan], index=dates)
    result, receipt = convert_levels(levels, curve, LevelReference(100., 10., "source_reference", "known reference"),
                                      time_basis="end_of_day_level")
    np.testing.assert_allclose(result.storage_m3[:3], [100e6, 62.5e6, 0])
    assert result.storage_m3.iloc[3:].isna().all()
    assert result.conversion_status.tolist()[-2:] == ["outside_geometry_domain", "missing_level"]
    assert receipt["out_of_domain_rows"] == 1
    assert not receipt["extrapolation_applied"]
    # User-visible DataFrames do not mutate the immutable scientific curve.
    copy = curve.table(); copy.loc[0, "storage_m3"] = 99
    assert curve.storage_m3[0] == 0


@pytest.mark.parametrize("changes", [dict(depth_m=(0, 5, 5)), dict(storage_m3=(0, 20, 10)),
    dict(area_m2=(0, -1, 2)), dict(storage_m3=(0, np.inf, 10)), dict(header_height_m=-99)])
def test_invalid_geometry_rejected(curve, changes):
    with pytest.raises(ValueError):
        GeometryCurve(**{**asdict(curve), **changes})


def test_regeom_units_placeholder_and_portable_identity(tmp_path):
    text = "GRanD ID= 3\nApproximated Geometry= Bowl\nDam Height (m)= -94.05\nDepth(m),Area(skm),Storage(mcm)\n0,0,0\n10,2,100\n"
    source = tmp_path / "3.csv"; source.write_text(text)
    curve = read_regeom(source, expected_grand_id=3)
    assert curve.header_height_m is None
    assert curve.storage_m3[-1] == 100e6 and curve.area_m2[-1] == 2e6
    copy = tmp_path / "relocated.csv"; copy.write_text(text)
    assert read_regeom(copy).signature == curve.signature
    with pytest.raises(ValueError, match="verified mapping"):
        read_regeom(source, expected_grand_id=4)
    converted, _ = convert_levels(pd.Series([100.], index=pd.date_range("2000-01-01", periods=1)),
                                  curve, LevelReference(100, 10, "source_reference", "independent coordinate"), time_basis="point_level")
    assert converted.storage_m3.iloc[0] == 100e6


def test_held_out_levels_cannot_move_calibration_reference(series_data, curve):
    frame, mask, selection = series_data
    levels = pd.Series(np.linspace(95., 100., len(frame)), index=frame.index)
    ref = calibration_maximum_reference(levels, curve, mask, fitting_signature=selection.signature, allow_exploratory=True)
    changed = levels.copy(); changed.loc[~mask] = 150.
    ref2 = calibration_maximum_reference(changed, curve, mask, fitting_signature=selection.signature, allow_exploratory=True)
    assert ref == ref2
    a, _ = convert_levels(levels, curve, ref, time_basis="point_level")
    b, _ = convert_levels(changed, curve, ref2, time_basis="point_level")
    pd.testing.assert_series_equal(a.loc[mask, "storage_m3"], b.loc[mask, "storage_m3"])
    assert b.loc[~mask, "storage_m3"].isna().all()
    with pytest.raises(ValueError, match="opt-in"):
        calibration_maximum_reference(levels, curve, mask, fitting_signature=selection.signature)


def test_daily_mean_level_requires_explicit_proxy(curve):
    levels = pd.Series([100.], index=pd.date_range("2000-01-01", periods=1))
    reference = LevelReference(100, 10, "source_reference", "survey")
    with pytest.raises(ValueError, match="proxy"):
        convert_levels(levels, curve, reference, time_basis="daily_mean_level")
    _, receipt = convert_levels(levels, curve, reference, time_basis="daily_mean_level", allow_mean_level_proxy=True)
    assert receipt["storage_is_mean_level_proxy"]


@pytest.mark.parametrize("method", ["hazen", "linear"])
def test_supplied_storage_recipe_reproduces_existing_estimator(series_data, method):
    frame, mask, _ = series_data
    old = estimate_generalized_parameters(_observations(frame.loc[mask]), channel_capacity_m3s=100., quantile_method=method)
    new = build_targets("observed_monthly", frame, mask, capacity_m3=100e6, channel_capacity_m3s=100., quantile_method=method)
    pd.testing.assert_frame_equal(new["targets"], old.monthly_targets)
    assert new["target_estimation_ready"]
    mutated = frame.copy(); mutated.loc[~mask, ["storage_m3", "release_m3s"]] *= 100
    same = build_targets("observed_monthly", mutated, mask, capacity_m3=100e6, channel_capacity_m3s=100., quantile_method=method)
    pd.testing.assert_frame_equal(same["targets"], new["targets"])


def test_flow_only_does_not_invent_storage_observations(series_data):
    frame, mask, _ = series_data
    frame = frame.copy(); frame["storage_m3"] = np.nan
    frame.loc[mask & (frame.index.month == 1), "release_m3s"] = 0
    original = frame.copy()
    result = build_targets("flow_only", frame, mask, capacity_m3=100e6)
    np.testing.assert_array_equal(result["targets"].iloc[0, :3], [20e6, 30e6, 97e6])
    np.testing.assert_array_equal(result["targets"].iloc[0, 3:], [0, 0, 0])
    assert result["objective_variables"] == ("release",)
    assert result["target_estimation_ready"]
    pd.testing.assert_frame_equal(original, frame)


@pytest.mark.parametrize("method", ["storage_only", "ungauged_cr1"])
def test_release_prior_is_required_and_never_replaced_by_inflow(series_data, method):
    frame, mask, _ = series_data
    result = build_targets(method, frame, mask, capacity_m3=100e6)
    assert not result["target_estimation_ready"]
    assert result["targets"] is None
    assert any("needs_" in reason for reason in result["reasons"])
    prior = NaturalizedOutflow(frame.inflow_m3s.rename("naturalized"), "simulated_naturalized_outflow", "independent model run", True)
    ready = build_targets(method, frame, mask, capacity_m3=100e6, naturalized=prior)
    assert ready["target_estimation_ready"]
    assert ready["objective_variables"] == (("storage",) if method == "storage_only" else ())
    with pytest.raises(ValueError, match="naturalized"):
        NaturalizedOutflow(frame.inflow_m3s, "inflow", "raw input", True)


def test_coverage_masks_and_source_gates(series_data):
    frame, mask, _ = series_data
    wrong = mask.copy(); wrong.index = wrong.index + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="calendar"):
        build_targets("observed_monthly", frame, wrong, capacity_m3=100e6)
    with pytest.raises(ValueError, match="65 percent"):
        build_targets("observed_monthly", frame, pd.Series(frame.index.year == 2001, index=frame.index), capacity_m3=100e6)
    invalid = frame.copy(); invalid.loc[mask & (invalid.index.month == 3), "storage_m3"] = np.nan
    result = build_targets("observed_monthly", invalid, mask, capacity_m3=100e6, channel_capacity_m3s=100.)
    assert not result["target_estimation_ready"]
    gated = build_targets("flow_only", frame, mask, capacity_m3=100e6, source_gates=["unresolved capacity"])
    assert not gated["target_estimation_ready"] and "unresolved capacity" in gated["reasons"]
    assert capabilities(invalid, mask)["recommended_method"] == "flow_only"


def test_scoring_keeps_observation_support_and_valid_zeros(series_data):
    frame, _, selection = series_data
    frame = frame.copy(); frame.iloc[13, frame.columns.get_loc("release_m3s")] = 0.
    simulated = frame[["storage_m3", "release_m3s"]].copy()
    roles = pd.Series(selection.roles, index=frame.index)
    rows = score_native(frame, simulated, roles)
    assert rows[0]["release_pairs"] == 48 and rows[0]["release_nse"] == 1.
    simulated.iloc[13, 0] = np.nan
    with pytest.raises(ValueError, match="fixed scored support"):
        score_native(frame, simulated, roles)
    with pytest.raises(ValueError, match="Unsupported observation operator"):
        score_native(frame, simulated, roles, observation_basis="daily_mean_storage")


@pytest.fixture
def workspace():
    # A fresh, owned workspace child is portable and never targets scientific
    # data/checkpoints. Explicitly verify it before recursive fixture cleanup.
    parent = fleet.ROOT / "outputs"
    parent.mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="test_availability_", dir=parent)).resolve()
    yield folder
    assert folder.is_relative_to(parent.resolve()) and folder.name.startswith("test_availability_")
    shutil.rmtree(folder)


def fixture_config(workspace, series_data, *, supplements=None, gates=None):
    frame, _, selection = series_data
    frame.to_csv(workspace / "observations.csv", index_label="date", float_format="%.17g")
    (workspace / "selection.json").write_text(json.dumps(selection.as_dict()))
    model = {"model": {"capacity_m3": 100e6, "initial_storage_m3": 51e6, "initial_release_m3s": 10.,
                        "channel_capacity_m3s": 100., "dead_storage_fraction": .1, "inflow_correction": 1., "flood_mode": "storage"},
             "sampling": {"frequency": "monthly", "fixed_step_days": 30.5, "month_clock": "observed"},
             "root_policy": "nearest_previous_storage", "atol_storage_m3": 1e-4,
             "inflow_preprocessing": {"factor": 1., "offset_m3s": 0.}}
    (workspace / "model.json").write_text(json.dumps(model))
    cfg = {"kind": "dztr_data_availability_v1", "schema_version": 1, "methods": list(fleet.METHODS),
           "reservoirs": [{"reservoir_id": "sample", "name": "Sample", "GRAND_ID": None,
                            "observations_csv": "observations.csv", "model_config": "model.json", "selection_json": "selection.json",
                            "forcing_stage": "prepared", "source_gates": gates or []}],
           "supplemental_inputs": {"sample": supplements} if supplements else {}}
    path = workspace / "config.json"; path.write_text(json.dumps(cfg))
    return path


def test_pipeline_and_verified_baseline_reuse(workspace, series_data, monkeypatch):
    path = fixture_config(workspace, series_data)
    campaign = fleet.prepare(path, workspace / "campaign")
    summary = fleet.w.read(campaign / "report/summary.json")
    assert summary["reservoirs"] == 1 and summary["recipe_profiles"] == 5
    assert summary["methods"]["ungauged_cr1"]["valid_parameters"] == 0
    first = fleet.run_baselines(campaign)
    assert first["new_fixed_profile_attempts"] == 2 and first["new_optimizer_evaluations"] == 0
    report = pd.read_csv(campaign / "report/profile_inventory.csv")
    assert report.baseline_status.eq("completed").any()
    def must_not_rerun(*args, **kwargs):
        raise AssertionError("Verified simulation unexpectedly repeated")
    monkeypatch.setattr(fleet.app, "run_frame", must_not_rerun)
    again = fleet.run_baselines(campaign)
    assert again["verified_jobs_reused"] == 2 and again["new_fixed_profile_attempts"] == 0
    raw = campaign / "prepared/sample/observations_si.csv"
    raw.write_text(raw.read_text() + "\n")
    with pytest.raises(ValueError, match="Prepared input changed"):
        fleet.verify(campaign)


def test_pipeline_geometry_and_naturalized_recipe(workspace, series_data):
    frame, _, _ = series_data
    pd.DataFrame({"depth_m": [0, 10], "storage_m3": [0, 100e6]}).to_csv(workspace / "curve.csv", index=False)
    pd.DataFrame({"level_m": 90 + frame.storage_m3 / 1e7}, index=frame.index).to_csv(workspace / "levels.csv", index_label="date")
    pd.DataFrame({"naturalized_outflow_m3s": frame.inflow_m3s}, index=frame.index).to_csv(workspace / "naturalized.csv", index_label="date")
    supplements = {"geometry_csv": "curve.csv", "geometry_format": "surveyed_si", "geometry_evidence": "synthetic test reference",
                   "levels_csv": "levels.csv", "level_time_basis": "end_of_day_level",
                   "reference": {"level_m": 100., "depth_m": 10., "basis": "source_reference", "evidence": "known test datum"},
                   "naturalized_outflow": {"csv": "naturalized.csv", "source_kind": "simulated_naturalized_outflow",
                                           "evidence": "test model", "independent_of_target_responses": True}}
    path = fixture_config(workspace, series_data, supplements=supplements)
    out = fleet.prepare(path, workspace / "campaign")
    inventory = pd.read_csv(out / "report/profile_inventory.csv")
    assert inventory.parameters_ready.all()
    raw = fleet.read_frame(out / "prepared/sample/observations_si.csv")
    derived = fleet.read_frame(out / "prepared/sample/level_observations_si.csv")
    np.testing.assert_allclose(derived.storage_m3, raw.storage_m3, atol=1e-6)
    assert not inventory.calibration_ready.any()  # no new search adapter/job claimed


def test_source_gates_never_promoted_by_a_recipe(workspace, series_data):
    path = fixture_config(workspace, series_data, gates=["unresolved physical storage coordinate"])
    out = fleet.prepare(path, workspace / "campaign")
    profiles = fleet.w.read(out / "profile_registry.json")["profiles"]
    assert not any(p["parameters_ready"] or p["simulation_ready"] for p in profiles)
    assert all("unresolved physical storage coordinate" in p["reasons"] for p in profiles)


def test_geometry_storage_only_and_common_assessment_reference(workspace, series_data, monkeypatch):
    from types import SimpleNamespace
    from dztr.evaluation.metrics import objective_metric
    frame, mask, _ = series_data
    pd.DataFrame({"depth_m": [0, 10], "storage_m3": [0, 100e6]}).to_csv(workspace / "curve.csv", index=False)
    pd.DataFrame({"level_m": 90 + (frame.storage_m3 + 5e6) / 1e7}, index=frame.index).to_csv(workspace / "levels.csv", index_label="date")
    pd.DataFrame({"naturalized_outflow_m3s": frame.inflow_m3s}, index=frame.index).to_csv(workspace / "naturalized.csv", index_label="date")
    supplements = {"geometry_csv": "curve.csv", "geometry_format": "surveyed_si", "geometry_evidence": "test datum",
                   "levels_csv": "levels.csv", "level_time_basis": "point_level",
                   "reference": {"level_m": 100., "depth_m": 10., "basis": "source_reference", "evidence": "test"},
                   "naturalized_outflow": {"csv": "naturalized.csv", "source_kind": "simulated_naturalized_outflow",
                                           "evidence": "test", "independent_of_target_responses": True}}
    path = fixture_config(workspace, series_data, supplements=supplements)
    out = fleet.prepare(path, workspace / "with_supplied")
    def simulate(input_frame, *args, **kwargs):
        result = frame[["storage_m3", "release_m3s"]].reindex(input_frame.index).copy()
        result.storage_m3 += 5e6
        return SimpleNamespace(frame=result), 0.
    monkeypatch.setattr(fleet.app, "run_frame", simulate)
    fleet.run_baselines(out)
    performance = pd.read_csv(out / "report/performance_comparison.csv")
    row = performance[(performance.method == "level_geometry_monthly") & (performance.phase == "calibration")].iloc[0]
    expected = objective_metric(frame.loc[mask, "storage_m3"], frame.loc[mask, "storage_m3"] + 5e6)
    assert row.storage_nse == pytest.approx(expected)
    assert row.storage_reference_kind == "provided_storage_common_reference"
    # With supplied volume absent, the same level series supports storage-only
    # targets; it is still explicitly a geometry response proxy.
    missing = frame.copy(); missing.storage_m3 = np.nan
    missing.to_csv(workspace / "observations.csv", index_label="date")
    other = fleet.prepare(path, workspace / "without_supplied")
    record = next(p for p in fleet.w.read(other / "profile_registry.json")["profiles"] if p["method"] == "storage_only")
    assert record["parameters_ready"] and record["storage_target_source"] == "level_geometry"
