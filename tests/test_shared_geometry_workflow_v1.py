"""Shared estimator contracts, fixed support, no leakage and real continuation."""
import importlib.util
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from dztr.calibration.candidates import parameters_to_vector
from strategies.geometry_objective_v1 import band_nse
from workflows import data_availability_v2 as shared
from workflows import geometry_calibration_v1 as calibration
from workflows import reservoir_v2 as native
from workflows import run_manager_v3 as manager
from workflows.geometry_v1 import GeometryCurve
from workflows.storage_preparation_v1 import prepare_storage

spec = importlib.util.spec_from_file_location("prepared_test_inputs", native.ROOT / "tests/test_prepared_calibration_v1.py")
helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)


@pytest.fixture
def workspace():
    parent = native.ROOT / "outputs"
    folder = Path(tempfile.mkdtemp(prefix="test_shared_geometry_", dir=parent)).resolve()
    yield folder
    assert folder.is_relative_to(parent.resolve()) and folder.name.startswith("test_shared_geometry_")
    shutil.rmtree(folder)


def inputs(folder, *, changed_assessment=False, half_width=None):
    folder.mkdir(parents=True, exist_ok=True)
    path = helpers.make_inputs(folder)
    frame = shared.read_frame(folder / "observations.csv")
    depth = np.interp(frame.storage_m3, [0, 250e6, 1e9], [0, 5, 10])
    level = pd.Series(100 + depth + .03 * np.sin(np.arange(len(frame)) / 13), index=frame.index)
    selection = shared.YearSelection(**native.read(folder / "selection.json"))
    if changed_assessment:
        held = selection.mask(frame.index, "assessment")
        frame.loc[held, ["storage_m3", "release_m3s"]] *= 1.3
        level.loc[held] += 200
        frame.to_csv(folder / "observations.csv", index_label="date", float_format="%.17g")
    level.to_frame("level_m").to_csv(folder / "levels.csv", index_label="date", float_format="%.17g")
    pd.DataFrame({"depth_m": [0, 5, 10], "storage_m3": [0, 250e6, 1e9]}).to_csv(folder / "geometry.csv", index=False)
    cfg = native.read(path); cfg.update(kind="dztr_data_availability_v2", schema_version=2, methods=["observed_monthly", "level_geometry_monthly"])
    cfg["supplemental_inputs"] = {"sample": {"levels_csv": "levels.csv", "level_time_basis": "daily_mean_level", "allow_mean_level_proxy": True,
        "geometry_csv": "geometry.csv", "geometry_format": "surveyed_si", "geometry_evidence": "Synthetic test geometry, not a real survey",
        "level_reference": {"method": "calibration_storage_alignment", "level_datum": "test-datum"},
        "storage_uncertainty": {"kind": "calibration_offset_quantiles", "probabilities": [.05, .95]} if half_width is None else
                               {"kind": "fixed_offset_band", "half_width_m": half_width, "evidence": "Explicit test scenario"}}}
    calibration.write(path, cfg)
    return path


def cal_config(folder, campaign, *, objective="reference_band_nse", seeds=None):
    path = folder / "calibration.json"
    calibration.write(path, {"kind": "dztr_geometry_calibration_v1", "schema_version": 1, "campaign": str(campaign),
        "profiles": [{"reservoir_id": "sample", "method": "level_geometry_monthly"}],
        "scoring": "pooled", "storage_objective": objective,
        "optimization": {"seeds": seeds or [7], "population": 4, "evaluation_budget": 16, "workers": 1}})
    return path


def test_band_loss_has_independent_analytical_value_and_fixed_support():
    c = np.array([10., 20., 30.]); s = np.array([8., 25., 30.])
    assert band_nse(c, c - 2, c + 2, s) == pytest.approx(.955)
    assert band_nse(c, c, c, s) == pytest.approx(.855)
    assert band_nse(c, c - 2, c + 2, c) == 1.
    with pytest.raises(ValueError, match="fixed proxy support"):
        band_nse(c, [8, np.nan, 28], c + 2, s)
    with pytest.raises(ValueError, match="align"):
        band_nse(c, c, c, s[:-1])


def test_release_publication_recovers_transient_windows_lease(workspace, monkeypatch):
    source = workspace / "build"; source.mkdir(); (source / "receipt.json").write_text("verified")
    target = workspace / "release"
    original = Path.replace
    calls = []
    def replace(path, destination):
        calls.append(path)
        if len(calls) < 3:
            error = PermissionError("transient directory lease"); error.winerror = 5; raise error
        return original(path, destination)
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(calibration.time, "sleep", lambda _: None)
    calibration.publish_directory(source, target)
    assert len(calls) == 3 and (target / "receipt.json").read_text() == "verified"
    with pytest.raises(ValueError, match="overwrite"):
        calibration.publish_directory(source, target)


def test_shared_generation_and_calibration_ignore_held_response_changes(workspace):
    prepared = []
    for name, altered in (("original", False), ("altered", True)):
        folder = workspace / name; path = inputs(folder, changed_assessment=altered)
        campaign = shared.prepare(path, folder / "targets")
        cal = calibration.prepare(cal_config(folder, campaign), folder / "cal")
        objective, template, bounds, selection, _, _ = calibration.objective_for(cal, "sample")
        assert selection.roles.count("calibration") / len(selection.roles) >= .65
        result = objective.evaluate(parameters_to_vector(template)); assert result.feasible, result.reason
        reference = native.read(cal / "prepared/sample/level_conversion.json")
        prepared.append((template.monthly_targets, bounds.empirical.monthly_bounds, reference["reference"], reference["uncertainty"], result.objectives))
        table = pd.read_csv(campaign / "report/reservoir_profiles_by_GRAND_ID.csv")
        assert table.query("method=='level_geometry_monthly'").level_reference_method.iloc[0] == "calibration_storage_alignment"
    for a, b in zip(prepared[0][:2], prepared[1][:2]): pd.testing.assert_frame_equal(a, b)
    assert prepared[0][2:] == prepared[1][2:]


def test_reference_options_and_missing_overlap_rules(workspace):
    path = inputs(workspace / "inputs")
    cfg = native.read(path); spec = cfg["supplemental_inputs"]["sample"]
    spec["level_reference"] = {"method": "full_supply_capacity", "level_datum": "test-datum", "anchor_level_m": 110., "evidence": "Assumed full-supply tie", "allow_exploratory": True}
    spec["storage_uncertainty"] = {"kind": "fixed_offset_band", "half_width_m": .1, "evidence": "Exploratory reference perturbation"}
    frame = shared.read_frame(path.parent / "observations.csv"); frame.storage_m3 = np.nan
    frame.to_csv(path.parent / "observations.csv", index_label="date"); calibration.write(path, cfg)
    campaign = shared.prepare(path, workspace / "no_storage")
    receipt = native.read(campaign / "prepared/sample/level_conversion.json")
    assert not receipt["reference"]["uses_storage_observations"]
    assert receipt["reference"]["zero_level_m"] == 100.
    spec["level_reference"].pop("allow_exploratory"); calibration.write(path, cfg)
    with pytest.raises(ValueError, match="exploratory opt-in"):
        shared.prepare(path, workspace / "rejected")


def test_documented_reference_uses_matching_datum_and_mean_proxy_consent():
    index = pd.date_range("2000-01-01", periods=40)
    levels = pd.Series(np.linspace(102, 108, 40), index=index)
    storage = pd.Series(np.nan, index=index)
    curve = GeometryCurve((0, 10), (0, 100e6), None, "test", "0" * 64)
    spec = {"method": "documented_bottom", "level_datum": "gauge", "reference_datum": "geodetic",
            "zero_level_m": 99.5, "level_to_reference_offset_m": -.5,
            "datum_conversion_evidence": "Explicit test conversion", "evidence": "Known test bottom"}
    center, receipt, bands = prepare_storage(levels, storage, curve, pd.Series(True, index=index), "frozen",
        100e6, spec, {"kind": "none"}, time_basis="daily_mean_level", allow_mean_level_proxy=True)
    np.testing.assert_allclose(center.storage_m3, (levels - 100) * 1e7)
    np.testing.assert_array_equal(bands.storage_lower_m3, center.storage_m3)
    np.testing.assert_array_equal(bands.storage_upper_m3, center.storage_m3)
    assert receipt["storage_is_mean_level_proxy"] and not receipt["reference"]["uses_storage_observations"]
    bad = dict(spec); bad.pop("datum_conversion_evidence")
    with pytest.raises(ValueError, match="Different datums"):
        prepare_storage(levels, storage, curve, pd.Series(True, index=index), "frozen", 100e6,
            bad, {"kind": "none"}, time_basis="point_level", allow_mean_level_proxy=False)


def test_published_datum_mismatch_rejected(workspace):
    path = inputs(workspace / "inputs")
    metadata = path.parent / "datum.json"; calibration.write(metadata, {"level_datum": "different-datum"})
    cfg = native.read(path); cfg["supplemental_inputs"]["sample"]["datum_metadata_json"] = "datum.json"
    calibration.write(path, cfg)
    with pytest.raises(ValueError, match="published station datum"):
        shared.prepare(path, workspace / "rejected")


def test_unsupported_uncertainty_domain_blocks_calibration(workspace):
    path = inputs(workspace / "inputs", half_width=100.)
    campaign = shared.prepare(path, workspace / "targets")
    with pytest.raises(ValueError, match="fixed proxy support"):
        calibration.prepare(cal_config(workspace, campaign), workspace / "blocked")
    assert not (workspace / "blocked").exists()


def test_explicit_point_proxy_matches_native_costs(workspace):
    path = inputs(workspace / "inputs")
    campaign = shared.prepare(path, workspace / "targets")
    prepared = calibration.prepare(cal_config(workspace, campaign, objective="proxy_nse"), workspace / "cal")
    objective, template, _, _, _, _ = calibration.objective_for(prepared, "sample")
    result = objective.evaluate(parameters_to_vector(template))
    assert result.feasible
    np.testing.assert_allclose(result.objectives, [-result.metrics["release_nse"], -result.metrics["storage_nse"]])


def test_checkpoint_continuation_export_and_identity(workspace):
    path = inputs(workspace / "inputs")
    campaign = shared.prepare(path, workspace / "targets"); shared.run_baselines(campaign)
    cfg = cal_config(workspace, campaign, seeds=[7, 8])
    resumed = calibration.prepare(cfg, workspace / "resumed")
    first = calibration.calibrate(resumed, "sample", 7, 8)
    first_sha = native.sha(first / "manifest.json")
    calibration.calibrate(resumed, "sample", 7, 16)
    calibration.calibrate(resumed, "sample", 8, 16)
    fresh = calibration.prepare(cfg, workspace / "fresh"); calibration.calibrate(fresh, "sample", 7, 16)
    for name in ("trials.csv", "vectors.csv"):
        a = pd.read_csv(resumed / "jobs/sample/seed_7/budget_16" / name, float_precision="round_trip")
        b = pd.read_csv(fresh / "jobs/sample/seed_7/budget_16" / name, float_precision="round_trip")
        pd.testing.assert_frame_equal(a, b)
    assert native.sha(first / "manifest.json") == first_sha
    registry = calibration.finish(resumed, 16); native.verify_registry(registry)
    profile, parameters = native.load_profile(registry, grand_id=123)
    assert profile["parameterization"]["storage_objective"] == "reference_band_nse"
    assert not profile["selection_uses_assessment"] and not profile["convergence_established"]
    table = pd.read_csv(registry.parent / "reservoir_parameterization_by_GRAND_ID.csv")
    assert table.GRAND_ID.iloc[0] == 123 and table.storage_uncertainty_kind.iloc[0] == "calibration_offset_quantiles"
    assert calibration.finish(resumed, 16) == registry
    comparison = pd.read_csv(registry.parent / "baseline_comparison.csv")
    assert comparison.assessment_reference.eq("supplied_storage").all()
    assert len(comparison) == 6 and set(comparison.query("role=='seed'").seed) == {7, 8}
    identity = resumed / "jobs/sample/seed_7/identity.json"
    changed = native.read(identity); changed["population"] = 8; calibration.write(identity, changed)
    with pytest.raises(ValueError, match="Checkpoint identity changed"):
        calibration.calibrate(resumed, "sample", 7, 20)


def test_manager_shared_option_to_calibration_export_replay_and_resume(workspace):
    path = inputs(workspace / "inputs")
    calibration_spec = native.read(cal_config(workspace, path.parent))
    calibration_spec["campaign"] = "@targets.campaign"
    calibration_spec["optimization"]["evaluation_budget"] = 8
    cfg = {"kind": "dztr_run_manager_v3", "schema_version": 3, "stages": [
        {"id": "targets", "type": "parameterize", "config": str(path)},
        {"id": "baseline", "type": "baseline", "config": {"campaign": "@targets.campaign"}},
        {"id": "calibration", "type": "calibrate", "backend": "prepared_geometry_v1", "config": calibration_spec},
        {"id": "replay", "type": "simulate", "backend": "v2", "config": {"registry": "@calibration.registry", "forcing_csv": "@calibration.forcing_csv", "grand_id": 123}}]}
    config = workspace / "manager.json"; calibration.write(config, cfg)
    out = workspace / "managed"
    state = manager.run(config, out)
    assert state["status"] == "completed"
    assert all(s["status"] == "reused" for s in manager.run(config, out, resume=True)["stages"])
    receipt = native.read(out / "stages/replay/campaign/receipt.json")
    assert receipt["maximum_balance_residual_m3"] <= 1e-4
