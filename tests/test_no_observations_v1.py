"""Response isolation, physical assumptions, cached execution and route dispatch."""
import importlib.util
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from strategies.no_observations_v1 import parameters, recommend
from workflows import no_observations_v1 as workflow
from workflows import availability_comparison_v1 as comparison
from workflows import run_manager_v6 as manager
from workflows import reservoir_v2 as native

spec = importlib.util.spec_from_file_location("no_observation_inputs", native.ROOT / "tests/test_storage_only_v1.py")
helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)


@pytest.fixture
def workspace():
    folder = Path(tempfile.mkdtemp(prefix="test_no_observations_", dir=native.ROOT / "outputs")).resolve()
    yield folder
    assert folder.is_relative_to((native.ROOT / "outputs").resolve()) and folder.name.startswith("test_no_observations_")
    shutil.rmtree(folder)


def build_inputs(folder, *, gate=False):
    source = helpers.inputs(folder)
    frame = workflow.availability.read_frame(folder / "observations.csv")
    frame["inflow_m3s"] = 80.
    frame.to_csv(folder / "observations.csv", index_label="date", float_format="%.17g")
    cfg = native.read(source)
    if gate: cfg["reservoirs"][0]["source_gates"] = ["Unresolved flow timing"]
    workflow.write(source, cfg)
    prior = helpers.mean_spec(folder)
    donors = pd.read_csv(folder / "donors.csv")
    donors[["p05_ratio", "p30_ratio", "p97_ratio", "p99_ratio"]] = [.9, 1., 1.1, 1.2]
    donors.to_csv(folder / "donors.csv", index=False)
    pd.DataFrame({"GRAND_ID": [123], "DIS_AVG_LS": [80000], "mean_discharge_m3s": [80.]}).to_csv(folder / "grand.csv", index=False)
    return source, prior


def config(folder, campaign, prior):
    path = folder / "no_observations.json"
    workflow.write(path, {"kind": "dztr_no_observations_v1", "schema_version": 1, "campaign": str(campaign),
        "profiles": [{"reservoir_id": "sample", "storage_basis": "total_capacity"}], "release_prior": prior})
    return path


def test_fixed_targets_and_initial_states_ignore_both_responses(workspace):
    build_inputs(workspace)
    frame = workflow.availability.read_frame(workspace / "observations.csv")
    physical = native.read(workspace / "model.json")["model"]
    from dztr.schemas import MONTHLY_TARGET_COLUMNS
    prior = pd.DataFrame(np.tile([72., 80., 88.], (12, 1)), columns=MONTHLY_TARGET_COLUMNS[3:], index=range(1, 13))
    first = parameters(frame, physical, prior, storage_basis="total_capacity")
    changed = frame.copy(); changed.storage_m3 = -1e30; changed.release_m3s = 1e30
    second = parameters(changed, physical, prior, storage_basis="total_capacity")
    np.testing.assert_array_equal(first.monthly_targets, second.monthly_targets)
    assert first.initial_storage_m3 == second.initial_storage_m3 == .3 * physical["capacity_m3"]
    assert first.initial_release_m3s == second.initial_release_m3s == frame.inflow_m3s.iloc[0]
    assert physical["initial_storage_m3"] != first.initial_storage_m3
    physical["dead_storage_fraction"] = .25
    with pytest.raises(ValueError, match="Storage targets"):
        parameters(frame, physical, prior, storage_basis="total_capacity")
    active = parameters(frame, physical, prior, storage_basis="active_capacity")
    np.testing.assert_array_equal(active.monthly_targets.iloc[0, :3], [400e6, 475e6, 977.5e6])
    assert active.capacity_m3 == physical["capacity_m3"] and active.dead_storage_fraction == .25


@pytest.mark.parametrize("storage,release,level,prior,expected,status", [
    (True, True, False, False, "observed_monthly", "recommended"),
    (True, True, True, False, "level_geometry_monthly", "recommended"),
    (False, True, False, False, "flow_only", "recommended"),
    (True, False, False, True, "storage_only", "recommended"),
    (False, False, False, True, "ungauged_mean_flow", "recommended"),
    (False, False, False, False, "ungauged_mean_flow", "needs_release_prior"),
])
def test_recommendation_uses_availability_and_dependencies(storage, release, level, prior, expected, status):
    capabilities = {"storage_monthly_available": storage, "release_monthly_available": release,
                    "storage_source_kind": "level_geometry" if level else "provided_storage"}
    result = recommend(capabilities, inflow_usable=True, release_prior_usable=prior)
    assert result["method"] == expected and result["status"] == status
    capabilities["source_gates"] = ["Unresolved datum"]
    assert recommend(capabilities, inflow_usable=True, release_prior_usable=True)["status"] == "source_gated"
    assert recommend({}, inflow_usable=False, release_prior_usable=True)["status"] == "needs_inflow"


def test_preparation_and_job_reuse_are_response_independent(workspace, monkeypatch):
    targets, prior = build_inputs(workspace)
    upstream = workflow.availability.prepare(targets, workspace / "targets")
    cfg = config(workspace, upstream, prior)
    out = workflow.prepare(cfg, workspace / "experiment")
    forcing = workflow.availability.read_frame(out / "prepared/sample/forcing_si.csv")
    assert list(forcing.columns) == ["inflow_m3s"]
    receipt = workflow.fixed_job(out, "sample")
    assert receipt["status"] == "completed", receipt
    assert receipt["maximum_independent_balance_residual_m3"] < 1e-4
    def forbidden(*args, **kwargs): raise AssertionError("Completed fixed profiles must be reused")
    monkeypatch.setattr(workflow.application, "run_frame", forbidden)
    assert workflow.fixed_job(out, "sample") == receipt
    release = workflow.finish(out)
    profile, parameter = native.load_profile(release / "registry.json", identifier="sample")
    assert parameter.initial_storage_m3 == 300e6 and profile["evaluation_budget"] == 0
    assert not profile["parameterization"]["release_observations_used_for_fitting"]
    assert not profile["parameterization"]["storage_observations_used_for_fitting"]
    assert workflow.run(cfg, out) == release
    # Change every response and rebuild; no fitted target or initialization can change.
    other = workspace / "other"; other.mkdir()
    other_cfg, other_prior = build_inputs(other)
    values = workflow.availability.read_frame(other / "observations.csv")
    values[["storage_m3", "release_m3s"]] *= 1.3
    values.to_csv(other / "observations.csv", index_label="date", float_format="%.17g")
    new_upstream = workflow.availability.prepare(other_cfg, other / "targets")
    new_out = workflow.prepare(config(other, new_upstream, other_prior), other / "experiment")
    for name in ("monthly_targets_si.csv", "model_config.json", "forcing_si.csv"):
        assert native.sha(out / "prepared/sample" / name) == native.sha(new_out / "prepared/sample" / name)


def test_source_gate_and_missing_mean_remain_explicit(workspace):
    target, prior = build_inputs(workspace, gate=True)
    upstream = workflow.availability.prepare(target, workspace / "targets")
    out = workflow.run(config(workspace, upstream, prior), workspace / "gated")
    table = pd.read_csv(out / "reservoir_parameterization_all_by_GRAND_ID.csv")
    assert table.status.iloc[0] == "source_gated" and not table.parameters_available.iloc[0]
    cfg = native.read(target); cfg["reservoirs"][0]["source_gates"] = []; cfg["reservoirs"][0]["GRAND_ID"] = None
    workflow.write(target, cfg)
    upstream = workflow.availability.prepare(target, workspace / "new_targets")
    out = workflow.run(config(workspace, upstream, prior), workspace / "missing_mean")
    table = pd.read_csv(out / "reservoir_parameterization_all_by_GRAND_ID.csv")
    assert table.status.iloc[0] == "prior_or_parameter_gated" and not table.parameters_available.iloc[0]
    assert native.read(out / "summary.json")["optimizer_evaluations"] == 0


def test_failed_fixed_job_is_persistent_and_not_retried(workspace, monkeypatch):
    target, prior = build_inputs(workspace)
    upstream = workflow.availability.prepare(target, workspace / "targets")
    out = workflow.prepare(config(workspace, upstream, prior), workspace / "experiment")
    def failed(*args, **kwargs): raise ValueError("Synthetic no-root failure")
    monkeypatch.setattr(workflow.application, "run_frame", failed)
    first = workflow.fixed_job(out, "sample")
    assert first["status"] == "simulation_failed" and first["reason"] == "Synthetic no-root failure"
    def forbidden(*args, **kwargs): raise AssertionError("Known failure must remain saved")
    monkeypatch.setattr(workflow.application, "run_frame", forbidden)
    assert workflow.fixed_job(out, "sample") == first
    release = workflow.finish(out)
    table = pd.read_csv(release / "reservoir_parameterization_all_by_GRAND_ID.csv")
    assert table.parameters_available.iloc[0] and table.simulation_status.iloc[0] == "simulation_failed"


def test_manager_comparison_override_simulation_and_resume(workspace):
    target, prior = build_inputs(workspace)
    cfg = native.read(config(workspace, workspace, prior)); cfg["campaign"] = "@targets.campaign"
    comparative = {"kind": "dztr_availability_comparison_v1", "schema_version": 1,
        "no_observations_campaign": "@fixed.campaign", "observed_catalog": None, "diagnostic_cache": None,
        "flow_campaign": None, "storage_campaign": None, "flow_objective": "sqrt_release_nse", "budget": 64}
    managed = workspace / "manager.json"
    workflow.write(managed, {"kind": "dztr_run_manager_v6", "schema_version": 6, "stages": [
        {"id": "targets", "type": "parameterize", "config": str(target)},
        {"id": "fixed", "type": "no_observations", "config": cfg},
        {"id": "comparison", "type": "compare_routes", "config": comparative},
        {"id": "replay", "type": "simulate", "config": {"registry": "@comparison.registry",
            "forcing_csv": str(workspace / "observations.csv"), "reservoir_id": "sample", "method": "ungauged_mean_flow"}}]})
    design = manager.plan(managed)
    assert design["stages"][2]["depends_on"] == ["fixed"]
    result = manager.run(managed, workspace / "managed")
    assert result["status"] == "completed"
    report = Path(result["stages"][2]["outputs"]["campaign"])
    inventory = pd.read_csv(report / "reservoir_route_inventory_by_GRAND_ID.csv")
    assert len(inventory) == 1 and inventory.recommended_method.iloc[0] == "observed_monthly"
    assert inventory.status.iloc[0] == "recommended_profile_not_prepared"
    assert len(pd.read_csv(report / "reservoir_parameterization_all_by_GRAND_ID_and_method.csv")) == 4
    with pytest.raises(ValueError, match="unavailable"):
        comparison.resolve(report / "route_registry.json", identifier="sample")
    assert comparison.resolve(report / "route_registry.json", grand_id=123, method="ungauged_mean_flow")["reservoir_id"] == "sample"
    with pytest.raises(ValueError, match="positive GRAND"):
        comparison.resolve(report / "route_registry.json", grand_id=True)
    again = manager.run(managed, workspace / "managed", resume=True)
    assert all(r["status"] == "reused" for r in again["stages"])
    (report / "README.md").write_text("tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        comparison.resolve(report / "route_registry.json", identifier="sample", method="ungauged_mean_flow")


def test_prepared_forcing_tampering_is_rejected(workspace):
    target, prior = build_inputs(workspace)
    upstream = workflow.availability.prepare(target, workspace / "targets")
    out = workflow.prepare(config(workspace, upstream, prior), workspace / "experiment")
    with (out / "prepared/sample/forcing_si.csv").open("a") as stream: stream.write("\n")
    with pytest.raises(ValueError, match="Prepared artifact changed"): workflow.verify(out)


def test_annotated_trajectory_reader_keeps_only_required_numerical_fields(workspace):
    dates = pd.date_range("2000-01-01", periods=3, name="date")
    frame = pd.DataFrame({"inflow_m3s": 80., "storage_m3": 300e6, "release_m3s": 80.,
        "step_seconds": 86400., "balance_residual_m3": 0., "phase": ["warmup", "calibration", "assessment"],
        "regime": ["unclassified", "wet", "dry"]}, index=dates)
    path = workspace / "annotated.csv"; frame.to_csv(path, index_label="date", float_format="%.17g")
    loaded = comparison.read_simulation(path)
    assert "phase" not in loaded and "regime" not in loaded
    assert native.independent_balance(loaded, 1.) == 0
    frame.loc[dates[1], "storage_m3"] = np.nan
    frame.to_csv(path, index_label="date")
    with pytest.raises(ValueError, match="Nonfinite"): comparison.read_simulation(path)


def test_actual_inflow_only_input_recommends_and_applies_no_observation_route(workspace):
    target, prior = build_inputs(workspace)
    values = workflow.availability.read_frame(workspace / "observations.csv")[["inflow_m3s"]]
    values.to_csv(workspace / "observations.csv", index_label="date", float_format="%.17g")
    upstream = workflow.availability.prepare(target, workspace / "targets")
    release = workflow.run(config(workspace, upstream, prior), workspace / "fixed")
    assert native.read(release / "summary.json")["completed_simulations"] == 1
    report_cfg = workspace / "comparison.json"
    workflow.write(report_cfg, {"kind": "dztr_availability_comparison_v1", "schema_version": 1,
        "no_observations_campaign": str(release.parent), "observed_catalog": None, "diagnostic_cache": None,
        "flow_campaign": None, "storage_campaign": None, "flow_objective": "sqrt_release_nse", "budget": 64})
    report = comparison.report(report_cfg, workspace / "report")
    route = comparison.resolve(report / "route_registry.json", identifier="sample")
    assert route["profile_method"] == "ungauged_mean_flow"
    workflow.simulate(report / "route_registry.json", workspace / "observations.csv", workspace / "application", identifier="sample")
    assert native.read(workspace / "application/receipt.json")["responses_used"] is False
