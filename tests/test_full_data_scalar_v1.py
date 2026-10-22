"""Weighted full-data scientific contracts and exact scalar continuation."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from dztr.calibration.candidates import parameters_to_vector
from strategies.full_data_scalar_v1 import FullDataObjective, weights
from strategies.flow_only_v1 import nse
from strategies.representative_years import YearSelection, fit_selected_years
from workflows import full_data_comparison_v1 as workflow, reservoir_v2 as native

spec = importlib.util.spec_from_file_location("full_data_inputs", native.ROOT / "tests/test_prepared_calibration_v1.py")
helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)


@pytest.fixture
def workspace():
    root = native.ROOT / "outputs"
    path = Path(tempfile.mkdtemp(prefix="test_full_data_", dir=root)).resolve()
    yield path
    assert path.is_relative_to(root.resolve()) and path.name.startswith("test_full_data_")
    shutil.rmtree(path)


def setup_objective(folder, w=None, scoring="pooled"):
    helpers.make_inputs(folder)
    frame = workflow.prepared.availability.read_frame(folder / "observations.csv")
    selection = YearSelection(**native.read(folder / "selection.json"))
    model = native.read(folder / "model.json")
    template, bounds = fit_selected_years(native.observations(folder / "observations.csv"), selection, model["model"])
    metadata = {"physical_constants": model["model"], "sampling": model["sampling"], "root_policy": model["root_policy"],
        "atol_storage_m3": model["atol_storage_m3"], "initial_date": frame.index[0].date().isoformat()}
    return FullDataObjective(frame, selection, template, bounds, metadata, {"factor": 1., "offset_m3s": 0.},
        objective_weights=w or {"storage": 1, "release": 1}, scoring=scoring)


@pytest.mark.parametrize("value", [{"storage": 0, "release": 0}, {"storage": -1, "release": 1},
    {"storage": True, "release": 1}, {"storage": float("nan"), "release": 1}, {"release": 1}])
def test_invalid_weights_rejected(value):
    with pytest.raises(ValueError): weights(value)


def test_weights_normalize_without_overflow():
    assert weights({"storage": 2, "release": 1}) == {"storage": 2/3, "release": 1/3}
    assert weights({"storage": 1e308, "release": 1e308}) == {"storage": .5, "release": .5}
    assert weights({"storage": 0, "release": 4}) == {"storage": 0, "release": 1}


def test_missing_route_is_identified_only_for_unsigned_finite_forcing():
    frame = pd.DataFrame({"inflow_m3s": [1., 2.]})
    metadata = {"sampling": {"frequency": "daily", "allow_signed_inflow": False}}
    resolved, note = workflow.numerical_route(metadata, frame)
    assert resolved["route"] == "standard_native_v1" and "omitted" in note
    assert "route" not in metadata
    for values in ([-1., 2.], [np.nan, 2.], [np.inf, 2.]):
        with pytest.raises(ValueError): workflow.numerical_route(metadata, pd.DataFrame({"inflow_m3s": values}))
    for extra in ({"route": "signed_net_native_v1"}, {"forcing_policy": {}}, {"sampling": {"allow_signed_inflow": True}}):
        with pytest.raises(ValueError): workflow.numerical_route({**metadata, **extra}, frame)


@pytest.mark.parametrize("scoring", ["pooled", "equal_regime"])
def test_weighted_cost_and_zero_weight_do_not_mix_physical_units(workspace, scoring):
    obj = setup_objective(workspace, {"storage": 2, "release": 1}, scoring)
    sim = obj.frame.copy()
    sim.storage_m3 = sim.storage_m3 * .95
    sim.release_m3s = sim.release_m3s * 1.05
    expected = sum(obj.weights[v] * (1 - np.mean([nse(obj.frame[obj.column(v)].to_numpy()[g], sim[obj.column(v)].to_numpy()[g]) for g in obj.groups])) for v in obj.weights)
    assert obj.cost(sim) == pytest.approx(expected)
    obj.weights = weights({"storage": 0, "release": 1})
    original = obj.cost(sim)
    obj.frame.storage_m3 = np.nan; sim.storage_m3 = -1e30
    assert obj.cost(sim) == original
    with pytest.raises(ValueError, match="continuous"): obj.cost(sim.iloc[1:])


def test_full_data_release_only_retains_72_targets_and_calibration_only_bounds(workspace):
    obj = setup_objective(workspace, {"storage": 0, "release": 1})
    assert len(obj.lower) == len(obj.upper) == 72
    assert np.all(obj.upper > obj.lower)
    assert obj.frame.loc[~obj.mask, ["storage_m3", "release_m3s"]].isna().all().all()
    source = workflow.prepared.availability.read_frame(workspace / "observations.csv")
    source.loc[obj.selection.mask(source.index, "assessment"), ["storage_m3", "release_m3s"]] *= 1e4
    source.to_csv(workspace / "changed.csv", index_label="date", float_format="%.17g")
    t, b = fit_selected_years(native.observations(workspace / "changed.csv"), obj.selection, obj.metadata["physical_constants"])
    np.testing.assert_array_equal(obj.lower, b.lower_vector)
    np.testing.assert_array_equal(obj.upper, b.upper_vector)
    np.testing.assert_array_equal(obj.template.monthly_targets, t.monthly_targets)
    vector = parameters_to_vector(t); vector[0] = (obj.lower[0] + obj.upper[0])/2
    changed = obj.parameters(vector)
    assert changed.monthly_targets.iloc[0, 0] == pytest.approx(vector[0]*t.capacity_m3)


def comparison_config(folder):
    upstream = workflow.prepared.availability.prepare(helpers.make_inputs(folder), folder / "targets")
    cfg = {"kind": "dztr_full_data_comparison_v1", "schema_version": 1, "campaign": str(upstream),
        "profiles": [{"reservoir_id": "sample"}], "scoring": "pooled",
        "objectives": [{"id": "release_only", "weights": {"storage": 0, "release": 1}},
                       {"id": "joint", "weights": {"storage": 1, "release": 1}}],
        "optimization": {"seeds": [7], "horizon": 32, "evaluation_budget": 16, "workers": 1,
            "algorithms": [{"optimizer": "dds"}, {"optimizer": "ga", "ga": {"population_size": 4}}]}}
    path = folder / "comparison.json"; workflow.write(path, cfg)
    return path


def test_both_objectives_algorithms_continue_exactly_and_export_calibration_choices(workspace):
    cfg = comparison_config(workspace)
    out = workflow.prepare(cfg, workspace / "study")
    other = workflow.prepare(cfg, workspace / "fresh")
    for objective in ("release_only", "joint"):
        for algorithm in ("dds", "ga"):
            a = workflow.calibrate(out, "sample", objective, algorithm, 7, 7)
            b = workflow.calibrate(out, "sample", objective, algorithm, 7, 16)
            fresh = workflow.calibrate(other, "sample", objective, algorithm, 7, 16)
            result = native.read(b / "result.json")
            assert result == native.read(fresh / "result.json")
            assert result["trials"][:7] == native.read(a / "result.json")["trials"]
            assert result["trials"][0]["vector"] == parameters_to_vector(workflow.objective_for(out, "sample", objective).template).tolist()
    release = workflow.finish(out, 16)
    assert (release / "figures/sample_assessment.png").exists()
    assert native.read(release / "summary.json")["evaluations"] == 64
    selection = native.read(release / "selection.json")
    assert selection["selection_uses_assessment"] is False
    for objective in ("release_only", "joint"):
        for algorithm in ("dds", "ga"):
            profile, parameters = native.load_profile(release / objective / algorithm / "registry.json", identifier="sample")
            assert profile["adjustable_targets"] == 72 and profile["storage_observations_used_for_bounds"]
            assert profile["grand_id"] == 123
            assert "assessment" not in profile  # do not inherit stale upstream metrics
    workflow.checked(release)
    assert workflow.finish(out, 16) == release
    changed = native.read(out / "config.json"); changed["objectives"][0]["weights"]["storage"] = .2
    workflow.write(out / "config.json", changed)
    with pytest.raises(ValueError, match="preparation changed"): workflow.verify(out)


def test_config_rejects_duplicate_algorithms_and_wrong_settings(workspace):
    cfg = native.read(comparison_config(workspace))
    for entries in ([{"optimizer": "dds"}, {"optimizer": "dds"}], [{"optimizer": "ga", "perturbation": .2}], [{"ga": {}}]):
        changed = deepcopy(cfg); changed["optimization"]["algorithms"] = entries
        with pytest.raises(ValueError): workflow.normalize(changed)


def test_manager_full_data_handoff_simulation_and_reuse(workspace):
    from workflows import run_manager_v8 as manager
    config = comparison_config(workspace)
    cfg = native.read(config); cfg["objectives"] = cfg["objectives"][:1]
    cfg["optimization"].update(evaluation_budget=4, algorithms=[{"optimizer": "ga", "ga": {"population_size": 4}}])
    workflow.write(config, cfg)
    managed = workspace / "manager.json"
    workflow.write(managed, {"kind": "dztr_run_manager_v8", "schema_version": 8, "stages": [
        {"id": "fit", "type": "calibrate", "backend": "prepared_full_data_v1", "config": str(config)},
        {"id": "apply", "type": "simulate", "config": {"registry": "@fit.registry", "forcing_csv": "@fit.forcing_csv", "reservoir_id": "sample"}}
    ]})
    design = manager.plan(managed)
    assert design["stages"][0]["config"]["objectives"][0]["weights"] == {"storage": 0, "release": 1}
    result = manager.run(managed, workspace / "managed")
    assert result["status"] == "completed"
    assert Path(result["stages"][0]["outputs"]["parameter_table"]).exists()
    assert Path(result["stages"][1]["outputs"]["simulation_csv"]).exists()
    assert all(s["status"] == "reused" for s in manager.run(managed, workspace / "managed", resume=True)["stages"])
