"""DDS compatibility and optional GA across both physical calibration routes."""
import importlib.util
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from workflows import flow_calibration_v1 as old_flow, storage_calibration_v1 as old_storage
from workflows import flow_calibration_v2 as flow, storage_calibration_v2 as storage
from workflows import run_manager_v7 as manager, reservoir_v2 as native


def load_helpers(name):
    spec = importlib.util.spec_from_file_location("ga_" + name, native.ROOT / "tests" / name)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


FLOW_INPUTS = load_helpers("test_flow_calibration_v1.py")
STORAGE_INPUTS = load_helpers("test_storage_only_v1.py")


@pytest.fixture
def workspace():
    parent = native.ROOT / "outputs"; folder = Path(tempfile.mkdtemp(prefix="test_scalar_ga_", dir=parent)).resolve()
    yield folder
    assert folder.is_relative_to(parent.resolve()) and folder.name.startswith("test_scalar_ga_")
    shutil.rmtree(folder)


def inputs(folder, role, algorithm):
    helpers = FLOW_INPUTS if role == "flow" else STORAGE_INPUTS
    target = helpers.inputs(folder)
    upstream = flow.availability.prepare(target, folder / "targets")
    old_path = helpers.config(folder, upstream)
    old_cfg = native.read(old_path); old_cfg["objectives"] = [old_cfg["objectives"][0]]
    old_cfg["optimization"].update(seeds=[7], horizon=32, evaluation_budget=16, workers=1)
    flow.write(old_path, old_cfg)
    cfg = native.read(old_path); cfg.update(kind=f"dztr_{role}_calibration_v2", schema_version=2)
    if algorithm == "ga":
        cfg["optimization"].pop("perturbation")
        cfg["optimization"].update(optimizer="ga", ga={"population_size": 4})
    path = folder / (role + "_v2.json"); flow.write(path, cfg)
    return target, old_path, path


@pytest.mark.parametrize("role", ["flow", "storage"])
@pytest.mark.parametrize("algorithm", ["dds", "ga"])
def test_both_routes_continuation_and_calibration_only_export(workspace, role, algorithm):
    _, old_cfg, cfg = inputs(workspace, role, algorithm)
    module = flow if role == "flow" else storage
    objective = native.read(cfg)["objectives"][0]
    out = module.prepare(cfg, workspace / "experiment")
    obj = module.objective_for(out, "sample", objective)
    assert ("storage_m3" not in obj.frame) if role == "flow" else ("release_m3s" not in obj.frame)
    first = module.calibrate(out, "sample", objective, 7, 7)
    second = module.calibrate(out, "sample", objective, 7, 16)
    initial = native.read(first / "result.json"); result = native.read(second / "result.json")
    assert result["trials"][:7] == initial["trials"]
    fresh = module.prepare(cfg, workspace / "fresh")
    direct = module.calibrate(fresh, "sample", objective, 7, 16)
    assert result == native.read(direct / "result.json")
    if algorithm == "dds":
        previous = old_flow if role == "flow" else old_storage
        old = previous.prepare(old_cfg, workspace / "previous_version")
        reference = previous.calibrate(old, "sample", objective, 7, 16)
        assert result == native.read(reference / "result.json")
    release = module.finish(out, 16)
    profile, parameters = native.load_profile(release / objective / "registry.json", identifier="sample")
    assert profile["optimizer"] == algorithm.upper() and not profile["selection_uses_assessment"]
    assert profile["actual_evaluations"] == 16
    assert profile["dds_horizon"] == (32 if algorithm == "dds" else None)
    table = pd.read_csv(release / objective / "reservoir_parameterization_by_GRAND_ID.csv")
    assert table.optimizer.iloc[0] == algorithm.upper() and table.parameters_available.iloc[0]
    assert native.read(release / "summary.json")["evaluations"] == 16
    selection = native.read(release / "selection.json")["records"][0]
    assert selection["cost"] == min(t["cost"] for t in result["trials"] if t["cost"] is not None)
    # Both optimizers retain the same physical settings and the nonfitted targets.
    actual = parameters.monthly_targets
    fixed_columns = slice(0, 3) if role == "flow" else slice(3, 6)
    np.testing.assert_array_equal(actual.iloc[:, fixed_columns], obj.template.monthly_targets.iloc[:, fixed_columns])
    assert parameters.capacity_m3 == obj.template.capacity_m3
    assert parameters.dead_storage_fraction == obj.template.dead_storage_fraction
    assert parameters.channel_capacity_m3s == obj.template.channel_capacity_m3s


def test_manager_ga_and_default_dds_handoff_simulation_and_reuse(workspace):
    flow_folder = workspace / "flow"; flow_folder.mkdir()
    storage_folder = workspace / "storage"; storage_folder.mkdir()
    flow_target, _, flow_cfg = inputs(flow_folder, "flow", "ga")
    storage_target, _, storage_cfg = inputs(storage_folder, "storage", "dds")
    flow_spec = native.read(flow_cfg); flow_spec["campaign"] = "@flow_targets.campaign"
    storage_spec = native.read(storage_cfg); storage_spec["campaign"] = "@storage_targets.campaign"
    cfg = workspace / "manager.json"
    flow.write(cfg, {"kind": "dztr_run_manager_v7", "schema_version": 7, "stages": [
        {"id": "flow_targets", "type": "parameterize", "config": str(flow_target)},
        {"id": "storage_targets", "type": "parameterize", "config": str(storage_target)},
        {"id": "flow_fit", "type": "calibrate", "backend": "prepared_flow_v2", "config": flow_spec},
        {"id": "storage_fit", "type": "calibrate", "backend": "prepared_storage_v2", "config": storage_spec},
        {"id": "flow_application", "type": "simulate", "config": {"registry": "@flow_fit.registry", "forcing_csv": "@flow_fit.forcing_csv", "reservoir_id": "sample"}},
        {"id": "storage_application", "type": "simulate", "config": {"registry": "@storage_fit.registry", "forcing_csv": "@storage_fit.forcing_csv", "reservoir_id": "sample"}}
    ]})
    result = manager.run(cfg, workspace / "managed")
    assert result["status"] == "completed"
    for index, expected in ((2, "GA"), (3, "DDS")):
        registry = result["stages"][index]["outputs"]["registry"]
        profile, _ = native.load_profile(registry, identifier="sample")
        assert profile["optimizer"] == expected
    assert all(s["status"] == "reused" for s in manager.run(cfg, workspace / "managed", resume=True)["stages"])


def test_changed_optimizer_cannot_reuse_prepared_protocol(workspace):
    _, _, cfg = inputs(workspace, "flow", "ga")
    out = flow.prepare(cfg, workspace / "experiment")
    saved = native.read(out / "config.json"); saved["optimization"]["optimizer"] = "dds"
    flow.write(out / "config.json", saved)
    with pytest.raises(ValueError, match="preparation changed"): flow.verify(out)


def test_manager_plan_exposes_defaults_and_rejects_unknown_optimizer(workspace):
    _, _, cfg = inputs(workspace, "flow", "ga")
    managed = workspace / "manager.json"
    spec = native.read(cfg)
    flow.write(managed, {"kind": "dztr_run_manager_v7", "schema_version": 7,
        "stages": [{"id": "calibration", "type": "calibrate", "backend": "prepared_flow_v2", "config": spec}]})
    planned = manager.plan(managed)["stages"][0]["config"]["optimization"]
    assert planned["optimizer"] == "ga" and planned["ga"]["mutation_eta"] == 20.
    spec["optimization"]["optimizer"] = "unsupported"
    flow.write(managed, {"kind": "dztr_run_manager_v7", "schema_version": 7,
        "stages": [{"id": "calibration", "type": "calibrate", "backend": "prepared_flow_v2", "config": spec}]})
    with pytest.raises(ValueError, match="Choose optimizer"): manager.plan(managed)
