"""Release-only scientific contracts, continuation and managed integration."""
import importlib.util
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from strategies.dds_v1 import reflect, search
from strategies.flow_only_v1 import Evaluation, FlowObjective, fit, nse
from strategies.representative_years import YearSelection
from workflows import data_availability_v2 as availability
from workflows import flow_calibration_v1 as flow
from workflows import reservoir_v2 as native
from workflows import run_manager_v4 as manager

spec = importlib.util.spec_from_file_location("flow_test_inputs", native.ROOT / "tests/test_prepared_calibration_v1.py")
helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)


@pytest.fixture
def workspace():
    parent = native.ROOT / "outputs"
    folder = Path(tempfile.mkdtemp(prefix="test_flow_calibration_", dir=parent)).resolve()
    yield folder
    assert folder.is_relative_to(parent.resolve()) and folder.name.startswith("test_flow_calibration_")
    shutil.rmtree(folder)


def inputs(folder):
    path = helpers.make_inputs(folder, methods=["flow_only"])
    model = native.read(folder / "model.json")
    # A large synthetic reservoir avoids a known monthly zone-boundary root
    # gap, so integration tests exercise feasible continuous trajectories.
    model["model"].update(capacity_m3=1e11, initial_storage_m3=5e10)
    flow.write(folder / "model.json", model)
    cfg = native.read(path); cfg.update(kind="dztr_data_availability_v2", schema_version=2)
    flow.write(path, cfg)
    return path


def config(folder, campaign):
    path = folder / "flow.json"
    flow.write(path, {"kind": "dztr_flow_calibration_v1", "schema_version": 1, "campaign": str(campaign),
        "profiles": [{"reservoir_id": "sample"}], "objectives": ["sqrt_release_nse", "release_nse"], "scoring": "pooled",
        "bounds_policy": "monthly", "optimization": {"seeds": [7, 8], "horizon": 32, "evaluation_budget": 16, "perturbation": .2, "workers": 1}})
    return path


def test_scores_preserve_zero_and_fixed_observation_support():
    assert nse([0, 1, 4], [0, 4, 4], square_root=True) == pytest.approx(.5)
    assert nse([0, 1, 4], [0, 4, 4]) == pytest.approx(1 - 9 / (26 / 3))
    with pytest.raises(ValueError, match="fixed observation support"): nse([1, 2, 3], [1, np.nan, 3])
    with pytest.raises(ValueError, match="constant"): nse([1, 1], [1, 1])


def test_dds_reflection_and_exact_prefix_continuation():
    assert reflect(-.2, 0, 1) == pytest.approx(.2)
    assert reflect(1.2, 0, 1) == pytest.approx(.8)
    assert reflect(-2, 0, 1) == 0
    assert reflect(3, 0, 1) == 1
    def objective(v): return Evaluation(True, float(np.sum((v - .25) ** 2)))
    args = dict(seed=42, horizon=64, perturbation=.2)
    a = search(objective, [.8, .6], [0, 0], [1, 1], budget=8, **args)
    b = search(objective, [.8, .6], [0, 0], [1, 1], budget=24, state=a, **args)
    fresh = search(objective, [.8, .6], [0, 0], [1, 1], budget=24, **args)
    assert b == fresh and b["trials"][:8] == a["trials"]
    assert min(t["cost"] for t in b["trials"]) < b["trials"][0]["cost"]
    with pytest.raises(ValueError, match="settings changed"):
        search(objective, [.8, .6], [0, 0], [1, 1], budget=24, state=a, **{**args, "horizon": 128})
    with pytest.raises(ValueError, match="shrink"):
        search(objective, [.8, .6], [0, 0], [1, 1], budget=4, state=a, **args)


@pytest.mark.parametrize("objective", ["release_nse", "sqrt_release_nse"])
@pytest.mark.parametrize("scoring", ["pooled", "equal_regime"])
def test_hidden_storage_and_held_release_never_change_fitting(workspace, objective, scoring):
    inputs(workspace)
    frame = availability.read_frame(workspace / "observations.csv")
    plan = YearSelection(**native.read(workspace / "selection.json")); model = native.read(workspace / "model.json")
    def make(f):
        t, low, high = fit(f, plan, model["model"], quantile_method="hazen", storage_basis="total_capacity", bounds_policy="monthly")
        metadata = {"physical_constants": model["model"], "sampling": model["sampling"], "root_policy": model["root_policy"],
                    "atol_storage_m3": model["atol_storage_m3"], "initial_date": f.index[0].date().isoformat()}
        return FlowObjective(f, plan, t, low, high, metadata, {"factor": 1., "offset_m3s": 0.}, objective=objective, scoring=scoring)
    a = make(frame)
    changed = frame.copy(); changed.storage_m3 = -1e25
    changed.loc[plan.mask(frame.index, "assessment"), "release_m3s"] *= 100
    b = make(changed)
    np.testing.assert_array_equal(a.template.monthly_targets, b.template.monthly_targets)
    np.testing.assert_array_equal(a.lower, b.lower); np.testing.assert_array_equal(a.upper, b.upper)
    assert a.template.initial_storage_m3 == b.template.initial_storage_m3 == 30e9
    assert "storage_m3" not in a.frame
    vector = a.template.monthly_targets.iloc[:, 3:].to_numpy().ravel()
    x, y = a.evaluate(vector), b.evaluate(vector)
    assert x.feasible and y.feasible, (x.reason, y.reason)
    assert x.cost == y.cost
    np.testing.assert_array_equal(x.parameters.monthly_targets.iloc[:, :3], np.tile([20e9, 30e9, 97e9], (12, 1)))
    assert x.balance <= model["atol_storage_m3"]


def test_active_storage_adaptation_and_etienne_envelope_are_explicit(workspace):
    inputs(workspace); frame = availability.read_frame(workspace / "observations.csv")
    plan = YearSelection(**native.read(workspace / "selection.json")); physical = native.read(workspace / "model.json")["model"]
    physical["dead_storage_fraction"] = .25
    with pytest.raises(ValueError, match="dead storage"):
        fit(frame, plan, physical, quantile_method="hazen", storage_basis="total_capacity", bounds_policy="monthly")
    t, lo, hi = fit(frame, plan, physical, quantile_method="linear", storage_basis="active_capacity", bounds_policy="etienne_annual_envelope")
    np.testing.assert_allclose(t.monthly_targets.iloc[0, :3], [40e9, 47.5e9, 97.75e9])
    assert np.unique(lo.reshape(12, 3), axis=0).shape[0] == 1
    assert np.unique(hi.reshape(12, 3), axis=0).shape[0] == 1


def test_campaign_continuation_export_and_tamper_rejection(workspace):
    upstream = availability.prepare(inputs(workspace), workspace / "targets")
    cfg = config(workspace, upstream); out = flow.prepare(cfg, workspace / "campaign")
    training = availability.read_frame(out / "prepared/sample/observations_si.csv")
    assert training.storage_m3.isna().all()
    for o in ("release_nse", "sqrt_release_nse"):
        for seed in (7, 8):
            first = flow.calibrate(out, "sample", o, seed, 8)
            continued = flow.calibrate(out, "sample", o, seed, 16)
            assert native.read(continued / "result.json")["trials"][:8] == native.read(first / "result.json")["trials"]
    release = flow.finish(out, 16)
    summary = native.read(release / "summary.json")
    assert summary["selected_profiles"] == 2 and summary["evaluations"] == 64
    rows = pd.read_csv(release / "baseline_comparison.csv")
    assert len(rows) == 8 and rows.selected.sum() == 4
    assert rows.calibrated_storage_nse.notna().all()
    for o in ("release_nse", "sqrt_release_nse"):
        p, _ = native.load_profile(release / o / "registry.json", identifier="sample")
        assert p["parameterization"]["storage_observations_used_for_fitting"] is False
        assert p["physical_constants"]["initial_storage_m3"] == 30e9
    checkpoint = out / "jobs/sample/release_nse/seed_7/checkpoint.json"
    receipt = native.read(checkpoint); receipt["state"]["trials"][0]["cost"] = -12345
    flow.write(checkpoint, receipt)
    with pytest.raises(ValueError, match="identity or content changed"): flow.calibrate(out, "sample", "release_nse", 7, 24)


def test_managed_release_only_handoff_and_resume(workspace):
    cfg = config(workspace, workspace)
    target = inputs(workspace)
    spec = native.read(cfg); spec["campaign"] = "@targets.campaign"
    spec["objectives"] = ["sqrt_release_nse"]; spec["optimization"].update(seeds=[7], evaluation_budget=8)
    managed = workspace / "managed.json"
    flow.write(managed, {"kind": "dztr_run_manager_v4", "schema_version": 4, "stages": [
        {"id": "targets", "type": "parameterize", "config": str(target)},
        {"id": "calibration", "type": "calibrate", "backend": "prepared_flow_v1", "config": spec}]})
    design = manager.plan(managed); assert design["stages"][1]["depends_on"] == ["targets"]
    result = manager.run(managed, workspace / "managed")
    assert result["status"] == "completed"
    assert Path(result["stages"][1]["outputs"]["registry"]).is_file()
    native_profile = Path(result["stages"][1]["outputs"]["registry"])
    flow.simulate(native_profile, workspace / "observations.csv", workspace / "replay", identifier="sample")
    assert native.read(workspace / "replay/receipt.json")["observed_storage_used"] is False
    again = manager.run(managed, workspace / "managed", resume=True)
    assert all(r["status"] == "reused" for r in again["stages"])
