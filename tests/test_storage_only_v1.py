"""Naturalized priors, response isolation, physical storage bounds and DDS reuse."""
import importlib.util
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from strategies.storage_only_v1 import fit, StorageObjective
from strategies.representative_years import YearSelection
from workflows import data_availability_v2 as availability
from workflows import release_prior_v1 as priors
from workflows import reservoir_v2 as native
from workflows import storage_calibration_v1 as storage
from workflows import run_manager_v5 as manager

spec = importlib.util.spec_from_file_location("storage_test_inputs", native.ROOT / "tests/test_prepared_calibration_v1.py")
helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)


@pytest.fixture
def workspace():
    parent = native.ROOT / "outputs"; folder = Path(tempfile.mkdtemp(prefix="test_storage_only_", dir=parent)).resolve()
    yield folder
    assert folder.is_relative_to(parent.resolve()) and folder.name.startswith("test_storage_only_")
    shutil.rmtree(folder)


def inputs(folder):
    path = helpers.make_inputs(folder, methods=["storage_only"])
    model = native.read(folder / "model.json"); model["model"]["flood_mode"] = "inflow"
    storage.write(folder / "model.json", model)
    config = native.read(path); config.update(kind="dztr_data_availability_v2", schema_version=2)
    storage.write(path, config)
    dates = pd.date_range("2000-01-01", "2009-12-31", freq="D", name="date")
    values = 80 + 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    pd.DataFrame({"naturalized_outflow_m3s": values}, index=dates).to_csv(folder / "naturalized.csv", index_label="date", float_format="%.17g")
    return path


def prior_spec(folder, scope="calibration_overlap"):
    result = {"source_kind": "simulated_naturalized_outflow", "csv": str(folder / "naturalized.csv"), "format": "si_csv",
        "evidence": "Synthetic unregulated model-output fixture", "independent_of_target_responses": True,
        "scope": scope, "multiplier": 1.}
    if scope == "external_reference": result.update(reference_period={"start": "2000-01-01", "end": "2009-12-31"}, external_reference_rationale="Explicit synthetic retrospective reference test")
    return result


def config(folder, campaign, *, scope="calibration_overlap"):
    path = folder / "storage.json"
    storage.write(path, {"kind": "dztr_storage_calibration_v1", "schema_version": 1, "campaign": str(campaign),
        "profiles": [{"reservoir_id": "sample", "release_prior": prior_spec(folder, scope)}], "objectives": ["storage_nse"],
        "scoring": "pooled", "bounds_policy": "monthly",
        "optimization": {"seeds": [7, 8], "horizon": 32, "evaluation_budget": 16, "perturbation": .2, "workers": 1}})
    return path


def test_etienne_parser_ignores_observed_column_and_preserves_zero(workspace):
    path = workspace / "etienne.txt"
    path.write_text("date , sim flow, obs. flow\n20000101 0 secret\n20000102 2 other\n")
    values = priors.read_source(path, "etienne_crude")
    np.testing.assert_array_equal(values, [0, 2])
    path.write_text("date , sim flow, obs. flow\n20000101 -1 secret\n20000102 2 other\n")
    with pytest.raises(ValueError, match="nonnegative"): priors.read_source(path, "etienne_crude")


def test_prior_fits_calibration_support_only_and_external_scope_is_explicit(workspace):
    inputs(workspace); frame = availability.read_frame(workspace / "observations.csv")
    plan = YearSelection(**native.read(workspace / "selection.json"))
    def prepare(name, spec): return priors.prepare(spec, frame.index, plan, quantile_method="hazen", folder=workspace / name, pin=lambda p: p, base=workspace)
    first, receipt = prepare("first", prior_spec(workspace))
    natural = availability.read_frame(workspace / "naturalized.csv")
    years = set(frame.index[plan.mask(frame.index, "calibration")].year)
    natural.loc[~natural.index.year.isin(years), "naturalized_outflow_m3s"] += 1000
    natural.to_csv(workspace / "naturalized.csv", index_label="date", float_format="%.17g")
    second, changed = prepare("changed", prior_spec(workspace))
    np.testing.assert_array_equal(first, second); assert receipt["fitting_signature"] == changed["fitting_signature"]
    external, ext = prepare("external", prior_spec(workspace, "external_reference"))
    assert not np.array_equal(first, external) and ext["retrospective_external_climatology"]
    bad = prior_spec(workspace); bad["scope"] = "external_reference"
    with pytest.raises(ValueError, match="explicit period"): prepare("bad", bad)
    wrong = prior_spec(workspace); wrong["source_kind"] = "observed_release"
    with pytest.raises(ValueError, match="simulated naturalized"): prepare("observed", wrong)


@pytest.mark.parametrize("scoring", ["pooled", "equal_regime"])
def test_hidden_release_and_held_storage_never_change_storage_calibration(workspace, scoring):
    inputs(workspace); frame = availability.read_frame(workspace / "observations.csv")
    plan = YearSelection(**native.read(workspace / "selection.json")); model = native.read(workspace / "model.json")
    release, _ = priors.prepare(prior_spec(workspace), frame.index, plan, quantile_method="hazen", folder=workspace / "prior", pin=lambda p: p, base=workspace)
    def make(f):
        template, lo, hi, receipt = fit(f, plan, model["model"], release, quantile_method="hazen", bounds_policy="monthly")
        metadata = {"physical_constants": model["model"], "sampling": model["sampling"], "root_policy": model["root_policy"], "atol_storage_m3": model["atol_storage_m3"], "initial_date": f.index[0].date().isoformat()}
        return StorageObjective(f, plan, template, lo, hi, metadata, {"factor": 1., "offset_m3s": 0.}, objective="storage_nse", scoring=scoring)
    a = make(frame); changed = frame.copy(); changed.release_m3s = -1e25
    changed.loc[plan.mask(frame.index, "assessment"), "storage_m3"] += 1e12
    b = make(changed)
    assert "release_m3s" not in a.frame
    np.testing.assert_array_equal(a.template.monthly_targets, b.template.monthly_targets)
    np.testing.assert_array_equal(a.lower, b.lower); np.testing.assert_array_equal(a.upper, b.upper)
    assert a.template.initial_storage_m3 == b.template.initial_storage_m3 == frame.storage_m3.iloc[0]
    vector = a.template.monthly_targets.iloc[:, :3].to_numpy().ravel()
    x, y = a.evaluate(vector), b.evaluate(vector)
    assert x.feasible and y.feasible, (x.reason, y.reason)
    assert x.cost == y.cost
    np.testing.assert_array_equal(x.parameters.monthly_targets.iloc[:, 3:], release)
    assert x.balance <= model["atol_storage_m3"]


def test_physical_bound_intersection_is_recorded_and_cannot_change_capacity(workspace):
    inputs(workspace); frame = availability.read_frame(workspace / "observations.csv")
    plan = YearSelection(**native.read(workspace / "selection.json")); physical = native.read(workspace / "model.json")["model"]
    from dztr.schemas import MONTHLY_TARGET_COLUMNS
    release = pd.DataFrame(np.tile([50, 80, 100], (12, 1)), columns=MONTHLY_TARGET_COLUMNS[3:], index=range(1, 13))
    physical["dead_storage_fraction"] = .43
    template, low, high, receipt = fit(frame, plan, physical, release, quantile_method="hazen", bounds_policy="etienne_annual_envelope")
    assert receipt["bound_cells_changed"] > 0 and template.capacity_m3 == physical["capacity_m3"]
    assert (template.monthly_targets.iloc[:, 0] > .43 * template.capacity_m3).all()
    physical["dead_storage_fraction"] = .9
    with pytest.raises(ValueError, match="No physically feasible intersection"): fit(frame, plan, physical, release, quantile_method="hazen", bounds_policy="monthly")


def test_real_checkpoint_continuation_export_and_hidden_release(workspace):
    upstream = availability.prepare(inputs(workspace), workspace / "targets")
    cfg = config(workspace, upstream); out = storage.prepare(cfg, workspace / "calibration")
    training = availability.read_frame(out / "prepared/sample/observations_si.csv")
    assert training.release_m3s.isna().all()
    for seed in (7, 8):
        first = storage.calibrate(out, "sample", "storage_nse", seed, 8)
        second = storage.calibrate(out, "sample", "storage_nse", seed, 16)
        assert native.read(second / "result.json")["trials"][:8] == native.read(first / "result.json")["trials"]
    release = storage.finish(out, 16)
    assert native.read(release / "summary.json")["selected_profiles"] == 1
    profile, _ = native.load_profile(release / "storage_nse/registry.json", identifier="sample")
    assert profile["parameterization"]["release_observations_used_for_fitting"] is False
    assert len(pd.read_csv(release / "baseline_comparison.csv")) == 4
    storage.simulate(release / "storage_nse/registry.json", workspace / "observations.csv", workspace / "replay", identifier="sample")
    assert native.read(workspace / "replay/receipt.json")["observed_release_used"] is False


def test_missing_prior_and_source_gate_are_exported_without_fake_parameters(workspace):
    target = inputs(workspace); source = native.read(target)
    source["reservoirs"][0]["source_gates"] = ["unresolved storage datum"]
    storage.write(target, source)
    upstream = availability.prepare(target, workspace / "targets")
    out = storage.prepare(config(workspace, upstream), workspace / "calibration")
    record = native.read(out / "preparation.json")["reservoirs"][0]
    assert not record["calibration_ready"] and record["status"] == "source_gated"


def test_missing_prior_publishes_explicit_empty_inventory(workspace):
    upstream = availability.prepare(inputs(workspace), workspace / "targets")
    cfg = config(workspace, upstream); c = native.read(cfg); del c["profiles"][0]["release_prior"]; storage.write(cfg, c)
    out = storage.prepare(cfg, workspace / "calibration"); release = storage.finish(out, 16)
    table = pd.read_csv(release / "storage_nse/reservoir_parameterization_by_GRAND_ID.csv")
    assert len(table) == 1 and table.status.iloc[0] == "needs_naturalized_release_prior"
    assert not table.parameters_available.iloc[0]
    assert pd.read_csv(release / "baseline_comparison.csv").empty


def test_monthly_prior_from_shared_estimator_hands_off_without_duplicate_spec(workspace):
    target = inputs(workspace); frame = availability.read_frame(workspace / "observations.csv")
    pd.DataFrame({"naturalized_outflow_m3s": 80 + 15 * np.sin(np.arange(len(frame)) / 12)}, index=frame.index).to_csv(workspace / "monthly_naturalized.csv", index_label="date")
    c = native.read(target); c["supplemental_inputs"] = {"sample": {"naturalized_outflow": {
        "csv": "monthly_naturalized.csv", "source_kind": "simulated_naturalized_outflow", "evidence": "Synthetic monthly natural-lake simulation",
        "independent_of_target_responses": True}}}; storage.write(target, c)
    upstream = availability.prepare(target, workspace / "targets")
    cfg = config(workspace, upstream); c = native.read(cfg); del c["profiles"][0]["release_prior"]; storage.write(cfg, c)
    out = storage.prepare(cfg, workspace / "calibration")
    assert native.read(out / "preparation.json")["reservoirs"][0]["calibration_ready"]
    prior = native.read(out / "prepared/sample/release_prior/receipt.json")
    assert prior["source_frequency"] == "monthly" and prior["scope"] == "calibration_overlap"


def mean_spec(folder, policy="grand_only"):
    donors = pd.DataFrame([{"reservoir_id": rid, "GRAND_ID": gid, "quantile_method": "hazen", "p05_ratio": .1,
        "p30_ratio": .5, "p97_ratio": 1.5, "p99_ratio": 2., "reference_years": "2011,2012"}
        for rid, gid in (("sample", 123), ("a", 124), ("b", 125), ("c", 126))])
    donors.to_csv(folder / "donors.csv", index=False)
    pd.DataFrame({"GRAND_ID": [123], "DIS_AVG_LS": [10000], "mean_discharge_m3s": [10.]}).to_csv(folder / "grand.csv", index=False)
    return {"source_kind": "mean_flow_scaled_donor_prior", "donor_table": str(folder / "donors.csv"), "grand_table": str(folder / "grand.csv"),
        "mean_policy": policy, "minimum_donors": 3, "release_target_limit": "source_channel", "evidence": "Synthetic donor relationship"}


def test_grand_mean_units_target_exclusion_and_channel_limit(workspace):
    inputs(workspace); frame = availability.read_frame(workspace / "observations.csv")
    plan = YearSelection(**native.read(workspace / "selection.json")); model = native.read(workspace / "model.json")
    context = {"sampling": model["sampling"], "physical_constants": model["model"]}
    spec = mean_spec(workspace)
    donors = pd.read_csv(workspace / "donors.csv"); donors.loc[donors.reservoir_id == "sample", ["p05_ratio", "p30_ratio", "p97_ratio", "p99_ratio"]] = [100, 200, 300, 400]
    donors.to_csv(workspace / "donors.csv", index=False)
    result, receipt = priors.prepare(spec, frame.index, plan, quantile_method="hazen", folder=workspace / "prior",
        pin=lambda p: p, base=workspace, frame=frame, metadata=context, reservoir_id="sample", grand_id=123)
    np.testing.assert_array_equal(result.iloc[0], [1, 5, 15])
    assert receipt["mean_discharge_m3s"] == 10 and receipt["donor_count"] == 3
    assert "sample" not in receipt["donor_reservoir_ids"]
    grand = pd.read_csv(workspace / "grand.csv"); grand.mean_discharge_m3s = 10000.; grand.to_csv(workspace / "grand.csv", index=False)
    with pytest.raises(ValueError, match="conversion disagrees"):
        priors.prepare(spec, frame.index, plan, quantile_method="hazen", folder=workspace / "wrong_units",
            pin=lambda p: p, base=workspace, frame=frame, metadata=context, reservoir_id="sample", grand_id=123)
    grand.mean_discharge_m3s = 10.; grand.to_csv(workspace / "grand.csv", index=False)
    context["physical_constants"]["channel_capacity_m3s"] = 8.
    limited, trunc = priors.prepare(spec, frame.index, plan, quantile_method="hazen", folder=workspace / "limited",
        pin=lambda p: p, base=workspace, frame=frame, metadata=context, reservoir_id="sample", grand_id=123)
    np.testing.assert_array_equal(limited.iloc[0], [1, 5, 8]); assert trunc["target_levels_truncated"] == 1


def test_mean_inflow_fallback_is_calibration_only_and_explicit(workspace):
    inputs(workspace); frame = availability.read_frame(workspace / "observations.csv")
    plan = YearSelection(**native.read(workspace / "selection.json")); model = native.read(workspace / "model.json")
    spec = mean_spec(workspace, "grand_then_calibration_inflow")
    def make(f, name): return priors.prepare(spec, f.index, plan, quantile_method="hazen", folder=workspace / name,
        pin=lambda p: p, base=workspace, frame=f, metadata={"sampling": model["sampling"], "physical_constants": model["model"]}, reservoir_id="sample", grand_id=None)
    result, receipt = make(frame, "first")
    changed = frame.copy(); changed.release_m3s = -1e20; changed.storage_m3 *= 1000
    changed.loc[~plan.mask(frame.index, "calibration"), "inflow_m3s"] += 1e12
    other, changed_receipt = make(changed, "changed")
    np.testing.assert_array_equal(result, other)
    assert receipt["mean_source"] == changed_receipt["mean_source"] == "calibration_effective_inflow_proxy"
    assert receipt["mean_discharge_m3s"] == changed_receipt["mean_discharge_m3s"]
    spec["mean_policy"] = "grand_only"
    with pytest.raises(ValueError, match="No positive mean discharge"): make(frame, "strict")


def test_manager_resolves_donor_and_grand_paths_relative_to_config(workspace):
    target = inputs(workspace); cfg = native.read(config(workspace, workspace))
    prior = mean_spec(workspace); prior.update(donor_table="donors.csv", grand_table="grand.csv")
    cfg["profiles"][0]["release_prior"] = prior; cfg["campaign"] = "@targets.campaign"
    storage.write(workspace / "mean.json", cfg)
    path = workspace / "manager.json"
    storage.write(path, {"kind": "dztr_run_manager_v5", "schema_version": 5, "stages": [
        {"id": "targets", "type": "parameterize", "config": str(target)},
        {"id": "calibration", "type": "calibrate", "backend": "prepared_storage_v1", "config": "mean.json"}]})
    planned = manager.plan(path)["stages"][1]["config"]["profiles"][0]["release_prior"]
    assert Path(planned["donor_table"]) == workspace / "donors.csv"
    assert Path(planned["grand_table"]) == workspace / "grand.csv"


def test_manager_storage_only_handoff_replay_and_resume(workspace):
    target = inputs(workspace); cfg = native.read(config(workspace, workspace))
    cfg["campaign"] = "@targets.campaign"; cfg["optimization"].update(seeds=[7], evaluation_budget=8)
    managed = workspace / "manager.json"
    storage.write(managed, {"kind": "dztr_run_manager_v5", "schema_version": 5, "stages": [
        {"id": "targets", "type": "parameterize", "config": str(target)},
        {"id": "calibration", "type": "calibrate", "backend": "prepared_storage_v1", "config": cfg},
        {"id": "replay", "type": "simulate", "config": {"registry": "@calibration.registry", "forcing_csv": "@calibration.forcing_csv", "reservoir_id": "sample"}}]})
    result = manager.run(managed, workspace / "managed")
    assert result["status"] == "completed" and Path(result["stages"][1]["outputs"]["registry"]).is_file()
    again = manager.run(managed, workspace / "managed", resume=True)
    assert all(r["status"] == "reused" for r in again["stages"])
