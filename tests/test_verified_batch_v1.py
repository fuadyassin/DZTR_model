"""Verified imports retain original identities, signed routes and source gates."""
import copy
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from workflows import verified_batch_v1 as batch
from workflows import profile_application_v1 as app
from workflows import reservoir_v2 as native


@pytest.fixture(scope="module")
def template():
    root = native.ROOT / "outputs/calibrated_paper_cohort_v1_20261017_final"
    return batch.make_config(root / "catalog.json", root,
       ["ex14_Gardiner", "ex8_Flaming_gorge", "ex35_Bull_Lake", "ex15_Bennett"])


@pytest.fixture
def workspace():
    parent = native.ROOT / "outputs"
    folder = Path(tempfile.mkdtemp(prefix="test_verified_batch_", dir=parent)).resolve()
    yield folder
    assert folder.is_relative_to(parent.resolve()) and folder.name.startswith("test_verified_batch_")
    shutil.rmtree(folder)


def config(workspace, template, identifiers=None):
    cfg = copy.deepcopy(template)
    if identifiers is not None:
        cfg["reservoirs"] = [r for r in cfg["reservoirs"] if r["reservoir_id"] in identifiers]
    path = workspace / "config.json"
    native.write(path, cfg)
    return path


def test_plan_is_read_only_and_preserves_explicit_deferred_inventory(workspace, template, monkeypatch):
    path = config(workspace, template)
    monkeypatch.setattr(batch, "verify_searches", lambda *a: pytest.fail("Planning must not replay checkpoint analysis"))
    design = batch.plan(path)
    assert len(design["stages"]) == 4
    assert [r["planned_status"] for r in design["stages"]].count("importable") == 2
    assert [r["planned_status"] for r in design["stages"]].count("deferred") == 2
    assert design["new_optimizer_evaluations"] == design["new_simulations"] == 0
    assert not (workspace / "run").exists()


def test_real_batch_import_resume_registry_and_signed_solver_dispatch(workspace, template, monkeypatch):
    path = config(workspace, template)
    out = workspace / "run"
    monkeypatch.setattr(app, "run_frame", lambda *a, **k: pytest.fail("Import must not simulate"))
    state = batch.run(path, out, until="ex14_Gardiner")
    assert state["status"] == "stopped_at_requested_stage"
    state = batch.run(path, out, resume=True)
    assert state["status"] == "completed"
    assert state["summary"]["imported_profiles"] == 2
    assert state["summary"]["imported_standard_profiles"] == 1
    assert state["summary"]["imported_signed_profiles"] == 1
    assert state["summary"]["separate_interval_studies"] == state["summary"]["source_gated"] == 1
    assert state["summary"]["verified_seed_searches"] == 6
    assert state["summary"]["new_optimizer_evaluations"] == state["summary"]["new_simulations"] == 0
    registry = out / "release/registry.json"
    native.verify_registry(registry)
    catalog = app.read_catalog(out / "release/catalog.json")
    assert len(catalog["reservoirs"]) == 4
    entry, metadata, parameters = app.load_profile(out / "release/catalog.json", reservoir_id="ex8_Flaming_gorge")
    assert entry["route"] == metadata["route"] == "signed_net_native_v1"
    assert metadata["forcing_policy"]["interpretation"] == "exploratory_net_boundary_flux"
    signed_receipt = native.read(out / "jobs/ex8_Flaming_gorge/receipt.json")["result"]
    for seed in signed_receipt["checkpoint_records"]:
        original = Path(seed["original_identity_file"])
        copied = out / "release/identities/ex8_Flaming_gorge" / f"seed_{seed['seed']}.json"
        assert native.sha(original) == native.sha(copied) == seed["original_identity_sha256"]
        assert seed["checkpoint_copied_or_reidentified"] is False
    table = pd.read_csv(out / "release/reservoir_parameterization_by_GRAND_ID.csv", float_precision="round_trip")
    assert len(table) == 4 and table.parameter_values_available.sum() == 2
    bennett = table.loc[table.reservoir_id.eq("ex15_Bennett")].iloc[0]
    assert bennett.inflow_preprocessing_factor == 1.45
    assert pd.isna(bennett.storage_critical_m3_month_01)
    comparison = pd.read_csv(out / "release/report/baseline_comparison_all_seeds_and_selected.csv")
    assert len(comparison) == 16 and not comparison.selection_uses_assessment.any()
    for variable in ("storage", "release"):
        np.testing.assert_allclose(comparison[variable + "_nse_gain"], comparison[variable + "_nse"] - comparison["baseline_" + variable + "_nse"])
    def forbidden(*args, **kwargs): raise AssertionError("Completed scientific import must not repeat")
    monkeypatch.setattr(batch, "verify_searches", forbidden)
    original_write = batch.write_status
    status_writes = []
    def counted_write(*args):
        status_writes.append(1)
        return original_write(*args)
    monkeypatch.setattr(batch, "write_status", counted_write)
    again = batch.run(path, out, resume=True)
    assert all(r["status"] == "reused" for r in again["stages"])
    assert len(status_writes) == 1  # Verified reuse needs one final status update.
    # The existing application dispatches imported signed profiles, independently of importing.
    monkeypatch.undo()
    result = app.simulate(out / "release/catalog.json", workspace / "signed_replay", reservoir_id="ex8_Flaming_gorge")
    receipt = native.read(result / "simulation_receipt.json")
    assert receipt["route"] == "signed_net_native_v1" and receipt["maximum_independent_balance_residual_m3"] <= 1e-4


def test_changed_contract_is_retained_as_incompatible_with_blank_parameters(workspace, template):
    path = config(workspace, template, ["ex14_Gardiner"])
    cfg = native.read(path)
    cfg["reservoirs"][0]["expected_model"]["physical_constants"]["capacity_m3"] *= 1.1
    native.write(path, cfg)
    state = batch.run(path, workspace / "run")
    assert state["summary"]["imported_profiles"] == 0 and state["summary"]["incompatible"] == 1
    table = pd.read_csv(workspace / "run/release/reservoir_parameterization_by_GRAND_ID.csv")
    assert len(table) == 1 and not table.parameter_values_available.iloc[0]
    assert table.storage_critical_m3_month_01.isna().all()
    comparison = pd.read_csv(workspace / "run/release/report/baseline_comparison_all_seeds_and_selected.csv")
    assert comparison.empty and "storage_nse_gain" in comparison


def test_deferred_promotion_unknown_ids_and_duplicate_ids_fail(workspace, template):
    path = config(workspace, template, ["ex15_Bennett"])
    cfg = native.read(path); cfg["reservoirs"][0]["action"] = "import_verified"; native.write(path, cfg)
    with pytest.raises(ValueError, match="cannot be promoted"): batch.plan(path)
    cfg["reservoirs"][0]["action"] = "retain_deferred"; cfg["reservoirs"][0]["reservoir_id"] = "unknown"; native.write(path, cfg)
    with pytest.raises(ValueError, match="paper scope"): batch.plan(path)
    cfg["reservoirs"].append(dict(cfg["reservoirs"][0])); native.write(path, cfg)
    with pytest.raises(ValueError, match="unique reservoir"): batch.plan(path)


def test_missing_input_contract_never_becomes_an_import(workspace, template):
    path = config(workspace, template, ["ex14_Gardiner"])
    cfg = native.read(path)
    cfg["reservoirs"][0]["inputs"]["observations_csv"]["path"] = "data/missing_requested_input.csv"
    native.write(path, cfg)
    design = batch.plan(path)
    assert design["stages"][0]["planned_status"] == "incompatible"
    assert "observations_csv_unavailable" in design["stages"][0]["reasons"]


def test_imported_artifact_tampering_blocks_resume(workspace, template):
    path = config(workspace, template, ["ex14_Gardiner"])
    out = workspace / "run"
    batch.run(path, out)
    target = out / "release/profiles/ex14_Gardiner/monthly_targets_si.csv"
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="Imported profile artifact changed"):
        batch.run(path, out, resume=True)
    assert native.read(out / "status.json")["status"] == "failed" and not (out / "manager.lock").exists()


@pytest.mark.parametrize("winerror", [5, 32])
def test_status_retries_transient_windows_atomic_replace_locks(workspace, monkeypatch, winerror):
    original = native.write
    calls = []
    def locked(path, value):
        calls.append(1)
        if len(calls) < 3:
            error = PermissionError("transient reader lock")
            error.winerror = winerror
            raise error
        return original(path, value)
    monkeypatch.setattr(native, "write", locked)
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)
    batch.write_status(workspace, {"status": "completed"})
    assert len(calls) == 3
    assert native.read(workspace / "status.json")["status"] == "completed"


@pytest.mark.parametrize("winerror,attempts", [(5, 5), (None, 1)])
def test_persistent_or_nontransient_status_errors_are_reported(workspace, monkeypatch, winerror, attempts):
    calls = []
    def locked(path, value):
        calls.append(1)
        error = PermissionError("still locked")
        if winerror is not None: error.winerror = winerror
        raise error
    monkeypatch.setattr(native, "write", locked)
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError, match="still locked"):
        batch.write_status(workspace, {"status": "completed"})
    assert len(calls) == attempts


def test_manager_lock_is_released_even_when_status_cannot_be_saved(workspace, template, monkeypatch):
    path = config(workspace, template, ["ex14_Gardiner"])
    def failed(*args): raise PermissionError("status file cannot be saved")
    monkeypatch.setattr(batch, "write_status", failed)
    with pytest.raises(PermissionError, match="cannot be saved"):
        batch.run(path, workspace / "run")
    assert not (workspace / "run/manager.lock").exists()
