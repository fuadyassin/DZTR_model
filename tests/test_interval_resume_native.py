"""Native checkpoint continuation and explicitly reviewed adapter migration."""
from copy import deepcopy

import numpy as np
import pytest

import dztr.calibration.interval_resume as adapter
from dztr.calibration.checkpoint import load_optimizer_checkpoint, save_optimizer_checkpoint
from dztr.calibration.optimizer import optimize_nsga2
from test_interval_resume import frozen_seed, freeze_parent, write_json, SyntheticObjective


def native(seed, adapter_hash=None):
    report = deepcopy(seed.prior)
    identity = report["checkpoint_identity"]
    identity["code_sha256"] = adapter._module_hashes()
    if adapter_hash is not None:
        identity["code_sha256"][adapter.APPROVED_ADAPTER] = adapter_hash
    report["code_sha256"] = deepcopy(identity["code_sha256"])
    # This synthetic fixture starts a different native search identity; the
    # production writer correctly refuses to overwrite the old identity.
    (seed.directory / "optimizer_checkpoint.npz").unlink()
    save_optimizer_checkpoint(seed.directory / "optimizer_checkpoint.npz", seed.state, identity)
    write_json(seed.directory / "summary.json", report)
    freeze_parent(seed.directory)
    return report


def migration(seed, old_hash):
    proof = (seed.directory.parent / "verification.json").resolve()
    return {"old_sha256": old_hash, "new_sha256": adapter._module_hashes()[adapter.APPROVED_ADAPTER],
            "prior_verification_path": str(proof), "prior_verification_sha256": adapter._sha256(proof)}


def test_native_resume_exact_and_chainable(frozen_seed):
    seed = frozen_seed
    original = native(seed)
    report = adapter.resume_interval_seed(seed.directory, seed.policy, seed.output)
    state = load_optimizer_checkpoint(seed.output / "optimizer_checkpoint.npz", report["checkpoint_identity"])
    uninterrupted = optimize_nsga2(SyntheticObjective(), seed.baseline,
        type(seed.settings)(**{**seed.settings.__dict__, "evaluation_budget": 256}))
    np.testing.assert_array_equal([p.vector for p in state.evaluated], [p.vector for p in uninterrupted.evaluated])
    assert [p.objectives for p in state.evaluated] == [p.objectives for p in uninterrupted.evaluated]
    assert len(seed.objective.calls) == 128
    extension = report["checkpoint_identity"]["identity_extension"]
    assert extension["origin_kind"] == "native_with_adapter"
    assert extension["origin_code_sha256"] == original["code_sha256"]
    assert extension["origin_identity_sha256"] == adapter._digest(original["checkpoint_identity"])
    assert extension["approved_added_modules"] == {} and extension["adapter_code_migrations"] == []
    freeze_parent(seed.output)
    seed.objective.forbidden.update(p.vector.tobytes() for p in state.evaluated)
    seed.objective.calls.clear()
    second = adapter.resume_interval_seed(seed.output, seed.policy, seed.output.parent / "next",
                                          total_evaluation_budget=272, checkpoints=(272,))
    assert second["new_search_evaluations"] == len(seed.objective.calls) == 16
    assert second["checkpoint_identity"]["identity_extension"]["origin_identity_sha256"] == extension["origin_identity_sha256"]


def test_native_old_adapter_requires_explicit_verified_migration(frozen_seed):
    seed = frozen_seed
    original = native(seed, "1" * 64)
    with pytest.raises(ValueError, match="code hash drift"):
        adapter.resume_interval_seed(seed.directory, seed.policy, seed.output)
    assert seed.loader_calls == seed.objective.calls == []
    approved = migration(seed, "1" * 64)
    report = adapter.resume_interval_seed(seed.directory, seed.policy, seed.output,
                                          approved_adapter_migration=approved)
    assert len(seed.objective.calls) == report["new_search_evaluations"] == 128
    receipt = report["migration_receipt"]
    assert not receipt["historical_hashes_unchanged"] and receipt["historical_numerical_hashes_unchanged"]
    assert receipt["adapter_code_migrations"] == [approved]
    assert report["checkpoint_identity"]["identity_extension"]["origin_code_sha256"] == original["code_sha256"]
    freeze_parent(seed.output)
    continued = adapter.resume_interval_seed(seed.output, seed.policy, seed.output.parent / "next",
                                              total_evaluation_budget=272, checkpoints=(272,))
    assert continued["migration_receipt"]["adapter_code_migrations"] == [approved]


@pytest.mark.parametrize("field", ["old_sha256", "new_sha256", "prior_verification_path", "prior_verification_sha256"])
def test_migration_mismatch_refused_before_model_loading(frozen_seed, field):
    seed = frozen_seed
    native(seed, "1" * 64)
    approved = migration(seed, "1" * 64)
    approved[field] = "2" * 64
    with pytest.raises(ValueError, match="migration"):
        adapter.resume_interval_seed(seed.directory, seed.policy, seed.output,
                                      approved_adapter_migration=approved)
    assert seed.loader_calls == seed.objective.calls == []


def test_adapter_migration_cannot_authorize_optimizer_change(frozen_seed, monkeypatch):
    seed = frozen_seed
    native(seed, "1" * 64)
    approved = migration(seed, "1" * 64)
    hashes = adapter._module_hashes()
    hashes["calibration/optimizer.py"] = "2" * 64
    monkeypatch.setattr(adapter, "_module_hashes", lambda: hashes)
    with pytest.raises(ValueError, match="code hash drift"):
        adapter.resume_interval_seed(seed.directory, seed.policy, seed.output,
                                      approved_adapter_migration=approved)
    assert seed.loader_calls == seed.objective.calls == []


@pytest.mark.parametrize("field", ["identity_extension", "migration_receipt"])
def test_half_removed_chain_is_not_treated_as_native(frozen_seed, field):
    seed = frozen_seed
    native(seed)
    report = adapter.resume_interval_seed(seed.directory, seed.policy, seed.output)
    identity = deepcopy(report["checkpoint_identity"])
    del identity[field]
    with pytest.raises(ValueError, match="both its bridge"):
        adapter._extended_identity(identity, adapter._module_hashes())


def test_unnecessary_migration_refused(frozen_seed):
    seed = frozen_seed
    native(seed)
    approved = migration(seed, adapter._module_hashes()[adapter.APPROVED_ADAPTER])
    with pytest.raises(ValueError, match="unnecessary"):
        adapter.resume_interval_seed(seed.directory, seed.policy, seed.output,
                                      approved_adapter_migration=approved)
    assert seed.loader_calls == seed.objective.calls == []


def test_schema1_chain_retains_original_identity(frozen_seed):
    origin = deepcopy(frozen_seed.prior["checkpoint_identity"])
    current = adapter._module_hashes()
    extension = {"schema_version": 1, "approved_added_modules": {adapter.APPROVED_ADAPTER: current[adapter.APPROVED_ADAPTER]},
                 "historical_code_sha256": origin["code_sha256"],
                 "previous_identity_sha256": adapter._digest(origin), "origin_identity_sha256": adapter._digest(origin)}
    extension["bridge_sha256"] = adapter._digest(extension)
    identity = {**origin, "code_sha256": current, "identity_extension": extension,
                "migration_receipt": {"bridge_sha256": extension["bridge_sha256"],
                                      "previous_identity_sha256": adapter._digest(origin)}}
    continued = adapter._extended_identity(identity, current)
    assert continued["identity_extension"]["origin_kind"] == "pre_adapter"
    assert continued["identity_extension"]["origin_identity_sha256"] == adapter._digest(origin)
    assert continued["identity_extension"]["origin_code_sha256"] == origin["code_sha256"]
