"""Synthetic exact continuation, provenance refusal and cumulative receipts."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dztr import __version__
import dztr.calibration.interval_resume as adapter
from dztr.calibration.checkpoint import load_optimizer_checkpoint, save_optimizer_checkpoint
from dztr.calibration.optimizer import NSGA2Settings, optimize_nsga2
from dztr.calibration.stopping import StoppingPolicy


class SyntheticBounds:
    lower_vector = np.zeros(72)
    upper_vector = np.ones(72)

    def summary(self):
        return {"synthetic": True, "dimension": 72, "calibration_only": True}


class SyntheticObjective:
    def __init__(self, forbidden=()):
        self.bounds = SyntheticBounds()
        self.calls = []
        self.forbidden = {np.asarray(vector).tobytes() for vector in forbidden}

    def evaluate(self, vector):
        assert vector.tobytes() not in self.forbidden, "Inherited trial must never be reevaluated."
        self.calls.append(vector.copy())
        feasible = int(float(np.sum(vector)) * 1e8) % 7 != 0
        costs = (float(np.mean((vector - .2) ** 2) - .8),
                 float(np.mean((vector - .8) ** 2) - .8))
        return SimpleNamespace(feasible=feasible, objectives=costs,
                               reason=None if feasible else "Synthetic strict-root rejection")

    def evaluate_validation(self, *args, **kwargs):
        pytest.fail("Resume must never call held-out validation.")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def freeze_parent(directory):
    write_json(directory.parent / "verification.json", {
        "status": "passed", "artifact_sha256": {
            f"{directory.name}/{name}": adapter._sha256(directory / name)
            for name in adapter.ARTIFACT_NAMES},
    })


@pytest.fixture
def frozen_seed(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    directory = parent / "seed_42"
    directory.mkdir(parents=True)
    policy = tmp_path / "policy.json"
    config = tmp_path / "config.json"
    source = tmp_path / "observations.csv"
    write_json(policy, {"policy_id": "synthetic_v1"})
    write_json(config, {"reservoir": "synthetic"})
    source.write_text("date,storage\n2000-01-01,1\n", encoding="utf-8")
    consumed = {str(path.resolve()): adapter._sha256(path) for path in (policy, config, source)}
    periods = {"simulation_start": "1999-12-31",
               "calibration": {"start": "2000-01-01", "end": "2000-12-31"},
               "validation": {"start": "2001-01-01", "end": "2001-12-31"}}
    baseline = np.full(72, .5)
    settings = NSGA2Settings(seed=42, population_size=16, evaluation_budget=128)
    problem = SyntheticObjective()
    states = []
    result = optimize_nsga2(problem, baseline, settings, checkpoint_callback=states.append)
    state = states[-1]
    old_codes = {name: digest for name, digest in adapter._module_hashes().items()
                 if name != adapter.APPROVED_ADAPTER}
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "dztr": __version__}
    identity = {"policy_id": "synthetic_v1", "consumed_files_sha256": consumed,
                "code_sha256": old_codes, "software_versions": versions,
                "nonbudget_settings": {name: value for name, value in asdict(settings).items()
                                       if name != "evaluation_budget"},
                "root_policy": "reject", "initial_storage_offset_capacity_fraction": 0.,
                "periods": periods, "bounds": problem.bounds.summary()}
    stopping = StoppingPolicy(population_size=16, check_interval=16, window_evaluations=64,
        minimum_evaluations=96, patience=2, max_evaluations=128,
        normalized_hv_tolerance=.0025, front_tolerance=.01, extrema_tolerance=.005)
    report = {**{name: deepcopy(identity[name]) for name in (
        "policy_id", "consumed_files_sha256", "code_sha256", "software_versions",
        "root_policy", "initial_storage_offset_capacity_fraction", "periods", "bounds")},
        "optimization": asdict(settings), "checkpoint_identity": identity,
        "search_evaluations": 128, "checkpoint_evaluations": 128, "checkpoint_saved": True,
        "final_population_ids": list(state.population_ids), "stopping_policy": asdict(stopping),
        "baseline_evaluation_id": 0, "baseline_included_in_budget": True,
        "conditional_hindcast": True, "validation_label": "segmented conditional hindcast"}
    for name in adapter.ARTIFACT_NAMES:
        (directory / name).write_text("month,value\n1,1\n", encoding="utf-8")
    pd.DataFrame([adapter._trial_row(point) for point in result.evaluated]).to_csv(
        directory / "evaluations.csv", index=False)
    adapter._vectors_frame(result.evaluated).to_csv(directory / "evaluated_vectors.csv")
    pd.DataFrame([asdict(record) for record in result.history]).to_csv(directory / "history.csv", index=False)
    (directory / "optimizer_checkpoint.npz").unlink()
    save_optimizer_checkpoint(directory / "optimizer_checkpoint.npz", state, identity)
    write_json(directory / "summary.json", report)
    freeze_parent(directory)
    resumed = SyntheticObjective(forbidden=problem.calls)
    context = {"policy": {"policy_id": "synthetic_v1"}, "config": {"reservoir": "synthetic"},
               "consumed_files_sha256": consumed, "periods": periods, "bounds": resumed.bounds}
    loader_calls = []

    def loader(path):
        loader_calls.append(path)
        return resumed, baseline.copy(), object(), context

    monkeypatch.setattr(adapter, "load_interval_problem", loader)
    return SimpleNamespace(directory=directory, policy=policy, source=source, settings=settings,
        baseline=baseline, state=state, prior=report, objective=resumed, loader_calls=loader_calls,
        context=context, output=tmp_path / "resumed" / "seed_42")


def resume(seed, **kwargs):
    return adapter.resume_interval_seed(seed.directory, seed.policy, seed.output, **kwargs)


def test_continuation_exactly_matches_uninterrupted_and_counts_only_new_calls(frozen_seed):
    seed = frozen_seed
    uninterrupted = SyntheticObjective()
    states = []
    full = optimize_nsga2(uninterrupted, seed.baseline, replace(seed.settings, evaluation_budget=256),
                         checkpoint_callback=states.append)
    callback_states = []
    report = resume(seed, progress_callback=callback_states.append)
    saved = load_optimizer_checkpoint(seed.output / "optimizer_checkpoint.npz", report["checkpoint_identity"])
    assert len(seed.objective.calls) == report["new_calls"] == report["new_search_evaluations"] == 128
    assert report["inherited_count"] == report["inherited_evaluations"] == 128
    assert report["total_evaluations"] == report["search_evaluations"] == 256
    assert report["baseline_reevaluated"] is False
    assert report["baseline_included_in_budget"] is True
    assert report["reporting_calibration_evaluations"] == report["reporting_validation_evaluations"] == 0
    assert report["new_rejected_evaluations"] > 0
    assert report["inherited_rejected_evaluations"] + report["new_rejected_evaluations"] == report["rejected_evaluations"]
    assert report["feasible_evaluations"] + report["rejected_evaluations"] == 256
    assert saved.population_ids == states[-1].population_ids
    assert saved.rng_state == states[-1].rng_state
    assert saved.history == full.history
    for continued, original in zip(saved.evaluated, full.evaluated):
        np.testing.assert_array_equal(continued.vector, original.vector)
        assert continued.objectives == original.objectives
        assert (continued.evaluation_id, continued.generation, continued.feasible, continued.reason) == (
            original.evaluation_id, original.generation, original.feasible, original.reason)
    np.testing.assert_array_equal(np.stack(seed.objective.calls), np.stack(uninterrupted.calls[128:]))
    assert [record.evaluations for record in callback_states] == list(range(144, 257, 16))
    assert [record.generation for record in callback_states] == list(range(8, 16))
    assert set(adapter.ARTIFACT_NAMES) == {path.name for path in seed.output.iterdir()}
    assert json.loads((seed.output / "summary.json").read_text()) == report
    assert pd.read_csv(seed.output / "evaluations.csv").evaluation_id.tolist() == list(range(256))
    assert pd.read_csv(seed.output / "history.csv").evaluations.tolist() == list(range(16, 257, 16))
    for name in ("evaluations.csv", "evaluated_vectors.csv", "history.csv"):
        previous = (seed.directory / name).read_bytes()
        assert (seed.output / name).read_bytes()[:len(previous)] == previous
    for name in ("target_sample_support.csv", "monthly_targets_si.csv", "monthly_bounds_si.csv"):
        assert (seed.output / name).read_bytes() == (seed.directory / name).read_bytes()
    old_codes = seed.prior["checkpoint_identity"]["code_sha256"]
    current_codes = report["checkpoint_identity"]["code_sha256"]
    assert set(current_codes) - set(old_codes) == {adapter.APPROVED_ADAPTER}
    assert all(current_codes[name] == value for name, value in old_codes.items())
    assert report["migration_receipt"]["historical_hashes_unchanged"] is True
    assert report["stopping_controls_search"] is report["optimizer_convergence_established"] is False
    assert report["termination"] == "budget_reached"
    expected_stopping = deepcopy(seed.prior["stopping_policy"])
    expected_stopping["max_evaluations"] = 256
    expected_stopping["hypervolume_reference"] = list(expected_stopping["hypervolume_reference"])
    assert report["stopping_policy"] == expected_stopping


@pytest.mark.parametrize("artifact", ["summary.json", "optimizer_checkpoint.npz", "evaluations.csv", "evaluated_vectors.csv", "history.csv"])
def test_frozen_artifact_hash_drift_refused_before_problem_loading(frozen_seed, artifact):
    seed = frozen_seed
    path = seed.directory / artifact
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="artifact SHA256"):
        resume(seed)
    assert seed.loader_calls == seed.objective.calls == []
    assert not seed.output.exists()


@pytest.mark.parametrize("change", ["historic_code", "extra_module", "missing_module", "adapter_changed"])
def test_code_identity_drift_and_unapproved_module_refused(frozen_seed, monkeypatch, change):
    hashes = adapter._module_hashes()
    if change == "historic_code":
        hashes["calibration/optimizer.py"] = "0" * 64
    elif change == "extra_module":
        hashes["calibration/unreviewed.py"] = "0" * 64
    elif change == "missing_module":
        del hashes["calibration/optimizer.py"]
    else:
        del hashes[adapter.APPROVED_ADAPTER]
    monkeypatch.setattr(adapter, "_module_hashes", lambda: hashes)
    with pytest.raises(ValueError, match="code hash|approved"):
        resume(frozen_seed)
    assert frozen_seed.loader_calls == frozen_seed.objective.calls == []


def test_consumed_input_drift_refused_before_loading(frozen_seed):
    frozen_seed.source.write_bytes(frozen_seed.source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="input SHA256 drift"):
        resume(frozen_seed)
    assert frozen_seed.loader_calls == frozen_seed.objective.calls == []


def test_summary_checkpoint_identity_mismatch_refused_even_with_updated_artifact_receipt(frozen_seed):
    seed = frozen_seed
    report = deepcopy(seed.prior)
    report["checkpoint_identity"]["policy_id"] = "changed"
    write_json(seed.directory / "summary.json", report)
    freeze_parent(seed.directory)
    with pytest.raises(ValueError, match="identity disagree"):
        resume(seed)
    assert seed.loader_calls == seed.objective.calls == []


def test_checkpoint_identity_mismatch_refused_before_loading(frozen_seed):
    seed = frozen_seed
    report = deepcopy(seed.prior)
    report["checkpoint_identity"]["nonbudget_settings"]["seed"] = 43
    write_json(seed.directory / "summary.json", report)
    freeze_parent(seed.directory)
    with pytest.raises(ValueError, match="Checkpoint identity"):
        resume(seed)
    assert seed.loader_calls == seed.objective.calls == []


def test_csv_cost_mismatch_refused_even_with_updated_artifact_receipt(frozen_seed):
    seed = frozen_seed
    path = seed.directory / "evaluations.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[0, "negative_release_nse"] += .1
    frame.to_csv(path, index=False)
    freeze_parent(seed.directory)
    with pytest.raises(ValueError, match="costs/metadata"):
        resume(seed)
    assert seed.loader_calls == seed.objective.calls == []


def test_existing_output_refused_before_verification_and_loading(frozen_seed, monkeypatch):
    seed = frozen_seed
    seed.output.mkdir(parents=True)
    marker = seed.output / "preserve.txt"
    marker.write_text("original", encoding="utf-8")
    monkeypatch.setattr(adapter, "_verified_artifacts", lambda *_: pytest.fail("No artifact read after overwrite refusal."))
    with pytest.raises(FileExistsError, match="fresh"):
        resume(seed)
    assert marker.read_text() == "original"
    assert list(seed.output.iterdir()) == [marker]
    assert seed.loader_calls == seed.objective.calls == []


def test_stopping_policy_drift_refused_before_loading(frozen_seed):
    values = deepcopy(frozen_seed.prior["stopping_policy"])
    values["max_evaluations"] = 256
    values["front_tolerance"] = 100.
    with pytest.raises(ValueError, match="tolerances remain frozen"):
        resume(frozen_seed, stopping_policy=StoppingPolicy(**values))
    assert frozen_seed.loader_calls == frozen_seed.objective.calls == []


@pytest.mark.parametrize("budget", [128, 127, 255, True])
def test_invalid_or_nonextended_budget_refused_before_loading(frozen_seed, budget):
    with pytest.raises(ValueError, match="budget|boundary"):
        resume(frozen_seed, total_evaluation_budget=budget)
    assert frozen_seed.loader_calls == frozen_seed.objective.calls == []


def test_loaded_bounds_identity_drift_refused_without_evaluations(frozen_seed):
    frozen_seed.context["periods"] = {"changed": True}
    with pytest.raises(ValueError, match="Loaded problem identity drift"):
        resume(frozen_seed)
    assert len(frozen_seed.loader_calls) == 1
    assert frozen_seed.objective.calls == []


def test_chained_resume_requires_same_approved_adapter_and_preserves_original_identity(frozen_seed):
    seed = frozen_seed
    report = resume(seed)
    first = seed.output
    freeze_parent(first)
    first_state = load_optimizer_checkpoint(first / "optimizer_checkpoint.npz", report["checkpoint_identity"])
    seed.objective.forbidden.update(point.vector.tobytes() for point in first_state.evaluated)
    seed.objective.calls.clear()
    second = first.parent / "second_seed"
    continued = adapter.resume_interval_seed(first, seed.policy, second, total_evaluation_budget=272,
                                            checkpoints=(128, 256, 272))
    assert continued["inherited_evaluations"] == 256
    assert continued["new_search_evaluations"] == len(seed.objective.calls) == 16
    assert continued["search_evaluations"] == 272
    assert continued["checkpoint_identity"]["identity_extension"]["origin_identity_sha256"] == (
        report["checkpoint_identity"]["identity_extension"]["origin_identity_sha256"])
    assert continued["checkpoint_identity"]["code_sha256"] == report["checkpoint_identity"]["code_sha256"]


def test_chained_adapter_hash_drift_refused(frozen_seed, monkeypatch):
    seed = frozen_seed
    resume(seed)
    freeze_parent(seed.output)
    hashes = adapter._module_hashes()
    hashes[adapter.APPROVED_ADAPTER] = "0" * 64
    monkeypatch.setattr(adapter, "_module_hashes", lambda: hashes)
    seed.objective.calls.clear()
    seed.loader_calls.clear()
    with pytest.raises(ValueError, match="code hash drift"):
        adapter.resume_interval_seed(seed.output, seed.policy, seed.output.parent / "changed",
                                     total_evaluation_budget=272, checkpoints=(272,))
    assert seed.loader_calls == seed.objective.calls == []


def test_stopping_suggestion_never_shortens_the_fixed_continuation(frozen_seed):
    seed = frozen_seed
    report = deepcopy(seed.prior)
    for name in ("normalized_hv_tolerance", "front_tolerance", "extrema_tolerance"):
        report["stopping_policy"][name] = 100.
    write_json(seed.directory / "summary.json", report)
    freeze_parent(seed.directory)
    continued = resume(seed)
    assert continued["stopping_assessment"]["suggested_stop_evaluations"] is not None
    assert continued["search_evaluations"] == 256
    assert len(seed.objective.calls) == 128
    assert continued["termination"] == "budget_reached"
    assert continued["stopping_controls_search"] is False
