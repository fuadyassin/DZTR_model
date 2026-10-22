"""Exact NSGA-II continuation and non-pickle, identity-checked persistence."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import dztr.calibration.checkpoint as checkpoint_module
from dztr.calibration.checkpoint import load_optimizer_checkpoint, save_optimizer_checkpoint
from dztr.calibration.optimizer import NSGA2Settings, optimize_nsga2


IDENTITY = {"observations_sha256": "a" * 64,
            "search_code_sha256": {"optimizer.py": "b" * 64},
            "software_versions": {"numpy": np.__version__},
            "configuration": {"objective_order": ["release", "storage"]}}


class Objective:
    """Deterministic coordinate-based rejection survives objective reconstruction."""

    def __init__(self, *, fixed=False):
        self.baseline = np.repeat([0.2, 0.5, 0.8, 1.0, 2.0, 3.0], 12).astype(float)
        self.baseline[36:] *= 1e6
        lower, upper = self.baseline.copy(), self.baseline.copy()
        if not fixed:
            lower[0], upper[0] = 0.1, 0.3
            lower[12], upper[12] = 0.4, 0.6
            lower[36], upper[36] = 0.5e6, 1.5e6
        self.bounds = SimpleNamespace(lower_vector=lower, upper_vector=upper)
        self.calls = []

    def evaluate(self, vector):
        vector = np.asarray(vector)
        self.calls.append(vector.copy())
        widths = self.bounds.upper_vector - self.bounds.lower_vector
        unit = np.divide(vector - self.bounds.lower_vector, widths,
                         out=np.zeros(72), where=widths > 0)
        if unit[0] > 0.65:
            return SimpleNamespace(feasible=False, objectives=(np.inf, np.inf),
                                   reason="Synthetic physical rejection")
        return SimpleNamespace(feasible=True, objectives=(
            -unit[0] + 0.1 * unit[36] ** 2,
            -(1 - unit[0]) + 0.1 * unit[12] ** 2), reason=None)

    def evaluate_validation(self, *args, **kwargs):
        raise AssertionError("Checkpointing and stopping must use calibration only.")


def settings(**changes):
    options = {"seed": 73, "population_size": 8, "evaluation_budget": 32}
    options.update(changes)
    return NSGA2Settings(**options)


def run(objective, options=None, **kwargs):
    return optimize_nsga2(objective, objective.baseline,
                          settings() if options is None else options, **kwargs)


def assert_candidates_equal(first, second):
    assert [(item.evaluation_id, item.generation, item.objectives, item.feasible, item.reason)
            for item in first] == [
                (item.evaluation_id, item.generation, item.objectives, item.feasible, item.reason)
                for item in second]
    np.testing.assert_array_equal(np.stack([item.vector for item in first]),
                                  np.stack([item.vector for item in second]))


def assert_results_equal(first, second):
    assert first.settings == second.settings
    assert first.history == second.history
    assert first.termination == second.termination
    assert_candidates_equal(first.evaluated, second.evaluated)
    assert_candidates_equal(first.final_population, second.final_population)
    assert_candidates_equal(first.front, second.front)


def test_resumed_generation_matches_uninterrupted_trials_survivors_history_and_rng_exactly():
    complete_objective, complete_states = Objective(), []
    complete = run(complete_objective, checkpoint_callback=complete_states.append)
    partial_objective, partial_states = Objective(), []
    partial = run(partial_objective, settings(evaluation_budget=16),
                  checkpoint_callback=partial_states.append)
    resumed_objective, resumed_states = Objective(), []
    resumed = run(resumed_objective, resume_state=partial_states[-1],
                  checkpoint_callback=resumed_states.append)
    assert_results_equal(complete, resumed)
    assert partial.evaluations == len(partial_objective.calls) == 16
    assert resumed.evaluations == 32
    assert len(resumed_objective.calls) == 16
    np.testing.assert_array_equal(resumed_objective.calls, complete_objective.calls[16:])
    assert [state.evaluations for state in resumed_states] == [24, 32]
    assert resumed_states[-1].population_ids == complete_states[-1].population_ids
    assert resumed_states[-1].rng_state == complete_states[-1].rng_state
    assert any(not trial.feasible for trial in resumed.evaluated)
    assert len({trial.vector.tobytes() for trial in resumed.evaluated}) == 32
    fixed = complete_objective.bounds.lower_vector == complete_objective.bounds.upper_vector
    np.testing.assert_array_equal(np.stack(complete_objective.calls)[:, fixed],
                                  np.tile(complete_objective.baseline[fixed], (32, 1)))


def test_callback_interruption_retains_a_completed_generation_for_exact_continuation():
    expected = run(Objective())
    completed = []

    class Interrupted(RuntimeError):
        pass

    def interrupt_after_snapshot(state):
        completed.append(state)
        if state.evaluations == 16:
            raise Interrupted("Interrupted after durable generation boundary")

    objective = Objective()
    with pytest.raises(Interrupted):
        run(objective, checkpoint_callback=interrupt_after_snapshot)
    assert len(objective.calls) == completed[-1].evaluations == 16
    assert completed[-1].complete_generation
    resumed_objective = Objective()
    actual = run(resumed_objective, resume_state=completed[-1])
    assert_results_equal(expected, actual)
    assert len(resumed_objective.calls) == 16


def test_checkpoint_is_emitted_before_stopping_and_a_true_flag_prevents_future_calls():
    objective, event_order, saved = Objective(), [], []

    def checkpoint(state):
        saved.append(state)
        event_order.append(("save", state.evaluations))

    def stop(state):
        event_order.append(("stop", state.evaluations))
        return state.evaluations == 16

    result = run(objective, checkpoint_callback=checkpoint, stopping_callback=stop)
    assert len(objective.calls) == result.evaluations == 16
    assert event_order == [("save", 8), ("stop", 8), ("save", 16), ("stop", 16)]
    assert saved[-1].evaluations == 16
    assert "stop" in result.termination.lower()
    assert result.history[-1].evaluations == 16


def test_disabled_stopping_preserves_the_candidate_stream():
    original, observed = Objective(), Objective()
    expected = run(original)
    checks = []

    def continue_search(state):
        checks.append(state.evaluations)
        return False

    actual = run(observed, stopping_callback=continue_search)
    assert_results_equal(expected, actual)
    assert checks == [8, 16, 24, 32]


def test_resume_rechecks_a_saved_stop_before_any_model_call_without_saving_twice():
    states = []
    run(Objective(), settings(evaluation_budget=16), checkpoint_callback=states.append)
    objective, saved_again, checked = Objective(), [], []

    def stopping(state):
        checked.append(state.evaluations)
        return True

    result = run(objective, resume_state=states[-1],
                 checkpoint_callback=saved_again.append, stopping_callback=stopping)
    assert result.evaluations == 16
    assert objective.calls == []
    assert saved_again == []
    assert checked == [16]
    assert "stop" in result.termination.lower()


@pytest.mark.parametrize("changes", [
    {"seed": 74}, {"population_size": 4}, {"crossover_probability": 0.7},
    {"crossover_eta": 17}, {"mutation_eta": 21}, {"local_initialization_fraction": 0.25},
    {"local_initialization_scale": 0.1}, {"evaluation_budget": 8},
])
def test_incompatible_resume_settings_are_rejected_before_objective_calls(changes):
    snapshots = []
    run(Objective(), settings(evaluation_budget=16), checkpoint_callback=snapshots.append)
    objective = Objective()
    with pytest.raises(ValueError):
        run(objective, settings(**changes), resume_state=snapshots[-1])
    assert objective.calls == []


@pytest.mark.parametrize("change", ["lower", "upper", "baseline"])
def test_changed_bounds_or_baseline_cannot_resume_the_old_search(change):
    snapshots = []
    run(Objective(), settings(evaluation_budget=16), checkpoint_callback=snapshots.append)
    objective = Objective()
    if change == "lower":
        objective.bounds.lower_vector[0] -= 0.01
    elif change == "upper":
        objective.bounds.upper_vector[0] += 0.01
    else:
        objective.baseline[0] += 0.01
    with pytest.raises(ValueError):
        run(objective, resume_state=snapshots[-1])
    assert objective.calls == []


@pytest.mark.parametrize("change", ["population_ids", "generation", "rng_state", "evaluation_ids", "history_count"])
def test_inconsistent_resume_state_is_rejected_before_evaluation(change):
    states = []
    run(Objective(), settings(evaluation_budget=16), checkpoint_callback=states.append)
    state = states[-1]
    changes = {"population_ids": {"population_ids": (0,) * 8},
               "generation": {"generation": 99},
               "rng_state": {"rng_state": {}},
               "evaluation_ids": {"evaluated": state.evaluated[:-1] + (replace(state.evaluated[-1], evaluation_id=0),)},
               "history_count": {"history": state.history[:-1] + (replace(state.history[-1], evaluations=15),)}}
    objective = Objective()
    with pytest.raises(ValueError):
        invalid = replace(state, **changes[change])
        run(objective, resume_state=invalid)
    assert objective.calls == []


def test_partial_final_generation_is_not_a_resumable_snapshot():
    states = []
    partial = run(Objective(), settings(evaluation_budget=19), checkpoint_callback=states.append)
    assert partial.evaluations == 19
    assert [state.evaluations for state in states] == [8, 16]
    objective = Objective()
    with pytest.raises(ValueError):
        state = replace(states[-1], complete_generation=False, boundary_type="partial_generation")
        run(objective, resume_state=state)
    assert objective.calls == []


def test_duplicate_exhaustion_in_fixed_space_never_fabricates_a_full_generation_checkpoint():
    objective, states, checks = Objective(fixed=True), [], []
    result = run(objective, checkpoint_callback=states.append,
                 stopping_callback=lambda state: checks.append(state.evaluations) or False)
    assert result.evaluations == len(objective.calls) == 1
    assert states == checks == []


def test_saved_checkpoint_round_trips_without_pickle_and_continues_exactly(tmp_path):
    states = []
    run(Objective(), settings(evaluation_budget=16), checkpoint_callback=states.append)
    path = tmp_path / "search_state.npz"
    save_optimizer_checkpoint(path, states[-1], IDENTITY)
    with np.load(path, allow_pickle=False) as archive:
        assert archive.files
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
    restored = load_optimizer_checkpoint(path, IDENTITY)
    assert restored.settings == states[-1].settings
    assert restored.rng_state == states[-1].rng_state
    assert restored.population_ids == states[-1].population_ids
    assert_candidates_equal(restored.evaluated, states[-1].evaluated)
    objective = Objective()
    actual = run(objective, resume_state=restored)
    assert_results_equal(run(Objective()), actual)
    assert len(objective.calls) == 16


@pytest.mark.parametrize("changed", ["observations_sha256", "search_code_sha256", "software_versions", "configuration"])
def test_checkpoint_identity_mismatch_prevents_model_calls(tmp_path, changed):
    states = []
    run(Objective(), settings(evaluation_budget=16), checkpoint_callback=states.append)
    path = tmp_path / "state.npz"
    save_optimizer_checkpoint(path, states[-1], IDENTITY)
    identity = {**IDENTITY, changed: "Different identity"}
    objective = Objective()
    with pytest.raises(ValueError):
        restored = load_optimizer_checkpoint(path, identity)
        run(objective, resume_state=restored)
    assert objective.calls == []


def test_truncated_checkpoint_is_rejected_before_search(tmp_path):
    states = []
    run(Objective(), settings(evaluation_budget=16), checkpoint_callback=states.append)
    path = tmp_path / "state.npz"
    save_optimizer_checkpoint(path, states[-1], IDENTITY)
    path.write_bytes(path.read_bytes()[:100])
    objective = Objective()
    with pytest.raises(ValueError):
        restored = load_optimizer_checkpoint(path, IDENTITY)
        run(objective, resume_state=restored)
    assert objective.calls == []


def test_valid_npz_with_modified_candidate_data_fails_the_internal_checksum(tmp_path):
    states = []
    run(Objective(), settings(evaluation_budget=16), checkpoint_callback=states.append)
    path = tmp_path / "state.npz"
    save_optimizer_checkpoint(path, states[-1], IDENTITY)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    vectors = next(value for value in arrays.values()
                   if value.ndim == 2 and value.shape == (16, 72) and value.dtype.kind == "f")
    vectors[0, 0] += 1e-6
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    objective = Objective()
    with pytest.raises(ValueError):
        restored = load_optimizer_checkpoint(path, IDENTITY)
        run(objective, resume_state=restored)
    assert objective.calls == []


def test_failed_atomic_replacement_preserves_the_previous_complete_checkpoint(tmp_path, monkeypatch):
    states = []
    run(Objective(), checkpoint_callback=states.append)
    path = tmp_path / "state.npz"
    save_optimizer_checkpoint(path, states[1], IDENTITY)
    original = path.read_bytes()

    def fail_replace(*args, **kwargs):
        raise OSError("Simulated atomic replacement failure")

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failure"):
        save_optimizer_checkpoint(path, states[-1], IDENTITY)
    assert path.read_bytes() == original
    assert {item.name for item in tmp_path.iterdir()} == {"state.npz"}


@pytest.mark.parametrize("winerror", [5, 32])
def test_transient_windows_lock_retries_atomic_replacement(tmp_path, monkeypatch, winerror):
    states = []
    run(Objective(), checkpoint_callback=states.append)
    path = tmp_path / "state.npz"
    save_optimizer_checkpoint(path, states[1], IDENTITY)
    original_replace = checkpoint_module.os.replace
    attempts, sleeps = [], []

    def locked_replace(source, destination):
        attempts.append(destination)
        if len(attempts) < 3:
            error = PermissionError("Simulated transient Windows file lock")
            error.winerror = winerror
            raise error
        return original_replace(source, destination)

    monkeypatch.setattr(checkpoint_module.os, "replace", locked_replace)
    monkeypatch.setattr(checkpoint_module.time, "sleep", sleeps.append)
    save_optimizer_checkpoint(path, states[-1], IDENTITY)
    assert len(attempts) == 3
    assert sleeps == [0.1, 0.1]
    restored = load_optimizer_checkpoint(path, IDENTITY)
    assert_candidates_equal(restored.evaluated, states[-1].evaluated)
    assert {item.name for item in tmp_path.iterdir()} == {"state.npz"}


def test_permanent_windows_lock_preserves_old_checkpoint_after_bounded_retries(tmp_path, monkeypatch):
    states = []
    run(Objective(), checkpoint_callback=states.append)
    path = tmp_path / "state.npz"
    save_optimizer_checkpoint(path, states[1], IDENTITY)
    original = path.read_bytes()
    attempts, sleeps = [], []

    def locked_replace(source, destination):
        attempts.append(destination)
        error = PermissionError("Simulated permanent Windows file lock")
        error.winerror = 5
        raise error

    monkeypatch.setattr(checkpoint_module.os, "replace", locked_replace)
    monkeypatch.setattr(checkpoint_module.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError):
        save_optimizer_checkpoint(path, states[-1], IDENTITY)
    assert len(attempts) == 5
    assert sleeps == [0.1] * 4
    assert path.read_bytes() == original
    assert {item.name for item in tmp_path.iterdir()} == {"state.npz"}
