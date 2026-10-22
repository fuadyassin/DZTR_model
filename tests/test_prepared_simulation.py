"""Exact scientific parity against frozen, independently loaded original code.

Reference sources preserve the pre-cache public solvers and complete simulation
loop. They are intentionally not built from the optimized arithmetic helpers.
"""

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from dztr.calibration.bounds import BOUND_COLUMNS, CalibrationBounds
from dztr.calibration.checkpoint import load_optimizer_checkpoint, save_optimizer_checkpoint
import dztr.calibration.objectives as objective_module
from dztr.calibration.objectives import CalibrationObjective
from dztr.calibration.optimizer import NSGA2Settings, optimize_nsga2
from dztr.calibration.periods import DatePeriod
from dztr.data import normalize_observations
from dztr.model import fixed_point_step, solve_implicit_step
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters
import dztr.simulation.runner as runner_module
from dztr.simulation import simulate_reservoir


REFERENCE_FOLDER = Path(__file__).parent / "reference_cases" / "uncached_simulation"


def _load_reference(filename, name):
    spec = importlib.util.spec_from_file_location(name, REFERENCE_FOLDER / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def original():
    provenance = json.loads((REFERENCE_FOLDER / "provenance.json").read_text(encoding="utf-8"))
    # Support the explicit source-hash map without depending on provenance prose.
    digests = provenance.get("source_sha256", provenance.get("sha256"))
    assert isinstance(digests, dict), "Frozen references require recorded source SHA256."
    for filename in ("release.py", "balance.py", "runner.py"):
        actual = hashlib.sha256((REFERENCE_FOLDER / filename).read_bytes()).hexdigest()
        assert actual == digests[filename], f"Frozen reference changed: {filename}"
    release = _load_reference("release.py", "dztr.model._original_monthly_release_reference")
    balance = _load_reference("balance.py", "dztr.model._original_balance_reference")
    for name in ("_finite", "_monthly_values", "_release_for_values", "_timestep"):
        setattr(balance, name, getattr(release, name))
    runner = _load_reference("runner.py", "dztr.simulation._original_runner_reference")
    runner.dztr_release, runner.dztr_zone = release.dztr_release, release.dztr_zone
    runner.fixed_point_step, runner.solve_implicit_step = balance.fixed_point_step, balance.solve_implicit_step
    assert balance.solve_implicit_step is not solve_implicit_step
    assert balance.fixed_point_step is not fixed_point_step
    return runner, balance, release


def parameters(**changes):
    month = np.arange(1, 13, dtype=float)
    targets = np.column_stack((
        35.0 + 0.35 * month, 58.0 + 0.2 * month, 82.0 + 0.15 * month,
        0.0001 + 0.000002 * month, 0.0002 + 0.000003 * month,
        0.00035 + 0.000004 * month,
    ))
    values = dict(
        monthly_targets=pd.DataFrame(targets, index=pd.Index(range(1, 13), name="month"),
                                     columns=MONTHLY_TARGET_COLUMNS),
        capacity_m3=100.0, initial_storage_m3=60.0,
        initial_release_m3s=0.0002, channel_capacity_m3s=0.001,
        inflow_correction=0.93,
    )
    values.update(changes)
    return ReservoirParameters(**values)


def observations(inflows, start="1999-12-29"):
    dates = pd.date_range(start, periods=len(inflows), freq="D", name="date")
    ordinal = np.arange(len(inflows), dtype=float)
    return normalize_observations(pd.DataFrame({
        "inflow_m3s": np.asarray(inflows),
        "storage_m3": 55.0 + np.sin(ordinal / 2.0) * 4.0,
        "release_m3s": 0.0002 + ordinal * 0.000001,
    }, index=dates))


def assert_simulations_equal(actual, expected):
    pd.testing.assert_frame_equal(actual.frame, expected.frame, check_exact=True)
    assert actual.parameters is expected.parameters
    assert actual.solver == expected.solver
    assert actual.overflow_mode == expected.overflow_mode
    assert actual.issues == expected.issues


@pytest.mark.parametrize("solver", ["implicit", "matlab_fixed_point"])
@pytest.mark.parametrize("flood_mode", ["storage", "inflow"])
@pytest.mark.parametrize("overflow_mode", ["conservative", "matlab"])
@pytest.mark.parametrize("start", ["1999-12-29", "2000-02-26"])
def test_complete_daily_frames_are_exact_across_calendar_boundaries(
    original, solver, flood_mode, overflow_mode, start,
):
    reference, _, _ = original
    params = parameters(flood_mode=flood_mode)
    obs = observations([0.00024, 0.00023, 0.00027, 0.00021,
                        0.00026, 0.00028, 0.00022, 0.00025], start)
    options = dict(solver=solver, overflow_mode=overflow_mode, max_iterations=13)
    actual = simulate_reservoir(obs, params, **options)
    expected = reference.simulate_reservoir(obs, params, **options)
    assert_simulations_equal(actual, expected)
    assert obs.frame.index.equals(actual.frame.index)
    if start == "2000-02-26":
        assert pd.Timestamp("2000-02-29") in actual.frame.index
    else:
        assert {12, 1} == set(actual.frame.index.month)


@pytest.mark.parametrize("solver", ["implicit", "matlab_fixed_point"])
@pytest.mark.parametrize("flood_mode", ["storage", "inflow"])
@pytest.mark.parametrize("overflow_mode", ["conservative", "matlab"])
def test_spill_outputs_and_balance_diagnostics_remain_exact(
    original, solver, flood_mode, overflow_mode,
):
    reference, _, _ = original
    params = parameters(initial_storage_m3=95.0, flood_mode=flood_mode)
    # Isolate the first spill event. Carrying its enhanced release into the
    # following day can legitimately create multiple inflow-mode roots.
    obs = observations([0.003] * 2, "2000-01-31")
    options = dict(solver=solver, overflow_mode=overflow_mode, max_iterations=9)
    actual = simulate_reservoir(obs, params, **options)
    expected = reference.simulate_reservoir(obs, params, **options)
    assert_simulations_equal(actual, expected)
    assert actual.frame.overflow_volume_m3.iloc[1:].gt(0).all()
    assert actual.frame.release_m3s.iloc[1:].gt(params.channel_capacity_m3s).all()


@pytest.mark.parametrize("iterations", [1, 2, 17, 1000])
@pytest.mark.parametrize("overflow_mode", ["conservative", "matlab"])
def test_fixed_iterations_preserve_negative_storage_and_issues(
    original, iterations, overflow_mode,
):
    reference, _, _ = original
    params = parameters(initial_storage_m3=1.0, initial_release_m3s=0.1)
    obs = observations([0.0] * 4)
    options = dict(solver="matlab_fixed_point", overflow_mode=overflow_mode,
                   max_iterations=iterations)
    actual = simulate_reservoir(obs, params, **options)
    expected = reference.simulate_reservoir(obs, params, **options)
    assert_simulations_equal(actual, expected)
    assert actual.frame.storage_m3.iloc[1:].lt(0).all()
    assert any("negative storage" in issue for issue in actual.issues)


def model_parameters():
    targets = pd.DataFrame(np.tile([200.0, 500.0, 800.0, 1.0, 4.0, 7.0], (12, 1)),
                           index=range(1, 13), columns=MONTHLY_TARGET_COLUMNS)
    return ReservoirParameters(targets, 1000.0, 350.0, 2.5, 10.0)


@pytest.mark.parametrize("arguments, changes, message", [
    ((50.0, 100.0, 0.0, 0.0, 1, 10.0), {}, "negative storage"),
    ((200.0, 0.1, 0.35, 0.35, 1, 1000.0), {}, "No feasible implicit"),
    ((750.0, 1.0, 1.0, 20.0, 1, 10.0), {"flood_mode": "inflow"}, "Multiple implicit"),
    ((750.0, 1.0, 8.0, 8.0, 1, 10.0), {"channel_capacity_m3s": 3.0}, "Multiple implicit"),
])
def test_public_implicit_solver_error_messages_are_identical(original, arguments, changes, message):
    _, reference, _ = original
    params = replace(model_parameters(), **changes)
    positional, dt = arguments[:5], arguments[5]
    errors = []
    for solver in (solve_implicit_step, reference.solve_implicit_step):
        with pytest.raises(ValueError, match=message) as caught:
            solver(*positional, params, dt_seconds=dt)
        errors.append(str(caught.value))
    assert errors[0] == errors[1]


@pytest.mark.parametrize("iterations", [1, 2, 3, 1000])
def test_public_fixed_iteration_storage_release_stagger_is_exact(original, iterations):
    _, reference, _ = original
    params = model_parameters()
    actual = fixed_point_step(350.0, 2.5, 5.0, 5.0, 1, params,
                              dt_seconds=10.0, max_iterations=iterations)
    expected = reference.fixed_point_step(350.0, 2.5, 5.0, 5.0, 1, params,
                                          dt_seconds=10.0, max_iterations=iterations)
    assert actual == expected
    if iterations == 1:
        assert actual == pytest.approx((373.125, 2.73125))


def test_new_call_observes_monthly_table_edits_and_other_candidates(original):
    reference, _, _ = original
    obs = observations([0.00025] * 6, "2000-01-01")
    first = parameters()
    first_before = first.monthly_targets.copy(deep=True)
    original_output = simulate_reservoir(obs, first)
    first.monthly_targets.loc[1, list(MONTHLY_TARGET_COLUMNS[3:])] *= 1.1
    edited_output = simulate_reservoir(obs, first)
    assert_simulations_equal(edited_output, reference.simulate_reservoir(obs, first))
    assert not original_output.frame.release_m3s.equals(edited_output.frame.release_m3s)
    alternate = replace(parameters(), monthly_targets=first_before)
    alternate.monthly_targets.loc[1, list(MONTHLY_TARGET_COLUMNS[:3])] *= 0.9
    alternate_before = alternate.monthly_targets.copy(deep=True)
    alternate_output = simulate_reservoir(obs, alternate)
    assert_simulations_equal(alternate_output, reference.simulate_reservoir(obs, alternate))
    assert_simulations_equal(simulate_reservoir(obs, first), edited_output)
    pd.testing.assert_frame_equal(alternate.monthly_targets, alternate_before, check_exact=True)


def test_unused_corrupted_month_is_not_prepared_eagerly(original):
    reference, _, _ = original
    params = parameters()
    # The immutable dataclass contains a mutable DataFrame. The historical
    # implementation reads only months actually used, even after this edit.
    params.monthly_targets.drop(index=6, inplace=True)
    obs = observations([0.00025] * 5, "2000-01-01")
    assert_simulations_equal(simulate_reservoir(obs, params), reference.simulate_reservoir(obs, params))


@pytest.mark.parametrize("start, count, used_months", [
    ("1999-12-31", 2, {12, 1}),
    ("1999-12-30", 400, set(range(1, 13))),
])
def test_each_encountered_month_is_read_once_and_cache_is_local_to_the_call(
    original, monkeypatch, start, count, used_months,
):
    reference, _, _ = original
    original_read = runner_module._monthly_values
    calls = []

    def counted(month, reservoir):
        calls.append(int(month))
        return original_read(month, reservoir)

    monkeypatch.setattr(runner_module, "_monthly_values", counted)
    params = parameters()
    obs = observations([0.00025] * count, start)
    actual = simulate_reservoir(obs, params)
    assert_simulations_equal(actual, reference.simulate_reservoir(obs, params))
    assert set(calls) == used_months
    assert len(calls) == len(used_months) <= 12
    assert all(calls.count(month) == 1 for month in used_months)
    calls.clear()
    simulate_reservoir(obs, params)
    assert set(calls) == used_months
    assert len(calls) == len(used_months)


@pytest.mark.parametrize("arguments, changes, message", [
    ((50.0, 100.0, 0.0, 0.0, 10.0), {}, "negative storage"),
    ((200.0, 0.1, 0.35, 0.35, 1000.0), {}, "No feasible implicit"),
    ((750.0, 1.0, 1.0, 20.0, 10.0), {"flood_mode": "inflow"}, "Multiple implicit"),
    ((750.0, 1.0, 8.0, 8.0, 10.0), {"channel_capacity_m3s": 3.0}, "Multiple implicit"),
])
def test_daily_solver_preserves_dated_discontinuity_and_negative_storage_errors(
    original, arguments, changes, message,
):
    reference, _, _ = original
    storage, release, previous_inflow, inflow, reference_dt = arguments
    scale = reference_dt / 86400.0
    base = replace(model_parameters(), **changes)
    targets = base.monthly_targets.copy(deep=True)
    targets.loc[:, list(MONTHLY_TARGET_COLUMNS[3:])] *= scale
    params = replace(base, monthly_targets=targets, initial_storage_m3=storage,
                     initial_release_m3s=release * scale,
                     channel_capacity_m3s=base.channel_capacity_m3s * scale)
    obs = observations([previous_inflow * scale, inflow * scale], "2000-01-31")
    errors = []
    for simulate in (simulate_reservoir, reference.simulate_reservoir):
        with pytest.raises(ValueError, match=message) as caught:
            simulate(obs, params)
        errors.append(str(caught.value))
    assert errors[0] == errors[1]
    assert errors[0].startswith("Simulation failed on 2000-02-01:")


def search_problem():
    obs = observations([0.00025] * 15, "1999-12-27")
    period = DatePeriod("1999-12-29", "2000-01-08")
    template = parameters()
    # Overlap is deliberate: box-valid trials can fail physical target order.
    # Those rejections must consume the same budget and preserve RNG lineage.
    lower = np.tile([25.0, 45.0, 70.0, 0.00005, 0.00015, 0.00025], (12, 1))
    upper = np.tile([55.0, 75.0, 95.0, 0.00025, 0.00035, 0.00045], (12, 1))
    bounds = CalibrationBounds(pd.DataFrame(np.c_[lower, upper], index=range(1, 13),
                                           columns=BOUND_COLUMNS),
                               template.capacity_m3, template.dead_storage_fraction, period)
    vector = template.monthly_targets.to_numpy(copy=True)
    vector[:, :3] /= template.capacity_m3
    baseline = vector.T.reshape(72)
    objective = CalibrationObjective(obs, period, template, warmup_steps=2, bounds=bounds)
    return objective, baseline


def assert_trials_equal(actual, expected):
    assert [(item.evaluation_id, item.generation, item.objectives, item.feasible, item.reason)
            for item in actual] == [
                (item.evaluation_id, item.generation, item.objectives, item.feasible, item.reason)
                for item in expected]
    np.testing.assert_array_equal(np.stack([item.vector for item in actual]),
                                  np.stack([item.vector for item in expected]))


def assert_search_equal(actual, expected):
    assert actual.settings == expected.settings
    assert actual.history == expected.history
    assert actual.termination == expected.termination
    assert_trials_equal(actual.evaluated, expected.evaluated)
    assert_trials_equal(actual.final_population, expected.final_population)
    assert_trials_equal(actual.front, expected.front)


@pytest.mark.parametrize("seed", [23, 73])
def test_seeded_search_costs_rejections_survivors_history_and_rng_match_original(
    original, monkeypatch, seed,
):
    reference, _, _ = original
    settings = NSGA2Settings(seed=seed, population_size=4, evaluation_budget=24)
    current, baseline = search_problem()
    current_states = []
    actual = optimize_nsga2(current, baseline, settings, checkpoint_callback=current_states.append)
    old, old_baseline = search_problem()
    old_states = []
    with monkeypatch.context() as context:
        context.setattr(objective_module, "simulate_reservoir", reference.simulate_reservoir)
        expected = optimize_nsga2(old, old_baseline, settings, checkpoint_callback=old_states.append)
    assert objective_module.simulate_reservoir is simulate_reservoir
    assert_search_equal(actual, expected)
    assert any(trial.feasible for trial in actual.evaluated)
    assert any(not trial.feasible for trial in actual.evaluated)
    assert [state.evaluations for state in current_states] == [state.evaluations for state in old_states]
    for current_state, old_state in zip(current_states, old_states):
        assert current_state.population_ids == old_state.population_ids
        assert current_state.rng_state == old_state.rng_state


def test_same_version_checkpoint_resume_reuses_saved_costs_and_matches_uninterrupted(tmp_path):
    settings = NSGA2Settings(seed=73, population_size=4, evaluation_budget=24)
    full_objective, baseline = search_problem()
    full_states = []
    expected = optimize_nsga2(full_objective, baseline, settings, checkpoint_callback=full_states.append)
    staged_objective, staged_baseline = search_problem()
    staged_states = []
    prefix = optimize_nsga2(staged_objective, staged_baseline, replace(settings, evaluation_budget=12),
                           checkpoint_callback=staged_states.append)
    identity = {"experiment": "monthly-preparation-parity", "version": "same-code-version"}
    checkpoint = tmp_path / "optimizer.npz"
    save_optimizer_checkpoint(checkpoint, staged_states[-1], identity)
    state = load_optimizer_checkpoint(checkpoint, identity)
    objective, resumed_baseline = search_problem()
    original_evaluate = objective.evaluate
    calls = []

    def counted(vector):
        calls.append(np.asarray(vector).copy())
        return original_evaluate(vector)

    objective.evaluate = counted
    resumed_states = []
    actual = optimize_nsga2(objective, resumed_baseline, settings,
                            resume_state=state, checkpoint_callback=resumed_states.append)
    assert prefix.evaluations == state.evaluations == 12
    assert len(calls) == 12
    np.testing.assert_array_equal(calls, np.stack([trial.vector for trial in expected.evaluated[12:]]))
    assert_search_equal(actual, expected)
    assert resumed_states[-1].population_ids == full_states[-1].population_ids
    assert resumed_states[-1].rng_state == full_states[-1].rng_state
