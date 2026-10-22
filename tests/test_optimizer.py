"""Independent Pareto definitions, counted budgets and normalized search tests."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dztr.calibration.objectives import CandidateEvaluation
from dztr.calibration.optimizer import NSGA2Settings, crowding_distance, nondominated_sort, optimize_nsga2
from dztr.calibration.periods import DatePeriod


PERIOD = DatePeriod("2000-01-01", "2000-01-05")


def candidate_box(release_scale=1.0):
    # Native units: 36 storage fractions, then 36 releases. Most targets are
    # fixed, with two search dimensions of very different numerical scales.
    baseline = np.repeat([0.2, 0.5, 0.8, 1.0, 2.0, 3.0], 12).astype(float)
    baseline[36:] *= release_scale
    lower, upper = baseline.copy(), baseline.copy()
    lower[0], upper[0] = 0.18, 0.22
    lower[36], upper[36] = 0.8 * release_scale, 1.2 * release_scale
    return SimpleNamespace(lower_vector=lower, upper_vector=upper), baseline


class SyntheticObjective:
    """Two costs with explicit tradeoffs; validation must never be requested."""

    def __init__(self, *, release_scale=1.0, reject_every=None, all_rejected=False, costs=None, failure=None):
        self.bounds, self.baseline = candidate_box(release_scale)
        self.calls = []
        self.reject_every = reject_every
        self.all_rejected = all_rejected
        self.costs = costs
        self.failure = failure

    def normalized(self, vector):
        lower, upper = self.bounds.lower_vector, self.bounds.upper_vector
        width = upper - lower
        return np.divide(vector - lower, width, out=np.zeros(72), where=width != 0)

    def evaluate(self, vector):
        self.calls.append(np.asarray(vector).copy())
        if self.failure is not None and len(self.calls) == 3:
            raise self.failure("Unexpected synthetic evaluation failure")
        unit = self.normalized(vector)
        rejected = self.all_rejected or (self.reject_every is not None and len(self.calls) % self.reject_every == 0)
        if rejected:
            return CandidateEvaluation(False, (np.inf, np.inf), {}, {}, "Synthetic rejected candidate", None, None, PERIOD)
        costs = (unit[0] ** 2 + 0.1 * unit[36] ** 2,
                 (1 - unit[0]) ** 2 + 0.1 * unit[36] ** 2) if self.costs is None else self.costs(unit)
        return CandidateEvaluation(True, tuple(costs), {}, {"release": 5, "storage": 5}, None, None, None, PERIOD)

    def evaluate_validation(self, *args, **kwargs):
        raise AssertionError("An optimizer must never inspect held-out validation.")


def settings(**changes):
    options = {"seed": 73, "population_size": 8, "evaluation_budget": 19}
    options.update(changes)
    return NSGA2Settings(**options)


def independent_nondominated_costs(costs):
    values = np.asarray(costs, dtype=float)
    return {
        tuple(row)
        for i, row in enumerate(values)
        if not any(np.all(other <= row) and np.any(other < row) for j, other in enumerate(values) if j != i)
    }


def test_nondominated_sort_minimizes_both_costs_and_preserves_duplicate_tradeoffs():
    costs = np.array([[0, 3], [1, 2], [2, 1], [3, 0], [2, 3], [3, 3], [1, 2]], dtype=float)
    fronts = nondominated_sort(costs)
    assert [set(front.tolist()) for front in fronts] == [{0, 1, 2, 3, 6}, {4}, {5}]
    negative_costs = np.array([[-1, -0.5], [-0.8, -0.8], [-0.5, -1], [0, 0]])
    assert set(nondominated_sort(negative_costs)[0].tolist()) == {0, 1, 2}


def test_crowding_preserves_both_endpoints_and_normalizes_objective_scales():
    costs = np.array([[-1, 0], [-0.75, -0.25], [-0.5, -0.5], [-0.25, -0.75], [0, -1]])
    distances = crowding_distance(costs)
    assert np.isinf(distances[[0, 4]]).all()
    np.testing.assert_allclose(distances[1:4], [1, 1, 1])
    transformed = costs * np.array([1e6, 0.01]) + np.array([10, -50])
    np.testing.assert_allclose(crowding_distance(transformed), distances, atol=1e-10)
    assert np.isinf(crowding_distance(costs[:1])).all()


def test_optimizer_uses_exact_budget_including_partial_generation_and_baseline():
    objective, history = SyntheticObjective(), []
    before = objective.baseline.copy()
    result = optimize_nsga2(objective, objective.baseline, settings(), progress_callback=history.append)
    assert len(objective.calls) == result.evaluations == 19
    assert len(result.evaluated) == 19
    assert [record.evaluations for record in result.history] == [8, 16, 19]
    assert [record.generation for record in result.history] == [0, 1, 2]
    assert history == list(result.history)
    np.testing.assert_array_equal(objective.calls[0], before)
    np.testing.assert_array_equal(result.evaluated[0].vector, before)
    np.testing.assert_array_equal(objective.baseline, before)
    assert len(result.final_population) == 8
    assert result.termination


def test_search_is_reproducible_with_same_seed_and_changes_with_different_seed():
    first, second, third = SyntheticObjective(), SyntheticObjective(), SyntheticObjective()
    one = optimize_nsga2(first, first.baseline, settings())
    two = optimize_nsga2(second, second.baseline, settings())
    three = optimize_nsga2(third, third.baseline, settings(seed=74))
    np.testing.assert_array_equal(np.stack(first.calls), np.stack(second.calls))
    np.testing.assert_array_equal([item.objectives for item in one.evaluated], [item.objectives for item in two.evaluated])
    assert not np.array_equal(np.stack(first.calls[1:]), np.stack(third.calls[1:]))
    assert one.evaluations == two.evaluations == three.evaluations


def test_mixed_unit_rescaling_does_not_change_normalized_search_and_fixed_variables_stay_fixed():
    ordinary, scaled = SyntheticObjective(), SyntheticObjective(release_scale=1e6)
    first = optimize_nsga2(ordinary, ordinary.baseline, settings())
    second = optimize_nsga2(scaled, scaled.baseline, settings())
    ordinary_unit = np.stack([ordinary.normalized(vector) for vector in ordinary.calls])
    scaled_unit = np.stack([scaled.normalized(vector) for vector in scaled.calls])
    np.testing.assert_allclose(ordinary_unit, scaled_unit, rtol=0, atol=1e-12)
    np.testing.assert_allclose([item.objectives for item in first.evaluated],
                               [item.objectives for item in second.evaluated], atol=1e-12)
    for objective in (ordinary, scaled):
        calls = np.stack(objective.calls)
        lower, upper = objective.bounds.lower_vector, objective.bounds.upper_vector
        assert (calls >= lower).all() and (calls <= upper).all()
        fixed = lower == upper
        np.testing.assert_array_equal(calls[:, fixed], np.tile(lower[fixed], (len(calls), 1)))
        assert np.std(calls[:, 0]) > 0
        assert np.std(calls[:, 36]) > 0


def test_archive_matches_independent_nondominated_definition_over_all_feasible_calls():
    objective = SyntheticObjective(reject_every=3)
    result = optimize_nsga2(objective, objective.baseline, settings(evaluation_budget=35))
    assert any(not candidate.feasible for candidate in result.evaluated)
    assert all(candidate.feasible and np.isfinite(candidate.objectives).all() for candidate in result.front)
    feasible = [candidate.objectives for candidate in result.evaluated if candidate.feasible]
    actual = {tuple(candidate.objectives) for candidate in result.front}
    assert actual == independent_nondominated_costs(feasible)
    for candidate in result.evaluated:
        if not candidate.feasible:
            assert candidate.reason == "Synthetic rejected candidate"


def test_two_opposing_costs_remain_a_tradeoff_front_without_scalarization():
    objective = SyntheticObjective(costs=lambda unit: (-unit[0], -(1 - unit[0])))
    result = optimize_nsga2(objective, objective.baseline, settings(evaluation_budget=35))
    unique_evaluated_costs = {tuple(candidate.objectives) for candidate in result.evaluated}
    unique_front_costs = {tuple(candidate.objectives) for candidate in result.front}
    assert len(unique_front_costs) > 8
    assert unique_front_costs == unique_evaluated_costs
    for candidate in result.front:
        assert len(candidate.objectives) == 2
        assert sum(candidate.objectives) == pytest.approx(-1)


def test_all_rejected_search_raises_instead_of_returning_an_infinite_front():
    objective = SyntheticObjective(all_rejected=True)
    with pytest.raises(ValueError, match="feasible"):
        optimize_nsga2(objective, objective.baseline, settings())


def test_fully_fixed_space_stops_explicitly_without_duplicate_evaluations():
    objective = SyntheticObjective()
    objective.bounds = SimpleNamespace(lower_vector=objective.baseline.copy(), upper_vector=objective.baseline.copy())
    result = optimize_nsga2(objective, objective.baseline, settings())
    assert result.termination == "duplicate_exhausted"
    assert result.evaluations == len(objective.calls) == 1
    assert len(result.front) == 1
    np.testing.assert_array_equal(result.evaluated[0].vector, objective.baseline)


def test_declared_rejections_and_nonfinite_claimed_feasible_costs_cannot_enter_front():
    class BrokenCostsObjective(SyntheticObjective):
        def evaluate(self, vector):
            evaluated = super().evaluate(vector)
            if len(self.calls) % 3 == 0:
                return replace(evaluated, feasible=False, objectives=(-1e12, -1e12), reason="Declared rejection")
            if len(self.calls) % 3 == 2:
                return replace(evaluated, objectives=(np.nan, 0))
            return evaluated

    objective = BrokenCostsObjective()
    result = optimize_nsga2(objective, objective.baseline, settings())
    rejected = [candidate for candidate in result.evaluated if not candidate.feasible]
    assert rejected
    assert all(candidate.objectives == (np.inf, np.inf) and candidate.reason for candidate in rejected)
    assert all(candidate.feasible and np.isfinite(candidate.objectives).all() for candidate in result.front)
    assert {tuple(candidate.objectives) for candidate in result.front} == independent_nondominated_costs(
        [candidate.objectives for candidate in result.evaluated if candidate.feasible]
    )


def test_malformed_objective_cost_shape_raises_without_spending_remaining_budget():
    objective = SyntheticObjective(costs=lambda unit: (0, 1, 2))
    with pytest.raises(ValueError, match="exactly two"):
        optimize_nsga2(objective, objective.baseline, settings())
    assert len(objective.calls) == 1


@pytest.mark.parametrize("changes", [
    {"seed": -1}, {"seed": True},
    {"population_size": 0}, {"population_size": True}, {"population_size": 8.5},
    {"evaluation_budget": 7}, {"evaluation_budget": 19.5},
    {"crossover_probability": -0.1}, {"crossover_probability": 1.1},
    {"crossover_eta": 0}, {"crossover_eta": np.nan}, {"mutation_eta": -1},
    {"local_initialization_fraction": -0.1}, {"local_initialization_fraction": 1.1},
    {"local_initialization_scale": -1},
])
def test_invalid_static_search_settings_raise(changes):
    with pytest.raises(ValueError):
        settings(**changes)


@pytest.mark.parametrize("invalid_baseline", ["short", "nonfinite", "outside"])
def test_bad_baseline_is_rejected_before_any_objective_calls(invalid_baseline):
    objective = SyntheticObjective()
    baseline = objective.baseline.copy()
    if invalid_baseline == "short":
        baseline = baseline[:-1]
    elif invalid_baseline == "nonfinite":
        baseline[0] = np.inf
    else:
        baseline[0] = 0.3
    with pytest.raises(ValueError):
        optimize_nsga2(objective, baseline, settings())
    assert objective.calls == []


@pytest.mark.parametrize("error", [RuntimeError, ValueError])
def test_unexpected_evaluation_exceptions_are_not_swallowed(error):
    objective = SyntheticObjective(failure=error)
    with pytest.raises(error, match="Unexpected synthetic evaluation failure"):
        optimize_nsga2(objective, objective.baseline, settings())
    assert len(objective.calls) == 3
