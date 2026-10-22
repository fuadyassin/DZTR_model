"""Independent hypervolume geometry and honest nested calibration checkpoints."""

from types import SimpleNamespace

import numpy as np
import pytest

from dztr.calibration.convergence import (
    checkpoint_metrics,
    hypervolume_2d,
    normalize_parameter_vectors,
    normalized_parameter_spread,
    prefix_pareto_front,
)
from dztr.calibration.optimizer import EvaluatedCandidate, NSGA2Settings, optimize_nsga2


def point(identifier, costs, *, feasible=True):
    return EvaluatedCandidate(evaluation_id=identifier, generation=0, vector=np.zeros(72),
                              objectives=tuple(costs), feasible=feasible,
                              reason=None if feasible else "Rejected synthetic candidate")


@pytest.mark.parametrize("costs,expected", [
    ([], 0.0),
    ([[0, 0]], 1.0),
    ([[0, 0.5], [0.5, 0]], 0.75),
    ([[0, 0.5], [0.5, 0], [0.8, 0.8]], 0.75),
    ([[0, 0.5], [0.5, 0], [0, 0.5]], 0.75),
    ([[-1, 0.5], [0.5, -1]], 1.75),
    ([[1, 0], [0, 1], [2, -2], [-2, 2]], 0.0),
    ([[np.nan, 0], [0, np.inf], [-np.inf, 0], [0, 0]], 1.0),
])
def test_fixed_reference_hypervolume_matches_known_union_of_rectangles(costs, expected):
    assert hypervolume_2d(costs, reference=(1, 1)) == pytest.approx(expected)


def test_hypervolume_is_order_independent_and_respects_both_minimization_coordinates():
    costs = [[0, 0.5], [0.5, 0], [0.75, 0.75]]
    assert hypervolume_2d(costs) == pytest.approx(hypervolume_2d(costs[::-1]))
    assert hypervolume_2d([[0.5, 0.5]]) == pytest.approx(0.25)
    assert hypervolume_2d([[0.25, 0.5]]) == pytest.approx(0.375)
    assert hypervolume_2d([[0.5, 0.25]]) == pytest.approx(0.375)
    # Explicit reference is fixed externally, not recomputed from each front.
    assert hypervolume_2d([[0, 0]], reference=(2, 3)) == pytest.approx(6)


@pytest.mark.parametrize("reference", [(1,), (1, 1, 1), (np.nan, 1), (np.inf, 1)])
def test_invalid_hypervolume_reference_is_rejected(reference):
    with pytest.raises(ValueError):
        hypervolume_2d([[0, 0]], reference=reference)


def checkpoint_records():
    return (point(0, (-1, 0.4)), point(1, (-20, -20), feasible=False),
            point(2, (-0.8, -0.5)), point(3, (2, 2)), point(4, (-0.4, -0.9)))


def test_prefix_front_counts_evaluation_ids_and_filters_rejections_and_dominance():
    records = checkpoint_records()
    shuffled = (records[4], records[0], records[3], records[1], records[2])
    front = prefix_pareto_front(shuffled, 3)
    assert [candidate.evaluation_id for candidate in front] == [0, 2]
    assert all(candidate.feasible for candidate in front)
    complete = prefix_pareto_front(shuffled, 5)
    assert [candidate.evaluation_id for candidate in complete] == [0, 2, 4]
    # Independent strict dominance definition across the actual five-call prefix.
    feasible = [candidate for candidate in records if candidate.feasible]
    expected = {
        candidate.evaluation_id for candidate in feasible
        if not any(all(a <= b for a, b in zip(other.objectives, candidate.objectives))
                   and any(a < b for a, b in zip(other.objectives, candidate.objectives))
                   for other in feasible if other.evaluation_id != candidate.evaluation_id)
    }
    assert {candidate.evaluation_id for candidate in complete} == expected


def test_checkpoint_diagnostics_use_calibration_costs_and_allow_empty_feasible_prefixes():
    records = checkpoint_records()
    metrics = checkpoint_metrics(records, 5, reference=(1, 1))
    assert metrics["evaluations"] == 5
    assert metrics["feasible_count"] == 4
    assert metrics["rejected_count"] == 1
    assert metrics["front_size"] == 3
    assert metrics["max_release_nse"] == pytest.approx(1)
    assert metrics["max_storage_nse"] == pytest.approx(0.9)
    assert metrics["representative_ids"] == {"best_release": 0, "best_storage": 4, "balanced": 2}
    assert metrics["balanced_release_nse"] == pytest.approx(0.8)
    assert metrics["balanced_storage_nse"] == pytest.approx(0.5)
    # Three nonoverlapping vertical strips: widths0.2,0.4,1.4 and heights0.6,1.5,1.9.
    assert metrics["hypervolume"] == pytest.approx(0.2 * 0.6 + 0.4 * 1.5 + 1.4 * 1.9)
    np.testing.assert_array_equal(metrics["reference"], [1, 1])
    rejected = (point(0, (np.inf, np.inf), feasible=False),)
    empty = checkpoint_metrics(rejected, 1)
    assert empty["front_size"] == empty["feasible_count"] == 0
    assert empty["rejected_count"] == 1
    assert empty["hypervolume"] == 0
    assert empty["representative_ids"] is None
    assert empty["max_release_nse"] is None
    assert empty["max_storage_nse"] is None


def test_checkpoint_reports_tradeoff_points_outside_fixed_hypervolume_reference():
    records = (point(0, (0, 0.5)), point(1, (2, -0.2)))
    metrics = checkpoint_metrics(records, 2, reference=(1, 1))
    assert metrics["front_size"] == 2
    assert metrics["front_points_outside_reference"] == 1
    assert metrics["hypervolume"] == pytest.approx(0.5)


def test_nonfinite_claimed_feasible_record_counts_as_rejected_without_becoming_pareto():
    metrics = checkpoint_metrics((point(0, (np.nan, 0)),), 1)
    assert metrics["evaluations"] == metrics["rejected_count"] == 1
    assert metrics["feasible_count"] == metrics["front_size"] == 0
    assert metrics["hypervolume"] == 0


@pytest.mark.parametrize("checkpoint", [0, -1, 6, True, 2.5])
def test_invalid_checkpoint_counts_are_rejected(checkpoint):
    with pytest.raises(ValueError):
        prefix_pareto_front(checkpoint_records(), checkpoint)


@pytest.mark.parametrize("records", [
    (point(0, (0, 1)), point(0, (1, 0))),
    (point(0, (0, 1)), point(2, (1, 0))),
    (point(1, (0, 1)), point(2, (1, 0))),
])
def test_duplicate_or_noncontiguous_ids_cannot_misstate_evaluation_counts(records):
    with pytest.raises(ValueError):
        prefix_pareto_front(records, 1)


def mixed_unit_bounds(release_scale=1):
    baseline = np.repeat([0.2, 0.5, 0.8, 1, 2, 3], 12).astype(float)
    baseline[36:] *= release_scale
    lower, upper = baseline.copy(), baseline.copy()
    lower[0], upper[0] = 0.18, 0.22
    lower[36], upper[36] = 0.8 * release_scale, 1.2 * release_scale
    return SimpleNamespace(lower_vector=lower, upper_vector=upper), baseline


def test_normalized_spread_is_invariant_to_release_units_and_fixed_genes_have_zero_spread():
    bounds, baseline = mixed_unit_bounds()
    vectors = np.stack([bounds.lower_vector, baseline, bounds.upper_vector])
    normalized = normalize_parameter_vectors(vectors, bounds)
    active = bounds.upper_vector > bounds.lower_vector
    np.testing.assert_allclose(normalized[:, active], [[0, 0], [0.5, 0.5], [1, 1]], atol=1e-12)
    np.testing.assert_array_equal(normalized[:, ~active], np.zeros((3, 70)))
    spread = normalized_parameter_spread(vectors, bounds)
    np.testing.assert_array_equal(spread[active], [1, 1])
    np.testing.assert_array_equal(spread[~active], np.zeros(70))
    scaled_bounds, _ = mixed_unit_bounds(release_scale=1e6)
    scaled_vectors = vectors.copy()
    scaled_vectors[:, 36:] *= 1e6
    np.testing.assert_allclose(normalize_parameter_vectors(scaled_vectors, scaled_bounds), normalized, atol=1e-12)
    np.testing.assert_allclose(normalized_parameter_spread(scaled_vectors, scaled_bounds), spread, atol=1e-12)


@pytest.mark.parametrize("invalid", ["bare_vector", "nonfinite", "outside", "changed_fixed_gene"])
def test_parameter_normalization_rejects_shape_nonfinite_or_outside_bounds_without_clipping(invalid):
    bounds, baseline = mixed_unit_bounds()
    vectors = baseline[None, :].copy()
    if invalid == "bare_vector":
        vectors = baseline
    elif invalid == "nonfinite":
        vectors[0, 0] = np.nan
    elif invalid == "outside":
        vectors[0, 0] = 0.3
    else:
        vectors[0, 1] += 0.01
    with pytest.raises(ValueError):
        normalize_parameter_vectors(vectors, bounds)


class ToyProblem:
    def __init__(self):
        self.bounds, self.baseline = mixed_unit_bounds()
        self.calls = 0

    def evaluate(self, vector):
        self.calls += 1
        unit = normalize_parameter_vectors(np.asarray(vector)[None, :], self.bounds)[0]
        return SimpleNamespace(feasible=True, reason=None,
                               objectives=(unit[0] ** 2 + 0.1 * unit[36] ** 2,
                                           (1 - unit[0]) ** 2 + 0.1 * unit[36] ** 2))

    def evaluate_validation(self, *args, **kwargs):
        raise AssertionError("Convergence diagnostics cannot inspect validation.")


def test_nested_96_384_1024_checkpoints_match_standalone_search_prefixes_for_shared_seed():
    checkpoints, population_size = (96, 384, 1024), 16
    assert all(checkpoint % population_size == 0 for checkpoint in checkpoints)
    problem = ToyProblem()
    maximum = optimize_nsga2(problem, problem.baseline, NSGA2Settings(seed=42, population_size=population_size,
                                                                     evaluation_budget=max(checkpoints)))
    assert problem.calls == maximum.evaluations == 1024
    assert set(checkpoints).issubset({record.evaluations for record in maximum.history})
    volumes = []
    for checkpoint in checkpoints:
        diagnostics = checkpoint_metrics(maximum.evaluated, checkpoint, reference=(1, 1))
        assert diagnostics["evaluations"] == diagnostics["feasible_count"] == checkpoint
        volumes.append(diagnostics["hypervolume"])
        if checkpoint == max(checkpoints):
            continue
        smaller_problem = ToyProblem()
        standalone = optimize_nsga2(smaller_problem, smaller_problem.baseline,
                                   NSGA2Settings(seed=42, population_size=population_size, evaluation_budget=checkpoint))
        np.testing.assert_array_equal(np.stack([candidate.vector for candidate in maximum.evaluated[:checkpoint]]),
                                      np.stack([candidate.vector for candidate in standalone.evaluated]))
        np.testing.assert_array_equal([candidate.objectives for candidate in maximum.evaluated[:checkpoint]],
                                      [candidate.objectives for candidate in standalone.evaluated])
        prefix_ids = {candidate.evaluation_id for candidate in prefix_pareto_front(maximum.evaluated, checkpoint)}
        assert prefix_ids == {candidate.evaluation_id for candidate in standalone.front}
    assert all(after >= before - 1e-12 for before, after in zip(volumes, volumes[1:]))
