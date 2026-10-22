"""Calibration-only representative rules, ties and nondominated input checks."""

import numpy as np
import pytest

from dztr.calibration.optimizer import EvaluatedCandidate
from dztr.calibration.selection import select_pareto_representatives


def point(identifier, costs, *, feasible=True):
    return EvaluatedCandidate(evaluation_id=identifier, generation=0, vector=np.zeros(72),
                              objectives=tuple(costs), feasible=feasible, reason=None)


def test_representatives_match_two_minima_and_independent_normalized_ideal_distance():
    candidates = (point(10, (-0.4, -0.9)), point(8, (-1.0, 0.4)), point(2, (-0.8, -0.5)))
    selection = select_pareto_representatives(candidates)
    assert selection["best_release"] == 8
    assert selection["best_storage"] == 10
    costs = np.array([candidate.objectives for candidate in candidates])
    ideal = costs.min(axis=0)
    normalized = (costs - ideal) / (costs.max(axis=0) - ideal)
    independently_balanced = candidates[np.argmin(np.sqrt(np.sum(normalized ** 2, axis=1)))].evaluation_id
    assert selection["balanced"] == independently_balanced == 2
    # Objective-specific affine unit changes must not move the balanced role.
    scaled = tuple(point(candidate.evaluation_id, np.array(candidate.objectives) * [1e6, 0.01] + [10, -50])
                   for candidate in candidates)
    assert select_pareto_representatives(scaled) == selection


def test_identical_costs_have_zero_span_and_use_id_to_break_remaining_ties():
    candidates = (point(9, (-0.7, -0.8)), point(2, (-0.7, -0.8)), point(7, (-0.7, -0.8)))
    assert select_pareto_representatives(candidates) == {"best_release": 2, "best_storage": 2, "balanced": 2}


def test_single_front_member_can_fill_all_roles_without_division_by_zero():
    assert select_pareto_representatives((point(7, (-0.9, 0.3)),)) == {
        "best_release": 7, "best_storage": 7, "balanced": 7,
    }


def test_balanced_distance_ties_use_release_then_storage_before_id():
    candidates = (point(1, (1, 0)), point(12, (0, 1)))
    assert select_pareto_representatives(candidates) == {"best_release": 12, "best_storage": 1, "balanced": 12}


@pytest.mark.parametrize("candidates", [
    (),
    (point(1, (np.inf, np.inf), feasible=False),),
    (point(1, (np.nan, 0)),),
    (point(1, (0,)),),
    (point(1, (0, 0)), point(2, (1, 1))),
    (point(1, (0, 1)), point(1, (1, 0))),
])
def test_empty_infeasible_nonfinite_dominated_or_duplicate_id_lists_are_rejected(candidates):
    with pytest.raises(ValueError):
        select_pareto_representatives(candidates)
