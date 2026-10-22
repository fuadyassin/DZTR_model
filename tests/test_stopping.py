"""Independent archive checks for an offline, calibration-only stopping policy."""

from dataclasses import replace

import numpy as np
import pytest

from dztr.calibration.optimizer import EvaluatedCandidate
from dztr.calibration.stopping import StoppingPolicy, assess_stopping


def policy(**changes):
    settings = dict(population_size=4, check_interval=4, window_evaluations=4,
                    minimum_evaluations=8, patience=2, max_evaluations=20,
                    normalized_hv_tolerance=0.001, front_tolerance=0.01,
                    extrema_tolerance=0.01)
    settings.update(changes)
    return StoppingPolicy(**settings)


def archive(costs, *, rejected=()):
    rejected = set(rejected)
    return tuple(EvaluatedCandidate(
        evaluation_id=i, generation=i // 4,
        # Parameter spread deliberately grows while objective coordinates may plateau.
        vector=np.arange(72, dtype=float) + i * 1000.0,
        objectives=tuple(pair), feasible=i not in rejected,
        reason="Synthetic solver rejection" if i in rejected else None,
    ) for i, pair in enumerate(costs))


def checked(assessment, count):
    return next(row for row in assessment.history if row["evaluations"] == count)


def test_stable_objectives_need_minimum_budget_and_consecutive_windows_not_stable_parameters():
    records = archive([(0.1, 0.2)] * 12)
    assert assess_stopping(records[:4], policy()).status == "insufficient_history"
    first = assess_stopping(records[:8], policy())
    assert first.status == "continue_search"
    assert first.suggested_stop_evaluations is None
    stopped = assess_stopping(records, policy())
    assert stopped.status == "stopping_candidate"
    assert stopped.suggested_stop_evaluations == 12
    assert stopped.evaluations == 12
    assert checked(stopped, 8)["consecutive_stable_windows"] == 1
    assert checked(stopped, 12)["consecutive_stable_windows"] == 2
    assert checked(stopped, 12)["front_movement"] == pytest.approx(0)
    assert checked(stopped, 12)["normalized_hv_gain"] == pytest.approx(0)
    assert checked(stopped, 12)["newly_feasible_count"] == 4
    assert np.ptp(np.stack([row.vector for row in records]), axis=0).min() > 10000


def test_minimum_budget_does_not_count_stable_windows_before_it():
    records = archive([(0.1, 0.2)] * 16)
    settings = policy(minimum_evaluations=12)
    first = assess_stopping(records[:12], settings)
    assert first.status == "continue_search"
    assert checked(first, 12)["consecutive_stable_windows"] == 1
    assert assess_stopping(records, settings).suggested_stop_evaluations == 16


def test_flat_hypervolume_cannot_hide_a_continuing_storage_extreme_improvement():
    # A rectangle at x=reference contributes zero area, even while its y extreme improves.
    initial = [(0.9, 0.0), (1.0, -0.5), (0.95, 0.5), (1.0, 0.5)]
    records = archive(initial + [(1.0, -0.9)] + [(0.95, 0.5)] * 7)
    result = assess_stopping(records, policy())
    change = checked(result, 8)
    assert change["normalized_hv_gain"] == pytest.approx(0, abs=1e-14)
    assert change["extrema_drift"] == pytest.approx(0.4)
    assert not change["stable_window"]
    assert result.status == "continue_search"
    assert result.suggested_stop_evaluations is None


def test_nearly_flat_hypervolume_and_unchanged_extrema_cannot_hide_front_movement():
    initial = [(-1.0, 1.0), (0.0, 0.0), (1.0, -1.0), (0.5, 0.5)]
    # Adds only .5 * .00001 area; neither minimum objective changes.
    records = archive(initial + [(-0.5, 0.99999)] + [(0.5, 0.5)] * 7)
    result = assess_stopping(records, policy())
    change = checked(result, 8)
    assert change["normalized_hv_gain"] == pytest.approx(0.000005 / 4)
    assert change["extrema_drift"] == pytest.approx(0)
    # Independent nearest point: new(-.5,.99999) to old(-1,1), in fixed NSE units.
    assert change["front_movement"] == pytest.approx(0.5)
    assert not change["stable_window"]
    assert result.status == "continue_search"


def test_custom_reference_uses_its_fixed_theoretical_area_and_raw_front_coordinates():
    records = archive([(0.0, 0.5)] * 4 + [(-0.2, 0.5)] * 8)
    result = assess_stopping(records, policy(
        hypervolume_reference=(2.0, 3.0), normalized_hv_tolerance=0.05,
        front_tolerance=0.3, extrema_tolerance=0.3,
    ))
    change = checked(result, 8)
    # HV changes (2-(-.2))*(3-.5) - (2-0)*(3-.5) = .5.
    assert change["normalized_hv_gain"] == pytest.approx(0.5 / ((2 + 1) * (3 + 1)))
    assert change["front_movement"] == pytest.approx(0.2)
    assert change["extrema_drift"] == pytest.approx(0.2)
    assert result.status == "stopping_candidate"


def test_hypervolume_gain_equal_to_tolerance_is_not_a_stable_window():
    records = archive([(0.0, 0.0)] * 4 + [(-0.25, 0.0)] * 8)
    result = assess_stopping(records, policy(
        normalized_hv_tolerance=0.0625, front_tolerance=0.5, extrema_tolerance=0.5,
    ))
    assert checked(result, 8)["normalized_hv_gain"] == pytest.approx(0.0625)
    assert not checked(result, 8)["stable_window"]
    assert result.status == "continue_search"


@pytest.mark.parametrize("invalid_costs,rejected", [
    ([(np.inf, np.inf)] * 4, range(4, 8)),
    ([(np.nan, 0.1)] * 4, ()),
])
def test_a_window_of_failed_trials_does_not_advance_plateau_patience(invalid_costs, rejected):
    records = archive([(0.1, 0.2)] * 4 + invalid_costs + [(0.1, 0.2)] * 4,
                      rejected=rejected)
    result = assess_stopping(records, policy())
    assert checked(result, 8)["newly_feasible_count"] == 0
    assert not checked(result, 8)["stable_window"]
    assert result.status == "continue_search"
    assert result.suggested_stop_evaluations is None


@pytest.mark.parametrize("pair,rejected", [
    ((np.inf, np.inf), range(12)),
    ((np.nan, 0.1), ()),
    ((1.1, -0.5), ()),
])
def test_unusable_or_reference_excluded_fronts_cannot_claim_a_plateau(pair, rejected):
    result = assess_stopping(archive([pair] * 12, rejected=rejected), policy())
    assert result.status != "stopping_candidate"
    assert result.suggested_stop_evaluations is None
    assert not checked(result, 8)["stable_window"]
    if np.isfinite(pair).all():
        assert checked(result, 8)["front_points_outside_reference"] > 0


def test_continued_objective_improvement_reaches_a_hard_cap_without_claiming_convergence():
    records = archive([(0.5 - (i // 4) * 0.1, 0.2) for i in range(20)])
    running = assess_stopping(records[:16], policy())
    assert running.status == "continue_search"
    capped = assess_stopping(records, policy())
    assert capped.status == "budget_exhausted"
    assert capped.suggested_stop_evaluations is None
    assert all(not row["stable_window"] for row in capped.history if row["eligible"])


def test_renewed_progress_invalidates_an_earlier_stopping_candidate():
    records = archive([(0.1, 0.2)] * 12 + [(-0.5, 0.2)] * 4)
    result = assess_stopping(records, policy())
    assert result.suggested_stop_evaluations == 12
    assert result.status == "continue_search"
    assert result.candidate_invalidated
    assert checked(result, 16)["candidate_invalidated"]
    assert checked(result, 16)["consecutive_stable_windows"] == 0
    assert checked(result, 16)["front_movement"] == pytest.approx(0.6)


def test_an_extended_archive_cannot_create_a_first_suggestion_after_the_policy_cap():
    result = assess_stopping(archive([(0.1, 0.2)] * 12), policy(max_evaluations=8))
    assert result.status == "budget_exhausted"
    assert result.suggested_stop_evaluations is None
    assert not checked(result, 12)["eligible"]


def test_costs_better_than_perfect_nse_are_rejected_as_the_wrong_objective_domain():
    records = archive([(0.1, 0.2)] * 11 + [(-1.01, 0.0)])
    with pytest.raises(ValueError, match="negative-NSE"):
        assess_stopping(records, policy())


def test_default_minimum_and_patience_do_not_suggest_before_768_counted_trials():
    records = archive([(0.1, 0.2)] * 768)
    settings = StoppingPolicy()
    before = assess_stopping(records[:640], settings)
    assert before.status == "continue_search"
    assert before.suggested_stop_evaluations is None
    assert assess_stopping(records, settings).suggested_stop_evaluations == 768


def test_shuffled_records_are_counted_by_unique_evaluation_ids():
    records = archive([(0.1, 0.2)] * 12)
    expected = assess_stopping(records, policy())
    actual = assess_stopping(records[::-1], policy())
    assert actual.status == expected.status
    assert actual.history == expected.history
    with pytest.raises(ValueError):
        assess_stopping(records[:-1] + (records[0],), policy())
    with pytest.raises(ValueError):
        assess_stopping(records[:-1] + (replace(records[-1], evaluation_id=20),), policy())


@pytest.mark.parametrize("changes", [
    {"population_size": True}, {"population_size": 1},
    {"check_interval": 4.0}, {"check_interval": 0}, {"check_interval": 6},
    {"window_evaluations": 0}, {"window_evaluations": 6},
    {"minimum_evaluations": 4}, {"minimum_evaluations": 10},
    {"max_evaluations": 4}, {"max_evaluations": 22},
    {"patience": False}, {"patience": 0},
    {"hypervolume_reference": (1,)}, {"hypervolume_reference": (np.inf, 1)},
    {"hypervolume_reference": (-1, 1)},
    {"normalized_hv_tolerance": 0}, {"normalized_hv_tolerance": np.nan},
    {"front_tolerance": -1}, {"extrema_tolerance": np.inf},
])
def test_invalid_static_stopping_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        policy(**changes)
