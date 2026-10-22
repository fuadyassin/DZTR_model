"""Independent volume, operating-rule, spill, and segmented-clock checks."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dztr.data.intervals import DailyIntervalPolicy, prepare_daily_intervals
from dztr.model.release import dztr_release
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters
from dztr.simulation.intervals import (
    IntervalSimulationError, IntervalSimulationSettings, simulate_interval_reservoir,
    validate_interval_preparation,
)


DAY = 86400.


def parameters(*, release=2., flood_release=None, channel=None, flood_mode="storage"):
    flood_release = release if flood_release is None else flood_release
    channel = release if channel is None else channel
    table = pd.DataFrame(
        [[2e8, 5e8, 8e8, release, release, flood_release]] * 12,
        columns=MONTHLY_TARGET_COLUMNS, index=pd.Index(range(1, 13), name="month"),
    )
    return ReservoirParameters(table, 1e9, 4e8, 999., channel, .1, 1., flood_mode)


def preparation(means, *, storage=3e8, flags=(), net=False, periods=None, start="2009-01-01"):
    index = pd.date_range(start, periods=len(means), name="date")
    states = np.full(len(means), storage) if np.isscalar(storage) else storage
    frame = pd.DataFrame({"storage_m3": states, "inflow_m3s": means,
                          "release_m3s": np.full(len(means), 777.)}, index=index)
    policy = DailyIntervalPolicy("computed_net_inflow" if net else "physical_inflow",
                                 tuple(index[i] for i in flags))
    labels = None if periods is None else pd.Series(periods, index=index)
    return prepare_daily_intervals(frame, policy=policy, period_labels=labels)


def test_constant_release_uses_the_current_mean_volume_and_rule_initial_release():
    prepared = preparation([1000., 10., 100.])
    result = simulate_interval_reservoir(prepared, parameters())
    frame = result.frame
    expected = np.array([3e8, 3e8 + 8 * DAY, 3e8 + 106 * DAY])
    np.testing.assert_allclose(frame.storage_m3, expected, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(frame.release_endpoint_m3s, 2.)
    np.testing.assert_array_equal(frame.integrated_inflow_volume_m3.iloc[1:], [10 * DAY, 100 * DAY])
    np.testing.assert_array_equal(frame.interval_mean_total_outflow_m3s.iloc[1:], 2.)
    assert frame.interval_mean_total_outflow_m3s.iloc[0] != frame.interval_mean_total_outflow_m3s.iloc[0]
    assert "release_m3s" not in frame
    assert frame.balance_residual_m3.iloc[1:].abs().max() < 1e-7
    assert frame.release_rule_residual_m3s.eq(0).all()
    assert frame.interval_completed.tolist() == [False, True, True]


def test_signed_computed_net_mean_is_not_clipped_or_adjacent_averaged():
    prepared = preparation([50., -3., 4.], net=True)
    frame = simulate_interval_reservoir(prepared, parameters()).frame
    np.testing.assert_allclose(frame.storage_m3, [3e8, 3e8 - 5 * DAY, 3e8 - 3 * DAY])
    assert frame.integrated_inflow_volume_m3.iloc[1] == -3 * DAY
    assert frame.rule_driver_m3s.eq(0).all()
    for params in (replace(parameters(), flood_mode="inflow"),
                   replace(parameters(), inflow_correction=.9)):
        with pytest.raises(ValueError, match="Computed net flux"):
            simulate_interval_reservoir(prepared, params)


def test_physical_inflow_scale_is_applied_only_to_interval_volume():
    prepared = preparation([0., 10.])
    params = replace(parameters(), inflow_correction=.5)
    frame = simulate_interval_reservoir(prepared, params).frame
    assert frame.integrated_inflow_volume_m3.iloc[1] == 5 * DAY
    assert frame.rule_driver_m3s.iloc[1] == 10
    assert frame.storage_m3.iloc[1] == 3e8 + 3 * DAY


def test_capacity_spill_closes_balance_without_changing_or_carrying_release():
    prepared = preparation([2., 10., 0.], storage=1e9)
    params = parameters()
    frame = simulate_interval_reservoir(prepared, params).frame
    np.testing.assert_array_equal(frame.release_endpoint_m3s, 2.)
    np.testing.assert_array_equal(frame.storage_m3, [1e9, 1e9, 1e9 - 2 * DAY])
    assert frame.spill_volume_m3.iloc[1] == 8 * DAY
    assert frame.spill_volume_m3.iloc[2] == 0
    assert frame.integrated_release_volume_m3.iloc[1] == 2 * DAY
    assert frame.interval_mean_release_m3s.iloc[1] == 2
    assert frame.interval_mean_total_outflow_m3s.iloc[1] == 10
    assert frame.interval_mean_total_outflow_m3s.iloc[2] == 2
    assert frame.root_count.iloc[1] == 1
    assert abs(frame.balance_residual_m3.iloc[1:].sum()) < 1e-7
    for i, storage in enumerate(frame.storage_m3):
        assert frame.release_endpoint_m3s.iloc[i] == dztr_release(storage, 0., 1, params)


def test_quarantine_restarts_only_clean_blocks_and_carries_across_phases():
    prepared = preparation([2.] * 8, storage=np.array([3e8, 1., 1., 1., 1., 4e8, 1., 1.]),
                           flags=(4,), periods=["warmup"] * 2 + ["calibration"] * 4 + ["validation"] * 2)
    frame = simulate_interval_reservoir(prepared, parameters()).frame
    assert frame.storage_m3.iloc[0] == frame.storage_m3.iloc[1] == frame.storage_m3.iloc[2] == 3e8
    assert frame.storage_m3.iloc[3:5].isna().all()
    assert frame.storage_m3.iloc[5:].eq(4e8).all()
    assert frame.release_endpoint_m3s.dropna().eq(2).all()
    assert frame.interval_mean_total_outflow_m3s.iloc[[0, 3, 4, 5]].isna().all()
    assert not frame.storage_score_eligible.iloc[5]
    assert frame.interval_completed.iloc[2] and frame.interval_completed.iloc[6]
    assert not frame.source_audit_interval_eligible.iloc[2]
    assert not frame.source_audit_interval_eligible.iloc[6]


def test_prefix_ignores_future_numbers_and_preserves_next_day_quality_quarantine():
    prepared = preparation([2.] * 8, flags=(4,), net=True,
                           periods=["calibration"] * 4 + ["validation"] * 4)
    prepared.frame.loc[prepared.frame.index[4]:, "inflow_m3s"] = np.nan
    prepared.frame.loc[prepared.frame.index[4]:, "storage_m3"] = np.nan
    prepared.frame["release_m3s"] = prepared.frame.release_m3s.astype(object)
    prepared.frame.loc[prepared.frame.index[4]:, "release_m3s"] = "heldout malformed target"
    frame = simulate_interval_reservoir(prepared, parameters(), end_date=prepared.frame.index[3]).frame
    assert frame.storage_m3.iloc[:3].eq(3e8).all()
    assert frame.storage_m3.iloc[3:].isna().all()
    assert frame.interval_completed.sum() == 2
    assert frame.storage_endpoint_quarantined.iloc[3]
    with pytest.raises(IntervalSimulationError, match="2009-01-06"):
        simulate_interval_reservoir(prepared, parameters())


def test_noninitial_storage_and_mean_release_are_targets_not_forcing():
    prepared = preparation([2.] * 4)
    prepared.frame.loc[prepared.frame.index[1]:, "storage_m3"] = np.nan
    prepared.frame["release_m3s"] = np.nan
    frame = simulate_interval_reservoir(prepared, parameters()).frame
    assert frame.storage_m3.eq(3e8).all()
    assert frame.interval_completed.sum() == 3


def test_selected_start_adds_an_unscored_provisional_anchor():
    prepared = preparation([10.] * 5)
    frame = simulate_interval_reservoir(prepared, parameters(), start_date=prepared.frame.index[2],
                                        end_date=prepared.frame.index[3]).frame
    assert frame.storage_m3.iloc[:2].isna().all()
    assert frame.storage_m3.iloc[2] == 3e8
    assert frame.storage_m3.iloc[3] == 3e8 + 8 * DAY
    assert pd.isna(frame.storage_m3.iloc[4])
    assert frame.segment_initialization.iloc[2]
    assert not frame.storage_score_eligible.iloc[2]
    assert not frame.interval_completed.iloc[2]


def _ambiguous_preparation(previous_storage=None):
    params = parameters(release=2., flood_release=100., channel=20.)
    # Use critical release 1 and normal release 2, then a downward emergency jump.
    params.monthly_targets["release_critical_m3s"] = 1.
    predictor = 802e6
    slope = 98 / 3e8
    lower = (predictor - DAY / 2 * (2 - slope * 5e8)) / (1 + DAY / 2 * slope)
    upper = predictor - DAY / 2 * 20
    storage = 799e6 if previous_storage is None else previous_storage
    q0 = dztr_release(storage, 0., 1, params)
    mean = (predictor - storage + DAY / 2 * q0) / DAY
    return preparation([0., mean], storage=storage, net=True), params, (lower, upper)


def test_multiple_roots_reject_by_default_and_record_partial_candidates():
    prepared, params, expected = _ambiguous_preparation()
    with pytest.raises(IntervalSimulationError) as failure:
        simulate_interval_reservoir(prepared, params)
    error = failure.value
    assert error.detail["failure_code"] == "multiple_roots"
    assert error.detail["date"] == "2009-01-02"
    assert error.detail["root_count"] == 2
    np.testing.assert_allclose(error.detail["candidate_storage_m3"], expected, rtol=0, atol=1e-7)
    pd.testing.assert_index_equal(error.partial_result.frame.index, prepared.frame.index)
    assert error.partial_result.frame.storage_m3.iloc[0] == 799e6
    assert pd.isna(error.partial_result.frame.storage_m3.iloc[1])


@pytest.mark.parametrize("policy,index", [("minimum_storage", 0), ("maximum_storage", 1)])
def test_declared_root_policy_selects_existing_paired_root(policy, index):
    prepared, params, expected = _ambiguous_preparation()
    frame = simulate_interval_reservoir(prepared, params,
             settings=IntervalSimulationSettings(root_policy=policy)).frame
    assert frame.root_count.iloc[1] == 2 and frame.root_policy_applied.iloc[1]
    assert frame.storage_m3.iloc[1] == pytest.approx(expected[index], abs=1e-7)
    assert frame.release_endpoint_m3s.iloc[1] == dztr_release(frame.storage_m3.iloc[1], 0., 1, params)
    assert abs(frame.balance_residual_m3.iloc[1]) < 1e-4


def test_nearest_root_distance_tie_chooses_lower_storage():
    _, _, roots = _ambiguous_preparation()
    prepared, params, _ = _ambiguous_preparation(sum(roots) / 2)
    frame = simulate_interval_reservoir(prepared, params,
             settings=IntervalSimulationSettings(root_policy="nearest_previous_storage")).frame
    assert frame.root_policy_tie_break.iloc[1]
    assert frame.storage_m3.iloc[1] == pytest.approx(roots[0], abs=1e-7)


def test_capacity_spill_candidate_is_enumerated_alongside_in_bounds_roots():
    prepared, params, roots = _ambiguous_preparation()
    params = replace(params, capacity_m3=801e6)
    with pytest.raises(IntervalSimulationError) as failure:
        simulate_interval_reservoir(prepared, params)
    detail = failure.value.detail
    np.testing.assert_allclose(detail["candidate_storage_m3"], [roots[0], 801e6], atol=1e-7, rtol=0)
    np.testing.assert_allclose(detail["candidate_spill_volume_m3"], [0, 136000.], atol=1e-7, rtol=0)
    assert detail["root_count"] == 2
    maximum = simulate_interval_reservoir(prepared, params,
               settings=IntervalSimulationSettings("maximum_storage")).frame.iloc[1]
    assert maximum.storage_m3 == params.capacity_m3
    assert maximum.release_endpoint_m3s == 20
    assert maximum.spill_volume_m3 == pytest.approx(136000., abs=1e-7)
    assert abs(maximum.balance_residual_m3) < 1e-4
    minimum = simulate_interval_reservoir(prepared, params,
               settings=IntervalSimulationSettings("minimum_storage")).frame.iloc[1]
    assert minimum.storage_m3 == pytest.approx(roots[0], abs=1e-7)
    assert minimum.spill_volume_m3 == 0


def test_zero_spill_capacity_root_is_not_counted_twice():
    frame = simulate_interval_reservoir(preparation([2., 2.], storage=1e9), parameters()).frame
    assert frame.root_count.iloc[1] == 1
    assert frame.storage_m3.iloc[1] == 1e9
    assert frame.spill_volume_m3.iloc[1] == 0


@pytest.mark.parametrize("policy", ["reject", "minimum_storage", "maximum_storage", "nearest_previous_storage"])
def test_gap_remains_failure_for_every_declared_policy(policy):
    params = parameters(release=2000., channel=3000.)
    prepared = preparation([0., (260e6 - 100e6) / DAY], storage=100e6, net=True)
    with pytest.raises(IntervalSimulationError) as failure:
        simulate_interval_reservoir(prepared, params, settings=IntervalSimulationSettings(policy))
    assert failure.value.detail["failure_code"] == "no_root"
    assert failure.value.detail["root_count"] == 0


def test_negative_predictor_stops_without_advancing_or_skipping_later_block():
    prepared = preparation([0., -1., 0., 0., 0., 0.], storage=0., net=True, flags=(3,))
    with pytest.raises(IntervalSimulationError) as failure:
        simulate_interval_reservoir(prepared, parameters())
    error = failure.value
    assert error.detail["failure_code"] == "negative_predictor"
    assert error.partial_result.frame.storage_m3.iloc[0] == 0
    assert error.partial_result.frame.storage_m3.iloc[1:].isna().all()
    assert not error.partial_result.frame.interval_completed.any()


def test_initial_storage_stress_is_explicit_and_never_clipped():
    prepared = preparation([2., 2.])
    params = parameters()
    frame = simulate_interval_reservoir(prepared, params,
             settings=IntervalSimulationSettings(initial_storage_offset_capacity_fraction=.01)).frame
    assert frame.storage_m3.iloc[0] == 3e8 + .01 * params.capacity_m3
    prepared = preparation([2., 2.], storage=1e9)
    with pytest.raises(IntervalSimulationError) as failure:
        simulate_interval_reservoir(prepared, params,
              settings=IntervalSimulationSettings(initial_storage_offset_capacity_fraction=.01))
    assert failure.value.detail["failure_code"] == "initial_storage_out_of_bounds"
    assert failure.value.partial_result.frame.storage_m3.isna().all()


@pytest.mark.parametrize("field", ["simulation_interval_eligible", "storage_score_eligible", "segment_id"])
def test_mutated_preparation_masks_are_rejected(field):
    prepared = preparation([2.] * 4)
    prepared.frame.loc[prepared.frame.index[1], field] = False if field != "segment_id" else 9
    with pytest.raises(ValueError, match="derived field"):
        simulate_interval_reservoir(prepared, parameters())


@pytest.mark.parametrize("field", ["reported_inflow_volume_m3", "usable_inflow_volume_m3"])
def test_mutated_current_interval_volume_is_rejected_with_partial_result(field):
    prepared = preparation([2.] * 4)
    prepared.frame.loc[prepared.frame.index[1], field] += 1
    with pytest.raises(IntervalSimulationError) as failure:
        simulate_interval_reservoir(prepared, parameters())
    assert failure.value.detail["failure_code"] == "mutated_preparation"
    assert failure.value.partial_result.frame.storage_m3.iloc[0] == 3e8


def test_nonfinite_used_forcing_fails_with_date_and_completed_prefix():
    prepared = preparation([2.] * 4)
    prepared.frame.loc[prepared.frame.index[2], "inflow_m3s"] = np.inf
    with pytest.raises(IntervalSimulationError) as failure:
        simulate_interval_reservoir(prepared, parameters())
    assert failure.value.detail["date"] == "2009-01-03"
    assert failure.value.partial_result.frame.interval_completed.sum() == 1
    assert failure.value.partial_result.frame.storage_m3.iloc[2:].isna().all()


def test_complex_forcing_is_rejected_before_conversion_to_real():
    prepared = preparation([2.] * 4)
    prepared.frame["inflow_m3s"] = prepared.frame.inflow_m3s.astype(complex)
    prepared.frame.loc[prepared.frame.index[1], "inflow_m3s"] = 2 + 1j
    with pytest.raises(IntervalSimulationError, match="real data"):
        simulate_interval_reservoir(prepared, parameters())


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_equivalent_daily_datetime_resolutions_are_accepted(unit):
    prepared = preparation([2.] * 4)
    prepared.frame.index = prepared.frame.index.as_unit(unit)
    frame = simulate_interval_reservoir(prepared, parameters()).frame
    assert frame.interval_completed.sum() == 3
    assert frame.storage_m3.eq(3e8).all()


def test_returned_frames_and_parameters_are_independent():
    prepared = preparation([2.] * 4)
    prepared.frame["source_note"] = [{"values": []} for _ in range(4)]
    params = parameters()
    source = prepared.frame.copy(deep=True)
    result = simulate_interval_reservoir(prepared, params)
    pd.testing.assert_frame_equal(prepared.frame, source, check_exact=True)
    result.parameters.monthly_targets.iloc[0, 0] = 0
    result.frame.source_note.iloc[0]["values"].append("changed")
    result.frame.loc[result.frame.index[0], "reported_inflow_mean_m3s"] = 0
    assert params.monthly_targets.iloc[0, 0] == 2e8
    assert prepared.frame.source_note.iloc[0]["values"] == []
    assert prepared.frame.inflow_m3s.iloc[0] == 2


@pytest.mark.parametrize("offset", [True, np.nan, np.inf, 1j, "0.01"])
def test_invalid_settings_are_rejected(offset):
    with pytest.raises(ValueError, match="finite and real"):
        IntervalSimulationSettings(initial_storage_offset_capacity_fraction=offset)


def test_structural_validator_can_be_used_without_reading_numeric_observations():
    prepared = preparation([2.] * 4, net=True)
    prepared.frame["inflow_m3s"] = np.nan
    prepared.frame["storage_m3"] = np.nan
    prepared.frame["release_m3s"] = np.nan
    graph = validate_interval_preparation(prepared)
    assert graph.frame.simulation_interval_eligible.sum() == 3
