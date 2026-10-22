"""Independent flux, clock and admissibility checks for opt-in substeps."""

from dataclasses import replace
from math import ceil, fsum, isinf

import numpy as np
import pandas as pd
import pytest

from dztr.schemas import DataQuality, MONTHLY_TARGET_COLUMNS, ReservoirObservations, ReservoirParameters
from dztr.simulation.root_policy import POLICIES, simulate_root_policy_reservoir
from dztr.simulation.sampled import SamplingSettings
from dztr.simulation.substeps import (
    SubstepSettings,
    SubstepSimulationError,
    critical_substep_bounds,
    simulate_substep_reservoir,
)


DAY = 86400.


def _parameters(*, initial_storage=3.5e8, initial_release=2., releases=(1., 5., 9.),
                correction=1., flood_mode="storage", distinct_months=False):
    rows = [[2e8, 5e8, 8e8, *(tuple(20. + month for _ in range(3)) if distinct_months else releases)]
            for month in range(1, 13)]
    targets = pd.DataFrame(rows, columns=MONTHLY_TARGET_COLUMNS, index=pd.Index(range(1, 13), name="month"))
    return ReservoirParameters(targets, 1e9, initial_storage, initial_release, 50., .1, correction, flood_mode)


def _observations(index, inflow):
    index = pd.DatetimeIndex(index, name="date")
    frame = pd.DataFrame({"storage_m3": np.linspace(3e8, 4e8, len(index)), "inflow_m3s": inflow,
                          "release_m3s": np.linspace(2., 3., len(index))}, index=index)
    quality = DataQuality("available", len(frame), {column: int(frame[column].isna().sum()) for column in frame}, 0)
    return ReservoirObservations(frame, quality, "portable analytical fixture")


def _run(observations, parameters, n, *, policy="reject", sampling=None, record=True):
    return simulate_substep_reservoir(observations, parameters, sampling=sampling,
                                     settings=SubstepSettings(n, policy), record_internal=record)


CLOCKS = [
    (SamplingSettings(), pd.date_range("2019-12-30", periods=70)),
    (SamplingSettings("monthly"), pd.date_range("2019-12-01", periods=15, freq="MS")),
    (SamplingSettings("monthly", 30.5), pd.date_range("2019-12-01", periods=15, freq="MS")),
    (SamplingSettings("monthly", 30.5, "legacy", "2019-12-01"), pd.date_range("2019-12-01", periods=15, freq="MS")),
]


@pytest.mark.parametrize("sampling,index", CLOCKS)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("flood_mode", ["storage", "inflow"])
def test_one_substep_preserves_all_existing_root_policy_columns_exact(sampling, index, policy, flood_mode):
    observations = _observations(index, np.linspace(1., 3., len(index)))
    parameters = _parameters(distinct_months=True, flood_mode=flood_mode)
    expected = simulate_root_policy_reservoir(observations, parameters, sampling=sampling, policy=policy)
    actual = _run(observations, parameters, 1, sampling=sampling, policy=policy)
    pd.testing.assert_frame_equal(actual.simulation.frame[expected.frame.columns], expected.frame, check_exact=True)
    assert actual.simulation.parameters is parameters
    assert actual.simulation.overflow_mode == "conservative"
    assert actual.internal.index.names == ["date", "substep"]
    assert len(actual.internal) == len(index) - 1
    assert actual.internal.index.get_level_values("substep").tolist() == [1] * (len(index) - 1)


@pytest.mark.parametrize("n", [1, 2, 4, 7])
def test_constant_rule_integrates_linear_forcing_and_configured_initial_release(n):
    initial_inflow, final_inflow = 1000., 1e-9
    parameters = _parameters(initial_release=7., releases=(10., 10., 10.), correction=1.25)
    observations = _observations(pd.date_range("2020-01-01", periods=2), [initial_inflow, final_inflow])
    result = _run(observations, parameters, n)
    frame, internal = result.simulation.frame, result.internal
    h = DAY / n
    expected_inflow = DAY / 2. * parameters.inflow_correction * (initial_inflow + final_inflow)
    expected_release = h * (parameters.initial_release_m3s / 2. + (n - .5) * 10.)
    expected_storage = parameters.initial_storage_m3 + expected_inflow - expected_release
    assert frame.iloc[0].release_m3s == 7.
    assert internal.iloc[0].previous_release_m3s == 7.
    assert internal.iloc[0].previous_inflow_m3s == initial_inflow
    assert internal.iloc[-1].inflow_m3s == final_inflow
    np.testing.assert_allclose(internal.inflow_m3s, [initial_inflow + (final_inflow - initial_inflow) * j / n
                                                  if j != n else final_inflow for j in range(1, n + 1)],
                               rtol=0, atol=2e-13)
    np.testing.assert_allclose(internal.step_seconds, h, rtol=0, atol=0)
    np.testing.assert_allclose(internal.selected_root_release_m3s, 10., rtol=0, atol=0)
    np.testing.assert_allclose(frame.iloc[-1].storage_m3, expected_storage, rtol=0, atol=1e-6)
    np.testing.assert_allclose(frame.iloc[-1].integrated_inflow_volume_m3, expected_inflow, rtol=0, atol=2e-7)
    np.testing.assert_allclose(frame.iloc[-1].integrated_release_volume_m3, expected_release, rtol=0, atol=2e-7)
    np.testing.assert_allclose(frame.iloc[-1].interval_mean_release_m3s, expected_release / DAY, rtol=0, atol=1e-12)
    assert frame.iloc[-1].release_m3s == 10.
    assert abs(frame.iloc[-1].balance_residual_m3) < 1e-6
    if n > 1:
        # A full-interval endpoint trapezoid omits intermediate releases.
        assert abs(frame.iloc[-1].native_endpoint_balance_residual_m3) > 1.
        assert frame.iloc[-1].interval_mean_release_m3s != frame.iloc[-1].release_m3s


@pytest.mark.parametrize("n", [2, 4, 7])
def test_affine_normal_rule_matches_independent_closed_form_recurrence(n):
    parameters = _parameters(initial_release=2., correction=1.1)
    observations = _observations(pd.date_range("2020-01-01", periods=2), [6., 11.])
    result = _run(observations, parameters, n)
    h = DAY / n
    slope, intercept = 4. / 3e8, 1. - 4. / 3e8 * 2e8
    storage, release, previous_inflow = parameters.initial_storage_m3, parameters.initial_release_m3s, 6.
    expected_storage, expected_release = [], []
    volume_release = 0.
    for j in range(1, n + 1):
        inflow = 6. + 5. * j / n
        current_storage = (storage + h / 2. * (1.1 * (inflow + previous_inflow) - release)
                           - h / 2. * intercept) / (1. + h / 2. * slope)
        current_release = slope * current_storage + intercept
        volume_release += h / 2. * (release + current_release)
        expected_storage.append(current_storage)
        expected_release.append(current_release)
        storage, release, previous_inflow = current_storage, current_release, inflow
    np.testing.assert_allclose(result.internal.storage_m3, expected_storage, rtol=0, atol=1e-6)
    np.testing.assert_allclose(result.internal.release_m3s, expected_release, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.simulation.frame.iloc[-1].integrated_release_volume_m3,
                               volume_release, rtol=0, atol=2e-7)
    assert (result.internal.storage_m3.between(2e8, 5e8)).all()
    assert result.internal.release_rule_residual_m3s.abs().max() <= 1e-12
    assert result.internal.balance_residual_m3.abs().max() <= 1e-6


@pytest.mark.parametrize("n", [1, 2, 4, 7])
def test_overflow_carries_physical_release_and_spill_flux_across_native_boundaries(n):
    observations = _observations(pd.date_range("2020-01-01", periods=4), [1000.] * 4)
    parameters = replace(_parameters(initial_storage=9.9e8, initial_release=1.), channel_capacity_m3s=20.)
    result = _run(observations, parameters, n)
    h = DAY / n
    storage, release, previous_spill_increment = 9.9e8, 1., 0.
    for native_row in range(1, 4):
        volume_inflow = volume_release = volume_spill = excess_sum = 0.
        for substep in range(1, n + 1):
            observed = result.internal.loc[(observations.frame.index[native_row], substep)]
            predictor = storage + h / 2. * (2000. - release)
            raw_storage, raw_release = predictor - h / 2. * 20., 20.
            excess = max(raw_storage - parameters.capacity_m3, 0.)
            current_storage = min(raw_storage, parameters.capacity_m3)
            spill_increment = excess / h * 2.
            current_release = raw_release + spill_increment
            q_volume = h / 2. * (release + current_release)
            spill_volume = h / 2. * (previous_spill_increment + spill_increment)
            assert observed.previous_storage_m3 == storage
            assert observed.previous_release_m3s == release
            assert observed.selected_root_storage_m3 == raw_storage
            assert observed.selected_root_release_m3s == raw_release
            assert observed.storage_m3 == current_storage
            assert observed.release_m3s == current_release
            assert observed.overflow_volume_m3 == excess
            assert observed.spill_increment_m3s == spill_increment
            np.testing.assert_allclose(observed.integrated_release_volume_m3, q_volume, rtol=0, atol=1e-6)
            np.testing.assert_allclose(observed.integrated_spill_volume_m3, spill_volume, rtol=0, atol=1e-6)
            assert observed.release_rule_residual_m3s == 0.
            assert abs(observed.balance_residual_m3) <= 1e-6
            volume_inflow += h * 1000.
            volume_release += q_volume
            volume_spill += spill_volume
            excess_sum += excess
            storage, release, previous_spill_increment = current_storage, current_release, spill_increment
        row = result.simulation.frame.iloc[native_row]
        np.testing.assert_allclose(row.integrated_inflow_volume_m3, volume_inflow, rtol=0, atol=1e-6)
        np.testing.assert_allclose(row.integrated_release_volume_m3, volume_release, rtol=0, atol=1e-6)
        np.testing.assert_allclose(row.integrated_spill_volume_m3, volume_spill, rtol=0, atol=1e-6)
        np.testing.assert_allclose(row.overflow_volume_m3, excess_sum, rtol=0, atol=1e-6)
        assert row.storage_m3 == storage
        assert row.release_m3s == release
        assert row.overflow_substeps == n
        if n > 1 or native_row > 1:
            assert row.integrated_spill_volume_m3 != row.overflow_volume_m3
        assert abs(row.balance_residual_m3) <= 2e-6
    # There is no elapsed interval at the configured initial state.
    assert np.isnan(result.simulation.frame.iloc[0].integrated_spill_volume_m3)
    assert result.simulation.frame.iloc[0].cumulative_spill_volume_m3 == 0.


@pytest.mark.parametrize("sampling,index", CLOCKS)
@pytest.mark.parametrize("n", [2, 7])
def test_substeps_hold_destination_rule_month_and_declared_native_duration(sampling, index, n):
    index = index[:4]
    observations = _observations(index, np.full(4, 25.))
    parameters = _parameters(initial_release=7., distinct_months=True)
    result = _run(observations, parameters, n, sampling=sampling)
    if sampling.month_clock == "legacy":
        expected_months = (pd.Timestamp(sampling.legacy_start)
                           + pd.to_timedelta(np.arange(4) * sampling.fixed_step_days, unit="D")).month
    else:
        expected_months = index.month
    durations = np.diff(index.asi8) / 1e9 if sampling.fixed_step_days is None else np.full(3, sampling.fixed_step_days * DAY)
    for row in range(1, 4):
        internal = result.internal.xs(index[row], level="date")
        assert (internal.rule_month == expected_months[row]).all()
        assert (internal.release_m3s == 20. + expected_months[row]).all()
        np.testing.assert_allclose(internal.step_seconds, durations[row - 1] / n, rtol=0, atol=0)
        assert result.simulation.frame.iloc[row].rule_month == expected_months[row]
        assert result.simulation.frame.iloc[row].step_seconds == durations[row - 1]
        assert result.simulation.frame.iloc[row].substep_count == n
    assert result.internal.iloc[0].previous_release_m3s == 7.


def _gap_parameters(*, first_month_safe=False):
    parameters = _parameters(initial_storage=2.8e8, initial_release=0., releases=(10000., 10000., 10000.))
    if first_month_safe:
        targets = parameters.monthly_targets.copy()
        targets.loc[1, list(MONTHLY_TARGET_COLUMNS[3:])] = 1.
        parameters = replace(parameters, monthly_targets=targets)
    return parameters


@pytest.mark.parametrize("policy", POLICIES)
def test_substeps_do_not_force_a_root_when_a_gap_persists(policy):
    observations = _observations(pd.date_range("2020-02-01", periods=2), [0., 0.])
    with pytest.raises(SubstepSimulationError, match="No feasible implicit") as raised:
        _run(observations, _gap_parameters(), 4, policy=policy)
    detail = raised.value.detail
    assert detail["failure_code"] == "implicit_no_root"
    assert detail["substep"] == 1
    assert detail["substeps_per_interval"] == 4
    assert detail["step_seconds"] == 21600.
    assert detail["predictor_m3"] == 2.8e8
    assert 2.5e8 < detail["predictor_m3"] < 3.08e8
    assert len(raised.value.partial.simulation.frame) == 1
    assert len(raised.value.partial.internal) == 0


@pytest.mark.parametrize("policy", POLICIES)
def test_substeps_do_not_clip_or_fallback_from_negative_predictor(policy):
    parameters = _parameters(initial_storage=0., initial_release=100.)
    observations = _observations(pd.date_range("2020-01-01", periods=2), [0., 0.])
    with pytest.raises(SubstepSimulationError, match="negative storage") as raised:
        _run(observations, parameters, 4, policy=policy)
    assert raised.value.detail["predictor_m3"] < 0
    assert raised.value.detail["substep"] == 1
    assert len(raised.value.partial.simulation.frame) == 1


@pytest.mark.parametrize("record", [False, True])
def test_failure_preserves_exact_completed_prefix_without_inventing_native_endpoint(record):
    observations = _observations(pd.date_range("2020-01-30", periods=3), [0., 0., 0.])
    parameters = _gap_parameters(first_month_safe=True)
    expected_observations = ReservoirObservations(observations.frame.iloc[:2].copy(),
                                                DataQuality("available", 2, {column: 0 for column in observations.frame}, 0))
    expected = _run(expected_observations, parameters, 4, record=record)
    with pytest.raises(SubstepSimulationError) as raised:
        _run(observations, parameters, 4, record=record)
    error = raised.value
    assert pd.Timestamp(error.detail["date"]) == observations.frame.index[-1]
    assert error.detail["substep"] == 1
    assert error.detail["rule_month"] == 2
    pd.testing.assert_frame_equal(error.partial.simulation.frame, expected.simulation.frame, check_exact=True)
    if record:
        pd.testing.assert_frame_equal(error.partial.internal, expected.internal, check_exact=True)
    else:
        assert error.partial.internal is None


@pytest.mark.parametrize("record", [False, True])
def test_later_failed_part_retains_accepted_parts_without_scoring_unfinished_interval(record):
    parameters = replace(_gap_parameters(), initial_storage_m3=1.9e8)
    observations = _observations(pd.date_range("2020-02-01", periods=2), [0., 10000.])
    with pytest.raises(SubstepSimulationError) as raised:
        _run(observations, parameters, 4, record=record)
    error, detail = raised.value, raised.value.detail
    # Independent first two available-water solutions: S1=178e6, S2=180e6.
    assert detail["substep"] == 3
    assert detail["completed_substeps_in_failed_interval"] == 2
    np.testing.assert_allclose(detail["previous_storage_m3"], 1.8e8, rtol=0, atol=1e-6)
    np.testing.assert_allclose(detail["predictor_m3"], 2.75e8, rtol=0, atol=1e-6)
    np.testing.assert_allclose(detail["completed_interval_inflow_volume_m3"], 1.08e8, rtol=0, atol=1e-6)
    np.testing.assert_allclose(detail["completed_interval_release_volume_m3"], 1.18e8, rtol=0, atol=1e-6)
    assert len(error.partial.simulation.frame) == 1
    assert error.partial.simulation.frame.iloc[0].storage_m3 == 1.9e8
    if record:
        assert error.partial.internal.index.get_level_values("substep").tolist() == [1, 2]
        np.testing.assert_allclose(error.partial.internal.storage_m3, [1.78e8, 1.8e8], rtol=0, atol=1e-6)
    else:
        assert error.partial.internal is None


@pytest.mark.parametrize("bad", [-1., np.nan, np.inf])
def test_invalid_source_endpoint_rejects_whole_interval_before_positive_interpolated_parts(bad):
    observations = _observations(pd.date_range("2020-01-01", periods=3), [1., 2., bad])
    with pytest.raises(SubstepSimulationError) as raised:
        _run(observations, _parameters(), 4)
    error = raised.value
    assert error.detail["failure_code"] == "invalid_forcing"
    assert error.detail["substep"] == 1
    assert error.detail["completed_substeps_in_failed_interval"] == 0
    assert len(error.partial.simulation.frame) == 2
    assert len(error.partial.internal) == 4
    assert error.partial.internal.index.get_level_values("date").unique().tolist() == [observations.frame.index[1]]


@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("policy,root_index", [("minimum_storage", 0), ("maximum_storage", 1), ("nearest_previous_storage", 1)])
def test_multiple_root_policy_selects_analytical_pair_at_internal_duration(n, policy, root_index):
    h = DAY / n
    previous_storage, predictor = 8e8 + 5. * h, 8e8 + 30. * h
    parameters = _parameters(initial_storage=previous_storage, initial_release=150., flood_mode="inflow")
    observations = _observations(pd.date_range("2020-01-01", periods=2), [100., 100.])
    slope, intercept = 95. / 3e8, 5. - 95. / 3e8 * 5e8
    low_storage = (predictor - h / 2. * intercept) / (1. + h / 2. * slope)
    pairs = ((low_storage, 5. + 95. * ((low_storage - 5e8) / 3e8)), (8e8 + 20. * h, 20.))
    with pytest.raises(SubstepSimulationError, match="Multiple"):
        _run(observations, parameters, n)
    result = _run(observations, parameters, n, policy=policy)
    first = result.internal.iloc[0]
    assert first.root_count == 2
    assert first.root_policy_applied
    np.testing.assert_allclose(first.selected_root_storage_m3, pairs[root_index][0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(first.selected_root_release_m3s, pairs[root_index][1], rtol=0, atol=1e-9)
    assert first.selected_root_storage_m3 != (pairs[0][0] + pairs[1][0]) / 2.
    assert abs(first.selected_root_storage_m3 - predictor + h / 2. * first.selected_root_release_m3s) <= first.root_admissibility_tolerance_m3
    assert result.simulation.frame.iloc[-1].multiple_root_substeps == int((result.internal.root_count > 1).sum())
    assert result.simulation.frame.iloc[-1].root_policy_applied
    if n == 1:
        expected = simulate_root_policy_reservoir(observations, parameters, policy=policy)
        pd.testing.assert_frame_equal(result.simulation.frame[expected.frame.columns], expected.frame, check_exact=True)


def test_internal_tie_break_is_counted_even_if_native_endpoint_is_not_tied():
    h, predictor = DAY / 4., 8e8 + 30. * (DAY / 4.)
    slope, intercept = 95. / 3e8, 5. - 95. / 3e8 * 5e8
    low_storage = (predictor - h / 2. * intercept) / (1. + h / 2. * slope)
    previous_storage = (low_storage + 8e8 + 20. * h) / 2.
    previous_release = 200. - (predictor - previous_storage) / (h / 2.)
    parameters = _parameters(initial_storage=previous_storage, initial_release=previous_release, flood_mode="inflow")
    observations = _observations(pd.date_range("2020-01-01", periods=2), [100., 100.])
    result = _run(observations, parameters, 4, policy="nearest_previous_storage")
    assert result.internal.iloc[0].root_policy_tie_break
    np.testing.assert_allclose(result.internal.iloc[0].selected_root_storage_m3, low_storage, rtol=0, atol=1e-6)
    assert result.simulation.frame.iloc[-1].root_policy_tie_break_substeps == int(result.internal.root_policy_tie_break.sum())
    assert result.simulation.frame.iloc[-1].root_policy_tie_break


def test_cumulative_flux_ledger_closes_against_independently_summed_internal_volumes():
    observations = _observations(pd.date_range("2020-01-01", periods=15), np.linspace(6., 11., 15))
    parameters = _parameters(correction=1.1)
    result = _run(observations, parameters, 7)
    for date, native in result.simulation.frame.iloc[1:].iterrows():
        accepted = result.internal.loc[result.internal.index.get_level_values("date") <= date]
        incoming = fsum(accepted.integrated_inflow_volume_m3)
        outgoing = fsum(accepted.integrated_release_volume_m3)
        spill = fsum(accepted.integrated_spill_volume_m3)
        np.testing.assert_allclose(native.cumulative_inflow_volume_m3, incoming, rtol=0, atol=2e-9)
        np.testing.assert_allclose(native.cumulative_release_volume_m3, outgoing, rtol=0, atol=2e-9)
        assert native.cumulative_spill_volume_m3 == spill
        independently_closed = native.storage_m3 - parameters.initial_storage_m3 - (incoming - outgoing)
        np.testing.assert_allclose(native.cumulative_balance_residual_m3, independently_closed, rtol=0, atol=4e-9)
        assert abs(native.cumulative_balance_residual_m3) <= native.cumulative_roundoff_bound_m3


def test_internal_recording_changes_no_native_states_or_fluxes():
    observations = _observations(pd.date_range("2020-01-01", periods=3), [6., 11., 8.])
    parameters = _parameters(correction=1.1)
    recorded, unrecorded = _run(observations, parameters, 7), _run(observations, parameters, 7, record=False)
    assert unrecorded.internal is None
    pd.testing.assert_frame_equal(recorded.simulation.frame, unrecorded.simulation.frame, check_exact=True)


@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("bad", [-1., np.nan, np.inf])
def test_invalid_forcing_is_rejected_before_substeps(n, bad):
    observations = _observations(pd.date_range("2020-01-01", periods=2), [bad, 3.])
    with pytest.raises(ValueError, match="finite|nonnegative"):
        _run(observations, _parameters(), n)


@pytest.mark.parametrize("sampling,index", [
    (SamplingSettings(), pd.to_datetime(["2020-01-01", "2020-01-03"])),
    (SamplingSettings("monthly"), pd.to_datetime(["2020-01-01", "2020-03-01"])),
])
def test_missing_native_intervals_are_not_manufactured(sampling, index):
    with pytest.raises(ValueError, match="consecutive|missing"):
        _run(_observations(index, [1., 2.]), _parameters(), 4, sampling=sampling)


@pytest.mark.parametrize("count", [True, False, 0, -1, 1., 1.5, np.nan, "2", None])
def test_substep_count_requires_positive_integer(count):
    with pytest.raises((TypeError, ValueError)):
        SubstepSettings(count)


@pytest.mark.parametrize("policy", ["min", "automatic", None])
def test_root_selection_requires_recognized_declared_policy(policy):
    with pytest.raises((TypeError, ValueError)):
        SubstepSettings(2, policy)


def test_critical_bounds_include_zero_release_and_exact_thresholds():
    parameters = _parameters(releases=(10., 10., 10.))
    targets = parameters.monthly_targets.copy()
    for month, maximum_step in ((1, DAY), (2, DAY / 2.), (3, DAY / 2.1)):
        targets.loc[month, "storage_critical_m3"] = 1e8 + 10. * maximum_step
    targets.loc[4, list(MONTHLY_TARGET_COLUMNS[3:])] = 0.
    parameters = replace(parameters, monthly_targets=targets)
    bounds = critical_substep_bounds(parameters)
    assert set(bounds.index) == set(range(1, 13))
    for month in (1, 2, 3):
        available = parameters.monthly_targets.loc[month, "storage_critical_m3"] - 1e8
        assert bounds.loc[month, "available_volume_m3"] == available
        assert bounds.loc[month, "critical_release_m3s"] == 10.
        assert bounds.loc[month, "max_step_seconds"] == available / 10.
        assert bounds.loc[month, "minimum_substeps"] == max(1, ceil(DAY / (available / 10.)))
    assert bounds.loc[1, "minimum_substeps"] == 1
    assert bounds.loc[2, "minimum_substeps"] == 2
    assert bounds.loc[3, "minimum_substeps"] == 3
    assert isinf(bounds.loc[4, "max_step_seconds"])
    assert bounds.loc[4, "minimum_substeps"] == 1


@pytest.mark.parametrize("duration", [True, 0., -1., np.nan, np.inf])
def test_critical_bounds_reject_invalid_native_duration(duration):
    with pytest.raises((TypeError, ValueError)):
        critical_substep_bounds(_parameters(), duration)
