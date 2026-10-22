"""Explicit conventions choose actual roots and preserve strict mathematics."""

from dataclasses import replace
from math import ulp

import numpy as np
import pandas as pd
import pytest

from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.root_policy import RootPolicyCalibrationObjective
from dztr.calibration.sampled import SampledCalibrationObjective, default_sampled_split, estimate_sampled_bounds, select_sampled_period, target_observations
from dztr.model.balance import fixed_point_step, solve_implicit_step
from dztr.model.release import dztr_release
from dztr.parameterization import estimate_generalized_parameters
from dztr.schemas import DataQuality, MONTHLY_TARGET_COLUMNS, ReservoirObservations, ReservoirParameters
from dztr.simulation.root_policy import POLICIES, simulate_root_policy_reservoir, solve_root_policy_step
from dztr.simulation.sampled import SamplingSettings, simulate_sampled_reservoir


def _parameters(*, initial_storage=4e8, initial_release=1., inflow_enabled=False, large_release=False, distinct_months=False):
    rows = []
    for month in range(1, 13):
        releases = [100., 200., 300.] if large_release else [1., 2., 3.]
        if distinct_months:
            releases = [value + month / 10 for value in releases]
        rows.append([2e8, 5e8, 8e8, *releases])
    targets = pd.DataFrame(rows, columns=MONTHLY_TARGET_COLUMNS, index=pd.Index(range(1, 13), name="month"))
    return ReservoirParameters(targets, 1e9, initial_storage, initial_release, 500. if large_release else 20., .1, 1., "inflow" if inflow_enabled else "storage")


def _observations(index, inflow):
    index = pd.DatetimeIndex(index, name="date")
    frame = pd.DataFrame({"storage_m3": np.linspace(3e8, 5e8, len(index)), "inflow_m3s": inflow, "release_m3s": np.linspace(1., 2., len(index))}, index=index)
    return ReservoirObservations(frame, DataQuality("available", len(frame), {column: 0 for column in frame}, 0), "portable")


def _analytical_roots():
    predictor, half, dt = 802e6, 43200., 86400.
    slope = (100. - 2.) / (8e8 - 5e8)
    intercept = 2. - slope * 5e8
    flood_storage = (predictor - half * intercept) / (1 + half * slope)
    flood_release = 2. + ((flood_storage - 5e8) / (8e8 - 5e8)) * 98.
    emergency_storage = (predictor - half * (-8e8 / dt)) / (1 + half / dt)
    emergency_release = (emergency_storage - 8e8) / dt
    return ((flood_storage, flood_release), (emergency_storage, emergency_release))


def _multiple_inputs(previous_storage=None):
    roots = _analytical_roots()
    previous_storage = roots[0][0] if previous_storage is None else previous_storage
    previous_release = 100. - (802e6 - previous_storage) / 43200.
    assert previous_release >= 0
    return (previous_storage, previous_release, 0., 100., 1, _parameters(inflow_enabled=True))


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("dt", [86400., 30.5 * 86400.])
@pytest.mark.parametrize("storage", [5e7, 1e8 + 1000., 1.5e8, 3.5e8, 7e8, 9e8])
def test_unique_solution_uses_exact_strict_arithmetic(policy, dt, storage):
    inputs = (storage, 0., 0., 0., 1, _parameters())
    expected = solve_implicit_step(*inputs, dt_seconds=dt)
    actual = solve_root_policy_step(*inputs, dt_seconds=dt, policy=policy)
    assert (actual.storage_m3, actual.release_m3s) == expected
    assert actual.root_count == 1
    assert actual.policy_applied is False
    assert actual.tie_break_applied is False
    assert actual.candidate_storage_m3 == (expected[0],)
    assert actual.candidate_release_m3s == (expected[1],)


def test_default_and_explicit_reject_preserve_existing_multiple_root_error():
    inputs = _multiple_inputs()
    with pytest.raises(ValueError) as original:
        solve_implicit_step(*inputs)
    for kwargs in ({}, {"policy": "reject"}):
        with pytest.raises(ValueError) as actual:
            solve_root_policy_step(*inputs, **kwargs)
        assert str(actual.value) == str(original.value)
    with pytest.raises(ValueError) as unchanged:
        solve_implicit_step(*inputs)
    assert str(unchanged.value) == str(original.value)


@pytest.mark.parametrize("policy,index", [("minimum_storage", 0), ("maximum_storage", 1), ("nearest_previous_storage", 0)])
def test_multiple_policies_select_paired_analytical_roots(policy, index):
    inputs = _multiple_inputs()
    result = solve_root_policy_step(*inputs, policy=policy)
    expected = _analytical_roots()
    assert result.root_count == 2 and result.policy_applied and not result.tie_break_applied
    np.testing.assert_allclose(result.candidate_storage_m3, [pair[0] for pair in expected], rtol=0, atol=1e-6)
    np.testing.assert_allclose(result.candidate_release_m3s, [pair[1] for pair in expected], rtol=0, atol=1e-9)
    assert result.storage_m3 == result.candidate_storage_m3[index]
    assert result.release_m3s == result.candidate_release_m3s[index]
    assert result.release_m3s == dztr_release(result.storage_m3, 100., 1, inputs[-1])
    residual = result.storage_m3 - 802e6 + 43200. * result.release_m3s
    assert abs(residual) <= max(1e-4, 64 * ulp(802e6))
    assert result.storage_m3 != np.mean(result.candidate_storage_m3)


@pytest.mark.parametrize("offset,index,tie", [(0., 0, True), (2.5e-5, 0, True), (2e-4, 1, False)])
def test_nearest_distance_ties_choose_minimum_storage(offset, index, tie):
    roots = _analytical_roots()
    midpoint = (roots[0][0] + roots[1][0]) / 2
    result = solve_root_policy_step(*_multiple_inputs(midpoint + offset), policy="nearest_previous_storage")
    assert result.storage_m3 == result.candidate_storage_m3[index]
    assert result.tie_break_applied is tie


def test_nearest_can_choose_high_root_when_previous_state_is_closer():
    result = solve_root_policy_step(*_multiple_inputs(_analytical_roots()[1][0]), policy="nearest_previous_storage")
    assert result.storage_m3 == result.candidate_storage_m3[-1]
    assert not result.tie_break_applied


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("case", ["no_root", "negative_predictor"])
def test_no_policy_invents_a_root_for_gap_or_negative_predictor(policy, case):
    if case == "no_root":
        inputs, dt = (280e6, 0., 0., 0., 1, _parameters(large_release=True)), 30.5 * 86400.
    else:
        inputs, dt = (0., 100., 0., 0., 1, _parameters()), 86400.
    with pytest.raises(ValueError) as expected:
        solve_implicit_step(*inputs, dt_seconds=dt)
    with pytest.raises(ValueError) as actual:
        solve_root_policy_step(*inputs, dt_seconds=dt, policy=policy)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("sampling,index", [
    (SamplingSettings(), pd.date_range("2019-12-28", periods=68)),
    (SamplingSettings("monthly"), pd.date_range("2020-01-01", periods=14, freq="MS")),
    (SamplingSettings("monthly", 30.5), pd.date_range("2020-01-01", periods=14, freq="MS")),
    (SamplingSettings("monthly", 30.5, "legacy", "2020-01-01"), pd.date_range("2020-01-01", periods=14, freq="MS")),
])
@pytest.mark.parametrize("policy", POLICIES)
def test_unique_simulation_common_columns_exact_across_native_clocks(sampling, index, policy):
    observations = _observations(index, np.linspace(1., 3., len(index)))
    parameters = _parameters(distinct_months=True)
    expected = simulate_sampled_reservoir(observations, parameters, sampling=sampling)
    actual = simulate_root_policy_reservoir(observations, parameters, sampling=sampling, policy=policy)
    pd.testing.assert_frame_equal(actual.frame[expected.frame.columns], expected.frame, check_exact=True)
    assert (actual.frame.root_count.iloc[1:] == 1).all()
    assert not actual.frame.root_policy_applied.any()
    assert not actual.frame.root_policy_tie_break.any()


@pytest.mark.parametrize("frequency,days", [("daily", 1.), ("monthly", 30.5)])
@pytest.mark.parametrize("policy", POLICIES)
def test_conservative_overflow_arithmetic_and_conservation_exact(frequency, days, policy):
    index = pd.date_range("2020-01-01", periods=4, freq="D" if frequency == "daily" else "MS")
    observations = _observations(index, np.full(4, 1000.))
    parameters = _parameters(initial_storage=9.9e8)
    sampling = SamplingSettings(frequency, days)
    expected = simulate_sampled_reservoir(observations, parameters, sampling=sampling)
    result = simulate_root_policy_reservoir(observations, parameters, sampling=sampling, policy=policy)
    pd.testing.assert_frame_equal(result.frame[expected.frame.columns], expected.frame, check_exact=True)
    assert result.frame.overflow_volume_m3.max() > 0
    assert result.frame.storage_m3.max() <= parameters.capacity_m3
    assert result.frame.balance_residual_m3.abs().max() <= 1e-4
    audit_columns = ["selected_root_storage_m3", "selected_root_release_m3s", "root_admissibility_tolerance_m3"]
    assert result.frame[audit_columns].iloc[0].isna().all()
    for i in range(1, len(result.frame)):
        previous, current = result.frame.iloc[i - 1], result.frame.iloc[i]
        step_seconds = days * 86400.
        raw_storage, raw_release = solve_implicit_step(
            previous.storage_m3, previous.release_m3s, previous.inflow_m3s,
            current.inflow_m3s, current.name.month, parameters, dt_seconds=step_seconds)
        predictor = previous.storage_m3 + step_seconds / 2. * (
            parameters.inflow_correction * (previous.inflow_m3s + current.inflow_m3s)
            - previous.release_m3s)
        assert current.selected_root_storage_m3 == raw_storage
        assert current.selected_root_release_m3s == raw_release
        assert current.root_admissibility_tolerance_m3 == max(1e-4, 64. * ulp(max(abs(predictor), 1.)))
        excess = max(raw_storage - parameters.capacity_m3, 0.)
        assert current.overflow_volume_m3 == excess
        assert current.storage_m3 == min(raw_storage, parameters.capacity_m3)
        assert current.release_m3s == raw_release + excess / step_seconds * 2.


@pytest.mark.parametrize("policy", ["minimum_storage", "maximum_storage", "nearest_previous_storage"])
def test_selected_multiple_root_simulation_remains_on_rule_and_conserves_water(policy):
    inputs = _multiple_inputs()
    parameters = replace(inputs[-1], initial_storage_m3=inputs[0], initial_release_m3s=inputs[1])
    observations = _observations(pd.date_range("2020-01-01", periods=2), [0., 100.])
    with pytest.raises(ValueError, match="Multiple"):
        simulate_sampled_reservoir(observations, parameters)
    with pytest.raises(ValueError, match="Multiple"):
        simulate_root_policy_reservoir(observations, parameters)
    result = simulate_root_policy_reservoir(observations, parameters, policy=policy)
    assert result.frame.iloc[1].root_count == 2
    assert result.frame.iloc[1].root_policy_applied
    assert result.frame.balance_residual_m3.abs().max() <= 1e-4
    assert result.frame.release_rule_residual_m3s.iloc[1] == 0
    assert result.frame.iloc[1].release_m3s == dztr_release(result.frame.iloc[1].storage_m3, 100., 1, parameters)
    assert result.overflow_mode == "conservative"


def test_multiple_root_selection_precedes_capacity_clamp_and_conservative_spill():
    previous_storage, predictor, inflow = 9.99e8, 1.1e9, 10000.
    previous_release = inflow - (predictor - previous_storage) / 43200.
    parameters = _parameters(initial_storage=previous_storage, initial_release=previous_release, inflow_enabled=True)
    inputs = (previous_storage, previous_release, 0., inflow, 1, parameters)
    step = solve_root_policy_step(*inputs, policy="maximum_storage")
    assert step.root_count == 2
    assert step.candidate_storage_m3[0] < parameters.capacity_m3 < step.storage_m3
    excess = step.storage_m3 - parameters.capacity_m3
    observations = _observations(pd.date_range("2020-01-01", periods=2), [0., inflow])
    maximum = simulate_root_policy_reservoir(observations, parameters, policy="maximum_storage")
    minimum = simulate_root_policy_reservoir(observations, parameters, policy="minimum_storage")
    assert maximum.frame.iloc[1].storage_m3 == parameters.capacity_m3
    assert maximum.frame.iloc[1].overflow_volume_m3 == excess
    assert maximum.frame.iloc[1].release_m3s == step.release_m3s + excess / 86400. * 2.
    assert maximum.frame.iloc[1].selected_root_storage_m3 == step.storage_m3
    assert maximum.frame.iloc[1].selected_root_release_m3s == step.release_m3s
    assert maximum.frame.iloc[1].root_admissibility_tolerance_m3 == max(1e-4, 64. * ulp(predictor))
    assert minimum.frame.iloc[1].overflow_volume_m3 == 0
    assert minimum.frame.iloc[1].storage_m3 == step.candidate_storage_m3[0]
    assert minimum.frame.iloc[1].selected_root_storage_m3 == step.candidate_storage_m3[0]
    assert minimum.frame.iloc[1].selected_root_release_m3s == step.candidate_release_m3s[0]
    assert minimum.frame.iloc[1].root_admissibility_tolerance_m3 == max(1e-4, 64. * ulp(predictor))
    assert maximum.frame.balance_residual_m3.abs().max() <= 1e-4
    assert minimum.frame.balance_residual_m3.abs().max() <= 1e-4


@pytest.mark.parametrize("policy", POLICIES)
def test_no_root_simulation_remains_error_with_explicit_convention(policy):
    observations = _observations(pd.date_range("2020-01-01", periods=2, freq="MS"), [0., 0.])
    parameters = _parameters(initial_storage=280e6, initial_release=0., large_release=True)
    with pytest.raises(ValueError, match="No feasible"):
        simulate_root_policy_reservoir(observations, parameters, sampling=SamplingSettings("monthly", 30.5), policy=policy)


def test_explicit_root_convention_does_not_authorize_signed_forcing():
    observations = _observations(pd.date_range("2020-01-01", periods=2), [-1., 0.])
    with pytest.raises(ValueError, match="nonnegative forcing"):
        simulate_root_policy_reservoir(observations, _parameters(), policy="minimum_storage")
    with pytest.raises(ValueError, match="signed-forcing policy"):
        simulate_root_policy_reservoir(observations, _parameters(), sampling=SamplingSettings(allow_signed_inflow=True), policy="minimum_storage")


@pytest.mark.parametrize("policy", ["min", "nearest", "automatic", None])
def test_unrecognized_convention_rejected(policy):
    with pytest.raises(ValueError, match="policy must be"):
        solve_root_policy_step(4e8, 1., 1., 1., 1, _parameters(), policy=policy)


def _calibration_fixture(*, inflow_enabled=False):
    index = pd.date_range("2018-01-01", periods=60, freq="MS", name="date")
    year = index.year - 2018
    frame = pd.DataFrame({"storage_m3": 2e8 + year * 1e8 + index.month * 1e5,
                          "inflow_m3s": 2. + np.sin(np.arange(60)),
                          "release_m3s": 1. + year * .4 + index.month * .01}, index=index)
    observations = ReservoirObservations(frame, DataQuality("available", len(frame), {column: 0 for column in frame}, 0), "portable")
    sampling = SamplingSettings("monthly", 30.5)
    split = default_sampled_split(observations, sampling)
    fitting = target_observations(select_sampled_period(observations, split.calibration, sampling))
    estimates = estimate_generalized_parameters(fitting, channel_capacity_m3s=20., quantile_method="hazen")
    template = replace(_parameters(inflow_enabled=inflow_enabled), monthly_targets=estimates.monthly_targets)
    bounds = estimate_sampled_bounds(observations, split.calibration, template, sampling, quantile_method="hazen")
    return observations, sampling, split, template, bounds


@pytest.mark.parametrize("policy", ["minimum_storage", "maximum_storage", "nearest_previous_storage"])
def test_calibration_and_bounds_ignore_poisoned_validation_under_fixed_policy(policy):
    observations, sampling, split, template, bounds = _calibration_fixture(inflow_enabled=True)
    vector = parameters_to_vector(template)
    original = RootPolicyCalibrationObjective(observations, split, template, sampling=sampling, bounds=bounds, policy=policy)
    expected = original.evaluate(vector)
    assert expected.feasible, expected.reason
    altered = observations.frame.copy()
    validation = altered.index >= split.validation.start
    altered.loc[validation, "storage_m3"] = 9e8
    altered.loc[validation, "release_m3s"] = 9999.
    altered.loc[validation, "inflow_m3s"] = np.nan
    changed = replace(observations, frame=altered)
    new_bounds = estimate_sampled_bounds(changed, split.calibration, template, sampling, quantile_method="hazen")
    fitting = target_observations(select_sampled_period(changed, split.calibration, sampling))
    new_targets = estimate_generalized_parameters(fitting, channel_capacity_m3s=20., quantile_method="hazen")
    pd.testing.assert_frame_equal(new_bounds.monthly_bounds, bounds.monthly_bounds, check_exact=True)
    pd.testing.assert_frame_equal(new_targets.monthly_targets, template.monthly_targets, check_exact=True)
    objective = RootPolicyCalibrationObjective(changed, split, template, sampling=sampling, bounds=new_bounds, policy=policy)
    actual = objective.evaluate(vector)
    assert actual.feasible
    assert actual.objectives == expected.objectives
    assert actual.metrics == expected.metrics
    pd.testing.assert_frame_equal(actual.simulation.frame, expected.simulation.frame, check_exact=True)
    assert objective.root_policy == policy
    with pytest.raises(ValueError, match="finite forcing"):
        objective.evaluate_validation(vector)
    assert objective.root_policy == policy


def test_unique_calibration_and_separate_validation_match_strict_objective_exactly():
    observations, sampling, split, template, bounds = _calibration_fixture()
    vector = parameters_to_vector(template)
    strict = SampledCalibrationObjective(observations, split, template, sampling=sampling, bounds=bounds)
    declared = RootPolicyCalibrationObjective(observations, split, template, sampling=sampling, bounds=bounds, policy="maximum_storage")
    for expected, actual in ((strict.evaluate(vector), declared.evaluate(vector)),
                             (strict.evaluate_validation(vector), declared.evaluate_validation(vector))):
        assert expected.feasible and actual.feasible
        assert actual.objectives == expected.objectives
        assert actual.metrics == expected.metrics
        assert actual.period == expected.period
        assert actual.paired_counts == expected.paired_counts
        pd.testing.assert_frame_equal(actual.simulation.frame[expected.simulation.frame.columns], expected.simulation.frame, check_exact=True)


@pytest.mark.parametrize("policy", ["minimum_storage", "maximum_storage", "nearest_previous_storage"])
def test_validation_under_policy_carries_whole_continuous_state(policy):
    observations, sampling, split, template, bounds = _calibration_fixture(inflow_enabled=True)
    objective = RootPolicyCalibrationObjective(observations, split, template, sampling=sampling, bounds=bounds, policy=policy)
    evaluation = objective.evaluate_validation(parameters_to_vector(template))
    assert evaluation.feasible, evaluation.reason
    full = simulate_root_policy_reservoir(observations, template, sampling=sampling, policy=policy)
    pd.testing.assert_frame_equal(evaluation.simulation.frame, full.frame, check_exact=True)
    assert evaluation.period == split.validation
    assert evaluation.paired_counts == {"release": 24, "storage": 24}
    validation_observations = select_sampled_period(observations, split.validation, sampling)
    restarted = simulate_root_policy_reservoir(validation_observations, template, sampling=sampling, policy=policy)
    assert restarted.frame.iloc[0].storage_m3 != full.frame.loc[split.validation.start, "storage_m3"]


# Frozen first failed steps; no fixtures depend on ignored output folders.
_KNOWN_FAILURES = [
    ("Glen Canyon", 29921548977.2106, 2998.7486281077877, 3155.0109442829116, 3318.1907371278344, 6, 86400.,
     33304000000., .1, 2600., [26316820906.207207, 28503425126.9448, 30011811068.7648, 229.08328892928006, 543.6834545664001, 1318.1492088576001],
     [(29934446981.110535, 3175.8881482118354), (30014701303.29064, 1318.1492088576001)]),
    ("Fort Randall", 4924408179.426898, 557.8991224789247, 716.42, 1050.55, 4, 2635200.,
     5800000000., .19648275862068965, 2000., [4520700000., 4944400000., 5276800000., 321.68, 711.8820000000001, 907.836],
     [(5215528692.313664, 988.1233115778703), (5321315254.048666, 907.836)]),
]


@pytest.mark.parametrize("case", _KNOWN_FAILURES, ids=[row[0] for row in _KNOWN_FAILURES])
def test_frozen_failure_roots_and_literal_1000_iterations(case):
    name, s, q, previous_inflow, inflow, month, dt, capacity, dead, channel, target_row, roots = case
    targets = pd.DataFrame([target_row] * 12, columns=MONTHLY_TARGET_COLUMNS, index=pd.Index(range(1, 13), name="month"))
    parameters = ReservoirParameters(targets, capacity, s, q, channel, dead, 1., "inflow")
    inputs = (s, q, previous_inflow, inflow, month, parameters)
    with pytest.raises(ValueError, match="Multiple"):
        solve_implicit_step(*inputs, dt_seconds=dt)
    low = solve_root_policy_step(*inputs, dt_seconds=dt, policy="minimum_storage")
    high = solve_root_policy_step(*inputs, dt_seconds=dt, policy="maximum_storage")
    assert (low.storage_m3, low.release_m3s) == roots[0]
    assert (high.storage_m3, high.release_m3s) == roots[1]
    assert high.storage_m3 < capacity
    assert low.storage_m3 < capacity
    fixed = fixed_point_step(*inputs, dt_seconds=dt, max_iterations=1000)
    assert fixed == roots[-1]  # Evidence for these two individual steps only.
    predictor = s + dt / 2 * (inflow + previous_inflow - q)
    iterate_q = dztr_release(predictor, inflow, month, parameters, dt)
    tail = []
    for iteration in range(1000):
        iterate_s = predictor - dt / 2 * iterate_q
        iterate_q = dztr_release(iterate_s, inflow, month, parameters, dt)
        if iteration >= 900:
            tail.append((iterate_s, iterate_q))
    assert len(set(tail)) == 1
    assert tail[-1] == fixed


def test_fixed_iteration_cycle_is_reference_behavior_and_never_a_no_root_fallback():
    parameters, predictor, dt = _parameters(large_release=True), 280e6, 30.5 * 86400.
    inputs = (predictor, 0., 0., 0., 1, parameters)
    q = dztr_release(predictor, 0., 1, parameters, dt)
    tail = []
    for iteration in range(1000):
        s = predictor - dt / 2 * q
        q = dztr_release(s, 0., 1, parameters, dt)
        if iteration >= 900:
            tail.append((s, q))
    assert len(set(tail)) == 2
    assert all(tail[index] == tail[index + 2] for index in range(len(tail) - 2))
    assert (s, q) == fixed_point_step(*inputs, dt_seconds=dt, max_iterations=1000)
    assert abs(s - predictor + dt / 2 * q) > 1e6
    for policy in POLICIES:
        with pytest.raises(ValueError, match="No feasible"):
            solve_root_policy_step(*inputs, dt_seconds=dt, policy=policy)
