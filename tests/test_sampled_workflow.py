"""Native-sampling balance and calibration-period checks on portable records."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.sampled import (
    SampledCalibrationObjective, default_sampled_split, estimate_sampled_bounds,
    run_sampled_baseline, select_sampled_period, validate_split,
)
from dztr.model.release import _monthly_values, _release_for_values
from dztr.schemas import DataQuality, MONTHLY_TARGET_COLUMNS, ReservoirObservations, ReservoirParameters
from dztr.simulation.runner import simulate_reservoir
from dztr.simulation.sampled import LegacyReservoirParameters, SamplingSettings, simulate_sampled_reservoir


def _observations(dates, *, inflow=None, storage=None, release=None):
    dates = pd.DatetimeIndex(dates, name="date")
    n = len(dates)
    frame = pd.DataFrame({
        "storage_m3": np.linspace(3e8, 5e8, n) if storage is None else storage,
        "inflow_m3s": np.linspace(1, 3, n) if inflow is None else inflow,
        "release_m3s": np.linspace(1, 2, n) if release is None else release,
    }, index=dates)
    return ReservoirObservations(frame, DataQuality("available", n, {key: int(frame[key].isna().sum()) for key in frame}, 0), "portable")


def _parameters(*, initial_storage=4e8, initial_release=1, distinct_months=False):
    rows = [[2e8, 5e8, 8e8, 1 + (month / 10 if distinct_months else 0), 2 + (month / 10 if distinct_months else 0), 3 + (month / 10 if distinct_months else 0)] for month in range(1, 13)]
    monthly = pd.DataFrame(rows, index=pd.Index(range(1, 13), name="month"), columns=MONTHLY_TARGET_COLUMNS)
    return ReservoirParameters(monthly, 1e9, initial_storage, initial_release, 20, 0.1, 1, "storage")


def _legacy(parameters, **updates):
    values = {key: getattr(parameters, key) for key in ("monthly_targets", "capacity_m3", "initial_storage_m3", "initial_release_m3s", "channel_capacity_m3s", "dead_storage_fraction", "inflow_correction", "flood_mode")}
    values.update(updates)
    return LegacyReservoirParameters(**values)


@pytest.mark.parametrize("solver", ["implicit", "matlab_fixed_point"])
@pytest.mark.parametrize("overflow", ["conservative", "matlab"])
def test_daily_delegate_complete_frame_exact(solver, overflow):
    observations = _observations(pd.date_range("2019-12-28", periods=68))
    parameters = _parameters(distinct_months=True)
    expected = simulate_reservoir(observations, parameters, solver=solver, overflow_mode=overflow, max_iterations=7)
    actual = simulate_sampled_reservoir(observations, parameters, sampling=SamplingSettings(), solver=solver, overflow_mode=overflow, max_iterations=7)
    pd.testing.assert_frame_equal(actual.frame, expected.frame, check_exact=True)
    assert actual.issues == expected.issues


@pytest.mark.parametrize("fixed_days", [None, 30.5])
def test_monthly_balance_integrates_elapsed_or_explicit_duration(fixed_days):
    observations = _observations(pd.date_range("2020-01-01", periods=3, freq="MS"), inflow=[1, 2, 3])
    parameters = _parameters(initial_storage=1e6, initial_release=0)
    result = simulate_sampled_reservoir(observations, parameters, sampling=SamplingSettings("monthly", fixed_days))
    days = [31, 29] if fixed_days is None else [30.5, 30.5]
    expected = [1e6, 1e6 + days[0] * 86400 * 1.5, 1e6 + days[0] * 86400 * 1.5 + days[1] * 86400 * 2.5]
    np.testing.assert_array_equal(result.frame.storage_m3, expected)
    np.testing.assert_array_equal(result.frame.release_m3s, [0, 0, 0])
    np.testing.assert_array_equal(result.frame.step_seconds.iloc[1:], np.asarray(days) * 86400)
    assert result.frame.balance_residual_m3.abs().max() == 0


def test_legacy_month_clock_explicitly_drifts_from_calendar_targets():
    observations = _observations(pd.date_range("2020-01-01", periods=4, freq="MS"))
    parameters = _parameters(distinct_months=True)
    observed = simulate_sampled_reservoir(observations, parameters, sampling=SamplingSettings("monthly", 30.5))
    legacy = simulate_sampled_reservoir(observations, parameters, sampling=SamplingSettings("monthly", 30.5, "legacy", "2020-01-01"))
    assert observed.frame.rule_month.tolist() == [1, 2, 3, 4]
    assert legacy.frame.rule_month.tolist() == [1, 1, 3, 4]
    assert observed.frame.iloc[1].release_m3s != legacy.frame.iloc[1].release_m3s


def _literal_fixed_point(observations, parameters, step, iterations, months, overflow):
    months = np.asarray(months)
    storage = [parameters.initial_storage_m3]
    release = [parameters.initial_release_m3s]
    forcing = observations.frame.inflow_m3s.to_numpy()
    for i in range(1, len(forcing)):
        predictor = storage[-1] + step / 2 * (parameters.inflow_correction * (forcing[i] + forcing[i-1]) - release[-1])
        values = _monthly_values(int(months[i]), parameters)
        q = _release_for_values(predictor, forcing[i], values, step)
        for _ in range(iterations):
            s = predictor - step / 2 * q
            q = _release_for_values(s, forcing[i], values, step)
        if overflow == "matlab" and s > parameters.capacity_m3:
            q += (s - parameters.capacity_m3) / step
            s = parameters.capacity_m3
        storage.append(s)
        release.append(q)
    return np.array(storage), np.array(release)


@pytest.mark.parametrize("overflow", ["matlab", "none"])
@pytest.mark.parametrize("iterations", [1, 2, 7])
def test_unmodified_legacy_table_dead_equality_and_overflow_replay(overflow, iterations):
    parameters = _parameters(initial_storage=9e8, initial_release=1)
    targets = parameters.monthly_targets.copy()
    targets["storage_critical_m3"] = 1e8  # Literal legacy dead == critical.
    legacy_parameters = _legacy(parameters, monthly_targets=targets)
    observations = _observations(pd.date_range("2020-01-01", periods=4, freq="MS"), inflow=[100, 100, 100, 100])
    sampling = SamplingSettings("monthly", 30.5, "legacy", "2020-01-01")
    result = simulate_sampled_reservoir(observations, legacy_parameters, sampling=sampling, solver="matlab_fixed_point", overflow_mode=overflow, max_iterations=iterations)
    expected_storage, expected_release = _literal_fixed_point(observations, legacy_parameters, 30.5 * 86400, iterations, result.frame.rule_month, overflow)
    np.testing.assert_array_equal(result.frame.storage_m3, expected_storage)
    np.testing.assert_array_equal(result.frame.release_m3s, expected_release)
    if overflow == "none":
        assert result.frame.storage_m3.max() > parameters.capacity_m3
    else:
        assert result.frame.storage_m3.max() <= parameters.capacity_m3


def test_signed_forcing_requires_explicit_policy_and_stays_signed():
    observations = _observations(pd.date_range("2020-01-01", periods=3, freq="MS"), inflow=[-0.1, -0.1, -0.1])
    parameters = _parameters(initial_storage=1e6, initial_release=0)
    with pytest.raises(ValueError, match="signed values"):
        simulate_sampled_reservoir(observations, parameters, sampling=SamplingSettings("monthly", 30.5))
    result = simulate_sampled_reservoir(observations, parameters, sampling=SamplingSettings("monthly", 30.5, allow_signed_inflow=True))
    np.testing.assert_array_equal(result.frame.inflow_m3s, [-0.1, -0.1, -0.1])
    assert any("Signed" in issue for issue in result.issues)


def test_legacy_dead_equals_critical_executes_dead_branch_without_repair():
    parameters = _parameters(initial_storage=1e8, initial_release=0)
    targets = parameters.monthly_targets.copy()
    targets["storage_critical_m3"] = 1e8
    parameters = _legacy(parameters, monthly_targets=targets)
    observations = _observations(pd.date_range("2020-01-01", periods=3, freq="MS"), inflow=[0, 0, 0])
    result = simulate_sampled_reservoir(observations, parameters, sampling=SamplingSettings("monthly", 30.5), solver="matlab_fixed_point", overflow_mode="none", max_iterations=2)
    np.testing.assert_array_equal(result.frame.storage_m3, [1e8, 1e8, 1e8])
    np.testing.assert_array_equal(result.frame.release_m3s, [0, 0, 0])
    assert (parameters.monthly_targets.storage_critical_m3 == 1e8).all()


@pytest.mark.parametrize("dates,frequency", [(["2020-01-01", "2020-03-01"], "monthly"), (["2020-01-01", "2020-01-03"], "daily")])
def test_missing_native_intervals_are_rejected(dates, frequency):
    with pytest.raises(ValueError, match="missing intervals are not filled"):
        simulate_sampled_reservoir(_observations(pd.to_datetime(dates)), _parameters(), sampling=SamplingSettings(frequency))


def _history():
    dates = pd.date_range("2018-01-01", periods=60, freq="MS")
    year = dates.year - 2018
    return _observations(dates, storage=2e8 + year * 1e8 + dates.month * 1e5,
                         inflow=2 + np.sin(np.arange(60)), release=1 + year * 0.4 + dates.month * 0.01)


def test_default_split_native_warmup_and_final_partial_year():
    observations = _history()
    split = default_sampled_split(observations, SamplingSettings("monthly"))
    assert split.warmup_rows == 12
    assert split.calibration.start == pd.Timestamp("2019-01-01")
    assert split.calibration.end == pd.Timestamp("2020-12-01")
    partial = replace(observations, frame=observations.frame.iloc[:-3].copy())
    assert default_sampled_split(partial, SamplingSettings("monthly")).validation.end == pd.Timestamp("2022-09-01")
    with pytest.raises(ValueError, match="warmup_rows disagrees"):
        validate_split(observations, replace(split, warmup_rows=365), SamplingSettings("monthly"))


def test_calibration_bounds_ignore_warmup_and_validation_measurements():
    observations = _history()
    sampling = SamplingSettings("monthly")
    split = default_sampled_split(observations, sampling)
    altered = observations.frame.copy()
    outside = (altered.index < split.calibration.start) | (altered.index > split.calibration.end)
    altered.loc[outside, "storage_m3"] = 9e8
    altered.loc[outside, "release_m3s"] = 9000
    changed = replace(observations, frame=altered)
    expected = estimate_sampled_bounds(observations, split.calibration, _parameters(), sampling)
    actual = estimate_sampled_bounds(changed, split.calibration, _parameters(), sampling)
    pd.testing.assert_frame_equal(actual.monthly_bounds, expected.monthly_bounds, check_exact=True)


def test_calibration_cost_ignores_validation_forcing_and_validation_checks_it():
    observations = _history()
    sampling = SamplingSettings("monthly")
    split = default_sampled_split(observations, sampling)
    template = _parameters()
    vector = parameters_to_vector(template)
    expected = SampledCalibrationObjective(observations, split, template, sampling=sampling).evaluate(vector)
    altered = observations.frame.copy()
    validation = altered.index >= split.validation.start
    altered.loc[validation, "inflow_m3s"] = np.nan
    altered.loc[validation, "storage_m3"] = 9e8
    altered.loc[validation, "release_m3s"] = 9000
    objective = SampledCalibrationObjective(replace(observations, frame=altered), split, template, sampling=sampling)
    actual = objective.evaluate(vector)
    assert expected.feasible and actual.feasible
    assert actual.objectives == expected.objectives
    pd.testing.assert_frame_equal(actual.simulation.frame, expected.simulation.frame, check_exact=True)
    with pytest.raises(ValueError, match="finite forcing"):
        objective.evaluate_validation(vector)


def test_validation_carries_continuous_state_and_scores_only_validation():
    observations = _history()
    sampling = SamplingSettings("monthly")
    split = default_sampled_split(observations, sampling)
    template = _parameters()
    objective = SampledCalibrationObjective(observations, split, template, sampling=sampling)
    evaluation = objective.evaluate_validation(parameters_to_vector(template))
    assert evaluation.feasible
    full = simulate_sampled_reservoir(observations, template, sampling=sampling)
    pd.testing.assert_frame_equal(evaluation.simulation.frame, full.frame, check_exact=True)
    assert evaluation.period == split.validation
    assert evaluation.paired_counts == {"release": 24, "storage": 24}
    validation_observations = select_sampled_period(observations, split.validation, sampling)
    restarted = simulate_sampled_reservoir(validation_observations, template, sampling=sampling)
    assert restarted.frame.iloc[0].storage_m3 != evaluation.simulation.frame.loc[split.validation.start, "storage_m3"]


def test_baseline_targets_ignore_validation_and_warmup_data():
    observations = _history()
    sampling = SamplingSettings("monthly")
    split = default_sampled_split(observations, sampling)
    model = {key: getattr(_parameters(), key) for key in ("capacity_m3", "initial_storage_m3", "initial_release_m3s", "channel_capacity_m3s", "dead_storage_fraction", "inflow_correction", "flood_mode")}
    expected = run_sampled_baseline(observations, split, sampling=sampling, model=model)
    altered = observations.frame.copy()
    outside = (altered.index < split.calibration.start) | (altered.index > split.calibration.end)
    altered.loc[outside, "storage_m3"] *= 1.5
    altered.loc[outside, "release_m3s"] *= 1.5
    actual = run_sampled_baseline(replace(observations, frame=altered), split, sampling=sampling, model=model)
    pd.testing.assert_frame_equal(actual.estimates.monthly_targets, expected.estimates.monthly_targets, check_exact=True)
    pd.testing.assert_frame_equal(actual.simulation.frame, expected.simulation.frame, check_exact=True)
    assert actual.summary()["parameter_estimation_period"] == split.calibration.as_dict()
    assert actual.table().period.value_counts().to_dict() == {"calibration": 24, "validation": 24, "spinup": 12}
