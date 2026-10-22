"""Calibration isolation, interval-mean total outflow and pooled clean blocks."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import dztr.calibration.intervals as objective_module
from dztr.calibration.bounds import DEFAULT_PERCENTILE_BOUNDS
from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.intervals import (
    IntervalCalibrationObjective, estimate_interval_bounds, fit_interval_targets,
)
from dztr.calibration.periods import DatePeriod
from dztr.data.intervals import DailyIntervalPolicy, prepare_daily_intervals
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters
from dztr.simulation.intervals import IntervalSimulationSettings, simulate_interval_reservoir


CALIBRATION = DatePeriod("2000-01-01", "2000-01-08")
VALIDATION = DatePeriod("2000-01-09", "2000-01-14")


def template():
    return ReservoirParameters(
        pd.DataFrame(np.tile([20e6, 50e6, 80e6, 1., 2., 3.], (12, 1)),
                     index=pd.Index(range(1, 13), name="month"), columns=MONTHLY_TARGET_COLUMNS),
        capacity_m3=100e6, initial_storage_m3=60e6, initial_release_m3s=999.,
        channel_capacity_m3s=4.,
    )


def preparation(quarantine=()):
    dates = pd.date_range("1999-12-29", VALIDATION.end, name="date")
    sequence = np.arange(len(dates), dtype=float)
    frame = pd.DataFrame({
        "storage_m3": 40e6 + sequence * 10000,
        "inflow_m3s": 3. + (sequence % 5) * .3,
        "release_m3s": 1. + sequence * .1,
    }, index=dates)
    return prepare_daily_intervals(frame, policy=DailyIntervalPolicy(
        quarantined_interval_end_dates=quarantine,
    ))


def problem(prepared=None, **options):
    return IntervalCalibrationObjective(
        preparation() if prepared is None else prepared,
        CALIBRATION, template(), **options,
    )


def negative_nse(observed, predicted):
    observed, predicted = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    return np.sum((observed - predicted) ** 2) / np.sum((observed - np.mean(observed)) ** 2) - 1


def test_objective_scores_daily_mean_total_outflow_and_storage_with_independent_formula():
    prepared = preparation()
    parameters = template()
    evaluated = problem(prepared).evaluate(parameters_to_vector(parameters))
    assert evaluated.feasible, evaluated.reason
    assert evaluated.period == CALIBRATION
    assert evaluated.metrics["conditional_hindcast"] is True
    assert any("conditional_hindcast" in issue for issue in evaluated.simulation.issues)
    observed = prepared.frame.loc[CALIBRATION.start:CALIBRATION.end]
    simulated = evaluated.simulation.frame.loc[observed.index]
    expected = [
        negative_nse(observed.release_m3s, simulated.interval_mean_total_outflow_m3s),
        negative_nse(observed.storage_m3, simulated.storage_m3),
    ]
    np.testing.assert_allclose(evaluated.objectives, expected)
    endpoint_cost = negative_nse(observed.release_m3s, simulated.release_endpoint_m3s)
    assert abs(evaluated.objectives[0] - endpoint_cost) > 1e-7
    assert evaluated.paired_counts == {"release": 8, "storage": 8}
    assert evaluated.simulation.frame.loc[VALIDATION.start:, "storage_m3"].isna().all()


def test_clean_blocks_are_pooled_once_and_missing_storage_release_pairs_are_independent():
    prepared = preparation(("2000-01-04",))
    prepared.frame.loc["2000-01-02", "release_m3s"] = np.nan
    prepared.frame.loc["2000-01-07", "storage_m3"] = np.nan
    evaluated = problem(prepared).evaluate(parameters_to_vector(template()))
    assert evaluated.feasible, evaluated.reason
    observed = prepared.frame.loc[CALIBRATION.start:CALIBRATION.end]
    simulated = evaluated.simulation.frame.loc[observed.index]
    expected = []
    for label, column, simulated_column in (
        ("release", "release_m3s", "interval_mean_total_outflow_m3s"),
        ("storage", "storage_m3", "storage_m3"),
    ):
        mask = observed[f"{label}_score_eligible"] & observed[column].notna()
        expected.append(negative_nse(observed.loc[mask, column], simulated.loc[mask, simulated_column]))
        assert evaluated.paired_counts[label] == int(mask.sum()) == 4
    np.testing.assert_allclose(evaluated.objectives, expected)
    assert simulated.loc["2000-01-05", "interval_mean_total_outflow_m3s"] != simulated.loc["2000-01-05", "interval_mean_total_outflow_m3s"]
    assert not prepared.frame.loc["2000-01-05", "storage_score_eligible"]


@pytest.mark.parametrize("column,value", [
    ("inflow_m3s", np.nan), ("inflow_m3s", np.inf), ("inflow_m3s", "malformed"),
    ("storage_m3", "malformed"), ("release_m3s", "malformed"),
])
def test_future_invalid_values_cannot_change_calibration_and_fail_only_explicit_validation(column, value):
    source = preparation()
    changed = deepcopy(source)
    if isinstance(value, str):
        changed.frame[column] = changed.frame[column].astype(object)
    changed.frame.loc[VALIDATION.start:, column] = value
    vector = parameters_to_vector(template())
    original = problem(source).evaluate(vector)
    alternative_problem = problem(changed)
    alternative = alternative_problem.evaluate(vector)
    assert alternative.feasible, alternative.reason
    np.testing.assert_array_equal(alternative.objectives, original.objectives)
    pd.testing.assert_frame_equal(
        alternative.simulation.frame.loc[:CALIBRATION.end],
        original.simulation.frame.loc[:CALIBRATION.end],
        check_dtype=False,  # Preserved source column may be upcast by a future malformed cell.
    )
    model_columns = original.simulation.frame.columns.difference(["reported_inflow_mean_m3s"])
    pd.testing.assert_frame_equal(
        alternative.simulation.frame.loc[:CALIBRATION.end, model_columns],
        original.simulation.frame.loc[:CALIBRATION.end, model_columns],
    )
    with pytest.raises(ValueError, match="inflow|observations|nonmissing"):
        alternative_problem.evaluate_validation(vector, VALIDATION)


def test_future_storage_and_release_are_not_constructor_requirements():
    source = preparation()
    source.frame.loc[VALIDATION.start:, ["storage_m3", "release_m3s"]] = np.nan
    objective = problem(source)
    assert objective.evaluate(parameters_to_vector(template())).feasible
    with pytest.raises(ValueError, match="at least two"):
        objective.evaluate_validation(parameters_to_vector(template()), VALIDATION)


def test_validation_carries_clean_calibration_boundary_and_is_explicit():
    source = preparation()
    labels = pd.Series(np.where(source.frame.index <= CALIBRATION.end, "calibration", "validation"), index=source.frame.index)
    source = prepare_daily_intervals(source.frame[["storage_m3", "inflow_m3s", "release_m3s"]], period_labels=labels)
    objective = problem(source)
    vector = parameters_to_vector(template())
    calibrated = objective.evaluate(vector)
    held_out = objective.evaluate_validation(vector, VALIDATION)
    assert held_out.feasible, held_out.reason
    assert held_out.period == VALIDATION
    assert held_out.paired_counts == {"release": 6, "storage": 6}
    pd.testing.assert_frame_equal(held_out.simulation.frame.loc[:CALIBRATION.end], calibrated.simulation.frame.loc[:CALIBRATION.end])
    assert held_out.simulation.frame.loc[VALIDATION.start, "interval_mean_total_outflow_m3s"] > 0
    assert held_out.simulation.frame.loc[VALIDATION.start, "storage_m3"] != source.frame.loc[VALIDATION.start, "storage_m3"]
    assert not source.frame.loc[VALIDATION.start, "source_audit_interval_eligible"]
    assert source.frame.loc[VALIDATION.start, "simulation_interval_eligible"]


def test_explicit_midblock_simulation_start_initializes_once_and_excludes_that_label():
    source = preparation()
    original = deepcopy(source.frame)
    objective = problem(source, simulation_start=CALIBRATION.start)
    result = objective.evaluate(parameters_to_vector(template()))
    assert result.feasible, result.reason
    assert result.paired_counts == {"release": 7, "storage": 7}
    assert np.isnan(result.simulation.frame.loc[CALIBRATION.start, "interval_mean_total_outflow_m3s"])
    pd.testing.assert_frame_equal(source.frame, original, check_exact=True)


def test_source_flag_after_calibration_end_still_quarantines_its_preceding_endpoint():
    source = preparation(("2000-01-09",))
    objective = problem(source)
    result = objective.evaluate(parameters_to_vector(template()))
    assert result.feasible, result.reason
    assert result.paired_counts == {"release": 7, "storage": 7}
    assert np.isnan(result.simulation.frame.loc[CALIBRATION.end, "storage_m3"])
    assert not source.frame.loc[CALIBRATION.end, "storage_target_eligible"]


def test_caller_mutations_cannot_change_snapshot_targets_template_or_forcing():
    source, parameters = preparation(), template()
    objective = IntervalCalibrationObjective(source, CALIBRATION, parameters)
    vector = parameters_to_vector(parameters)
    original = objective.evaluate(vector)
    source.frame.loc[:, "inflow_m3s"] = 999.
    source.frame.loc[:, "storage_m3"] = 999.
    source.frame.loc[:, "release_m3s"] = 999.
    parameters.monthly_targets.loc[:, "release_normal_m3s"] = 999.
    alternative = objective.evaluate(vector)
    np.testing.assert_array_equal(alternative.objectives, original.objectives)
    pd.testing.assert_frame_equal(alternative.simulation.frame, original.simulation.frame)


def test_invalid_candidate_receives_infinite_costs_without_simulation(monkeypatch):
    objective = problem()
    monkeypatch.setattr(objective_module, "simulate_interval_reservoir", lambda *args, **kwargs: pytest.fail("Invalid candidate reached solver"))
    result = objective.evaluate(np.zeros(72))
    assert not result.feasible
    assert result.reason
    assert result.metrics == {}
    assert result.simulation is None
    assert np.isinf(result.objectives).all()


def test_strict_ambiguous_root_is_rejected_without_partially_scoring():
    parameters = replace(template(),
        monthly_targets=pd.DataFrame(np.tile([200e6, 500e6, 800e6, 1., 2., 100.], (12, 1)),
            index=pd.Index(range(1, 13), name="month"), columns=MONTHLY_TARGET_COLUMNS),
        capacity_m3=1e9, initial_storage_m3=799e6, channel_capacity_m3s=20.,
    )
    dates = pd.date_range("2000-01-01", periods=4, name="date")
    previous_storage = 799e6
    previous_release = 2. + (previous_storage - 500e6) / 300e6 * 98.
    desired_predictor = 802e6
    inflow = (desired_predictor - previous_storage + 43200 * previous_release) / 86400
    source = prepare_daily_intervals(pd.DataFrame({
        "storage_m3": previous_storage + np.arange(4) * 1000,
        "inflow_m3s": np.full(4, inflow),
        "release_m3s": np.arange(4, dtype=float) + 10.,
    }, index=dates))
    objective = IntervalCalibrationObjective(source, DatePeriod(dates[0], dates[-1]), parameters)
    result = objective.evaluate(parameters_to_vector(parameters))
    assert not result.feasible
    assert "multiple" in result.reason.lower()
    assert np.isinf(result.objectives).all()
    assert result.metrics == {}
    assert result.simulation is not None


@pytest.mark.parametrize("label", ["storage", "release"])
@pytest.mark.parametrize("mode", ["constant", "one_pair"])
def test_undefined_observed_nse_is_rejected_before_candidate_evaluation(label, mode):
    source = preparation()
    column = "storage_m3" if label == "storage" else "release_m3s"
    if mode == "constant":
        source.frame.loc[CALIBRATION.start:CALIBRATION.end, column] = 123.
    else:
        source.frame.loc[CALIBRATION.start:CALIBRATION.end, column] = np.nan
        source.frame.loc[CALIBRATION.start, column] = 123.
    with pytest.raises(ValueError, match=f"nonconstant {label}"):
        problem(source)


def annual_preparation():
    dates = pd.date_range("2000-01-01", "2001-01-31", name="date")
    day = dates.day.to_numpy(dtype=float)
    frame = pd.DataFrame({
        "storage_m3": 30e6 + day * 1e6,
        "inflow_m3s": np.where(dates.day == 3, -.2, 4.),
        "release_m3s": 1. + day * .1,
    }, index=dates)
    frame.loc["2000-01-14":"2000-01-15", "storage_m3"] = 9e8
    frame.loc["2000-01-14":"2000-01-16", "release_m3s"] = 9e5
    return prepare_daily_intervals(frame, policy=DailyIntervalPolicy(
        flow_semantics="computed_net_inflow", quarantined_interval_end_dates=("2000-01-15",),
    ))


ANNUAL_CALIBRATION = DatePeriod("2000-01-01", "2000-12-31")


def test_target_percentiles_use_only_clean_calibration_with_separate_sample_support():
    source = annual_preparation()
    source.frame.loc["2000-01-21", "storage_m3"] = np.nan
    source.frame.loc["2000-01-22", "release_m3s"] = np.nan
    original = deepcopy(source.frame)
    fitted = fit_interval_targets(source, ANNUAL_CALIBRATION, channel_capacity_m3s=5.)
    assert fitted.period == ANNUAL_CALIBRATION
    assert fitted.estimates.quantile_method == "hazen"
    assert fitted.estimates.channel_capacity_m3s == 5.
    january = source.frame.loc["2000-01-01":"2000-01-31"]
    storage = january.loc[january.storage_target_eligible, "storage_m3"].dropna()
    release = january.loc[january.simulation_interval_eligible, "release_m3s"].dropna()
    np.testing.assert_allclose(fitted.estimates.monthly_targets.loc[1].iloc[:3], np.percentile(storage, [10, 45, 85], method="hazen"))
    np.testing.assert_allclose(fitted.estimates.monthly_targets.loc[1].iloc[3:], np.percentile(release, [10, 45, 85], method="hazen"))
    assert fitted.support.loc[1, "storage_sample_count"] == len(storage) == 28
    assert fitted.support.loc[1, "release_sample_count"] == len(release) == 26
    pd.testing.assert_frame_equal(source.frame, original, check_exact=True)
    source.frame.loc["2001-01-01":, "storage_m3"] = np.nan
    source.frame["release_m3s"] = source.frame.release_m3s.astype(object)
    source.frame.loc["2001-01-01":, "release_m3s"] = "malformed"
    changed = fit_interval_targets(source, ANNUAL_CALIBRATION, channel_capacity_m3s=5.)
    pd.testing.assert_frame_equal(changed.estimates.monthly_targets, fitted.estimates.monthly_targets)
    pd.testing.assert_frame_equal(changed.support, fitted.support)


def test_interval_bounds_use_clean_calibration_samples_and_matching_physical_constants():
    source = annual_preparation()
    bounds = estimate_interval_bounds(source, ANNUAL_CALIBRATION, template())
    assert bounds.period == ANNUAL_CALIBRATION
    assert bounds.capacity_m3 == template().capacity_m3
    assert bounds.dead_storage_fraction == template().dead_storage_fraction
    january = source.frame.loc["2000-01-01":"2000-01-31"]
    for variable, stem, suffix, mask in (
        ("storage_m3", "storage", "m3", january.storage_target_eligible),
        ("release_m3s", "release", "m3s", january.simulation_interval_eligible),
    ):
        sample = january.loc[mask, variable].to_numpy()
        for level, (lower, upper) in zip(("critical", "normal", "flood"), DEFAULT_PERCENTILE_BOUNDS):
            expected = np.percentile(sample, [lower, upper], method="hazen")
            assert bounds.monthly_bounds.loc[1, f"lower_{stem}_{level}_{suffix}"] == expected[0]
            assert bounds.monthly_bounds.loc[1, f"upper_{stem}_{level}_{suffix}"] == expected[1]
    source.frame.loc["2001-01-01":, ["storage_m3", "release_m3s", "inflow_m3s"]] = np.nan
    changed = estimate_interval_bounds(source, ANNUAL_CALIBRATION, template())
    pd.testing.assert_frame_equal(changed.monthly_bounds, bounds.monthly_bounds)
    with pytest.raises(ValueError, match="match the calibration period"):
        IntervalCalibrationObjective(source, CALIBRATION, template(), bounds=bounds)
    mismatch = replace(bounds, capacity_m3=110e6)
    with pytest.raises(ValueError, match="same capacity"):
        IntervalCalibrationObjective(source, ANNUAL_CALIBRATION, template(), bounds=mismatch)


@pytest.mark.parametrize("unit", ["s", "ms", "us"])
def test_daily_labels_with_other_datetime_resolutions_are_accepted(unit):
    source = preparation()
    source.frame.index = source.frame.index.as_unit(unit)
    source.segments["start"] = source.segments.start.dt.as_unit(unit)
    source.segments["end"] = source.segments.end.dt.as_unit(unit)
    result = problem(source).evaluate(parameters_to_vector(template()))
    assert result.feasible, result.reason


@pytest.mark.parametrize("column", ["storage_m3", "release_m3s", "inflow_m3s"])
def test_complex_selected_inputs_are_rejected_without_discarding_imaginary_components(column):
    source = preparation()
    source.frame[column] = source.frame[column].astype(complex)
    source.frame.loc[CALIBRATION.start, column] += 1j
    with pytest.raises(ValueError, match="real numeric|numeric inflow"):
        problem(source)


@pytest.mark.parametrize("column", ["storage_target_eligible", "simulation_interval_eligible"])
def test_target_fit_and_bounds_cannot_readmit_quarantine_by_mutating_masks(column):
    source = annual_preparation()
    source.frame.loc["2000-01-15", column] = True
    with pytest.raises(ValueError, match="source-quality graph"):
        fit_interval_targets(source, ANNUAL_CALIBRATION)
    with pytest.raises(ValueError, match="source-quality graph"):
        estimate_interval_bounds(source, ANNUAL_CALIBRATION, template())


def test_objective_rejects_mutated_score_masks_and_bad_simulation_start():
    source = preparation(("2000-01-04",))
    source.frame.loc["2000-01-04", "release_score_eligible"] = True
    with pytest.raises(ValueError, match="source-quality graph"):
        problem(source)
    with pytest.raises(ValueError, match="on or before"):
        problem(simulation_start="2000-01-02")


def test_release_scoring_includes_explicit_spill_in_total_outflow(monkeypatch):
    source = preparation()
    raw = simulate_interval_reservoir(source, template(), end_date=CALIBRATION.end)
    modified = raw.frame.copy(deep=True)
    dates = source.frame.loc[CALIBRATION.start:CALIBRATION.end].index
    modified.loc[dates, "interval_mean_total_outflow_m3s"] = source.frame.loc[dates, "release_m3s"]
    modified.loc[dates, "interval_mean_release_m3s"] = .1
    modified.loc[dates, "release_endpoint_m3s"] = 1000.
    monkeypatch.setattr(objective_module, "simulate_interval_reservoir", lambda *args, **kwargs: replace(raw, frame=modified))
    result = problem(source).evaluate(parameters_to_vector(template()))
    assert result.feasible
    assert result.objectives[0] == -1.
