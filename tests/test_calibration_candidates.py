"""Candidate ordering, raw empirical boxes, and calibration-only fitting."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from dztr.calibration.bounds import CalibrationBounds, estimate_calibration_bounds
from dztr.calibration.candidates import parameters_from_vector, parameters_to_vector
from dztr.calibration.periods import DatePeriod
from dztr.data import normalize_observations
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters


@pytest.fixture
def template():
    table = pd.DataFrame(np.tile([250, 500, 750, 0.5, 1.5, 2.5], (12, 1)), index=range(1, 13), columns=MONTHLY_TARGET_COLUMNS)
    return ReservoirParameters(table, 1000.0, 400.0, 1.25, 0.1,
                               dead_storage_fraction=0.1, inflow_correction=0.92, flood_mode="inflow")


def raw_bounds_table():
    table = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
    for side, values in (("lower", [150, 400, 650, 0.2, 0.8, 1.5]), ("upper", [350, 600, 850, 1.0, 2.0, 3.0])):
        for column, value in zip(MONTHLY_TARGET_COLUMNS, values):
            table[f"{side}_{column}"] = float(value)
    return table


@pytest.fixture
def bounds(template):
    return CalibrationBounds(raw_bounds_table(), template.capacity_m3, template.dead_storage_fraction,
                             DatePeriod("2001-01-01", "2001-12-31"), source="training")


def history():
    index = pd.date_range("2000-01-01", "2002-12-31", name="date")
    day = index.day.to_numpy(dtype=float)
    return normalize_observations(pd.DataFrame({
        "storage_m3": 200 + 18 * day,
        "release_m3s": 0.5 + 0.1 * day,
        "inflow_m3s": np.full(len(index), 1.0),
    }, index=index), source="three_year_fixture")


def test_vector_roundtrip_exact_matlab_order_and_mixed_units(template):
    table = template.monthly_targets.copy()
    for i, column in enumerate(MONTHLY_TARGET_COLUMNS):
        table[column] += np.arange(12) * (2 if i < 3 else 0.01)
    seasonal = replace(template, monthly_targets=table)
    vector = parameters_to_vector(seasonal)
    assert vector.shape == (72,)
    for i, column in enumerate(MONTHLY_TARGET_COLUMNS):
        expected = table[column].to_numpy() / (template.capacity_m3 if i < 3 else 1)
        np.testing.assert_allclose(vector[12 * i:12 * (i + 1)], expected)
    decoded = parameters_from_vector(vector, template)
    pd.testing.assert_frame_equal(decoded.monthly_targets, seasonal.monthly_targets)
    for field in ("capacity_m3", "initial_storage_m3", "initial_release_m3s", "channel_capacity_m3s", "dead_storage_fraction", "inflow_correction", "flood_mode"):
        assert getattr(decoded, field) == getattr(template, field)


def test_candidate_decoding_does_not_mutate_template_or_alias_input(template):
    original = template.monthly_targets.copy(deep=True)
    vector = parameters_to_vector(template)
    vector[0] = 0.3
    decoded = parameters_from_vector(vector, template)
    vector[0] = 0.2
    decoded.monthly_targets.iloc[0, 0] = 310
    pd.testing.assert_frame_equal(template.monthly_targets, original)
    assert decoded.monthly_targets.iloc[0, 0] == 310
    with pytest.raises(FrozenInstanceError):
        decoded.capacity_m3 = 2000


@pytest.mark.parametrize("bad", [np.zeros(71), np.zeros(73), np.zeros((6, 12)), np.full(72, np.nan), np.full(72, np.inf), np.full(72, 1j), [10 ** 500] * 72, "bad"])
def test_vector_rejects_bad_shape_or_nonfinite_values(template, bounds, bad):
    with pytest.raises(ValueError, match="Candidate"):
        parameters_from_vector(bad, template)
    assert not bounds.contains(bad)
    with pytest.raises(ValueError, match="Candidate"):
        bounds.validate(bad, template)


@pytest.mark.parametrize("change", ["tied_storage", "above_capacity", "below_dead", "unordered_release"])
def test_candidate_physical_constraints_are_not_silently_repaired(template, change):
    vector = parameters_to_vector(template)
    if change == "tied_storage":
        vector[12] = vector[0]
    elif change == "above_capacity":
        vector[24] = 1.1
    elif change == "below_dead":
        vector[0] = 0.05
    else:
        vector[36] = 3.0
    original = vector.copy()
    with pytest.raises(ValueError):
        parameters_from_vector(vector, template)
    np.testing.assert_array_equal(vector, original)


def test_release_candidates_are_not_globally_capped_at_channel_capacity(template):
    vector = parameters_to_vector(template)
    assert vector[36:].min() > template.channel_capacity_m3s
    decoded = parameters_from_vector(vector, template)
    assert decoded.monthly_targets.release_flood_m3s.max() == 2.5


def test_bounds_vectors_use_the_same_mixed_unit_order_and_are_detached(bounds):
    expected_lower = np.repeat([0.15, 0.4, 0.65, 0.2, 0.8, 1.5], 12)
    expected_upper = np.repeat([0.35, 0.6, 0.85, 1.0, 2.0, 3.0], 12)
    np.testing.assert_array_equal(bounds.lower_vector, expected_lower)
    np.testing.assert_array_equal(bounds.upper_vector, expected_upper)
    changed = bounds.lower_vector
    changed[:] = 0
    np.testing.assert_array_equal(bounds.lower_vector, expected_lower)


def test_bounds_validate_inclusive_box_and_preserve_template(bounds, template):
    for vector in (bounds.lower_vector, bounds.upper_vector, parameters_to_vector(template)):
        assert bounds.contains(vector)
        decoded = bounds.validate(vector, template)
        assert decoded.channel_capacity_m3s == template.channel_capacity_m3s
    outside = bounds.lower_vector
    outside[0] -= 1e-3
    assert not bounds.contains(outside)
    with pytest.raises(ValueError, match="outside"):
        bounds.validate(outside, template)


@pytest.mark.parametrize("field,value", [("capacity_m3", 1100), ("dead_storage_fraction", 0.05)])
def test_bounds_require_matching_capacity_and_dead_storage(bounds, template, field, value):
    altered = replace(template, **{field: value})
    with pytest.raises(ValueError, match="must match"):
        bounds.validate(parameters_to_vector(template), altered)


def test_box_membership_does_not_claim_physical_storage_feasibility(template):
    raw = raw_bounds_table()
    raw["lower_storage_critical_m3"] = 90
    raw["upper_storage_critical_m3"] = 500
    raw["upper_storage_flood_m3"] = 1100
    bounds = CalibrationBounds(raw, 1000, 0.1, DatePeriod("2001-01-01", "2001-12-31"))
    assert bounds.monthly_bounds.lower_storage_critical_m3.iloc[0] == 90
    assert bounds.monthly_bounds.upper_storage_flood_m3.iloc[0] == 1100
    for changes in ({0: 0.1}, {0: 0.45, 12: 0.45}, {24: 1.05}):
        vector = parameters_to_vector(template)
        for index, value in changes.items():
            vector[index] = value
        assert bounds.contains(vector)
        with pytest.raises(ValueError, match="Storage targets"):
            bounds.validate(vector, template)


def test_box_membership_does_not_claim_release_order_is_valid(bounds, template):
    vector = parameters_to_vector(template)
    vector[36], vector[48] = 1.0, 0.8
    assert bounds.contains(vector)
    with pytest.raises(ValueError, match="Release targets"):
        bounds.validate(vector, template)


def test_bounds_copies_input_and_carries_portable_provenance(template):
    raw = raw_bounds_table()
    bounds = CalibrationBounds(raw, 1000, 0.1, DatePeriod("2001-01-01", "2001-12-31"), source="fixture")
    raw.iloc[0, 0] = 999
    assert bounds.monthly_bounds.iloc[0, 0] == 150
    provenance = bounds.summary()
    assert provenance["estimation_period"] == {"start": "2001-01-01", "end": "2001-12-31"}
    assert provenance["quantile_method"] == "hazen"
    assert provenance["percentile_bounds"] == {
        "critical": {"lower": 5.0, "upper": 35.0},
        "normal": {"lower": 35.0, "upper": 75.0},
        "flood": {"lower": 75.0, "upper": 95.0},
    }
    assert provenance["source"] == "fixture"
    with pytest.raises(FrozenInstanceError):
        bounds.capacity_m3 = 2000


@pytest.mark.parametrize("problem", ["nan", "negative", "lower_above_upper", "dead", "capacity", "ties", "release_order", "missing_column", "duplicate_column"])
def test_invalid_or_entirely_infeasible_bound_boxes_raise(problem):
    raw = raw_bounds_table()
    if problem == "nan":
        raw.iloc[0, 0] = np.nan
    elif problem == "negative":
        raw.iloc[0, 0] = -1
    elif problem == "lower_above_upper":
        raw.loc[1, "lower_storage_critical_m3"] = 500
    elif problem == "dead":
        raw.loc[1, ["lower_storage_critical_m3", "upper_storage_critical_m3"]] = [50, 100]
    elif problem == "capacity":
        raw.loc[1, ["lower_storage_flood_m3", "upper_storage_flood_m3"]] = [1001, 1100]
    elif problem == "ties":
        for side in ("lower", "upper"):
            for column in MONTHLY_TARGET_COLUMNS[:3]:
                raw.loc[1, f"{side}_{column}"] = 500
    elif problem == "release_order":
        raw.loc[1, ["lower_release_critical_m3s", "upper_release_critical_m3s"]] = [5, 6]
    elif problem == "missing_column":
        raw = raw.drop(columns="lower_storage_critical_m3")
    else:
        raw = pd.concat([raw, raw[["lower_release_critical_m3s"]]], axis=1)
    with pytest.raises(ValueError):
        CalibrationBounds(raw, 1000, 0.1, DatePeriod("2001-01-01", "2001-12-31"))


def test_derived_bounds_match_all_monthly_cdf_definitions_and_si_units(template):
    obs = history()
    period = DatePeriod("2001-01-01", "2001-12-31")
    bounds = estimate_calibration_bounds(obs, period, template)
    fitting = obs.frame.loc[period.start:period.end]
    for month in range(1, 13):
        for variable, stem, suffix in (("storage_m3", "storage", "m3"), ("release_m3s", "release", "m3s")):
            sample = fitting.loc[fitting.index.month == month, variable].to_numpy()
            for level, pair in zip(("critical", "normal", "flood"), ((5, 35), (35, 75), (75, 95))):
                for side, percentile in zip(("lower", "upper"), pair):
                    assert bounds.monthly_bounds.loc[month, f"{side}_{stem}_{level}_{suffix}"] == pytest.approx(np.percentile(sample, percentile, method="hazen"))
    assert bounds.period == period
    assert bounds.source == obs.source
    assert bounds.quantile_method == "hazen"
    assert bounds.monthly_bounds.upper_release_flood_m3s.min() > template.channel_capacity_m3s


def test_spinup_and_validation_observations_cannot_change_calibration_bounds(template):
    obs = history()
    period = DatePeriod("2001-01-01", "2001-12-31")
    before = estimate_calibration_bounds(obs, period, template)
    changed = obs.frame.copy(deep=True)
    outside = changed.index.year != 2001
    changed.loc[outside, ["storage_m3", "release_m3s"]] = [99999, 88888]
    after = estimate_calibration_bounds(normalize_observations(changed, source=obs.source), period, template)
    pd.testing.assert_frame_equal(before.monthly_bounds, after.monthly_bounds)
    np.testing.assert_array_equal(before.lower_vector, after.lower_vector)
    np.testing.assert_array_equal(before.upper_vector, after.upper_vector)


def test_derived_bounds_support_an_explicit_linear_method(template):
    obs = history()
    period = DatePeriod("2001-01-01", "2001-12-31")
    bounds = estimate_calibration_bounds(obs, period, template, quantile_method="linear")
    sample = obs.frame.loc["2001-01-01":"2001-01-31", "storage_m3"].to_numpy()
    assert bounds.monthly_bounds.loc[1, "lower_storage_critical_m3"] == np.percentile(sample, 5, method="linear")
    assert bounds.quantile_method == "linear"


@pytest.mark.parametrize("variable", ["storage_m3", "release_m3s"])
def test_missing_calibration_month_raises_instead_of_using_overall_quantiles(template, variable):
    obs = history()
    changed = obs.frame.copy()
    changed.loc[(changed.index.year == 2001) & (changed.index.month == 6), variable] = np.nan
    with pytest.raises(ValueError, match="missing bounds in months \\[6\\]"):
        estimate_calibration_bounds(normalize_observations(changed), DatePeriod("2001-01-01", "2001-12-31"), template)


def test_derived_bounds_enforce_complete_calibration_date_coverage(template):
    obs = history()
    broken = normalize_observations(obs.frame.drop(index=pd.Timestamp("2001-06-15")))
    with pytest.raises(ValueError, match="consecutive daily coverage"):
        estimate_calibration_bounds(broken, DatePeriod("2001-01-01", "2001-12-31"), template)


def test_constant_storage_record_has_no_strictly_ordered_feasible_box(template):
    obs = history()
    changed = obs.frame.copy()
    changed["storage_m3"] = 500
    with pytest.raises(ValueError, match="no physically feasible candidate"):
        estimate_calibration_bounds(normalize_observations(changed), DatePeriod("2001-01-01", "2001-12-31"), template)
