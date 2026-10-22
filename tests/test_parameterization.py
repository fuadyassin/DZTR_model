"""Independent quantile checks and regressions against the bundled workbook."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dztr.data import normalize_observations, read_reservoir_observations
from dztr.parameterization import DEFAULT_PERCENTILES, MONTH_NAMES, estimate_generalized_parameters, estimate_percentiles


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def saved_catalog():
    return pd.read_excel(ROOT / "Extended_GRanD_Dams_v1_1_DZTR.xlsx")


def complete_year():
    dates = pd.date_range("2001-01-01", "2001-12-31", freq="D")
    values = np.arange(len(dates), dtype=float) + 1
    return normalize_observations(pd.DataFrame({"storage_m3": values * 1e6, "inflow_m3s": values * 2, "release_m3s": values * 3}, index=dates))


def test_linear_quantiles_p99_missing_samples_and_sample_counts():
    observations = normalize_observations(pd.DataFrame({
        "Date": ["2001-01-01", "2001-01-02", "2001-01-04"],
        "storage_m3": [1e6, 3e6, np.nan], "inflow_m3s": [2, np.nan, 6], "release_m3s": [10, 20, 30],
    }))
    frame = estimate_percentiles(observations, percentiles=(10, 45, 85, 99)).frame
    indexed = frame.set_index(["variable", "month", "percentile"])
    assert indexed.loc[("storage_m3", 0, 45), "value"] == pytest.approx(1.9e6)
    assert indexed.loc[("storage_m3", 1, 10), "value"] == pytest.approx(1.2e6)
    assert indexed.loc[("release_m3s", 0, 99), "value"] == pytest.approx(29.8)
    assert indexed.loc[("inflow_m3s", 0, 85), "sample_count"] == 2
    assert np.isnan(indexed.loc[("release_m3s", 2, 10), "value"])
    assert indexed.loc[("release_m3s", 2, 10), "sample_count"] == 0


@pytest.mark.parametrize("percentiles", [(), (10, 10), (-1, 50), (50, 101), (np.nan,), (np.inf,)])
def test_invalid_percentile_requests_raise(percentiles):
    with pytest.raises(ValueError):
        estimate_percentiles(complete_year(), percentiles=percentiles)


def test_missing_variable_has_nan_quantiles_and_zero_samples():
    with pytest.warns(UserWarning):
        observations = normalize_observations(pd.DataFrame({"Date": ["2001-01-01"], "storage_m3": [1]}))
    estimates = estimate_percentiles(observations).frame
    inflow = estimates.loc[estimates.variable == "inflow_m3s"]
    assert inflow.value.isna().all()
    assert inflow.sample_count.eq(0).all()
    with pytest.raises(ValueError):
        estimate_generalized_parameters(observations)


@pytest.mark.parametrize("value", [0, np.nan])
def test_placeholder_and_unavailable_cannot_produce_parameters(value):
    observations = normalize_observations(pd.DataFrame({
        "Date": ["2001-01-01", "2001-01-02"], "storage_m3": [value, value],
        "inflow_m3s": [value, value], "release_m3s": [value, value],
    }))
    with pytest.raises(ValueError):
        estimate_percentiles(observations)
    with pytest.raises(ValueError):
        estimate_generalized_parameters(observations)


def test_generalized_parameters_have_six_monthly_targets_and_p99_channel_capacity():
    observations = complete_year()
    result = estimate_generalized_parameters(observations)
    assert result.monthly_targets.index.tolist() == list(range(1, 13))
    assert result.monthly_targets.shape == (12, 6)
    january = observations.frame.loc[observations.frame.index.month == 1]
    assert result.monthly_targets.loc[1, "storage_critical_m3"] == pytest.approx(np.percentile(january.storage_m3, 10))
    assert result.monthly_targets.loc[1, "storage_normal_m3"] == pytest.approx(np.percentile(january.storage_m3, 45))
    assert result.monthly_targets.loc[1, "storage_flood_m3"] == pytest.approx(np.percentile(january.storage_m3, 85))
    assert result.monthly_targets.loc[1, "release_normal_m3s"] == pytest.approx(np.percentile(january.release_m3s, 45))
    assert result.channel_capacity_m3s == pytest.approx(np.percentile(observations.frame.release_m3s, 99))


@pytest.mark.parametrize("name,grand_id", [("Gardiner", 253), ("Trinity", 131)])
def test_all_780_historical_quantiles_reproduce_saved_workbook(name, grand_id):
    observations = read_reservoir_observations(ROOT / "ResDataInflowStorageRelease" / f"{name}_{grand_id}.xlsx")
    assert observations.quality.status == "available"
    estimates = estimate_percentiles(observations).frame.set_index(["variable", "month", "percentile"])
    saved = saved_catalog().set_index("GRAND_ID").loc[grand_id]
    expected, actual = [], []
    for metric, variable, factor in (("S", "storage_m3", 1e6), ("I", "inflow_m3s", 1), ("Q", "release_m3s", 1)):
        for percentile in range(5, 101, 5):
            for month, suffix in enumerate(("", *(f"_{name}" for name in MONTH_NAMES))):
                expected.append(saved[f"{metric}{percentile}{suffix}"] * factor)
                actual.append(estimates.loc[(variable, month, percentile), "value"])
    assert len(actual) == 780
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-6)
    assert 99 in DEFAULT_PERCENTILES


@pytest.mark.parametrize("name,grand_id", [("Gardiner", 253), ("Trinity", 131)])
def test_historical_generalized_targets_reproduce_saved_workbook(name, grand_id):
    observations = read_reservoir_observations(ROOT / "ResDataInflowStorageRelease" / f"{name}_{grand_id}.xlsx")
    targets = estimate_generalized_parameters(observations).monthly_targets
    saved = saved_catalog().set_index("GRAND_ID").loc[grand_id]
    expected, actual = [], []
    for month, month_name in enumerate(MONTH_NAMES, start=1):
        for zone, percentile in (("critical", 10), ("normal", 45), ("flood", 85)):
            expected.extend([saved[f"S{percentile}_{month_name}"] * 1e6, saved[f"Q{percentile}_{month_name}"]])
            actual.extend([targets.loc[month, f"storage_{zone}_m3"], targets.loc[month, f"release_{zone}_m3s"]])
    assert len(actual) == 72
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-6)
