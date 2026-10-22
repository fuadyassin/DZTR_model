"""Independent expected values for the MATLAB objective functions."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dztr.data import normalize_observations
from dztr.evaluation import SUPPORTED_METRICS, evaluate_simulation, objective_metric


@pytest.mark.parametrize("metric,expected", [
    ("nse", -0.5),
    ("rmse", 1.0),
    ("nrmse", 0.5),
    ("pbias", 50.0),
    ("kger", 1.0),
    ("kgea", 1.0),
    ("kgeb", 1.5),
    ("kge", 0.5),
    ("kges", 0.5),
])
def test_independent_expected_metric_values(metric, expected):
    # A constant +1 bias leaves variance and correlation unchanged.
    assert objective_metric([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], metric) == pytest.approx(expected)


def test_lnse_matches_logarithm_of_shifted_observations():
    x = np.log(np.array([0.0, 1.0, 2.0]) + 0.01)
    y = np.log(np.array([0.0, 1.5, 3.0]) + 0.01)
    expected = 1 - np.sum((x - y) ** 2) / np.sum((x - np.mean(x)) ** 2)
    assert objective_metric([0, 1, 2], [0, 1.5, 3], "lnse") == pytest.approx(expected)


def test_kge_components_use_sample_standard_deviation_and_pearson_r():
    observed, simulated = [1, 2, 4], [4, 1, 3]
    # Both centered sums of squares are 14/3; centered cross sum is -2/3.
    assert objective_metric(observed, simulated, "kger") == pytest.approx(-1 / 7)
    assert objective_metric(observed, simulated, "kgea") == pytest.approx(1)
    assert objective_metric(observed, simulated, "kgeb") == pytest.approx(8 / 7)
    expected = 1 - np.sqrt(((-1 / 7) - 1) ** 2 + (8 / 7 - 1) ** 2)
    assert objective_metric(observed, simulated, "kge") == pytest.approx(expected)


def test_nan_pairs_are_removed_together_and_pbias_is_signed():
    assert objective_metric([1, np.nan, 3, 4], [2, 10, np.nan, 5], "rmse") == pytest.approx(1)
    assert objective_metric([1, np.nan, 3, 4], [2, 10, np.nan, 5], "pbias") == pytest.approx(40)
    assert objective_metric([1, 2, 3], [0.5, 1, 1.5], "pbias") == pytest.approx(-50)


@pytest.mark.parametrize("metric", sorted(SUPPORTED_METRICS))
def test_empty_and_missing_only_vectors_return_nan(metric):
    assert np.isnan(objective_metric([], [], metric))
    assert np.isnan(objective_metric([np.nan, 1], [2, np.nan], metric))


@pytest.mark.parametrize("metric", ["nse", "lnse", "nrmse", "kger", "kgea", "kge", "kges"])
def test_constant_observations_return_nan_for_undefined_metrics(metric):
    assert np.isnan(objective_metric([2, 2, 2], [1, 2, 3], metric))


def test_degenerate_denominators_and_single_sample_are_handled_without_warnings():
    for metric in ("pbias", "kgeb", "kge", "kges"):
        assert np.isnan(objective_metric([-1, 0, 1], [0, 1, 2], metric))
    for metric in ("nse", "nrmse", "kger", "kgea", "kge"):
        assert np.isnan(objective_metric([2], [3], metric))
    assert objective_metric([2], [3], "kgeb") == pytest.approx(1.5)
    assert objective_metric([2], [3], "rmse") == 1
    assert objective_metric([1, 2, 3], [2, 2, 2], "kgea") == 0
    assert np.isnan(objective_metric([1, 2, 3], [2, 2, 2], "kge"))


@pytest.mark.parametrize("observed,simulated", [([1, 2], [1]), ([[1, 2]], [1, 2]), ([[1, 2]], [[1, 2]]), ([1, np.inf], [1, np.nan]), (["bad"], [1])])
def test_invalid_shape_numeric_values_and_infinities_raise(observed, simulated):
    with pytest.raises(ValueError):
        objective_metric(observed, simulated)


def test_series_indexes_must_align_and_unknown_metrics_raise_even_when_empty():
    with pytest.raises(ValueError, match="index"):
        objective_metric(pd.Series([1, 2], index=[0, 1]), pd.Series([1, 2], index=[1, 0]))
    with pytest.raises(ValueError, match="Unknown metric"):
        objective_metric([], [], "unknown")


@pytest.mark.parametrize("values", [[-0.01, 1], [-1, 1]])
def test_lnse_invalid_domain_raises(values):
    with pytest.raises(ValueError, match="greater than -0.01"):
        objective_metric(values, [0, 1], "lnse")
    with pytest.raises(ValueError, match="greater than -0.01"):
        objective_metric([0, 1], values, "lnse")


def evaluation_inputs():
    observed = normalize_observations(pd.DataFrame({
        "Date": pd.date_range("2001-01-01", periods=4),
        "storage_m3": [1, 2, 3, 4], "inflow_m3s": [1, 1, 1, 1], "release_m3s": [1, 2, 3, 4],
    }))
    frame = pd.DataFrame({"storage_m3": [100, 3, 4, 5], "release_m3s": [100, 3, 4, 5]}, index=observed.frame.index, dtype=float)
    # Evaluation needs only the result's frame, independent of solver selection.
    return observed, SimpleNamespace(frame=frame)


def test_evaluate_simulation_excludes_the_count_of_warmup_rows():
    observed, simulated = evaluation_inputs()
    metrics = evaluate_simulation(observed, simulated, warmup_steps=1)
    assert set(metrics) == {"release_nse", "release_kge", "release_pbias", "release_rmse", "storage_nse", "storage_kge"}
    assert metrics["release_rmse"] == pytest.approx(1)
    assert metrics["release_pbias"] == pytest.approx(100 / 3)
    assert metrics["release_nse"] == pytest.approx(-0.5)
    assert metrics["storage_nse"] == pytest.approx(-0.5)
    assert metrics["release_kge"] == pytest.approx(2 / 3)
    assert metrics["storage_kge"] == pytest.approx(2 / 3)
    assert evaluate_simulation(observed, simulated)["release_rmse"] > 40
    assert all(np.isnan(value) for value in evaluate_simulation(observed, simulated, warmup_steps=4).values())


def test_storage_and_release_have_independent_missing_pairs():
    observed, simulated = evaluation_inputs()
    simulated.frame.loc[simulated.frame.index[1], "release_m3s"] = np.nan
    simulated.frame.loc[simulated.frame.index[2], "storage_m3"] = np.nan
    metrics = evaluate_simulation(observed, simulated, warmup_steps=1)
    assert metrics["release_pbias"] == pytest.approx(200 / 7)
    assert metrics["storage_nse"] == pytest.approx(0)
    assert metrics["release_nse"] == pytest.approx(-3)


@pytest.mark.parametrize("warmup", [-1, 1.5, True, "1"])
def test_warmup_requires_nonnegative_integer(warmup):
    with pytest.raises(ValueError, match="warmup_steps"):
        evaluate_simulation(*evaluation_inputs(), warmup_steps=warmup)


def test_evaluation_rejects_misaligned_indexes_missing_columns_and_infinities():
    observed, simulated = evaluation_inputs()
    simulated.frame = simulated.frame.iloc[::-1]
    with pytest.raises(ValueError, match="index"):
        evaluate_simulation(observed, simulated)
    observed, simulated = evaluation_inputs()
    simulated.frame = simulated.frame.drop(columns="storage_m3")
    with pytest.raises(ValueError, match="columns"):
        evaluate_simulation(observed, simulated)
    observed, simulated = evaluation_inputs()
    simulated.frame.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinities"):
        evaluate_simulation(observed, simulated, warmup_steps=1)
