"""Leakage, state continuity and scoring safeguards for split baseline runs."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from dztr.calibration.baseline import json_ready, run_baseline
from dztr.calibration.benchmark import run_configured_baseline
from dztr.calibration.periods import CalibrationValidationSplit, DatePeriod
from dztr.data import normalize_observations
from dztr.evaluation import objective_metric
from dztr.simulation import simulate_reservoir


MODEL = {"capacity_m3": 80e6, "initial_storage_m3": 55e6, "initial_release_m3s": 2.5}


def synthetic_observations():
    dates = pd.date_range("1999-12-02", "2001-01-10", name="date")
    frame = pd.DataFrame({
        "storage_m3": 50e6 + (dates.day.to_numpy() - 16) * 1e6,
        "inflow_m3s": np.full(len(dates), 2.2),
        "release_m3s": 2.0 + dates.day.to_numpy() * 0.03,
    }, index=dates)
    return normalize_observations(frame, source="synthetic_reference")


def split():
    return CalibrationValidationSplit(
        DatePeriod("2000-01-01", "2000-12-31"),
        DatePeriod("2001-01-04", "2001-01-10"),
        warmup_steps=30,
    )


@pytest.fixture(scope="module")
def baseline():
    return run_baseline(synthetic_observations(), split(), **MODEL)


@pytest.mark.parametrize("period", ["spinup", "validation"])
def test_spinup_and_validation_observations_cannot_change_fit_or_trajectory(baseline, period):
    original = synthetic_observations()
    changed = original.frame.copy(deep=True)
    boundary = split()
    selected = (
        changed.index < boundary.calibration.start
        if period == "spinup"
        else changed.index >= boundary.validation.start
    )
    changed.loc[selected, "storage_m3"] = 1e10
    changed.loc[selected, "release_m3s"] = 1e6
    alternative_input = normalize_observations(changed)
    before = alternative_input.frame.copy(deep=True)
    alternative = run_baseline(alternative_input, boundary, **MODEL)
    pd.testing.assert_frame_equal(alternative.estimates.monthly_targets, baseline.estimates.monthly_targets)
    pd.testing.assert_frame_equal(alternative.quantiles.frame, baseline.quantiles.frame)
    assert alternative.estimates.channel_capacity_m3s == baseline.estimates.channel_capacity_m3s
    pd.testing.assert_frame_equal(alternative.simulation.frame, baseline.simulation.frame)
    pd.testing.assert_frame_equal(alternative_input.frame, before)
    if period == "validation":
        assert alternative.metrics["calibration"] == baseline.metrics["calibration"]
        assert alternative.metrics["validation"]["release_pbias"] != baseline.metrics["validation"]["release_pbias"]


def test_inferred_channel_capacity_uses_only_calibration_release_p99(baseline):
    source = synthetic_observations().frame
    calibration_release = source.loc["2000-01-01":"2000-12-31", "release_m3s"]
    expected = np.percentile(calibration_release, 99, method="hazen")
    assert baseline.estimates.channel_capacity_m3s == pytest.approx(expected)
    assert baseline.simulation.parameters.channel_capacity_m3s == pytest.approx(expected)
    inferred = baseline.quantiles.frame
    capacity = inferred.loc[(inferred.variable == "release_m3s") & (inferred.month == 0) & (inferred.percentile == 99)]
    assert capacity.sample_count.iloc[0] == len(calibration_release)
    assert capacity.value.iloc[0] == pytest.approx(expected)


def test_validation_carries_continuous_state_instead_of_resetting(baseline):
    boundary = baseline.split.validation.start
    previous = boundary - pd.Timedelta(days=1)
    simulation = baseline.simulation.frame
    parameters = baseline.simulation.parameters
    inflow = baseline.observations.frame.inflow_m3s
    expected_change = 86400 * 0.5 * (
        parameters.inflow_correction * (inflow.loc[previous] + inflow.loc[boundary])
        - simulation.loc[previous, "release_m3s"] - simulation.loc[boundary, "release_m3s"]
    )
    assert simulation.loc[boundary, "overflow_volume_m3"] == 0
    assert simulation.loc[boundary, "storage_m3"] - simulation.loc[previous, "storage_m3"] == pytest.approx(expected_change, abs=1e-5)
    validation = normalize_observations(baseline.observations.frame.loc[boundary:baseline.split.validation.end])
    reset = simulate_reservoir(validation, parameters)
    assert reset.frame.storage_m3.iloc[0] == MODEL["initial_storage_m3"]
    assert abs(simulation.loc[boundary, "storage_m3"] - reset.frame.storage_m3.iloc[0]) > 1e6


def test_entire_short_validation_is_scored_without_an_additional_year_skip(baseline):
    validation_dates = baseline.split.validation
    observed = baseline.observations.frame.loc[validation_dates.start:validation_dates.end, "release_m3s"]
    simulated = baseline.simulation.frame.loc[validation_dates.start:validation_dates.end, "release_m3s"]
    assert len(observed) == 7
    assert baseline.paired_counts["validation"] == {"release": 7, "storage": 7}
    assert baseline.paired_counts["calibration"] == {"release": 366, "storage": 366}
    assert baseline.metrics["validation"]["release_nse"] == pytest.approx(objective_metric(observed, simulated, "nse"))
    assert np.isfinite(list(baseline.metrics["validation"].values())).all()


def test_missing_measurements_have_independent_counts_and_keep_the_simulation(baseline):
    source = synthetic_observations().frame.copy()
    source.loc[pd.Timestamp("2000-02-02"), "release_m3s"] = np.nan
    source.loc[pd.Timestamp("2000-04-04"), "storage_m3"] = np.nan
    source.loc[pd.Timestamp("2001-01-05"), "release_m3s"] = np.nan
    source.loc[pd.Timestamp("2001-01-08"), "storage_m3"] = np.nan
    result = run_baseline(normalize_observations(source), split(), **MODEL)
    assert result.paired_counts["calibration"] == {"release": 365, "storage": 365}
    assert result.paired_counts["validation"] == {"release": 6, "storage": 6}
    assert len(result.simulation.frame) == len(baseline.simulation.frame)
    validation_release = source.loc["2001-01-04":"2001-01-10", "release_m3s"].dropna()
    simulated = result.simulation.frame.loc[validation_release.index, "release_m3s"]
    assert result.metrics["validation"]["release_rmse"] == pytest.approx(np.sqrt(np.mean((simulated - validation_release) ** 2)))


def test_daily_labels_and_summary_include_spinup_gap_and_both_scored_periods(baseline):
    daily = baseline.daily_table()
    assert daily.period.value_counts().to_dict() == {"calibration": 366, "spinup": 30, "validation": 7, "unscored": 3}
    assert daily.loc["1999-12-02", "period"] == "spinup"
    assert daily.loc["2000-01-01", "period"] == "calibration"
    assert daily.loc["2000-12-31", "period"] == "calibration"
    assert daily.loc["2001-01-01":"2001-01-03", "period"].eq("unscored").all()
    assert daily.loc["2001-01-04":"2001-01-10", "period"].eq("validation").all()
    summary = baseline.summary()
    assert summary["periods"]["spinup"] == {"start": "1999-12-02", "end": "1999-12-31", "daily_rows": 30}
    assert summary["periods"]["simulation"]["daily_rows"] == len(daily)
    assert summary["parameter_estimation_period"] == {"start": "2000-01-01", "end": "2000-12-31"}
    assert summary["diagnostics"]["simulation"]["max_abs_balance_residual_m3"] < 1e-4
    pd.testing.assert_series_equal(daily.observed_storage_m3, baseline.observations.frame.storage_m3, check_names=False)


@pytest.mark.parametrize("missing", ["1999-12-02", "1999-12-15", "2001-01-02", "2001-01-10"])
def test_missing_spinup_gap_or_validation_dates_reject_the_continuous_window(missing):
    frame = synthetic_observations().frame.drop(index=pd.Timestamp(missing))
    with pytest.raises(ValueError, match="complete consecutive daily coverage"):
        run_baseline(normalize_observations(frame), split(), **MODEL)


@pytest.mark.parametrize("missing", ["1999-12-02", "2000-03-03", "2001-01-02", "2001-01-08"])
def test_missing_inflow_is_rejected_in_every_simulated_period(missing):
    frame = synthetic_observations().frame.copy()
    frame.loc[pd.Timestamp(missing), "inflow_m3s"] = np.nan
    with pytest.raises(ValueError, match="inflow"):
        run_baseline(normalize_observations(frame), split(), **MODEL)


@pytest.mark.parametrize("column", ["storage_m3", "release_m3s"])
def test_calibration_monthly_coverage_cannot_be_supplied_by_other_periods(column):
    frame = synthetic_observations().frame.copy()
    # December spinup has observations, but cannot supply calibration December.
    frame.loc["2000-12-01":"2000-12-31", column] = np.nan
    with pytest.raises(ValueError, match="missing targets in months \\[12\\]"):
        run_baseline(normalize_observations(frame), split(), **MODEL)


def test_undefined_validation_scores_become_json_null_without_nonstandard_tokens():
    frame = synthetic_observations().frame.copy()
    frame.loc["2001-01-04":"2001-01-10", "storage_m3"] = 50e6
    frame.loc["2001-01-04":"2001-01-10", "release_m3s"] = 2.2
    result = run_baseline(normalize_observations(frame), split(), **MODEL)
    assert np.isnan(result.metrics["validation"]["release_nse"])
    assert np.isnan(result.metrics["validation"]["storage_kge"])
    report = json_ready(result.summary())
    encoded = json.dumps(report, allow_nan=False)
    assert "NaN" not in encoded
    decoded = json.loads(encoded)
    assert decoded["metrics"]["validation"]["release_nse"] is None
    assert decoded["metrics"]["validation"]["storage_kge"] is None
    assert decoded["metrics"]["validation"]["release_rmse"] is not None


def test_configured_artifacts_are_portable_reproducible_and_leave_inputs_unchanged(tmp_path):
    source = tmp_path / "observations.csv"
    synthetic_observations().frame.to_csv(source, index_label="date")
    config = {
        "reservoir": {"name": "Synthetic"}, "observations_file": "observations.csv",
        "periods": {
            "calibration": split().calibration.as_dict(), "validation": split().validation.as_dict(), "warmup_steps": 30,
        },
        "model": MODEL, "quantile_method": "hazen", "target_percentiles": [10, 45, 85],
        "simulation": {"solver": "implicit", "overflow_mode": "conservative"},
        "protocol": {"date_interpretation": "Synthetic regression periods; no published reservoir dates are asserted."},
    }
    configuration = tmp_path / "config.json"
    configuration.write_text(json.dumps(config), encoding="utf-8")
    input_bytes, config_bytes = source.read_bytes(), configuration.read_bytes()
    output = tmp_path / "artifacts"
    report = run_configured_baseline(configuration, tmp_path, output)
    assert {path.name for path in output.iterdir()} == {
        "daily_simulation.csv", "monthly_targets_si.csv", "calibration_quantiles_si.csv",
        "metrics.csv", "resolved_config.json", "summary.json",
    }
    assert json.loads((output / "resolved_config.json").read_text(encoding="utf-8")) == config
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == report
    assert report["source_sha256"] == {
        "observations": hashlib.sha256(input_bytes).hexdigest(), "configuration": hashlib.sha256(config_bytes).hexdigest(),
    }
    assert source.read_bytes() == input_bytes
    assert configuration.read_bytes() == config_bytes
    daily = pd.read_csv(output / "daily_simulation.csv")
    assert daily.period.eq("validation").sum() == 7
    assert pd.read_csv(output / "monthly_targets_si.csv").shape == (12, 7)
    quantiles = pd.read_csv(output / "calibration_quantiles_si.csv")
    assert quantiles.loc[quantiles.month == 0, "sample_count"].eq(366).all()
    assert pd.read_csv(output / "metrics.csv").period.tolist() == ["calibration", "validation"]


def test_configured_artifact_paths_cannot_overwrite_source_observations(tmp_path):
    source = tmp_path / "daily_simulation.csv"
    synthetic_observations().frame.to_csv(source, index_label="date")
    configuration = tmp_path / "config.json"
    configuration.write_text(json.dumps({
        "observations_file": source.name,
        "periods": {
            "calibration": split().calibration.as_dict(),
            "validation": split().validation.as_dict(),
            "warmup_steps": split().warmup_steps,
        },
        "model": MODEL,
    }), encoding="utf-8")
    observation_bytes, configuration_bytes = source.read_bytes(), configuration.read_bytes()
    files_before = {path.name for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="overwrite source observations or configuration"):
        run_configured_baseline(configuration, tmp_path, source.parent)
    assert source.read_bytes() == observation_bytes
    assert configuration.read_bytes() == configuration_bytes
    assert {path.name for path in tmp_path.iterdir()} == files_before
