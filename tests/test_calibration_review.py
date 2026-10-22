"""Portable candidate-review artifacts and input overwrite protection."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import dztr.calibration.review as review_module
from dztr.calibration.baseline import run_baseline
from dztr.calibration.periods import CalibrationValidationSplit, DatePeriod
from dztr.calibration.review import run_calibration_review
from dztr.data import normalize_observations
from dztr.schemas import MONTHLY_TARGET_COLUMNS


MODEL = {"capacity_m3": 80e6, "initial_storage_m3": 55e6, "initial_release_m3s": 2.5}


def synthetic_source():
    dates = pd.date_range("1999-12-02", "2001-01-10", name="date")
    return normalize_observations(pd.DataFrame({
        "storage_m3": 50e6 + (dates.day.to_numpy() - 16) * 1e6,
        "inflow_m3s": np.full(len(dates), 2.2),
        "release_m3s": 2.0 + dates.day.to_numpy() * 0.03,
    }, index=dates))


def selection():
    return CalibrationValidationSplit(DatePeriod("2000-01-01", "2000-12-31"),
                                       DatePeriod("2001-01-04", "2001-01-10"), warmup_steps=30)


def write_inputs(folder, *, source_name="observations.csv", config_name="config.json"):
    source = folder / source_name
    synthetic_source().frame.to_csv(source, index_label="date")
    config = {
        "observations_file": source.name, "reservoir": {"name": "Synthetic review"},
        "periods": {"calibration": selection().calibration.as_dict(),
                    "validation": selection().validation.as_dict(), "warmup_steps": 30},
        "quantile_method": "hazen", "target_percentiles": [10, 45, 85], "model": MODEL,
        "simulation": {"solver": "implicit", "overflow_mode": "conservative"},
    }
    configuration = folder / config_name
    configuration.write_text(json.dumps(config), encoding="utf-8")
    return source, configuration, config


def test_review_exports_native_candidate_space_and_matches_baseline_without_matlab(tmp_path, monkeypatch):
    def unexpected_matlab_load(*args, **kwargs):
        raise AssertionError("A portable review must not load optional MATLAB inputs.")

    monkeypatch.setattr(review_module, "_load_mat", unexpected_matlab_load)
    monkeypatch.setattr(review_module, "read_matlab_observations", unexpected_matlab_load)
    source, configuration, config = write_inputs(tmp_path)
    source_bytes, configuration_bytes = source.read_bytes(), configuration.read_bytes()
    output = tmp_path / "review"
    report = run_calibration_review(configuration, tmp_path, output)
    assert report["passed"]
    assert report["historical_reference_check_performed"] is False
    assert report["reference_checks"] == []
    assert report["objective_order"] == ["negative_release_nse", "negative_storage_nse"]
    assert report["objective_direction"] == "minimize"
    assert report["bounds"]["estimation_period"] == selection().calibration.as_dict()
    assert report["bounds"]["percentile_bounds"] == {
        "critical": {"lower": 5.0, "upper": 35.0},
        "normal": {"lower": 35.0, "upper": 75.0},
        "flood": {"lower": 75.0, "upper": 95.0},
    }
    assert report["source_sha256"] == {
        "observations": hashlib.sha256(source_bytes).hexdigest(),
        "configuration": hashlib.sha256(configuration_bytes).hexdigest(),
    }
    assert {path.name for path in output.iterdir()} == {
        "bounds_si.csv", "candidate_space.csv", "baseline_targets_si.csv", "candidate_vectors.csv",
        "candidate_checks.csv", "reference_checks.csv", "summary.json", "resolved_config.json",
    }
    baseline = run_baseline(synthetic_source(), selection(), **MODEL)
    for period in ("calibration", "validation"):
        expected_costs = [-baseline.metrics[period]["release_nse"], -baseline.metrics[period]["storage_nse"]]
        actual = report["baseline"][period]
        assert actual["feasible"]
        assert actual["reason"] is None
        np.testing.assert_allclose(actual["objectives"], expected_costs, atol=1e-12)
        assert actual["paired_counts"] == baseline.paired_counts[period]
    space = pd.read_csv(output / "candidate_space.csv")
    assert space.shape == (72, 7)
    assert space.vector_index.tolist() == list(range(72))
    assert space.target_column.tolist() == list(np.repeat(MONTHLY_TARGET_COLUMNS, 12))
    assert space.month.tolist() == list(np.tile(np.arange(1, 13), 6))
    assert space.unit.tolist() == ["fraction_of_capacity"] * 36 + ["m3/s"] * 36
    fitted = synthetic_source().frame.loc["2000-01-01":"2000-12-31"]
    for block, column in enumerate(MONTHLY_TARGET_COLUMNS):
        variable = "storage_m3" if column.startswith("storage_") else "release_m3s"
        lower_p, upper_p = ((5, 35), (35, 75), (75, 95))[block % 3]
        scale = MODEL["capacity_m3"] if block < 3 else 1
        for month in range(1, 13):
            row = space.iloc[block * 12 + month - 1]
            sample = fitted.loc[fitted.index.month == month, variable]
            expected_limits = np.percentile(sample, [lower_p, upper_p], method="hazen") / scale
            np.testing.assert_allclose([row.lower, row.upper], expected_limits, atol=1e-12)
            assert row.baseline == pytest.approx(baseline.estimates.monthly_targets.loc[month, column] / scale)
    assert ((space.baseline >= space.lower) & (space.baseline <= space.upper)).all()
    vectors = pd.read_csv(output / "candidate_vectors.csv")
    assert vectors.candidate.tolist() == ["generalized_baseline"]
    np.testing.assert_allclose(vectors.filter(regex=r"^x\d{2}$").iloc[0], space.baseline, atol=1e-12)
    checks = pd.read_csv(output / "candidate_checks.csv")
    assert checks.role.tolist() == ["starting_candidate"]
    assert checks.inside_percentile_box.all()
    assert checks.physically_valid_within_bounds.all()
    assert pd.read_csv(output / "reference_checks.csv").empty

    def reject_nonstandard_json(token):
        raise AssertionError(f"Nonstandard JSON token: {token}")

    for path in output.glob("*.json"):
        content = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonstandard_json)
        assert content == (report if path.name == "summary.json" else config)
    assert source.read_bytes() == source_bytes
    assert configuration.read_bytes() == configuration_bytes


@pytest.mark.parametrize("collision", ["observations", "configuration"])
def test_review_output_cannot_overwrite_source_or_config(tmp_path, collision):
    source, configuration, _ = write_inputs(
        tmp_path,
        source_name="candidate_space.csv" if collision == "observations" else "observations.csv",
        config_name="summary.json" if collision == "configuration" else "config.json",
    )
    observation_bytes, configuration_bytes = source.read_bytes(), configuration.read_bytes()
    files_before = {path.name for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="overwrite source observations or configuration"):
        run_calibration_review(configuration, tmp_path, tmp_path)
    assert source.read_bytes() == observation_bytes
    assert configuration.read_bytes() == configuration_bytes
    assert {path.name for path in tmp_path.iterdir()} == files_before
