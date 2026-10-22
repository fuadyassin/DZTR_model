"""Verify scientific input interpretation and missing-data safeguards."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dztr.data import find_reservoir_file, normalize_observations, read_reservoir_observations


def legacy_input():
    return pd.DataFrame({
        "year": [2001, 2001, 2001], "month": [1, 1, 1], "day": [3, 1, 4],
        "Storage(mil_m3)": [3.0, 1.0, np.nan],
        "Inflow(cms)": [6.0, 2.0, 8.0], "Outflow(cms)": [9.0, 3.0, 12.0],
    })


def test_legacy_units_dates_missing_data_and_source_are_preserved():
    raw = legacy_input()
    before = raw.copy(deep=True)
    result = normalize_observations(raw, source="reference")
    pd.testing.assert_frame_equal(raw, before)
    assert result.frame.index.equals(pd.DatetimeIndex(["2001-01-01", "2001-01-03", "2001-01-04"], name="date"))
    assert list(result.frame.columns) == ["storage_m3", "inflow_m3s", "release_m3s"]
    assert result.frame.storage_m3.iloc[:2].tolist() == [1e6, 3e6]
    assert np.isnan(result.frame.storage_m3.iloc[2])
    assert result.quality.row_count == 3
    assert result.quality.missing_days == 1
    assert result.quality.missing_counts["storage_m3"] == 1
    assert result.quality.status == "available"
    assert result.source == "reference"


@pytest.mark.parametrize("storage_column,storage_unit", [("Storage(mcm)", None), ("Storage", "mcm")])
def test_hydrometric_date_and_storage_aliases(storage_column, storage_unit):
    raw = pd.DataFrame({
        "Date": ["2001-02-01", "2001-02-02"], storage_column: [2.5, 4.0],
        "Inflow(cms)": [1, 2], "Outflow(cms)": [3, 4],
    })
    normalized = normalize_observations(raw, storage_unit=storage_unit)
    assert normalized.frame.storage_m3.tolist() == [2.5e6, 4e6]


def test_bare_units_are_required_and_explicit_units_are_supported():
    raw = pd.DataFrame({"Date": ["2001-01-01"], "Storage": [2], "Inflow": [3], "Outflow": [4]})
    with pytest.raises(ValueError, match="unit"):
        normalize_observations(raw)
    result = normalize_observations(raw, storage_unit="million_m3", flow_unit="m3/s")
    assert result.frame.iloc[0].tolist() == [2e6, 3, 4]
    with pytest.raises(ValueError, match="conflict"):
        normalize_observations(legacy_input(), storage_unit="m3")


def test_missing_variables_remain_missing_and_generate_a_quality_issue():
    with pytest.warns(UserWarning, match="Missing observation"):
        result = normalize_observations(pd.DataFrame({"Date": ["2001-01-01"], "Storage(mcm)": [1]}))
    assert result.frame.inflow_m3s.isna().all()
    assert result.frame.release_m3s.isna().all()
    assert result.quality.missing_counts["release_m3s"] == 1
    assert result.quality.issues


def test_duplicate_dates_and_conflicting_date_representations_are_rejected():
    duplicated = legacy_input()
    duplicated.loc[1, "day"] = 3
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_observations(duplicated)
    conflicting = legacy_input()
    conflicting["Date"] = ["2002-01-03", "2002-01-01", "2002-01-04"]
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_observations(conflicting)


@pytest.mark.parametrize("bad_value,match", [(-1, "negative"), (np.inf, "infinite"), ("bad", "nonnumeric")])
def test_invalid_measurements_are_rejected(bad_value, match):
    raw = legacy_input()
    raw["Inflow(cms)"] = raw["Inflow(cms)"].astype(object)
    raw.loc[0, "Inflow(cms)"] = bad_value
    with pytest.raises(ValueError, match=match):
        normalize_observations(raw)


@pytest.mark.parametrize("date", ["2001-02-30", "2001-01-01 12:00:00", "2001-01-01T00:00:00Z", None])
def test_invalid_or_subdaily_dates_are_rejected(date):
    with pytest.raises(ValueError):
        normalize_observations(pd.DataFrame({"Date": [date], "storage_m3": [1], "inflow_m3s": [2], "release_m3s": [3]}))


def test_complete_zero_placeholders_and_missing_only_records_are_distinct():
    raw = pd.DataFrame({"Date": ["2001-01-01", "2001-01-02"], "Storage": [0, 0], "Inflow": [0, 0], "Outflow": [0, 0]})
    result = normalize_observations(raw)
    assert result.is_placeholder
    assert result.quality.status == "placeholder"
    assert result.frame.eq(0).all().all()
    missing = raw.copy()
    missing[["Storage", "Inflow", "Outflow"]] = np.nan
    result = normalize_observations(missing)
    assert not result.is_placeholder
    assert result.quality.status == "unavailable"
    assert result.frame.isna().all().all()


def test_zero_release_with_nonzero_storage_is_available():
    raw = pd.DataFrame({"Date": ["2001-01-01", "2001-01-02"], "storage_m3": [1, 2], "inflow_m3s": [0, 0], "release_m3s": [0, 0]})
    result = normalize_observations(raw)
    assert not result.is_placeholder
    assert result.quality.status == "available"


def test_find_reservoir_file_uses_exact_id_and_rejects_ambiguity(tmp_path):
    expected = tmp_path / "Gardiner_253.xlsx"
    expected.touch()
    (tmp_path / "Other_1253.xlsx").touch()
    assert find_reservoir_file(tmp_path, 253) == expected
    with pytest.raises(FileNotFoundError):
        find_reservoir_file(tmp_path, 131)
    (tmp_path / "Other_253.csv").touch()
    with pytest.raises(ValueError):
        find_reservoir_file(tmp_path, 253)


def test_csv_reader_normalizes_units_without_filling_gaps(tmp_path):
    path = tmp_path / "observations.csv"
    legacy_input().to_csv(path, index=False)
    result = read_reservoir_observations(path)
    assert result.quality.missing_days == 1
    assert result.frame.storage_m3.iloc[0] == 1e6
    assert result.source == str(path)
