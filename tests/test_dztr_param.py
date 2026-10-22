"""Compatibility regressions for legacy percentile and workbook workflows."""

import numpy as np
import pandas as pd
import pytest

from dztr_param.columns import create_extended_columns
from dztr_param.data_updater import update_extended_data
from dztr_param.file_reader import check_and_read_files
from dztr_param.percentiles import calculate_percentiles


def raw_observations():
    return pd.DataFrame({
        "year": [2001, 2001, 2001], "month": [1, 1, 1], "day": [1, 3, 4],
        "Storage(mil_m3)": [1, 3, 5], "Inflow(cms)": [2, 6, np.nan], "Outflow(cms)": [3, 9, 15],
    })


def test_schema_and_percentiles_agree_and_preserve_legacy_storage_units():
    wide = calculate_percentiles(raw_observations())
    schema = create_extended_columns()
    assert wide.shape == (21, 39)
    assert len(schema) == 822
    assert len(schema) == len(set(schema))
    assert wide.at["P45", "Storage(mil_m3)_all"] == pytest.approx(2.8)
    assert wide.at["P99", "Outflow(cms)_all"] == pytest.approx(14.88)
    for metric in ("Q", "S", "I"):
        for percentile in (65, 99, 100):
            assert f"{metric}{percentile}" in schema
            assert f"{metric}{percentile}_Dec" in schema
    assert wide["Inflow(cms)_Feb"].isna().all()


def test_updater_handles_new_ids_gaps_and_extra_metadata_without_mutation(tmp_path):
    raw = raw_observations()
    raw.to_excel(tmp_path / "New_999.xlsx", index=False)
    catalog = pd.DataFrame({"GRAND_ID": [253], "RES_NAME": ["existing"], "DAM_NAME": ["dam"], "CAP_MCM": [1234.0]})
    before = catalog.copy(deep=True)
    result = update_extended_data(catalog, [(999, "New reservoir", "New dam")], folder=tmp_path)
    pd.testing.assert_frame_equal(catalog, before)
    assert result.loc[result.GRAND_ID == 253, "CAP_MCM"].iloc[0] == 1234
    row = result.loc[result.GRAND_ID == 999].iloc[0]
    assert row.RES_NAME == "New reservoir"
    assert row.DAM_NAME == "New dam"
    assert row.S45 == pytest.approx(2.8)
    assert row.Q99 == pytest.approx(14.88)
    assert np.isnan(row.Q45_Feb)


def test_updater_preserves_names_for_integer_id_and_sets_nan_parameters(tmp_path):
    raw_observations().to_excel(tmp_path / "Existing_253.xlsx", index=False)
    catalog = pd.DataFrame({"GRAND_ID": [253], "RES_NAME": ["existing"], "DAM_NAME": ["existing dam"], "Q45_Feb": [999.0]})
    result = update_extended_data(catalog, [np.int64(253)], folder=tmp_path)
    assert result.loc[0, "RES_NAME"] == "existing"
    assert result.loc[0, "DAM_NAME"] == "existing dam"
    assert np.isnan(result.loc[0, "Q45_Feb"])
    result = update_extended_data(catalog, [(253, "renamed", "renamed dam")], folder=tmp_path)
    assert result.loc[0, "RES_NAME"] == "renamed"
    assert result.loc[0, "DAM_NAME"] == "renamed dam"


def test_updater_rejects_duplicate_catalog_ids(tmp_path):
    catalog = pd.DataFrame({"GRAND_ID": [253, 253]})
    with pytest.raises(ValueError, match="duplicate GRAND_ID"):
        update_extended_data(catalog, [253], folder=tmp_path)


@pytest.mark.parametrize("value", [0, np.nan])
def test_updater_skips_placeholder_and_unavailable_records(tmp_path, value):
    raw = raw_observations()
    raw[["Storage(mil_m3)", "Inflow(cms)", "Outflow(cms)"]] = value
    raw.to_excel(tmp_path / "Placeholder_999.xlsx", index=False)
    catalog = pd.DataFrame({"GRAND_ID": [253], "RES_NAME": ["existing"]})
    with pytest.warns(UserWarning, match="Skipping GRAND_ID"):
        result = update_extended_data(catalog, [999], folder=tmp_path)
    assert result.GRAND_ID.tolist() == [253]
    assert result.Q10.isna().all()


def test_raw_legacy_reader_warns_for_missing_and_ambiguous_files(tmp_path):
    with pytest.warns(UserWarning):
        assert check_and_read_files(253, tmp_path) is None
    raw = raw_observations()
    raw.to_excel(tmp_path / "Gardiner_253.xlsx", index=False)
    actual = check_and_read_files(253, tmp_path)
    pd.testing.assert_frame_equal(actual, raw)
    raw.to_excel(tmp_path / "Duplicate_253.xlsx", index=False)
    with pytest.warns(UserWarning):
        assert check_and_read_files(253, tmp_path) is None


def test_legacy_reader_and_updater_accept_csv_observations(tmp_path):
    raw_observations().to_csv(tmp_path / "Gardiner_253.csv", index=False)
    raw = check_and_read_files(253, tmp_path)
    pd.testing.assert_frame_equal(raw, raw_observations())
    catalog = pd.DataFrame({"GRAND_ID": [253], "RES_NAME": ["Gardiner"]})
    updated = update_extended_data(catalog, [253], folder=tmp_path)
    assert updated.loc[0, "S45"] == pytest.approx(2.8)
    assert updated.loc[0, "Q99"] == pytest.approx(14.88)
