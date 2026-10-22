"""Portable source-reconciliation checks without reservoir model evaluations."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

scipy_io = pytest.importorskip("scipy.io")

import dztr.integrations.world_observations as module
from dztr.integrations.world_observations import WorldObservationConflict, read_world_observations


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source(tmp_path, book, *, dates=None, values=None, monthly=False, aggregation="none", factor=1e6, order=("year", "month", "day"), calendar_source=None, origin=None, selection=None):
    folder = tmp_path / "example"
    folder.mkdir()
    workbook, mat, writer = (folder / name for name in ("res_info.xlsx", "res_info_out.mat", "writer.m"))
    pd.DataFrame(book).to_excel(workbook, header=False, index=False)
    if dates is None:
        dates = pd.to_datetime(pd.DataFrame(np.asarray(book)[:, :3], columns=["year", "month", "day"]))
    dates = pd.DatetimeIndex(dates)
    origin = str(dates[0].date()) if origin is None else origin
    writer.write_text(f"strdate2 = '{origin}';\nres_info_out.stoobs = resdata(:,4).*{factor};\n", encoding="utf-8")
    if values is None:
        values = np.asarray(book, dtype=float)[:, 3:6].copy()
        values[:, 0] *= factor
    serial = (dates - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1) + 719529
    if monthly:
        serial = float(serial[0]) + np.arange(len(dates)) * 30.5
    info = {"timed": np.asarray(serial)}
    info.update({field: np.asarray(values)[:, column] for column, field in enumerate(("stoobs", "inflow", "outflow"))})
    scipy_io.savemat(mat, {"res_info_out": info})
    mapping = {"storage_factor": factor, "aggregation": aggregation, "date_column_order": list(order) if order else None, "year_offset": None, "calendar_source": calendar_source or ("workbook_month_labels" if monthly else "writer_generated_row_order"), "workbook_legacy_start": origin, "measurement_columns": [4, 5, 6]}
    return {
        "reservoir_id": "fixture", "folder": str(folder), "writer_file": str(writer),
        "source_sha256": {str(path): _digest(path) for path in (workbook, mat, writer)},
        "sampling": {"frequency": "monthly" if monthly else "daily", "timestep_days": 30.5 if monthly else 1, "legacy_start": origin},
        "observation_mapping": mapping, "selection": selection or {},
    }


def _daily_book():
    return [[2000, 1, 1, 10, 2, 3], [2000, 1, 2, 11, 4, 5], [2000, 1, 3, 12, 6, 7]]


def test_headerless_first_record_units_and_raw_flow_preprocessing(tmp_path):
    spec = _source(tmp_path, _daily_book())
    spec["inflow_preprocessing"] = {"factor": 1.45, "offset_m3s": 8}
    observations, proof = read_world_observations(spec)
    assert list(observations.frame.index) == list(pd.date_range("2000-01-01", periods=3))
    assert observations.frame.iloc[0].tolist() == [10e6, 2, 3]
    assert proof["inflow_preprocessing_applied"] is False
    assert observations.frame.inflow_m3s.tolist() == [2, 4, 6]
    assert proof["selected_mat_rows_1based"] == {"first": 1, "last": 3}
    assert proof["status"] == "verified"


def test_falcon_month_day_year_inference(tmp_path):
    book = [[1, 30, 2000, 10, 2, 3], [1, 31, 2000, 11, 4, 5], [2, 1, 2000, 12, 6, 7]]
    spec = _source(tmp_path, book, dates=pd.date_range("2000-01-30", periods=3), order=None)
    observations, proof = read_world_observations(spec)
    assert proof["calendar_mapping"]["date_column_order"] == ["month", "day", "year"]
    assert observations.frame.index[0] == pd.Timestamp("2000-01-30")


def test_float_components_do_not_allow_overflowing_month_alternative(tmp_path):
    book = [[2003.0, 1.0, float(day), 10, 2, 3] for day in (1, 2, 3)]
    spec = _source(tmp_path, book, order=None)
    _, proof = read_world_observations(spec)
    assert proof["calendar_mapping"]["date_column_order"] == ["year", "month", "day"]


def test_daily_generated_clock_preserves_ordinal_rows_and_records_labels(tmp_path):
    book = [[80, 1, 1, 10, 2, 3], [87, 1, 2, 11, 4, 5], [80, 1, 1, 12, 6, 7]]
    spec = _source(tmp_path, book, dates=pd.date_range("1980-01-01", periods=3), order=None)
    observations, proof = read_world_observations(spec)
    assert observations.frame.storage_m3.tolist() == [10e6, 11e6, 12e6]
    assert observations.frame.index.equals(pd.date_range("1980-01-01", periods=3, name="date"))
    assert proof["calendar_mapping"]["year_offset"] == 1900
    assert [row["workbook_row_1based"] for row in proof["workbook_calendar_label_discrepancies"]] == [2, 3]
    assert proof["status"] == "review_required"


def test_generated_daily_subset_uses_full_workbook_origin(tmp_path):
    book = _daily_book()
    values = np.asarray(book, dtype=float)[1:, 3:6]
    values[:, 0] *= 1e6
    spec = _source(tmp_path, book, dates=pd.date_range("2000-01-02", periods=2), values=values, origin="2000-01-01")
    observations, _ = read_world_observations(spec)
    assert observations.frame.storage_m3.tolist() == [11e6, 12e6]


def test_month_labels_replace_fractional_clock_and_source_rows_select(tmp_path):
    book = [[2000, month, 0, month, month + 10, month + 20] for month in range(1, 6)]
    spec = _source(tmp_path, book, dates=pd.date_range("2000-01-01", periods=4, freq="MS"), values=np.asarray(book)[:4, 3:6] * [1e6, 1, 1], monthly=True, selection={"t1": 1, "t2": 3, "start": "20000101", "end": "20000331"})
    observations, proof = read_world_observations(spec)
    assert observations.frame.index.equals(pd.date_range("2000-01-01", periods=3, freq="MS", name="date"))
    assert observations.frame.storage_m3.tolist() == [1e6, 2e6, 3e6]
    assert proof["legacy_clock"]["nonmidnight_count"] == 2
    assert observations.quality.missing_days == 0
    assert proof["missing_observation_periods"] == 0


def test_dekadal_unweighted_means_and_mat_values_are_retained(tmp_path):
    book = [[2001, month, day, storage, flow, 3] for month in (1, 2) for day, storage, flow in ((1, 9, 2), (10, 12, 5), (20, 15, 8))]
    saved = np.array([[12e6 + 1e-6, 5, 3], [12e6, 5, 3]])
    spec = _source(tmp_path, book, dates=pd.date_range("2001-01-01", periods=2, freq="MS"), values=saved, monthly=True, aggregation="unweighted_monthly_mean")
    observations, proof = read_world_observations(spec)
    np.testing.assert_array_equal(observations.frame.to_numpy(), saved)
    assert proof["aggregation_group_counts"] == {"2001-01": 3, "2001-02": 3}
    assert proof["workbook_mat_comparison"]["storage_m3"]["matching"]


def test_aggregation_propagates_missing_measurements_like_matlab(tmp_path):
    book = [[2001, 1, 1, 9, 2, 3], [2001, 1, 10, np.nan, 5, 3], [2001, 1, 20, 15, 8, 3]]
    spec = _source(tmp_path, book, dates=["2001-01-01"], values=[[np.nan, 5, 3]], monthly=True, aggregation="unweighted_monthly_mean")
    observations, proof = read_world_observations(spec)
    assert np.isnan(observations.frame.iloc[0].storage_m3)
    assert proof["fields"]["storage_m3"]["missing_count"] == 1


@pytest.mark.parametrize("monthly", [False, True])
def test_calendar_gaps_preserved_and_reported_in_sampling_units(tmp_path, monthly):
    dates = pd.to_datetime(["2000-01-01", "2000-03-01"] if monthly else ["2000-01-01", "2000-01-03"])
    book = [[date.year, date.month, 0 if monthly else date.day, 10 + i, 2, 3] for i, date in enumerate(dates)]
    spec = _source(tmp_path, book, dates=dates, monthly=monthly, calendar_source="workbook_month_labels" if monthly else "workbook")
    observations, proof = read_world_observations(spec)
    assert observations.frame.index.equals(pd.DatetimeIndex(dates, name="date"))
    assert len(observations.frame) == 2
    assert proof["missing_observation_periods"] == 1
    assert observations.quality.missing_days == (0 if monthly else 1)


def test_negative_missing_and_all_zero_forcing_remain_visible(tmp_path):
    book = [[2000, 1, 1, -10, 0, 3], [2000, 1, 2, np.nan, 0, -1], [2000, 1, 3, 12, 0, 0]]
    spec = _source(tmp_path, book)
    observations, proof = read_world_observations(spec)
    assert observations.frame.iloc[0].storage_m3 == -10e6
    assert np.isnan(observations.frame.iloc[1].storage_m3)
    assert observations.frame.iloc[1].release_m3s == -1
    assert proof["fields"]["inflow_m3s"]["zero_count"] == 3
    assert proof["status"] == "review_required"
    assert any("entirely zero" in issue for issue in observations.quality.issues)


def test_source_selection_compares_only_used_rows(tmp_path):
    values = np.asarray(_daily_book(), dtype=float)[:, 3:6] * [1e6, 1, 1]
    values[-1, 0] += 100
    spec = _source(tmp_path, _daily_book(), values=values, selection={"t1": 1, "t2": 2})
    observations, _ = read_world_observations(spec)
    assert len(observations.frame) == 2


@pytest.mark.parametrize("column", [0, 1, 2])
def test_stale_mat_measurement_conflict_has_provenance(tmp_path, column):
    values = np.asarray(_daily_book(), dtype=float)[:, 3:6] * [1e6, 1, 1]
    values[0, column] += 10
    spec = _source(tmp_path, _daily_book(), values=values)
    with pytest.raises(WorldObservationConflict, match="Selected MAT measurements conflict") as caught:
        read_world_observations(spec)
    comparison = caught.value.provenance["workbook_mat_comparison"]
    assert sum(item["mismatch_count"] for item in comparison.values()) == 1
    assert caught.value.provenance["status"] == "conflict"


def test_tuyen_elevation_volume_conflict_does_not_guess_conversion(tmp_path):
    book = [[2009, 1, 1, 1.1292e9, 2, 3], [2009, 1, 2, 1.1302e9, 4, 5]]
    spec = _source(tmp_path, book, factor=1, values=[[112.92, 2, 3], [113.02, 4, 5]])
    spec["reservoir_id"] = "ex43"
    with pytest.raises(WorldObservationConflict, match="no inferred scale or hypsometry") as caught:
        read_world_observations(spec)
    assert caught.value.provenance["units"]["workbook_storage_factor"] == 1


@pytest.mark.parametrize("file", ["res_info.xlsx", "res_info_out.mat", "writer.m"])
def test_hash_guard_before_reading_each_required_source(tmp_path, file):
    spec = _source(tmp_path, _daily_book())
    path = Path(spec["folder"]) / file
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(WorldObservationConflict, match="Source SHA256 changed"):
        read_world_observations(spec)


def test_hash_guard_after_read_detects_mid_read_source_change(tmp_path, monkeypatch):
    spec = _source(tmp_path, _daily_book())
    original = module._read_mat
    def changed(path):
        arrays = original(path)
        writer = Path(spec["writer_file"])
        writer.write_text(writer.read_text() + "% edited during import\n")
        return arrays
    monkeypatch.setattr(module, "_read_mat", changed)
    with pytest.raises(WorldObservationConflict, match="Source SHA256 changed"):
        read_world_observations(spec)


def test_required_input_cannot_be_omitted_from_hash_manifest(tmp_path):
    spec = _source(tmp_path, _daily_book())
    del spec["source_sha256"][str(Path(spec["folder"]) / "res_info_out.mat")]
    with pytest.raises(WorldObservationConflict, match="must all be pinned"):
        read_world_observations(spec)


@pytest.mark.parametrize("selection", [{"t1": 0}, {"t2": 4}, {"t1": 3, "t2": 2}, {"t1": True}, {"t1": "find(...)"}, {"start": "20010101"}])
def test_unresolved_invalid_or_empty_source_selection_rejected(tmp_path, selection):
    spec = _source(tmp_path, _daily_book(), selection=selection)
    with pytest.raises(WorldObservationConflict):
        read_world_observations(spec)


def test_monthly_clock_and_first_month_conflicts_rejected(tmp_path):
    book = [[2000, 2, 0, 10, 2, 3], [2000, 3, 0, 11, 4, 5]]
    spec = _source(tmp_path, book, dates=pd.date_range("2000-01-01", periods=2, freq="MS"), monthly=True)
    with pytest.raises(WorldObservationConflict, match="First workbook month"):
        read_world_observations(spec)
    spec["sampling"]["timestep_days"] = 30
    with pytest.raises(WorldObservationConflict, match="discovered fixed timestep"):
        read_world_observations(spec)


def test_unpinned_storage_unit_inference_not_permitted(tmp_path):
    spec = _source(tmp_path, _daily_book())
    spec["observation_mapping"]["storage_factor"] = None
    with pytest.raises(WorldObservationConflict, match="evidenced by the writer"):
        read_world_observations(spec)


def test_multiple_parameter_writers_require_equal_observation_consensus(tmp_path):
    spec = _source(tmp_path, _daily_book())
    first = Path(spec["writer_file"])
    second = first.with_name("writer_variant.m")
    second.write_text(first.read_text() + "target_percentile=50;\n")
    spec["writer_file"] = None
    spec["writer_candidates"] = [str(first), str(second)]
    spec["source_sha256"][str(second)] = _digest(second)
    observations, proof = read_world_observations(spec)
    assert len(observations.frame) == 3
    assert proof["writer_file"] is None
    assert proof["observation_writer_consensus"]
    assert proof["observation_writer_candidates"] == [str(first), str(second)]
    second.write_text(second.read_text().replace("1000000.0", "1.0"))
    spec["source_sha256"][str(second)] = _digest(second)
    with pytest.raises(WorldObservationConflict, match="Multiple writers disagree"):
        read_world_observations(spec)
