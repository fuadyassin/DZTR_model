"""Date boundaries and daily coverage safeguards for split experiments."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from dztr.calibration.periods import CalibrationValidationSplit, DatePeriod, select_period
from dztr.data import normalize_observations
from dztr.schemas import DataQuality, ReservoirObservations


def source_observations():
    index = pd.date_range("1999-12-30", "2000-03-03", name="date")
    values = np.arange(1, len(index) + 1, dtype=float)
    frame = pd.DataFrame({"storage_m3": values * 1e6, "inflow_m3s": values + 1, "release_m3s": values + 2}, index=index)
    frame.loc[pd.Timestamp("2000-02-29"), "storage_m3"] = np.nan
    return normalize_observations(frame, source="fixture_reference")


@pytest.mark.parametrize("start,end", [
    ("2000-02-28", "2000-03-01"),
    (date(2000, 2, 28), date(2000, 3, 1)),
    (datetime(2000, 2, 28), datetime(2000, 3, 1)),
    (pd.Timestamp("2000-02-28"), pd.Timestamp("2000-03-01")),
    (np.datetime64("2000-02-28"), np.datetime64("2000-03-01")),
    ("2000-02-28T00:00:00", "2000-03-01 00:00:00"),
])
def test_explicit_dates_normalize_and_leap_day_counts_are_inclusive(start, end):
    period = DatePeriod(start, end)
    assert isinstance(period.start, pd.Timestamp)
    assert isinstance(period.end, pd.Timestamp)
    assert period.day_count == 3
    assert period.as_dict() == {"start": "2000-02-28", "end": "2000-03-01"}
    assert DatePeriod("2001-02-28", "2001-03-01").day_count == 2
    assert DatePeriod("2000-02-29", "2000-02-29").day_count == 1


@pytest.mark.parametrize("bad", [
    None, pd.NaT, np.datetime64("NaT"), np.nan, 20000101, 0.0, np.int64(20000101), True,
    "", "NaT", "today", "2000", "2000-02", "20000228", "2001-02-29", "2000-02-30",
    "2000-02-28T12:00:00", "2000-02-28T00:00:00.000000001", "2000-02-28T00:00:00Z",
    datetime(2000, 2, 28, 1), pd.Timestamp("2000-02-28", tz="UTC"),
])
def test_ambiguous_missing_numeric_timezone_and_subdaily_dates_are_rejected(bad):
    with pytest.raises(ValueError):
        DatePeriod(bad, "2000-03-01")
    with pytest.raises(ValueError):
        DatePeriod("2000-02-28", bad)


def test_date_period_order_and_immutability():
    with pytest.raises(ValueError, match="end"):
        DatePeriod("2000-03-01", "2000-02-28")
    period = DatePeriod("2000-02-28", "2000-03-01")
    with pytest.raises(FrozenInstanceError):
        period.start = pd.Timestamp("2000-01-01")


def test_chronological_split_accepts_adjacent_or_gapped_validation_and_numpy_integer_warmup():
    calibration = DatePeriod("1999-01-01", "2000-01-01")
    adjacent = CalibrationValidationSplit(calibration, DatePeriod("2000-01-02", "2000-12-31"))
    assert adjacent.warmup_steps == 365
    assert adjacent.simulation_period.as_dict() == {"start": "1998-01-01", "end": "2000-12-31"}
    gapped = CalibrationValidationSplit(calibration, DatePeriod("2000-02-01", "2000-12-31"), warmup_steps=np.int64(3))
    assert gapped.warmup_steps == 3
    assert isinstance(gapped.warmup_steps, int)
    assert gapped.simulation_period.as_dict() == {"start": "1998-12-29", "end": "2000-12-31"}
    with pytest.raises(FrozenInstanceError):
        gapped.warmup_steps = 5


@pytest.mark.parametrize("validation", [
    DatePeriod("1999-01-01", "1999-12-31"),
    DatePeriod("2000-01-01", "2000-01-31"),
    DatePeriod("2000-01-31", "2000-02-28"),
])
def test_split_rejects_overlap_or_reverse_chronology(validation):
    calibration = DatePeriod("2000-01-01", "2000-01-31")
    with pytest.raises(ValueError, match="after calibration"):
        CalibrationValidationSplit(calibration, validation, warmup_steps=0)


@pytest.mark.parametrize("warmup", [-1, 1.0, 1.5, True, "1", None])
def test_warmup_is_a_nonnegative_integer(warmup):
    with pytest.raises(ValueError, match="warmup_steps"):
        CalibrationValidationSplit(DatePeriod("2000-01-01", "2000-01-31"), DatePeriod("2000-02-01", "2000-02-29"), warmup_steps=warmup)


def test_spinup_can_exceed_calibration_length_and_zero_spinup_starts_at_calibration():
    calibration, validation = DatePeriod("2000-01-01", "2000-01-31"), DatePeriod("2000-02-01", "2000-02-29")
    long_spinup = CalibrationValidationSplit(calibration, validation, warmup_steps=366)
    assert long_spinup.simulation_period.start == pd.Timestamp("1998-12-31")
    assert long_spinup.calibration.day_count == 31
    no_spinup = CalibrationValidationSplit(calibration, validation, warmup_steps=0)
    assert no_spinup.simulation_period.start == calibration.start


def test_paper_protocol_interpretation_has_separate_trinity_spinup_and_no_overlap():
    split = CalibrationValidationSplit(DatePeriod("1971-01-01", "1985-12-31"), DatePeriod("1986-01-01", "2000-12-31"))
    assert split.simulation_period.as_dict() == {"start": "1970-01-01", "end": "2000-12-31"}
    assert split.calibration.start - split.simulation_period.start == pd.Timedelta(days=365)
    assert split.calibration.start > split.simulation_period.start


@pytest.mark.parametrize("unit", ["s", "ms", "us"])
def test_period_selection_accepts_equivalent_daily_datetime_resolutions(unit):
    source = source_observations()
    frame = source.frame.copy()
    # Pandas 1.x promotes to ns; Pandas 2.x retains these explicit resolutions.
    frame.index = pd.DatetimeIndex(frame.index.to_numpy().astype(f"datetime64[{unit}]"), name="date")
    normalized = normalize_observations(frame, source=source.source)
    period = DatePeriod("2000-01-01", "2000-01-03")
    selected = select_period(normalized, period)
    assert selected.frame.index.tolist() == [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-02"), pd.Timestamp("2000-01-03")]
    assert selected.frame.index.dtype == normalized.frame.index.dtype
    assert selected.quality.row_count == 3
    np.testing.assert_array_equal(selected.frame.to_numpy(), source.frame.loc["2000-01-01":"2000-01-03"].to_numpy())


def test_selection_is_inclusive_preserves_nan_values_and_does_not_mutate_source():
    source = source_observations()
    before = source.frame.copy(deep=True)
    period = DatePeriod("2000-02-28", "2000-03-01")
    selected = select_period(source, period)
    assert selected.frame.index.tolist() == [pd.Timestamp("2000-02-28"), pd.Timestamp("2000-02-29"), pd.Timestamp("2000-03-01")]
    assert selected.quality.row_count == 3
    assert selected.quality.missing_days == 0
    assert selected.quality.missing_counts == {"storage_m3": 1, "inflow_m3s": 0, "release_m3s": 0}
    assert np.isnan(selected.frame.loc[pd.Timestamp("2000-02-29"), "storage_m3"])
    assert selected.source == "fixture_reference"
    selected.frame.iloc[0, 0] = -1
    pd.testing.assert_frame_equal(source.frame, before)


def test_selected_quality_is_recomputed_instead_of_copying_full_record_counts():
    source = source_observations()
    stale = ReservoirObservations(source.frame, DataQuality("available", 100000, {"storage_m3": 1000}, 1000, ("stale issue",)), source.source)
    selected = select_period(stale, DatePeriod("2000-01-01", "2000-01-03"))
    assert selected.quality.row_count == 3
    assert selected.quality.missing_counts == {"storage_m3": 0, "inflow_m3s": 0, "release_m3s": 0}
    assert selected.quality.missing_days == 0
    assert "stale issue" not in selected.quality.issues


@pytest.mark.parametrize("period", [
    DatePeriod("1999-12-29", "2000-01-01"),
    DatePeriod("2000-03-01", "2000-03-04"),
    DatePeriod("1998-01-01", "1998-01-02"),
])
def test_missing_requested_endpoints_never_silently_shorten_selection(period):
    with pytest.raises(ValueError, match="complete consecutive daily coverage"):
        select_period(source_observations(), period)


def test_internal_calendar_gap_or_duplicate_is_rejected_without_filling():
    source = source_observations()
    gap = normalize_observations(source.frame.drop(index=pd.Timestamp("2000-02-29")), source=source.source)
    with pytest.raises(ValueError, match="complete consecutive daily coverage"):
        select_period(gap, DatePeriod("2000-02-28", "2000-03-01"))
    # Gaps outside the requested interval do not affect complete interval selection.
    selected = select_period(gap, DatePeriod("2000-01-01", "2000-01-03"))
    assert selected.quality.missing_days == 0
    duplicate = pd.concat([source.frame, source.frame.loc[[pd.Timestamp("2000-02-29")]]]).sort_index()
    duplicate_source = ReservoirObservations(duplicate, source.quality, source.source)
    with pytest.raises(ValueError, match="complete consecutive daily coverage"):
        select_period(duplicate_source, DatePeriod("2000-02-28", "2000-03-01"))


@pytest.mark.parametrize("status", ["unavailable", "placeholder"])
def test_unusable_source_status_cannot_be_selected(status):
    source = source_observations()
    unavailable = ReservoirObservations(source.frame, DataQuality(status, len(source.frame), {}, 0), source.source)
    with pytest.raises(ValueError, match=status):
        select_period(unavailable, DatePeriod("2000-01-01", "2000-01-03"))


def test_api_rejects_wrong_period_and_observation_types():
    period = DatePeriod("2000-01-01", "2000-01-03")
    with pytest.raises(TypeError, match="DatePeriod"):
        CalibrationValidationSplit(period.as_dict(), period, warmup_steps=0)
    with pytest.raises(TypeError, match="DatePeriod"):
        select_period(source_observations(), period.as_dict())
    with pytest.raises(TypeError, match="ReservoirObservations"):
        select_period(source_observations().frame, period)
