"""Explicit daily periods for calibration and validation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral, Number
import re

import numpy as np
import pandas as pd

from dztr.data import normalize_observations
from dztr.schemas import ReservoirObservations


def _calendar_date(value: object, field: str) -> pd.Timestamp:
    """Accept explicit calendar dates, with no inferred year/month boundaries."""
    if value is None or value is pd.NaT or isinstance(value, Number):
        raise ValueError(f"{field} must be an explicit calendar date, not a missing or numeric value.")
    if not isinstance(value, (str, date, datetime, pd.Timestamp, np.datetime64)):
        raise ValueError(f"{field} must be an ISO calendar date or a date/datetime value.")
    if isinstance(value, str):
        value = value.strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ].+)?", value):
            raise ValueError(f"{field} must specify an explicit ISO calendar date (YYYY-MM-DD).")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} contains an invalid calendar date: {value!r}.") from error
    if pd.isna(timestamp):
        raise ValueError(f"{field} cannot be a missing date.")
    if timestamp.tz is not None:
        raise ValueError(f"{field} must be a timezone-naive calendar date.")
    if timestamp != timestamp.normalize():
        raise ValueError(f"{field} must be at midnight; subdaily dates are unsupported.")
    return timestamp


@dataclass(frozen=True)
class DatePeriod:
    """Inclusive start/end dates; strings must supply ISO calendar dates."""

    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        start, end = _calendar_date(self.start, "start"), _calendar_date(self.end, "end")
        if end < start:
            raise ValueError("Period end must be on or after its start.")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def day_count(self) -> int:
        """Number of daily rows required, including both boundaries."""
        return int((self.end - self.start).days) + 1

    def as_dict(self) -> dict[str, str]:
        """Return portable ISO dates for experiment configuration and reports."""
        return {"start": self.start.date().isoformat(), "end": self.end.date().isoformat()}


@dataclass(frozen=True)
class CalibrationValidationSplit:
    """Chronological fit/validation periods and a preceding simulation spinup.

    Validation must start after calibration ends. A gap between the periods is
    allowed; the continuous simulation workflow must still cover that gap.
    ``warmup_steps`` counts simulated calendar days before calibration starts.
    These days are excluded from parameter estimation and from both scores;
    their ending state carries into calibration and onward into validation.
    """

    calibration: DatePeriod
    validation: DatePeriod
    warmup_steps: int = 365

    def __post_init__(self) -> None:
        if not isinstance(self.calibration, DatePeriod) or not isinstance(self.validation, DatePeriod):
            raise TypeError("calibration and validation must be DatePeriod objects.")
        if self.validation.start <= self.calibration.end:
            raise ValueError("Validation must start after calibration ends; periods cannot overlap or reverse order.")
        if not isinstance(self.warmup_steps, Integral) or isinstance(self.warmup_steps, bool) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer count of simulation days before calibration starts.")
        object.__setattr__(self, "warmup_steps", int(self.warmup_steps))

    @property
    def simulation_period(self) -> DatePeriod:
        """Continuous run dates, including spinup and any gap before validation."""
        return DatePeriod(
            self.calibration.start - pd.Timedelta(days=self.warmup_steps),
            self.validation.end,
        )


def select_period(observations: ReservoirObservations, period: DatePeriod) -> ReservoirObservations:
    """Select complete inclusive daily coverage and recompute subset quality.

    Every requested calendar date must be present exactly once and in order.
    Missing measurement values remain NaN; missing dates or boundaries raise an
    error. No records are filled, shortened, or inferred, and the source frame
    remains unchanged.
    """
    if not isinstance(observations, ReservoirObservations):
        raise TypeError("observations must be normalized ReservoirObservations.")
    if not isinstance(period, DatePeriod):
        raise TypeError("period must be a DatePeriod.")
    if observations.quality.status != "available":
        raise ValueError(f"Cannot select a calibration/validation period from {observations.quality.status} observations.")
    frame = observations.frame
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is not None:
        raise ValueError("Observations require a timezone-naive daily DatetimeIndex.")
    if frame.index.hasnans or not frame.index.equals(frame.index.normalize()):
        raise ValueError("Observation dates must be complete calendar dates at midnight.")
    selected = frame.loc[(frame.index >= period.start) & (frame.index <= period.end)]
    expected = pd.date_range(period.start, period.end, freq="D", name="date")
    if len(selected.index) != len(expected) or not (selected.index == expected).all():
        raise ValueError(
            f"Period {period.start.date().isoformat()} through {period.end.date().isoformat()} "
            f"requires complete consecutive daily coverage ({period.day_count} rows), including both boundaries."
        )
    return normalize_observations(selected.copy(deep=True), source=observations.source)
