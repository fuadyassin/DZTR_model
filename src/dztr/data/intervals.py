"""Prepare declared daily mean forcing and quarantined storage intervals.

This module records an accounting convention, not a physical measurement clock:
the mean labelled by day ``i`` supplies its nominal 86,400-second interval ending
at storage row ``i``. No interpolation, clipping, averaging of adjacent means,
simulation, or observation repair is performed here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
import re
from typing import ClassVar

import numpy as np
import pandas as pd


_OBSERVATION_COLUMNS = ("storage_m3", "inflow_m3s", "release_m3s")
_DERIVED_COLUMNS = frozenset({
    "interval_start", "interval_end", "interval_seconds",
    "reported_inflow_volume_m3", "usable_inflow_volume_m3",
    "source_transition_flagged", "storage_endpoint_quarantined", "segment_id",
    "segment_initialization", "simulation_interval_eligible",
    "source_audit_interval_eligible", "storage_target_eligible",
    "storage_score_eligible", "release_score_eligible", "period",
})


def _policy_date(value: object) -> pd.Timestamp:
    if isinstance(value, str):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
            raise ValueError("Quarantine date strings must use YYYY-MM-DD.")
    elif not isinstance(value, (date, datetime, pd.Timestamp, np.datetime64)):
        raise ValueError("Quarantine dates must be calendar dates, not numeric values.")
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid quarantine date {value!r}.") from exc
    if pd.isna(stamp) or stamp.tz is not None or stamp != stamp.normalize():
        raise ValueError("Quarantine dates must be finite timezone-naive midnight dates.")
    return stamp


@dataclass(frozen=True)
class DailyIntervalPolicy:
    """An explicit daily mean interpretation and storage-transition quarantine.

    ``physical_inflow`` rejects negative inflow. ``computed_net_inflow`` retains
    reported signs without treating them as an observed gross river inflow or
    attributing them to any particular loss process. Quarantine dates identify
    transition *end* labels; both observed storage endpoints are then excluded.
    The policy copies dates into an immutable, validated, sorted tuple.
    """

    flow_semantics: str = "physical_inflow"
    quarantined_interval_end_dates: tuple[pd.Timestamp, ...] = ()
    nominal_step_seconds: ClassVar[float] = 86400.0
    flow_alignment: ClassVar[str] = "interval_mean_end"

    def __post_init__(self) -> None:
        if (not isinstance(self.flow_semantics, str)
                or self.flow_semantics not in {"physical_inflow", "computed_net_inflow"}):
            raise ValueError("flow_semantics must be physical_inflow or computed_net_inflow.")
        raw_dates = self.quarantined_interval_end_dates
        if raw_dates is None or isinstance(raw_dates, (str, bytes)):
            raise ValueError("quarantined_interval_end_dates must be a collection of dates.")
        try:
            dates = tuple(_policy_date(value) for value in raw_dates)
        except TypeError as exc:
            raise ValueError("quarantined_interval_end_dates must be a collection of dates.") from exc
        if len(dates) != len(set(dates)):
            raise ValueError("Duplicate quarantine dates are ambiguous.")
        object.__setattr__(self, "quarantined_interval_end_dates", tuple(sorted(dates)))


@dataclass(frozen=True)
class IntervalPreparation:
    """Copied observations, eligibility masks, and contiguous clean segments.

    The container and policy are frozen; the returned DataFrames remain ordinary
    mutable DataFrames, independent of the caller's input. A segment's initial
    storage is provisional observed storage, never a validated physical state.
    """

    frame: pd.DataFrame
    segments: pd.DataFrame
    policy: DailyIntervalPolicy


def _validate_frame(frame: pd.DataFrame, policy: DailyIntervalPolicy) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if frame.columns.has_duplicates:
        raise ValueError("Duplicate observation columns are ambiguous.")
    collisions = _DERIVED_COLUMNS.intersection(frame.columns)
    if collisions:
        raise ValueError(f"Input contains reserved preparation columns: {sorted(collisions)}.")
    missing = set(_OBSERVATION_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing SI observation columns: {sorted(missing)}.")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Daily observations require a DatetimeIndex.")
    if len(index) < 2:
        raise ValueError("At least two daily storage rows are required.")
    if index.hasnans or index.tz is not None or not index.equals(index.normalize()):
        raise ValueError("Daily labels must be finite timezone-naive midnight dates.")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Daily labels must be sorted and unique.")
    if not ((index[1:] - index[:-1]) == pd.Timedelta(days=1)).all():
        raise ValueError("Daily labels must be consecutive; missing dates cannot be bridged.")
    for column in _OBSERVATION_COLUMNS:
        values = frame[column]
        if (not pd.api.types.is_numeric_dtype(values.dtype)
                or pd.api.types.is_bool_dtype(values.dtype)
                or pd.api.types.is_complex_dtype(values.dtype)):
            raise ValueError(f"{column} must contain normalized real numeric SI values.")
        numeric = values.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{column} must contain finite values; missing observations need review.")
        if ((column != "inflow_m3s" or policy.flow_semantics == "physical_inflow")
                and (numeric < 0).any()):
            if column == "inflow_m3s":
                raise ValueError("Negative inflow requires explicit computed_net_inflow semantics.")
            raise ValueError(f"{column} must be nonnegative.")
    dates = policy.quarantined_interval_end_dates
    if any(value not in index for value in dates):
        raise ValueError("Every quarantine end date must occur in the input calendar.")
    if index[0] in dates:
        raise ValueError("The first row has no preceding storage endpoint to quarantine.")


def _validated_periods(labels: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series | None:
    if labels is None:
        return None
    if not isinstance(labels, pd.Series):
        raise TypeError("period_labels must be a pandas Series with the observation index.")
    if not labels.index.equals(index):
        raise ValueError("period_labels must match the observation index exactly.")
    if not all(isinstance(value, str) and value.strip() for value in labels):
        raise ValueError("period_labels must contain nonempty strings without missing values.")
    return labels.copy(deep=True)


def _copied_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)
    # pandas' deep copy does not recursively copy Python objects inside cells.
    for column in frame.columns:
        if pd.api.types.is_object_dtype(frame[column].dtype):
            result[column] = pd.Series(
                [deepcopy(value) for value in frame[column]],
                index=frame.index, dtype=frame[column].dtype,
            )
    result.attrs = deepcopy(frame.attrs)
    return result


def prepare_daily_intervals(
    frame: pd.DataFrame,
    *,
    policy: DailyIntervalPolicy | None = None,
    period_labels: pd.Series | None = None,
) -> IntervalPreparation:
    """Retain all calendar rows and prepare daily mean interval eligibility.

    A flagged transition quarantines both endpoint storage observations. This
    also blocks every adjacent transition touching either endpoint. Each clean
    block becomes a separate segment; its first observed state initializes the
    block and is excluded from storage and release scores. Singleton blocks are
    retained and contain no usable interval. Period boundaries leave simulation
    continuity intact while excluding cross-period source-audit transitions.

    ``reported_inflow_volume_m3`` is the signed reported mean times 86,400 on
    every label, including the first label. ``usable_inflow_volume_m3`` is only
    present for eligible transitions. Neither implies a recovered hourly clock.
    Observations and additional input columns are copied without modification.
    """
    if policy is None:
        policy = DailyIntervalPolicy()
    if not isinstance(policy, DailyIntervalPolicy):
        raise TypeError("policy must be a DailyIntervalPolicy.")
    _validate_frame(frame, policy)
    periods = _validated_periods(period_labels, frame.index)
    result = _copied_frame(frame)
    index = result.index
    count = len(index)
    flagged = index.isin(policy.quarantined_interval_end_dates)
    quarantined = flagged.copy()
    quarantined[:-1] |= flagged[1:]
    clean = ~quarantined
    preceding_clean = np.r_[False, clean[:-1]]
    eligible = clean & preceding_clean
    initialization = clean & ~preceding_clean
    segment_numbers = np.cumsum(initialization)
    segment_ids = pd.array(segment_numbers, dtype="Int64")
    segment_ids[quarantined] = pd.NA

    result["interval_start"] = pd.Series(index, index=index).shift(1)
    result["interval_end"] = index
    result["interval_seconds"] = np.r_[np.nan, np.full(count - 1, policy.nominal_step_seconds)]
    with np.errstate(over="raise", invalid="raise"):
        try:
            volumes = result.inflow_m3s.to_numpy(dtype=float, na_value=np.nan) * policy.nominal_step_seconds
        except FloatingPointError as exc:
            raise ValueError("Reported inflow volumes exceed finite SI representation.") from exc
    result["reported_inflow_volume_m3"] = volumes
    result["usable_inflow_volume_m3"] = result.reported_inflow_volume_m3.where(eligible)
    result["source_transition_flagged"] = flagged
    result["storage_endpoint_quarantined"] = quarantined
    result["segment_id"] = segment_ids
    result["segment_initialization"] = initialization
    result["simulation_interval_eligible"] = eligible
    audit_eligible = eligible.copy()
    if periods is not None:
        result["period"] = periods
        audit_eligible &= periods.eq(periods.shift(1)).to_numpy(dtype=bool, na_value=False)
    result["source_audit_interval_eligible"] = audit_eligible
    result["storage_target_eligible"] = clean
    result["storage_score_eligible"] = eligible
    result["release_score_eligible"] = eligible

    records = []
    for segment_id in range(1, int(initialization.sum()) + 1):
        positions = np.flatnonzero(clean & (segment_numbers == segment_id))
        first, last = positions[0], positions[-1]
        record = {
            "segment_id": segment_id,
            "start": index[first],
            "end": index[last],
            "row_count": len(positions),
            "usable_intervals": len(positions) - 1,
            "initial_storage_m3": frame.storage_m3.iloc[first],
            "initialization_status": "provisional_observed_storage",
        }
        if periods is not None:
            record["period_start"] = periods.iloc[first]
            record["period_end"] = periods.iloc[last]
        records.append(record)
    segment_columns = [
        "segment_id", "start", "end", "row_count", "usable_intervals",
        "initial_storage_m3", "initialization_status",
    ]
    if periods is not None:
        segment_columns.extend(["period_start", "period_end"])
    segments = pd.DataFrame.from_records(records, columns=segment_columns)
    return IntervalPreparation(result, segments, policy)
