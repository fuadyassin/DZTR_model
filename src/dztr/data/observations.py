"""Normalize daily reservoir observations without inventing or filling records."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from dztr.schemas import DataQuality, ReservoirObservations


OBSERVATION_COLUMNS = ("storage_m3", "inflow_m3s", "release_m3s")


def _key(value: object) -> str:
    return str(value).strip().casefold().replace(" ", "").replace("³", "3")


# A None conversion factor denotes a column whose units must be supplied.
_ALIASES = {
    "storage_m3": ("storage_m3", 1.0),
    "storage(m3)": ("storage_m3", 1.0),
    "storage(m^3)": ("storage_m3", 1.0),
    "storage(mil_m3)": ("storage_m3", 1e6),
    "storage(mcm)": ("storage_m3", 1e6),
    "storage": ("storage_m3", None),
    "inflow_m3s": ("inflow_m3s", 1.0),
    "inflow(cms)": ("inflow_m3s", 1.0),
    "inflow(m3/s)": ("inflow_m3s", 1.0),
    "inflow(m^3/s)": ("inflow_m3s", 1.0),
    "inflow": ("inflow_m3s", None),
    "release_m3s": ("release_m3s", 1.0),
    "outflow_m3s": ("release_m3s", 1.0),
    "outflow(cms)": ("release_m3s", 1.0),
    "outflow(m3/s)": ("release_m3s", 1.0),
    "release(cms)": ("release_m3s", 1.0),
    "release(m3/s)": ("release_m3s", 1.0),
    "release": ("release_m3s", None),
    "outflow": ("release_m3s", None),
}

_STORAGE_UNITS = {
    "m3": 1.0,
    "m^3": 1.0,
    "mcm": 1e6,
    "mil_m3": 1e6,
    "millionm3": 1e6,
    "million_m3": 1e6,
}
_FLOW_UNITS = {"cms": 1.0, "m3s": 1.0, "m3/s": 1.0, "m^3/s": 1.0}


def _unit_factor(unit: str | None, units: dict[str, float], kind: str) -> float | None:
    if unit is None:
        return None
    try:
        return units[_key(unit)]
    except KeyError as exc:
        raise ValueError(f"Unsupported {kind} unit {unit!r}; accepted units: {', '.join(units)}") from exc


def _daily_index(data: pd.DataFrame) -> pd.DatetimeIndex:
    """Read actual dates and reject ambiguous, invalid, or non-daily timestamps."""
    date_columns = [column for column in data.columns if _key(column) == "date"]
    if len(date_columns) > 1:
        raise ValueError("Multiple date columns are ambiguous.")
    parts = {}
    for part in ("year", "month", "day"):
        matches = [column for column in data.columns if _key(column) == part]
        if len(matches) > 1:
            raise ValueError(f"Multiple {part} columns are ambiguous.")
        if matches:
            parts[part] = matches[0]

    candidates = []
    try:
        if date_columns:
            dates = data[date_columns[0]]
            if pd.api.types.is_numeric_dtype(dates.dtype) and len(dates):
                raise ValueError("Numeric date values are ambiguous; supply calendar dates.")
            candidates.append(pd.DatetimeIndex(pd.to_datetime(dates, errors="raise")))
        if len(parts) == 3:
            calendar = pd.DataFrame({part: data[column].to_numpy() for part, column in parts.items()})
            candidates.append(pd.DatetimeIndex(pd.to_datetime(calendar, errors="raise")))
        if isinstance(data.index, pd.DatetimeIndex):
            candidates.append(data.index)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid observation dates: {exc}") from exc

    if not candidates:
        if data.empty:
            return pd.DatetimeIndex([], name="date")
        raise ValueError("Observations require a Date/date column, year/month/day columns, or DatetimeIndex.")

    index = candidates[0]
    if any(not index.equals(other) for other in candidates[1:]):
        raise ValueError("Conflicting date representations in the observations.")
    if index.hasnans:
        raise ValueError("Observation dates contain missing values.")
    if index.tz is not None:
        raise ValueError("Daily observation dates must be timezone-naive calendar dates.")
    if not index.equals(index.normalize()):
        raise ValueError("Daily observation dates must be at midnight; subdaily timestamps are unsupported.")
    if index.has_duplicates:
        raise ValueError("Duplicate daily observation dates are not allowed.")
    return index.rename("date")


def _numeric_values(data: pd.DataFrame, column: object) -> np.ndarray:
    values = data[column]
    # Blank strings are missing observations, never zero-valued observations.
    if pd.api.types.is_object_dtype(values.dtype) or pd.api.types.is_string_dtype(values.dtype):
        blank = values.map(lambda value: isinstance(value, str) and not value.strip())
        values = values.mask(blank)
    numeric = pd.to_numeric(values, errors="coerce")
    if (values.notna() & numeric.isna()).any():
        raise ValueError(f"Observation column {column!r} contains nonnumeric values.")
    result = numeric.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(result).any():
        raise ValueError(f"Observation column {column!r} contains infinite values.")
    if (result < 0).any():
        raise ValueError(f"Observation column {column!r} contains negative values.")
    return result


def normalize_observations(
    data: pd.DataFrame,
    *,
    storage_unit: str | None = None,
    flow_unit: str | None = None,
    source: str | None = None,
) -> ReservoirObservations:
    """Return sorted daily observations in m³ and m³/s with quality information.

    Unit-bearing column names establish their units. Bare Storage/Inflow/Outflow
    columns require explicit units unless the entire input is an all-zero
    placeholder, or contains no measurements. Missing fields and calendar gaps
    remain missing. At least two rows and all three complete, zero-valued fields
    are required to classify a placeholder.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.columns.has_duplicates:
        raise ValueError("Duplicate observation column names are not allowed.")
    storage_factor = _unit_factor(storage_unit, _STORAGE_UNITS, "storage")
    flow_factor = _unit_factor(flow_unit, _FLOW_UNITS, "flow")
    index = _daily_index(data)
    selected = {}
    for column in data.columns:
        alias = _ALIASES.get(_key(column))
        if alias is None:
            continue
        canonical, factor = alias
        if canonical in selected:
            raise ValueError(f"Multiple columns represent {canonical!r}; select one before normalization.")
        selected[canonical] = (column, factor, _numeric_values(data, column))

    placeholder = (
        len(data) >= 2
        and len(selected) == len(OBSERVATION_COLUMNS)
        and all(np.equal(values, 0).all() for _, _, values in selected.values())
    )
    unavailable = not selected or all(np.isnan(values).all() for _, _, values in selected.values())
    issues = []
    normalized = {}
    for canonical in OBSERVATION_COLUMNS:
        if canonical not in selected:
            message = f"Missing observation variable {canonical}; retained as NaN."
            warnings.warn(message, UserWarning, stacklevel=2)
            issues.append(message)
            normalized[canonical] = np.full(len(data), np.nan)
            continue
        column, known_factor, values = selected[canonical]
        explicit_factor = storage_factor if canonical == "storage_m3" else flow_factor
        if known_factor is not None:
            if explicit_factor is not None and explicit_factor != known_factor:
                raise ValueError(f"Explicit unit conflicts with the unit-bearing column {column!r}.")
            factor = known_factor
        elif explicit_factor is not None:
            factor = explicit_factor
        elif placeholder or unavailable:
            factor = 1.0
            issues.append(f"Units unspecified for {column}; only zero/missing values were retained.")
        else:
            argument = "storage_unit" if canonical == "storage_m3" else "flow_unit"
            raise ValueError(f"Column {column!r} has no unit; provide {argument} explicitly.")
        normalized[canonical] = values * factor
        if np.isinf(normalized[canonical]).any():
            raise ValueError(f"Unit conversion overflow in {column!r}.")

    frame = pd.DataFrame(normalized, index=index).sort_index()
    missing_counts = {column: int(frame[column].isna().sum()) for column in OBSERVATION_COLUMNS}
    for column, count in missing_counts.items():
        if count and column in selected:
            issues.append(f"{column} has {count} missing observations.")
    missing_days = 0
    if len(frame):
        missing_days = int((frame.index[-1] - frame.index[0]).days + 1 - len(frame))
    if missing_days:
        issues.append(f"{missing_days} calendar days are missing; gaps were not filled.")
    status = "placeholder" if placeholder else "unavailable" if unavailable else "available"
    if placeholder:
        issues.append("All three observation variables are zero; classified as an unavailable placeholder.")
    elif unavailable:
        issues.append("No nonmissing observations are available.")
    quality = DataQuality(
        status=status,
        row_count=len(frame),
        missing_counts=missing_counts,
        missing_days=missing_days,
        issues=tuple(issues),
    )
    return ReservoirObservations(frame=frame, quality=quality, source=str(source) if source is not None else None)
