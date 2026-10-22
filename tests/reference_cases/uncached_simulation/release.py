"""DZTR's monthly piecewise storage-to-release relation, in SI units."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from dztr.schemas import ReservoirParameters


class _MonthlyValues(NamedTuple):
    dead: float
    critical: float
    normal: float
    flood: float
    qcritical: float
    qnormal: float
    qflood: float
    channel: float
    inflow_enabled: bool


def _finite(value: float, name: str, *, nonnegative: bool = False) -> float:
    value = float(value)
    if not isfinite(value) or (nonnegative and value < 0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return value


def _timestep(dt_seconds: float) -> float:
    dt = _finite(dt_seconds, "dt_seconds")
    if dt <= 0:
        raise ValueError("dt_seconds must be positive.")
    return dt


def _monthly_values(month: int, parameters: ReservoirParameters) -> _MonthlyValues:
    if isinstance(month, bool) or int(month) != month or not 1 <= month <= 12:
        raise ValueError("month must be an integer from 1 to 12.")
    row = parameters.monthly_targets.loc[int(month)]
    return _MonthlyValues(
        parameters.dead_storage_fraction * parameters.capacity_m3,
        float(row["storage_critical_m3"]),
        float(row["storage_normal_m3"]),
        float(row["storage_flood_m3"]),
        float(row["release_critical_m3s"]),
        float(row["release_normal_m3s"]),
        float(row["release_flood_m3s"]),
        parameters.channel_capacity_m3s,
        parameters.flood_mode == "inflow",
    )


def _zone(storage: float, values: _MonthlyValues) -> int:
    for zone, limit in enumerate(values[:4]):
        if storage <= limit:
            return zone
    return 4


def _release_for_values(storage: float, inflow: float, values: _MonthlyValues, dt: float) -> float:
    """Literal MATLAB DZTR branches; input validation is done by callers."""
    if storage <= values.dead:
        return 0.0
    if storage <= values.critical:
        return min(values.qcritical, (storage - values.dead) / dt)
    if storage <= values.normal:
        return values.qcritical + (values.qnormal - values.qcritical) * (
            (storage - values.critical) / (values.normal - values.critical)
        )
    if storage <= values.flood:
        rise = values.qflood - values.qnormal
        if values.inflow_enabled:
            rise = max(inflow - values.qnormal, rise)
        return values.qnormal + rise * (
            (storage - values.normal) / (values.flood - values.normal)
        )
    return min(max((storage - values.flood) / dt, values.qflood), values.channel)


def dztr_zone(storage_m3: float, month: int, parameters: ReservoirParameters) -> int:
    """Return 0 (dead) through 4 (emergency), using MATLAB's exact boundaries."""
    storage = _finite(storage_m3, "storage_m3")
    return _zone(storage, _monthly_values(month, parameters))


def dztr_release(
    storage_m3: float,
    inflow_m3s: float,
    month: int,
    parameters: ReservoirParameters,
    dt_seconds: float = 86400.0,
) -> float:
    """Compute the MATLAB DZTR release law from six monthly SI targets.

    This is the operating rule before any simulation-level overflow treatment.
    Storage above capacity is accepted; negative storage yields zero release.
    The inflow option uses the uncorrected current inflow, as in MATLAB.
    Channel capacity applies only in the emergency zone.
    """
    storage = _finite(storage_m3, "storage_m3")
    inflow = _finite(inflow_m3s, "inflow_m3s", nonnegative=True)
    values = _monthly_values(month, parameters)
    return float(_release_for_values(storage, inflow, values, _timestep(dt_seconds)))
