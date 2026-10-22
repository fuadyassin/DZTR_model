"""Implicit trapezoidal balance and the historical MATLAB iteration scheme."""

from __future__ import annotations

from math import isfinite, ulp
from typing import TYPE_CHECKING

from .release import _finite, _monthly_values, _release_for_values, _timestep

if TYPE_CHECKING:
    from dztr.schemas import ReservoirParameters


def _step_inputs(
    previous_storage_m3: float,
    previous_release_m3s: float,
    previous_inflow_m3s: float,
    inflow_m3s: float,
    parameters: ReservoirParameters,
    dt_seconds: float,
) -> tuple[float, float, float]:
    storage = _finite(previous_storage_m3, "previous_storage_m3")
    release = _finite(previous_release_m3s, "previous_release_m3s", nonnegative=True)
    previous_inflow = _finite(previous_inflow_m3s, "previous_inflow_m3s", nonnegative=True)
    inflow = _finite(inflow_m3s, "inflow_m3s", nonnegative=True)
    dt = _timestep(dt_seconds)
    half_dt = dt / 2.0
    predictor = storage + half_dt * (
        parameters.inflow_correction * (inflow + previous_inflow) - release
    )
    if not isfinite(predictor):
        raise ValueError("The storage balance predictor is not finite.")
    return predictor, inflow, dt


def solve_implicit_step(
    previous_storage_m3: float,
    previous_release_m3s: float,
    previous_inflow_m3s: float,
    inflow_m3s: float,
    month: int,
    parameters: ReservoirParameters,
    dt_seconds: float = 86400.0,
    atol_storage_m3: float = 1e-4,
) -> tuple[float, float]:
    """Solve storage and release simultaneously before overflow handling.

    Every possible linear branch Q=a*S+b supplies one algebraic candidate.
    Candidates are checked against the actual operating rule, including exact
    zone boundaries. This avoids assuming continuity when the inflow option or
    a low channel capacity introduces a downward jump in release.

    Raises ValueError for negative storage, no feasible root, or multiple roots.
    Returned storage may exceed capacity; the simulation layer handles spill.
    """
    tolerance = _finite(atol_storage_m3, "atol_storage_m3", nonnegative=True)
    predictor, inflow, dt = _step_inputs(
        previous_storage_m3, previous_release_m3s, previous_inflow_m3s,
        inflow_m3s, parameters, dt_seconds,
    )
    values = _monthly_values(month, parameters)
    if predictor < 0:
        raise ValueError(
            "The implicit balance requires negative storage; decrease the time "
            "step or revise the previous release/initial state."
        )
    half_dt = dt / 2.0
    normal_slope = (values.qnormal - values.qcritical) / (values.normal - values.critical)
    flood_rise = values.qflood - values.qnormal
    if values.inflow_enabled:
        flood_rise = max(inflow - values.qnormal, flood_rise)
    flood_slope = flood_rise / (values.flood - values.normal)
    branches = (
        (0.0, 0.0),
        (1.0 / dt, -values.dead / dt),
        (0.0, values.qcritical),
        (normal_slope, values.qcritical - normal_slope * values.critical),
        (flood_slope, values.qnormal - flood_slope * values.normal),
        (0.0, values.qflood),
        (1.0 / dt, -values.flood / dt),
        (0.0, values.channel),
    )
    # Allow floating-point roundoff at large reservoir volumes without making
    # the requested absolute tolerance scale with the reservoir size.
    residual_tolerance = max(tolerance, 64.0 * ulp(max(abs(predictor), 1.0)))
    roots: list[tuple[float, float]] = []
    for slope, intercept in branches:
        storage = (predictor - half_dt * intercept) / (1.0 + half_dt * slope)
        if not isfinite(storage) or storage < 0:
            continue
        release = _release_for_values(storage, inflow, values, dt)
        residual = storage - predictor + half_dt * release
        if abs(residual) > residual_tolerance:
            continue
        if not any(
            abs(storage - old_storage) <= residual_tolerance
            and half_dt * abs(release - old_release) <= residual_tolerance
            for old_storage, old_release in roots
        ):
            roots.append((float(storage), float(release)))
    if not roots:
        raise ValueError(
            "No feasible implicit storage/release root exists. A discontinuity "
            "at a DZTR zone boundary can leave a gap; review targets, channel "
            "capacity, flood mode, and time step."
        )
    if len(roots) != 1:
        raise ValueError(
            f"Multiple implicit storage/release roots ({len(roots)}) exist. "
            "Review inflow-dependent flood release and channel capacity before "
            "choosing an operating policy."
        )
    return roots[0]


def fixed_point_step(
    previous_storage_m3: float,
    previous_release_m3s: float,
    previous_inflow_m3s: float,
    inflow_m3s: float,
    month: int,
    parameters: ReservoirParameters,
    dt_seconds: float = 86400.0,
    max_iterations: int = 1000,
) -> tuple[float, float]:
    """Reproduce SimRISM14.m's fixed iteration count, without a capacity clamp.

    There is deliberately no early convergence exit. As in MATLAB, the final
    storage uses the preceding release iterate, while the reported final
    release is evaluated from that storage. Negative raw storage is retained.
    """
    if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    predictor, inflow, dt = _step_inputs(
        previous_storage_m3, previous_release_m3s, previous_inflow_m3s,
        inflow_m3s, parameters, dt_seconds,
    )
    values = _monthly_values(month, parameters)
    release = _release_for_values(predictor, inflow, values, dt)
    for _ in range(int(max_iterations)):
        storage = predictor - dt / 2.0 * release
        release = _release_for_values(storage, inflow, values, dt)
    return float(storage), float(release)
