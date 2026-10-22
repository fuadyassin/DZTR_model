"""Daily reservoir simulation with explicit numerical compatibility choices."""

from numbers import Integral

import numpy as np
import pandas as pd

from dztr.model.balance import (
    _fixed_point_inputs, _fixed_point_solution,
    _implicit_solution, _implicit_step_inputs,
)
from dztr.model.release import _finite, _monthly_values, _release_for_values, _timestep, _zone
from dztr.schemas import ReservoirObservations, ReservoirParameters, SimulationResult


def simulate_reservoir(
    observations: ReservoirObservations,
    parameters: ReservoirParameters,
    *,
    solver: str = "implicit",
    overflow_mode: str = "conservative",
    max_iterations: int = 1000,
    atol_storage_m3: float = 1e-4,
) -> SimulationResult:
    """Simulate continuous daily inflows using current-day monthly targets.

    The initial row contains configured storage and release, as in MATLAB.
    ``matlab_fixed_point`` preserves the original fixed iteration count and
    final-iterate ordering. ``overflow_mode='matlab'`` preserves its historical
    spill correction; the regular mode conserves the trapezoidal water balance.
    Neither numerical choice is inferred from how parameters were estimated.
    Monthly targets are read once per encountered month and held for this call.
    A new call reads current targets again; no values are cached across candidates.
    """
    if not isinstance(observations, ReservoirObservations) or not isinstance(parameters, ReservoirParameters):
        raise TypeError("Expected normalized ReservoirObservations and ReservoirParameters.")
    if observations.quality.status != "available":
        raise ValueError(f"Cannot simulate {observations.quality.status} observations.")
    if solver not in ("implicit", "matlab_fixed_point"):
        raise ValueError("solver must be 'implicit' or 'matlab_fixed_point'.")
    if overflow_mode not in ("conservative", "matlab"):
        raise ValueError("overflow_mode must be 'conservative' or 'matlab'.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if not np.isfinite(atol_storage_m3) or atol_storage_m3 <= 0:
        raise ValueError("atol_storage_m3 must be positive and finite.")
    frame = observations.frame
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.tz is not None or not index.equals(index.normalize()) or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Simulation requires sorted, unique daily calendar dates.")
    if len(index) < 2 or not np.all((index[1:] - index[:-1]) == pd.Timedelta(days=1)):
        raise ValueError("Simulation requires at least two consecutive daily rows; fill or resolve calendar gaps explicitly.")
    inflow = frame["inflow_m3s"].to_numpy(dtype=float)
    if not np.isfinite(inflow).all() or (inflow < 0).any():
        raise ValueError("Simulation requires finite, nonnegative inflow on every day.")
    count, dt = len(index), 86400.0
    storage = np.empty(count)
    release = np.empty(count)
    overflow = np.zeros(count)
    residual = np.zeros(count)
    rule_residual = np.full(count, np.nan)
    storage[0], release[0] = parameters.initial_storage_m3, parameters.initial_release_m3s
    months = index.month.to_numpy()
    monthly_values = [None] * 12

    def values_for_month(month):
        # Local and lazy: valid edits before a new simulation are visible, and
        # unused months are not read from a post-construction damaged table.
        values = monthly_values[month - 1]
        if values is None:
            values = _monthly_values(month, parameters)
            monthly_values[month - 1] = values
        return values

    for i in range(1, count):
        month = months[i]
        arguments = (storage[i - 1], release[i - 1], inflow[i - 1], inflow[i], parameters)
        try:
            if solver == "implicit":
                predictor, current_inflow, step_dt, tolerance = _implicit_step_inputs(
                    *arguments, dt, atol_storage_m3,
                )
                values = values_for_month(month)
                current_storage, current_release = _implicit_solution(
                    predictor, current_inflow, step_dt, values, tolerance,
                )
            else:
                predictor, current_inflow, step_dt = _fixed_point_inputs(
                    *arguments, dt, max_iterations,
                )
                values = values_for_month(month)
                current_storage, current_release = _fixed_point_solution(
                    predictor, current_inflow, step_dt, values, max_iterations,
                )
        except ValueError as exc:
            raise ValueError(f"Simulation failed on {index[i].date()}: {exc}") from exc
        # Check the coupled release law before intentionally applying spill.
        rule_residual[i] = current_release - float(_release_for_values(
            _finite(current_storage, "storage_m3"),
            _finite(inflow[i], "inflow_m3s", nonnegative=True), values, _timestep(dt),
        ))
        excess = max(current_storage - parameters.capacity_m3, 0.0)
        if excess:
            overflow[i] = excess
            current_storage = parameters.capacity_m3
            current_release += excess / dt * (2.0 if overflow_mode == "conservative" else 1.0)
        storage[i], release[i] = current_storage, current_release
        residual[i] = storage[i] - storage[i - 1] - dt / 2 * (
            parameters.inflow_correction * (inflow[i] + inflow[i - 1]) - release[i] - release[i - 1]
        )
    if not np.isfinite(storage).all() or not np.isfinite(release).all():
        raise ValueError("Simulation produced nonfinite storage or release.")
    issues = []
    if (storage < 0).any():
        issues.append("Legacy iteration produced negative storage; results require review.")
    if overflow_mode == "matlab" and overflow.any():
        issues.append("Legacy overflow correction does not conserve the trapezoidal balance; inspect balance_residual_m3.")
    if np.max(np.abs(residual[1:])) > atol_storage_m3:
        issues.append("Water-balance residual exceeds tolerance; legacy fixed iterations may not have converged.")
    result = pd.DataFrame({
        "storage_m3": storage, "release_m3s": release,
        "inflow_m3s": inflow,
        "effective_inflow_m3s": inflow * parameters.inflow_correction,
        "zone": [_zone(_finite(s, "storage_m3"), values_for_month(month))
                 for s, month in zip(storage, months)],
        "overflow_volume_m3": overflow,
        "balance_residual_m3": residual,
        "release_rule_residual_m3s": rule_residual,
    }, index=index.copy())
    return SimulationResult(result, parameters, solver, overflow_mode, tuple(issues))
