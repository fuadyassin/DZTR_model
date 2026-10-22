"""Explicit daily/monthly clocks and auditable original MATLAB replay inputs."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

from dztr.model.balance import _fixed_point_solution, _implicit_solution
from dztr.model.release import _monthly_values, _release_for_values, _zone
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirObservations, ReservoirParameters, SimulationResult
from .runner import simulate_reservoir


@dataclass(frozen=True)
class SamplingSettings:
    """Sampling is independent of the clock used to select operating targets.

    ``fixed_step_days=None`` integrates actual elapsed calendar time. A fixed
    30.5-day step retains the monthly MATLAB time interval. ``legacy`` target
    months come from a synthetic fixed-step clock, rather than observation dates.
    """

    frequency: str = "daily"
    fixed_step_days: float | None = None
    month_clock: str = "observed"
    legacy_start: str | None = None
    allow_signed_inflow: bool = False

    def __post_init__(self):
        if self.frequency not in ("daily", "monthly"):
            raise ValueError("frequency must be daily or monthly.")
        if self.month_clock not in ("observed", "legacy"):
            raise ValueError("month_clock must be observed or legacy.")
        if not isinstance(self.allow_signed_inflow, bool):
            raise TypeError("allow_signed_inflow must be a boolean.")
        if self.fixed_step_days is not None:
            if (isinstance(self.fixed_step_days, bool) or not isinstance(self.fixed_step_days, Real)
                    or not np.isfinite(self.fixed_step_days) or self.fixed_step_days <= 0):
                raise ValueError("fixed_step_days must be finite and positive.")
            if self.frequency == "daily" and self.fixed_step_days != 1:
                raise ValueError("Daily observations require one-day integration steps.")
        if self.month_clock == "legacy":
            if self.fixed_step_days is None or not isinstance(self.legacy_start, str):
                raise ValueError("Legacy month selection requires fixed_step_days and an explicit legacy_start.")
            value = pd.Timestamp(self.legacy_start)
            if pd.isna(value) or value.tz is not None or value != value.normalize():
                raise ValueError("legacy_start must be a timezone-naive calendar date.")


@dataclass(frozen=True)
class LegacyReservoirParameters:
    """Unmodified historical tables, without claiming modern physical validity.

    Original folders include critical storage below dead storage and targets
    above stated capacity. These inputs are retained only for explicit fixed-
    iteration MATLAB replay; modern estimation uses ReservoirParameters.
    """

    monthly_targets: pd.DataFrame
    capacity_m3: float
    initial_storage_m3: float
    initial_release_m3s: float
    channel_capacity_m3s: float
    dead_storage_fraction: float
    inflow_correction: float
    flood_mode: str

    def __post_init__(self):
        for name in ("capacity_m3", "initial_storage_m3", "initial_release_m3s",
                     "channel_capacity_m3s", "dead_storage_fraction", "inflow_correction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real number.")
        if self.capacity_m3 <= 0 or self.initial_storage_m3 < 0:
            raise ValueError("Legacy capacity must be positive and initial storage nonnegative.")
        if min(self.initial_release_m3s, self.channel_capacity_m3s, self.inflow_correction) < 0:
            raise ValueError("Legacy releases, channel capacity and inflow correction must be nonnegative.")
        if not 0 <= self.dead_storage_fraction < 1 or self.flood_mode not in ("storage", "inflow"):
            raise ValueError("Invalid legacy dead-storage fraction or flood mode.")
        frame = self.monthly_targets
        if (not isinstance(frame, pd.DataFrame) or not frame.index.is_unique
                or set(frame.index) != set(range(1, 13)) or frame.columns.has_duplicates
                or not set(MONTHLY_TARGET_COLUMNS).issubset(frame.columns)):
            raise ValueError("Legacy monthly table requires all six columns and months 1 through 12.")
        frame = frame.loc[list(range(1, 13)), list(MONTHLY_TARGET_COLUMNS)].astype(float).copy()
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError("Legacy targets must be finite.")
        if (frame.iloc[:, 3:].to_numpy() < 0).any():
            raise ValueError("Legacy release targets must be nonnegative.")
        object.__setattr__(self, "monthly_targets", frame)


def validate_sampling_index(index, frequency):
    """Reject missing samples; never manufacture daily records from months."""
    if (not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.tz is not None
            or not index.is_unique or not index.is_monotonic_increasing
            or not index.equals(index.normalize()) or len(index) < 2):
        raise ValueError("Sampling requires at least two sorted unique calendar dates at midnight.")
    if frequency == "daily":
        valid = np.all(np.diff(index.asi8) == pd.Timedelta(days=1).value)
    elif frequency == "monthly":
        valid = bool((index.day == 1).all() and np.all(np.diff(index.to_period("M").asi8) == 1))
    else:
        raise ValueError("frequency must be daily or monthly.")
    if not valid:
        raise ValueError(f"Observations require consecutive {frequency} samples; missing intervals are not filled.")


def simulate_sampled_reservoir(observations, parameters, *, sampling=None, solver="implicit",
                               overflow_mode="conservative", max_iterations=1000,
                               atol_storage_m3=1e-4):
    """Reuse the DZTR equations with explicit sampling and historical clocks."""
    sampling = SamplingSettings() if sampling is None else sampling
    if not isinstance(sampling, SamplingSettings):
        raise TypeError("sampling must be SamplingSettings.")
    if not isinstance(observations, ReservoirObservations) or observations.quality.status != "available":
        raise ValueError("Simulation requires available normalized observations.")
    if not isinstance(parameters, (ReservoirParameters, LegacyReservoirParameters)):
        raise TypeError("Expected regular or explicit legacy reservoir parameters.")
    if solver not in ("implicit", "matlab_fixed_point") or overflow_mode not in ("conservative", "matlab", "none"):
        raise ValueError("Invalid solver or overflow_mode.")
    if overflow_mode == "none" and not isinstance(parameters, LegacyReservoirParameters):
        raise ValueError("Omitting capacity overflow is restricted to explicit legacy replay.")
    if isinstance(parameters, LegacyReservoirParameters) and (solver != "matlab_fixed_point" or overflow_mode not in ("matlab", "none")):
        raise ValueError("Legacy tables require explicit matlab_fixed_point solver and historical overflow policy.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if isinstance(atol_storage_m3, bool) or not np.isfinite(atol_storage_m3) or atol_storage_m3 <= 0:
        raise ValueError("atol_storage_m3 must be finite and positive.")
    validate_sampling_index(observations.frame.index, sampling.frequency)
    if (sampling.frequency == "daily" and sampling.month_clock == "observed"
            and not sampling.allow_signed_inflow and isinstance(parameters, ReservoirParameters)):
        return simulate_reservoir(observations, parameters, solver=solver, overflow_mode=overflow_mode,
                                  max_iterations=max_iterations, atol_storage_m3=atol_storage_m3)
    index = observations.frame.index
    inflow = observations.frame.inflow_m3s.to_numpy(dtype=float)
    if not np.isfinite(inflow).all() or (not sampling.allow_signed_inflow and (inflow < 0).any()):
        raise ValueError("Forcing requires finite inflow; signed values need an explicit reviewed forcing policy.")
    if sampling.fixed_step_days is None:
        dt = np.r_[np.nan, np.diff(index.asi8) / 1e9]
    else:
        dt = np.full(len(index), sampling.fixed_step_days * 86400.0)
    months = index.month.to_numpy()
    if sampling.month_clock == "legacy":
        clock = pd.Timestamp(sampling.legacy_start) + pd.to_timedelta(
            np.arange(len(index)) * sampling.fixed_step_days, unit="D")
        months = clock.month.to_numpy()
    storage, release = np.empty(len(index)), np.empty(len(index))
    overflow, balance = np.zeros(len(index)), np.zeros(len(index))
    rule = np.full(len(index), np.nan)
    storage[0], release[0] = parameters.initial_storage_m3, parameters.initial_release_m3s
    values = {int(month): _monthly_values(int(month), parameters) for month in np.unique(months)}
    for i in range(1, len(index)):
        step = float(dt[i])
        predictor = storage[i - 1] + step / 2.0 * (
            parameters.inflow_correction * (inflow[i] + inflow[i - 1]) - release[i - 1])
        if not np.isfinite(predictor):
            raise ValueError(f"Nonfinite balance predictor at {index[i]}.")
        current = values[int(months[i])]
        try:
            if solver == "implicit":
                s, q = _implicit_solution(predictor, inflow[i], step, current, atol_storage_m3)
            else:
                s, q = _fixed_point_solution(predictor, inflow[i], step, current, max_iterations)
            rule[i] = q - _release_for_values(s, inflow[i], current, step)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(f"Simulation failed at {index[i]}: {error}") from error
        excess = max(s - parameters.capacity_m3, 0.0)
        overflow[i] = excess if overflow_mode != "none" else 0.0
        if excess and overflow_mode != "none":
            s = parameters.capacity_m3
            q += excess / step * (2.0 if overflow_mode == "conservative" else 1.0)
        storage[i], release[i] = s, q
        balance[i] = storage[i] - storage[i - 1] - step / 2.0 * (
            parameters.inflow_correction * (inflow[i] + inflow[i - 1]) - release[i] - release[i - 1])
    if not np.isfinite(storage).all() or not np.isfinite(release).all():
        raise ValueError("Simulation produced nonfinite states.")
    issues = []
    if (storage < 0).any():
        issues.append("Legacy iteration produced negative storage; results require review.")
    if overflow_mode == "matlab" and overflow.any():
        issues.append("Legacy overflow correction leaves a trapezoidal balance discrepancy.")
    if overflow_mode == "none" and (storage > parameters.capacity_m3).any():
        issues.append("The source omits overflow adjustment; storage above stated capacity was retained.")
    if np.max(np.abs(balance[1:])) > atol_storage_m3:
        issues.append("Water-balance residual exceeds the configured tolerance.")
    if sampling.allow_signed_inflow and (inflow < 0).any():
        issues.append("Signed source inflows were preserved under an explicit forcing policy.")
    result = pd.DataFrame({"storage_m3": storage, "release_m3s": release, "inflow_m3s": inflow,
        "effective_inflow_m3s": inflow * parameters.inflow_correction,
        "zone": [_zone(float(s), values[int(month)]) for s, month in zip(storage, months)],
        "overflow_volume_m3": overflow, "balance_residual_m3": balance,
        "release_rule_residual_m3s": rule, "step_seconds": dt, "rule_month": months}, index=index.copy())
    return SimulationResult(result, parameters, solver, overflow_mode, tuple(issues))
