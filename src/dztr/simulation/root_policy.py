"""Explicit experimental selection among valid DZTR trapezoidal balance roots.

The established strict solver remains the default. These conventions do not
change the release relation, add roots to a gap, or prove an operating policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, ulp

import numpy as np
import pandas as pd

from dztr.model.balance import _implicit_solution, _implicit_step_inputs
from dztr.model.release import _monthly_values, _release_for_values, _zone
from dztr.schemas import ReservoirObservations, ReservoirParameters, SimulationResult
from .sampled import SamplingSettings, validate_sampling_index


POLICIES = ("reject", "minimum_storage", "maximum_storage", "nearest_previous_storage")


@dataclass(frozen=True)
class RootPolicyStep:
    storage_m3: float
    release_m3s: float
    root_count: int
    policy_applied: bool
    tie_break_applied: bool
    candidate_storage_m3: tuple[float, ...]
    candidate_release_m3s: tuple[float, ...]


def _policy(value):
    if value not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}.")
    return value


def _candidates(predictor, inflow, dt, values, tolerance):
    # Keep the established solver's arithmetic and candidate validation literal.
    half_dt = dt / 2.0
    normal_slope = (values.qnormal - values.qcritical) / (values.normal - values.critical)
    flood_rise = values.qflood - values.qnormal
    if values.inflow_enabled:
        flood_rise = max(inflow - values.qnormal, flood_rise)
    flood_slope = flood_rise / (values.flood - values.normal)
    branches = (
        (0.0, 0.0), (1.0 / dt, -values.dead / dt), (0.0, values.qcritical),
        (normal_slope, values.qcritical - normal_slope * values.critical),
        (flood_slope, values.qnormal - flood_slope * values.normal),
        (0.0, values.qflood), (1.0 / dt, -values.flood / dt), (0.0, values.channel),
    )
    residual_tolerance = max(tolerance, 64.0 * ulp(max(abs(predictor), 1.0)))
    roots = []
    for slope, intercept in branches:
        storage = (predictor - half_dt * intercept) / (1.0 + half_dt * slope)
        if not isfinite(storage) or storage < 0:
            continue
        release = _release_for_values(storage, inflow, values, dt)
        residual = storage - predictor + half_dt * release
        if abs(residual) > residual_tolerance:
            continue
        if not any(abs(storage - old_s) <= residual_tolerance
                   and half_dt * abs(release - old_q) <= residual_tolerance for old_s, old_q in roots):
            roots.append((float(storage), float(release)))
    return sorted(roots), residual_tolerance


def _resolve(predictor, inflow, dt, values, tolerance, previous_storage, policy):
    try:
        storage, release = _implicit_solution(predictor, inflow, dt, values, tolerance)
        return RootPolicyStep(storage, release, 1, False, False, (storage,), (release,))
    except ValueError as error:
        if policy == "reject" or "Multiple implicit storage/release roots" not in str(error):
            raise
    roots, residual_tolerance = _candidates(predictor, inflow, dt, values, tolerance)
    if len(roots) < 2:
        raise ValueError("Root-policy candidates disagree with the strict solver; results require review.")
    tie = False
    if policy == "minimum_storage":
        selected = roots[0]
    elif policy == "maximum_storage":
        selected = roots[-1]
    else:
        distances = [abs(storage - previous_storage) for storage, _ in roots]
        minimum = min(distances)
        nearest = [root for root, distance in zip(roots, distances)
                   if distance - minimum <= residual_tolerance]
        tie = len(nearest) > 1
        selected = nearest[0]
    return RootPolicyStep(*selected, len(roots), True, tie,
                          tuple(storage for storage, _ in roots), tuple(release for _, release in roots))


def solve_root_policy_step(previous_storage_m3, previous_release_m3s, previous_inflow_m3s,
                           inflow_m3s, month, parameters, dt_seconds=86400.0,
                           atol_storage_m3=1e-4, *, policy="reject"):
    """Select only an accepted root, with a declared deterministic convention.

    ``nearest_previous_storage`` minimizes change from the preceding physical
    state. Distances tied within the solver's volume tolerance choose the lower
    storage. This is a numerical convention, not a recovered rule from the paper.
    Unique roots use the original solver exactly. Negative predictors and gaps
    remain errors for every policy.
    """
    policy = _policy(policy)
    if not isinstance(parameters, ReservoirParameters):
        raise TypeError("Root policies require physically valid ReservoirParameters.")
    predictor, inflow, dt, tolerance = _implicit_step_inputs(
        previous_storage_m3, previous_release_m3s, previous_inflow_m3s,
        inflow_m3s, parameters, dt_seconds, atol_storage_m3)
    return _resolve(predictor, inflow, dt, _monthly_values(month, parameters),
                    tolerance, float(previous_storage_m3), policy)


def simulate_root_policy_reservoir(observations, parameters, *, sampling=None,
                                   policy="reject", atol_storage_m3=1e-4):
    """Simulate a declared experimental policy with conservative spill.

    Native observations and integration/operating clocks are retained. No
    signed-inflow exception or historical invalid table is accepted here.
    Per-step columns record multiplicity, policy use and deterministic ties.
    """
    policy = _policy(policy)
    sampling = SamplingSettings() if sampling is None else sampling
    if not isinstance(sampling, SamplingSettings):
        raise TypeError("sampling must be SamplingSettings.")
    if not isinstance(observations, ReservoirObservations) or observations.quality.status != "available":
        raise ValueError("Simulation requires available normalized observations.")
    if not isinstance(parameters, ReservoirParameters):
        raise TypeError("Root policies require physically valid ReservoirParameters.")
    if sampling.allow_signed_inflow:
        raise ValueError("Root selection does not establish a signed-forcing policy.")
    if isinstance(atol_storage_m3, bool) or not np.isfinite(atol_storage_m3) or atol_storage_m3 <= 0:
        raise ValueError("atol_storage_m3 must be finite and positive.")
    frame, index = observations.frame, observations.frame.index
    validate_sampling_index(index, sampling.frequency)
    inflow = frame.inflow_m3s.to_numpy(dtype=float)
    if not np.isfinite(inflow).all() or (inflow < 0).any():
        raise ValueError("Root policies require finite, nonnegative forcing.")
    dt = (np.r_[np.nan, np.diff(index.asi8) / 1e9] if sampling.fixed_step_days is None
          else np.full(len(index), sampling.fixed_step_days * 86400.0))
    if sampling.frequency == "daily" and sampling.month_clock == "observed":
        dt = np.full(len(index), 86400.0)
    months = index.month.to_numpy()
    if sampling.month_clock == "legacy":
        months = (pd.Timestamp(sampling.legacy_start) + pd.to_timedelta(
            np.arange(len(index)) * sampling.fixed_step_days, unit="D")).month.to_numpy()
    storage, release = np.empty(len(index)), np.empty(len(index))
    overflow, residual = np.zeros(len(index)), np.zeros(len(index))
    rule_residual = np.full(len(index), np.nan)
    selected_storage = np.full(len(index), np.nan)
    selected_release = np.full(len(index), np.nan)
    admissibility_tolerance = np.full(len(index), np.nan)
    root_count = np.zeros(len(index), dtype=int)
    applied, tie = np.zeros(len(index), dtype=bool), np.zeros(len(index), dtype=bool)
    storage[0], release[0] = parameters.initial_storage_m3, parameters.initial_release_m3s
    monthly_values = {}

    def values_for_month(month):
        month = int(month)
        if month not in monthly_values:
            monthly_values[month] = _monthly_values(month, parameters)
        return monthly_values[month]

    for i in range(1, len(index)):
        step = float(dt[i])
        try:
            predictor, current_inflow, step, tolerance = _implicit_step_inputs(
                storage[i - 1], release[i - 1], inflow[i - 1], inflow[i],
                parameters, step, atol_storage_m3)
            values = values_for_month(months[i])
            result = _resolve(predictor, current_inflow, step, values, tolerance, storage[i - 1], policy)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(f"Simulation failed on {index[i].date()}: {error}") from error
        s, q = result.storage_m3, result.release_m3s
        selected_storage[i], selected_release[i] = s, q
        admissibility_tolerance[i] = max(tolerance, 64.0 * ulp(max(abs(predictor), 1.0)))
        rule_residual[i] = q - _release_for_values(s, current_inflow, values, step)
        root_count[i], applied[i], tie[i] = result.root_count, result.policy_applied, result.tie_break_applied
        excess = max(s - parameters.capacity_m3, 0.0)
        if excess:
            overflow[i] = excess
            s = parameters.capacity_m3
            q += excess / step * 2.0
        storage[i], release[i] = s, q
        residual[i] = storage[i] - storage[i - 1] - step / 2 * (
            parameters.inflow_correction * (inflow[i] + inflow[i - 1]) - release[i] - release[i - 1])
    if not np.isfinite(storage).all() or not np.isfinite(release).all():
        raise ValueError("Simulation produced nonfinite states.")
    issues = []
    if np.max(np.abs(residual[1:])) > atol_storage_m3:
        issues.append("Water-balance residual exceeds the configured absolute tolerance.")
    if applied.any():
        issues.append(f"Experimental {policy} convention selected among multiple valid roots.")
    result = pd.DataFrame({"storage_m3": storage, "release_m3s": release, "inflow_m3s": inflow,
        "effective_inflow_m3s": inflow * parameters.inflow_correction,
        "zone": [_zone(float(s), values_for_month(month)) for s, month in zip(storage, months)],
        "overflow_volume_m3": overflow, "balance_residual_m3": residual,
        "release_rule_residual_m3s": rule_residual, "step_seconds": dt, "rule_month": months,
        "root_count": root_count, "root_policy_applied": applied, "root_policy_tie_break": tie,
        "selected_root_storage_m3": selected_storage, "selected_root_release_m3s": selected_release,
        "root_admissibility_tolerance_m3": admissibility_tolerance}, index=index.copy())
    return SimulationResult(result, parameters, f"implicit_root_policy:{policy}", "conservative", tuple(issues))
