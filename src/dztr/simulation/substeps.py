"""Declared conservative substeps, separate from established solver identities.

Linear raw inflow and destination-row target months are explicit conventions.
The internal duration enters the literal DZTR release limits. Refinement is
therefore sensitivity of this scheme, not convergence to the original daily map.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, fsum, isfinite, ulp
from numbers import Integral

import numpy as np
import pandas as pd

from dztr.model.balance import _implicit_step_inputs
from dztr.model.release import _monthly_values, _release_for_values, _zone
from dztr.schemas import ReservoirObservations, ReservoirParameters, SimulationResult
from .root_policy import POLICIES, _resolve
from .sampled import SamplingSettings, validate_sampling_index


@dataclass(frozen=True)
class SubstepSettings:
    substeps_per_interval: int = 1
    root_policy: str = "reject"

    def __post_init__(self):
        if (isinstance(self.substeps_per_interval, bool)
                or not isinstance(self.substeps_per_interval, Integral)
                or self.substeps_per_interval < 1):
            raise ValueError("substeps_per_interval must be a positive integer.")
        if self.root_policy not in POLICIES:
            raise ValueError(f"root_policy must be one of {POLICIES}.")


@dataclass(frozen=True)
class SubstepResult:
    simulation: SimulationResult
    internal: pd.DataFrame | None


class SubstepSimulationError(ValueError):
    """A failed declared step, retaining completed intervals for honest scoring."""
    def __init__(self, detail, partial):
        self.detail = detail
        self.partial = partial
        super().__init__(f"Simulation failed on {detail['date']} at substep "
                         f"{detail['substep']}/{detail['substeps_per_interval']}: {detail['reason']}")


def critical_substep_bounds(parameters, native_step_seconds=86400.0):
    """All-month bounds for critical/normal continuity, not universal feasibility."""
    if not isinstance(parameters, ReservoirParameters):
        raise TypeError("Expected physically valid ReservoirParameters.")
    if (isinstance(native_step_seconds, bool) or not isfinite(native_step_seconds)
            or native_step_seconds <= 0):
        raise ValueError("native_step_seconds must be finite and positive.")
    rows = []
    for month in range(1, 13):
        values = _monthly_values(month, parameters)
        volume = values.critical - values.dead
        limit = volume / values.qcritical if values.qcritical > 0 else float("inf")
        required = max(1, ceil(native_step_seconds / limit))
        rows.append({"month": month, "available_volume_m3": volume,
                     "critical_release_m3s": values.qcritical,
                     "max_step_seconds": limit, "minimum_substeps": required})
    return pd.DataFrame(rows).set_index("month")


def _add(total, correction, value):
    increment = value - correction
    updated = total + increment
    return updated, (updated - total) - increment


def simulate_substep_reservoir(observations, parameters, *, sampling=None,
                              settings=None, atol_storage_m3=1e-4,
                              record_internal=False):
    """Integrate equal substeps with carried release and conservative overflow.

    No automatic refinement, root fallback, observation-state reset or missing
    sample filling occurs. Native ``release_m3s`` is an endpoint rate. The
    separate interval mean comes from summed trapezoidal release volumes.
    ``overflow_volume_m3`` sums clipping corrections, while integrated spill
    includes carried spill increments in both endpoints of each trapezoid.
    Internal tables are optional to keep fleet memory proportional to native
    rows. On failure, partial native rows contain only completed intervals.
    """
    settings = SubstepSettings() if settings is None else settings
    sampling = SamplingSettings() if sampling is None else sampling
    if not isinstance(settings, SubstepSettings) or not isinstance(sampling, SamplingSettings):
        raise TypeError("Expected SubstepSettings and SamplingSettings.")
    if not isinstance(record_internal, bool):
        raise TypeError("record_internal must be a boolean.")
    if not isinstance(parameters, ReservoirParameters):
        raise TypeError("Substeps require physically valid ReservoirParameters.")
    if not isinstance(observations, ReservoirObservations) or observations.quality.status != "available":
        raise ValueError("Substeps require available normalized observations.")
    if sampling.allow_signed_inflow:
        raise ValueError("Substeps do not establish a signed-forcing policy.")
    if (isinstance(atol_storage_m3, bool) or not isfinite(atol_storage_m3)
            or atol_storage_m3 <= 0):
        raise ValueError("atol_storage_m3 must be finite and positive.")
    frame, index = observations.frame, observations.frame.index
    validate_sampling_index(index, sampling.frequency)
    inflow = frame.inflow_m3s.to_numpy(dtype=float)
    if not isfinite(inflow[0]) or inflow[0] < 0:
        raise ValueError("Initial inflow must be finite and nonnegative.")
    durations = (np.r_[np.nan, np.diff(index.asi8) / 1e9] if sampling.fixed_step_days is None
                 else np.full(len(index), sampling.fixed_step_days * 86400.0))
    if sampling.frequency == "daily" and sampling.month_clock == "observed":
        durations = np.full(len(index), 86400.0)
    months = index.month.to_numpy()
    if sampling.month_clock == "legacy":
        months = (pd.Timestamp(sampling.legacy_start) + pd.to_timedelta(
            np.arange(len(index)) * sampling.fixed_step_days, unit="D")).month.to_numpy()
    count = int(settings.substeps_per_interval)
    monthly = {}

    def values_for_month(month):
        month = int(month)
        if month not in monthly:
            monthly[month] = _monthly_values(month, parameters)
        return monthly[month]

    s, q, previous_spill = parameters.initial_storage_m3, parameters.initial_release_m3s, 0.0
    initial = {"storage_m3": s, "release_m3s": q, "inflow_m3s": inflow[0],
        "effective_inflow_m3s": inflow[0] * parameters.inflow_correction,
        "zone": _zone(s, values_for_month(months[0])), "overflow_volume_m3": 0.0,
        "balance_residual_m3": 0.0, "release_rule_residual_m3s": np.nan,
        "root_count": 0, "root_policy_applied": False, "root_policy_tie_break": False,
        "selected_root_storage_m3": np.nan, "selected_root_release_m3s": np.nan,
        "root_admissibility_tolerance_m3": np.nan, "substep_count": 0,
        "substep_seconds": np.nan, "integrated_inflow_volume_m3": np.nan,
        "integrated_release_volume_m3": np.nan, "interval_mean_release_m3s": np.nan,
        "integrated_spill_volume_m3": np.nan, "max_substep_balance_residual_m3": np.nan,
        "max_substep_release_rule_residual_m3s": np.nan, "multiple_root_substeps": 0,
        "root_policy_tie_break_substeps": 0, "overflow_substeps": 0,
        "native_endpoint_balance_residual_m3": 0.0, "cumulative_inflow_volume_m3": 0.0,
        "cumulative_release_volume_m3": 0.0, "cumulative_spill_volume_m3": 0.0,
        "cumulative_balance_residual_m3": 0.0, "cumulative_roundoff_bound_m3": 0.0}
    rows, internal_rows, internal_index = [initial], [], []
    totals, corrections = np.zeros(4), np.zeros(4)
    worst_balance, worst_rule, ambiguous = 0.0, 0.0, 0

    def finish():
        result = pd.DataFrame(rows, index=index[:len(rows)].copy())
        # Match the established root-count array's platform integer dtype as
        # well as its values, including Windows' NumPy integer convention.
        result["root_count"] = np.asarray(result["root_count"], dtype=int)
        result["step_seconds"] = durations[:len(rows)]
        result["rule_month"] = months[:len(rows)]
        issues = ["Experimental linear-inflow substeps use internal duration in DZTR release limits."]
        if worst_balance > atol_storage_m3:
            issues.append("A substep water-balance residual exceeds the configured tolerance.")
        if ambiguous:
            issues.append(f"Experimental {settings.root_policy} convention selected among multiple roots.")
        internal = None
        if record_internal:
            internal = pd.DataFrame(internal_rows, index=pd.MultiIndex.from_tuples(
                internal_index, names=[index.name or "date", "substep"]))
        return SubstepResult(SimulationResult(result, parameters,
            f"implicit_substeps:{count}:{settings.root_policy}", "conservative", tuple(issues)), internal)

    for i in range(1, len(index)):
        native_dt, month = float(durations[i]), int(months[i])
        dt = native_dt / count
        values = values_for_month(month)
        start_s, start_q = s, q
        start_i, end_i = float(inflow[i - 1]), float(inflow[i])
        volumes_in, volumes_out, volumes_spill, abs_residuals = [], [], [], []
        clipping, rule_max, balance_max, tolerance_max = 0.0, 0.0, 0.0, 0.0
        multiple, ties, spills = 0, 0, 0
        prev_i = start_i
        for j in range(1, count + 1):
            # An invalid source endpoint invalidates its entire interval; never
            # integrate apparently valid interpolated parts of signed forcing.
            cur_i = (end_i if j == count or not isfinite(end_i) or end_i < 0
                     else start_i + (end_i - start_i) * (j / count))
            predictor = s + dt / 2.0 * (parameters.inflow_correction * (cur_i + prev_i) - q)
            try:
                predictor, current_i, step, tolerance = _implicit_step_inputs(
                    s, q, prev_i, cur_i, parameters, dt, atol_storage_m3)
                chosen = _resolve(predictor, current_i, step, values, tolerance, s, settings.root_policy)
            except (ValueError, ZeroDivisionError) as error:
                reason = str(error)
                code = ("implicit_no_root" if "No feasible implicit" in reason else
                        "implicit_multiple_roots" if "Multiple implicit" in reason else
                        "negative_predictor" if "negative storage" in reason else
                        "invalid_forcing" if not isfinite(end_i) or end_i < 0 else "invalid_step")
                detail = {"date": index[i].date().isoformat(), "interval_row": i,
                    "substep": j, "substeps_per_interval": count, "step_seconds": dt,
                    "rule_month": month, "failure_code": code, "reason": reason,
                    "previous_storage_m3": s, "previous_release_m3s": q,
                    "previous_inflow_m3s": prev_i, "inflow_m3s": cur_i,
                    "source_interval_inflow_m3s": end_i, "predictor_m3": predictor,
                    "completed_substeps_in_failed_interval": j - 1,
                    "completed_interval_inflow_volume_m3": fsum(volumes_in),
                    "completed_interval_release_volume_m3": fsum(volumes_out)}
                raise SubstepSimulationError(detail, finish()) from error
            raw_s, raw_q = chosen.storage_m3, chosen.release_m3s
            rule_residual = raw_q - _release_for_values(raw_s, cur_i, values, dt)
            admissibility = max(tolerance, 64.0 * ulp(max(abs(predictor), 1.0)))
            excess = max(raw_s - parameters.capacity_m3, 0.0)
            new_s, new_q = raw_s, raw_q
            spill_increment = 0.0
            if excess:
                new_s = parameters.capacity_m3
                spill_increment = excess / dt * 2.0
                new_q += spill_increment
            volume_in = dt / 2.0 * parameters.inflow_correction * (prev_i + cur_i)
            volume_out = dt / 2.0 * (q + new_q)
            volume_spill = dt / 2.0 * (previous_spill + spill_increment)
            residual = new_s - s - dt / 2.0 * (
                parameters.inflow_correction * (cur_i + prev_i) - new_q - q)
            if not all(isfinite(x) for x in (new_s, new_q, volume_in, volume_out, residual)):
                raise ValueError("Substep simulation produced nonfinite states or fluxes.")
            if record_internal:
                internal_rows.append({"previous_storage_m3": s, "previous_release_m3s": q,
                    "previous_inflow_m3s": prev_i, "storage_m3": new_s, "release_m3s": new_q,
                    "inflow_m3s": cur_i, "step_seconds": dt, "rule_month": month,
                    "predictor_m3": predictor, "selected_root_storage_m3": raw_s,
                    "selected_root_release_m3s": raw_q, "root_count": chosen.root_count,
                    "root_policy_applied": chosen.policy_applied,
                    "root_policy_tie_break": chosen.tie_break_applied,
                    "root_admissibility_tolerance_m3": admissibility,
                    "overflow_volume_m3": excess, "spill_increment_m3s": spill_increment,
                    "integrated_inflow_volume_m3": volume_in,
                    "integrated_release_volume_m3": volume_out,
                    "integrated_spill_volume_m3": volume_spill,
                    "balance_residual_m3": residual, "release_rule_residual_m3s": rule_residual})
                internal_index.append((index[i], j))
            volumes_in.append(volume_in)
            volumes_out.append(volume_out)
            volumes_spill.append(volume_spill)
            abs_residuals.append(abs(residual))
            clipping += excess
            multiple += int(chosen.policy_applied)
            ties += int(chosen.tie_break_applied)
            spills += int(excess > 0)
            balance_max = max(balance_max, abs(residual))
            rule_max = max(rule_max, abs(rule_residual))
            tolerance_max = max(tolerance_max, admissibility)
            worst_balance, worst_rule = max(worst_balance, balance_max), max(worst_rule, rule_max)
            ambiguous += int(chosen.policy_applied)
            s, q, prev_i, previous_spill = new_s, new_q, cur_i, spill_increment
        total_in, total_out, total_spill = fsum(volumes_in), fsum(volumes_out), fsum(volumes_spill)
        endpoint_residual = s - start_s - native_dt / 2.0 * (
            parameters.inflow_correction * (end_i + start_i) - q - start_q)
        residual = endpoint_residual if count == 1 else s - start_s - (total_in - total_out)
        for k, value in enumerate((total_in, total_out, total_spill, fsum(abs_residuals))):
            totals[k], corrections[k] = _add(totals[k], corrections[k], value)
        cumulative_residual = s - parameters.initial_storage_m3 - (totals[0] - totals[1])
        cumulative_bound = totals[3] + 64.0 * ulp(max(abs(totals[0]), abs(totals[1]), abs(s), 1.0))
        rows.append({"storage_m3": s, "release_m3s": q, "inflow_m3s": end_i,
            "effective_inflow_m3s": end_i * parameters.inflow_correction,
            "zone": _zone(s, values), "overflow_volume_m3": clipping,
            "balance_residual_m3": residual, "release_rule_residual_m3s": rule_residual,
            "root_count": chosen.root_count, "root_policy_applied": multiple > 0,
            "root_policy_tie_break": ties > 0, "selected_root_storage_m3": raw_s,
            "selected_root_release_m3s": raw_q, "root_admissibility_tolerance_m3": tolerance_max,
            "substep_count": count, "substep_seconds": dt,
            "integrated_inflow_volume_m3": total_in, "integrated_release_volume_m3": total_out,
            "interval_mean_release_m3s": total_out / native_dt,
            "integrated_spill_volume_m3": total_spill,
            "max_substep_balance_residual_m3": balance_max,
            "max_substep_release_rule_residual_m3s": rule_max,
            "multiple_root_substeps": multiple, "root_policy_tie_break_substeps": ties,
            "overflow_substeps": spills, "native_endpoint_balance_residual_m3": endpoint_residual,
            "cumulative_inflow_volume_m3": totals[0], "cumulative_release_volume_m3": totals[1],
            "cumulative_spill_volume_m3": totals[2], "cumulative_balance_residual_m3": cumulative_residual,
            "cumulative_roundoff_bound_m3": cumulative_bound})
    return finish()
