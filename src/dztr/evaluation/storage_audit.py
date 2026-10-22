"""Observed storage/flow consistency under declared measurement hypotheses.

This is an observation diagnostic, not a simulator or an estimator of losses.
Pass a single reporting period: shifting and integration never use other rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from dztr.simulation.sampled import validate_sampling_index


CONVENTIONS = ("endpoint_trapezoid", "interval_mean_end", "interval_mean_start")
LAGS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
INTERPRETATION = {
    "residual": "storage change - (inflow volume - observed release volume)",
    "positive": "Observed storage gain exceeds recorded net input under this hypothesis.",
    "negative": "Observed storage gain is less than recorded net input under this hypothesis.",
    "storage": "Arithmetic treats storage as endpoint states; measurement timing/datum remain unverified.",
    "monthly": "Monthly mean storage is not an endpoint state; these hypotheses do not establish monthly closure.",
    "lags": "+1 uses the preceding sample; -1 uses the following sample. Each flow is shifted separately.",
    "lag_support": "All five timing hypotheses use their common finite support within this period.",
    "causality": "A lead uses later observed data and is not a forecasting method.",
    "gaps": "Invalid intervals break cumulative segments; no whole-period closure is claimed across a gap.",
    "selection": "No preferred measurement convention, clock, correction, or lag is selected.",
    "causes": "Residuals do not identify evaporation, seepage, diversions, datum errors, or timing errors.",
}


@dataclass(frozen=True)
class StorageAuditResult:
    intervals: pd.DataFrame
    summary: dict
    monthly: pd.DataFrame
    extremes: pd.DataFrame
    flags: pd.DataFrame


def _finite_number(value, name, *, positive=False, nonnegative=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
            or (positive and value <= 0) or (nonnegative and value < 0)):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else 'nonnegative ' if nonnegative else ''}real number.")
    return float(value)


def _ledger(storage, inflow, release, durations, convention, inflow_lag=0, release_lag=0):
    incoming, outgoing = inflow.shift(inflow_lag), release.shift(release_lag)
    if convention == "endpoint_trapezoid":
        incoming = (incoming.shift(1) + incoming) / 2
        outgoing = (outgoing.shift(1) + outgoing) / 2
    elif convention == "interval_mean_start":
        incoming, outgoing = incoming.shift(1), outgoing.shift(1)
    incoming, outgoing = incoming.iloc[1:], outgoing.iloc[1:]
    delta = storage.diff().iloc[1:]
    valid = (np.isfinite(delta) & np.isfinite(incoming) & np.isfinite(outgoing))
    vin, vout = durations * incoming, durations * outgoing
    residual = delta - (vin - vout)
    valid &= np.isfinite(vin) & np.isfinite(vout) & np.isfinite(residual)
    frame = pd.DataFrame({"start_date": storage.index[:-1], "step_seconds": durations,
        "storage_change_m3": delta, "inflow_volume_m3": vin, "release_volume_m3": vout,
        "residual_m3": residual.where(valid), "residual_rate_m3s": (residual / durations).where(valid),
        "valid": valid.astype(bool), "negative_flow": (incoming < 0) | (outgoing < 0)}, index=delta.index)
    frame.index.name = "date"
    starts = valid & ~valid.shift(1, fill_value=False)
    segments = starts.cumsum().where(valid).astype("Int64")
    frame["segment_id"] = segments
    frame["cumulative_residual_m3"] = frame.residual_m3.groupby(segments).cumsum()
    return frame


def _statistics(frame, capacity=None, support=None):
    mask = frame.valid if support is None else frame.valid & support
    usable = frame.loc[mask]
    residual = usable.residual_m3.to_numpy(float)
    rate = usable.residual_rate_m3s.to_numpy(float)
    count = len(usable)
    complete = bool(len(frame) and mask.all())
    segments = []
    # Recompute segments for the requested support, including common-lag trims.
    labels = (mask & ~mask.shift(1, fill_value=False)).cumsum().where(mask)
    for _, segment in usable.groupby(labels.loc[usable.index]):
        segments.append({"start_date": str(segment.start_date.iloc[0].date()),
            "end_date": str(segment.index[-1].date()), "intervals": len(segment),
            "duration_seconds": float(segment.step_seconds.sum()),
            "closure_residual_m3": float(segment.residual_m3.sum())})
    result = {"total_intervals": len(frame), "valid_intervals": count,
        "invalid_or_excluded_intervals": len(frame) - count,
        "valid_duration_seconds": float(usable.step_seconds.sum()),
        "expected_duration_seconds": float(frame.step_seconds.sum()),
        "duration_coverage": float(usable.step_seconds.sum() / frame.step_seconds.sum()),
        "complete_closure": complete,
        "whole_period_closure_residual_m3": float(residual.sum()) if complete else None,
        "sum_residual_m3": float(residual.sum()) if count else None,
        "mean_residual_m3": float(np.mean(residual)) if count else None,
        "median_residual_m3": float(np.median(residual)) if count else None,
        "rmse_residual_m3": float(np.sqrt(np.mean(residual ** 2))) if count else None,
        "p95_abs_residual_m3": float(np.percentile(np.abs(residual), 95)) if count else None,
        "max_abs_residual_m3": float(np.max(np.abs(residual))) if count else None,
        "mean_residual_rate_m3s": float(np.mean(rate)) if count else None,
        "rmse_residual_rate_m3s": float(np.sqrt(np.mean(rate ** 2))) if count else None,
        "duration_weighted_residual_rate_m3s": float(residual.sum() / usable.step_seconds.sum()) if count else None,
        "negative_flow_intervals": int(usable.negative_flow.sum()), "segments": segments}
    result["rmse_residual_fraction_capacity"] = result["rmse_residual_m3"] / capacity if capacity and count else None
    return result


def audit_storage(frame, *, frequency="daily", capacity_m3=None, dead_storage_m3=None,
                  fixed_step_days=None, inflow_factor=1.0, inflow_offset_m3s=0.0,
                  balance_inflow_correction=1.0, lag_diagnostics=True):
    """Diagnose one period without choosing assumptions or changing observations.

    Missing values invalidate affected intervals; consecutive calendar samples
    are required. Signed flows remain in arithmetic and receive explicit flags.
    Configured forcing is reported separately from raw observations, when it
    differs, as ``correction * (factor * raw_inflow + offset)``.
    """
    columns = ["storage_m3", "inflow_m3s", "release_m3s"]
    if not isinstance(frame, pd.DataFrame) or list(frame.columns) != columns:
        raise ValueError("Expected exactly storage_m3, inflow_m3s, release_m3s in that order.")
    validate_sampling_index(frame.index, frequency)
    if not isinstance(lag_diagnostics, bool):
        raise TypeError("lag_diagnostics must be a boolean.")
    capacity = None if capacity_m3 is None else _finite_number(capacity_m3, "capacity_m3", positive=True)
    dead = None if dead_storage_m3 is None else _finite_number(dead_storage_m3, "dead_storage_m3", nonnegative=True)
    fixed = None if fixed_step_days is None else _finite_number(fixed_step_days, "fixed_step_days", positive=True)
    factor = _finite_number(inflow_factor, "inflow_factor")
    offset = _finite_number(inflow_offset_m3s, "inflow_offset_m3s")
    correction = _finite_number(balance_inflow_correction, "balance_inflow_correction", nonnegative=True)
    data = frame.astype(float).copy()
    storage, inflow, release = (data[name] for name in columns)
    finite = np.isfinite(data)
    flags = pd.DataFrame(index=data.index)
    for name in columns:
        flags["nonfinite_" + name.split("_")[0]] = ~finite[name]
    flags["negative_storage"] = storage.lt(0)
    flags["negative_inflow"] = inflow.lt(0)
    flags["negative_release"] = release.lt(0)
    flags["below_dead_storage"] = finite.storage_m3 & storage.lt(dead) if dead is not None else False
    flags["above_capacity"] = finite.storage_m3 & storage.gt(capacity) if capacity is not None else False
    flags["positive_release_below_dead"] = flags.below_dead_storage & finite.release_m3s & release.gt(0)
    calendar = pd.Series(np.diff(data.index.asi8) / 1e9, index=data.index[1:], dtype=float)
    clocks = {"calendar": calendar}
    if fixed is not None and not np.equal(calendar, fixed * 86400).all():
        clocks["source_fixed"] = pd.Series(fixed * 86400, index=calendar.index)
    forcings = {"raw": inflow}
    if correction * factor != 1 or correction * offset != 0:
        forcings["configured_balance"] = correction * (factor * inflow + offset)
    ledgers, records, monthly, extremes = [], [], [], []
    for forcing, values in forcings.items():
        for clock, durations in clocks.items():
            for convention in CONVENTIONS:
                name = f"{forcing}__{clock}__{convention}"
                ledger = _ledger(storage, values, release, durations, convention)
                ledgers.append(pd.concat({name: ledger}, names=["hypothesis"]))
                record = {"hypothesis": name, "forcing": forcing, "clock": clock, "convention": convention,
                    "statistics": _statistics(ledger, capacity), "lag_comparisons": []}
                if lag_diagnostics:
                    shifted = [_ledger(storage, values, release, durations, convention, a, b) for a, b in LAGS]
                    common = pd.concat([item.valid for item in shifted], axis=1).all(axis=1)
                    for (a, b), item in zip(LAGS, shifted):
                        record["lag_comparisons"].append({"inflow_lag_samples": a, "release_lag_samples": b,
                            "statistics": _statistics(item, capacity, common)})
                records.append(record)
                for month in range(1, 13):
                    part = ledger.loc[ledger.index.month == month]
                    if len(part):
                        # This is a seasonal aggregate, never a cumulative calendar segment.
                        stats = _statistics(part, capacity)
                        for field in ("segments", "whole_period_closure_residual_m3", "complete_closure"):
                            stats.pop(field)
                        monthly.append({"hypothesis": name, "destination_month": month, **stats})
                largest = ledger.loc[ledger.valid].residual_m3.abs().nlargest(10).index
                for date in largest:
                    row = ledger.loc[date]
                    extremes.append({"hypothesis": name, "date": date, **row.to_dict(),
                        "observed_storage_start_m3": float(storage.loc[row.start_date]),
                        "observed_storage_end_m3": float(storage.loc[date]),
                        "raw_inflow_start_m3s": float(inflow.loc[row.start_date]),
                        "raw_inflow_end_m3s": float(inflow.loc[date]),
                        "observed_release_start_m3s": float(release.loc[row.start_date]),
                        "observed_release_end_m3s": float(release.loc[date])})
    summary = {"rows": len(data), "intervals": len(data) - 1, "frequency": frequency,
        "first_date": str(data.index[0].date()), "last_date": str(data.index[-1].date()),
        "capacity_m3": capacity, "dead_storage_m3": dead,
        "configured_dead_exceeds_capacity": bool(capacity is not None and dead is not None and dead > capacity),
        "observation_flags": {name: int(flags[name].sum()) for name in flags},
        "storage_min_m3": float(storage[finite.storage_m3].min()) if finite.storage_m3.any() else None,
        "storage_max_m3": float(storage[finite.storage_m3].max()) if finite.storage_m3.any() else None,
        "configured_balance_inflow": {"factor": factor, "offset_m3s": offset, "correction": correction,
            "formula": "correction * (factor * raw_inflow + offset)"},
        "hypotheses": records, "interpretation": INTERPRETATION,
        "settings_selected": None, "model_ready_claimed": False, "search_evaluations": 0}
    flags.index.name = "date"
    extreme_columns = ["hypothesis", "date", "start_date", "step_seconds", "storage_change_m3",
        "inflow_volume_m3", "release_volume_m3", "residual_m3", "residual_rate_m3s", "valid",
        "negative_flow", "segment_id", "cumulative_residual_m3", "observed_storage_start_m3",
        "observed_storage_end_m3", "raw_inflow_start_m3s", "raw_inflow_end_m3s",
        "observed_release_start_m3s", "observed_release_end_m3s"]
    return StorageAuditResult(pd.concat(ledgers), summary, pd.DataFrame(monthly),
                             pd.DataFrame(extremes, columns=extreme_columns), flags)
