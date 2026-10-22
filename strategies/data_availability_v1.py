"""Composed target recipes; observation sources and numerical routes are distinct."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dztr.schemas import MONTHLY_TARGET_COLUMNS
from workflows.geometry_v1 import checked_mask, digest, validate_index

METHODS = ("observed_monthly", "level_geometry_monthly", "flow_only", "storage_only", "ungauged_cr1")
STORAGE_COLUMNS = list(MONTHLY_TARGET_COLUMNS[:3])
RELEASE_COLUMNS = list(MONTHLY_TARGET_COLUMNS[3:])


@dataclass(frozen=True)
class NaturalizedOutflow:
    values: pd.Series
    source_kind: str
    evidence: str
    independent_of_target_responses: bool

    def __post_init__(self):
        validate_index(self.values.index)
        if self.source_kind != "simulated_naturalized_outflow" or not self.evidence:
            raise ValueError("Declare simulated naturalized outflow and its model/source evidence")
        if not isinstance(self.independent_of_target_responses, bool):
            raise ValueError("Declare whether naturalized outflow depends on target responses")
        values = self.values.to_numpy(dtype=float)
        if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
            raise ValueError("Naturalized outflow must be nonnegative or missing")


def series_signature(series):
    return digest({"dates": [str(d) for d in series.index],
                   "values": [None if pd.isna(v) else float(v) for v in series]})


def _sample(series, mask):
    selected = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)[mask]
    if np.isinf(selected).any() or (selected[np.isfinite(selected)] < 0).any():
        raise ValueError("Target observations must be nonnegative or missing in fitting data")
    return selected[np.isfinite(selected)]


def empirical_targets(series, fit_mask, *, monthly, percentiles, method):
    mask = checked_mask(series.index, fit_mask)
    if method not in {"linear", "hazen"}:
        raise ValueError("Use the explicit linear or Hazen percentile estimator")
    if len(percentiles) != 3 or not all(0 <= p <= 100 for p in percentiles) or not np.all(np.diff(percentiles) > 0):
        raise ValueError("Use three increasing percentiles")
    values, counts = [], []
    for month in range(1, 13):
        selected = mask & (series.index.month == month) if monthly else mask
        sample = _sample(series, selected)
        counts.append(len(sample))
        values.append(np.percentile(sample, percentiles, method=method) if len(sample) else [np.nan] * 3)
    return np.asarray(values), counts


def capabilities(frame, fit_mask, *, storage_source_kind="provided_storage", source_gates=()):
    mask = checked_mask(frame.index, fit_mask)
    if storage_source_kind not in {"provided_storage", "level_geometry"}:
        raise ValueError("Declare supplied or level-derived storage")
    result = {"storage_source_kind": storage_source_kind, "source_gates": list(source_gates)}
    for name, column in (("storage", "storage_m3"), ("release", "release_m3s")):
        values = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
        counts = []
        for month in range(1, 13):
            counts.append(len(_sample(values, mask & (frame.index.month == month))))
        result[name + "_calibration_month_counts"] = counts
        result[name + "_monthly_available"] = all(counts)
    storage, release = result["storage_monthly_available"], result["release_monthly_available"]
    result["recommended_method"] = (("level_geometry_monthly" if storage_source_kind == "level_geometry" else "observed_monthly")
                                    if storage and release else "storage_only" if storage else "flow_only" if release else "ungauged_cr1")
    return result


def build_targets(method, frame, fit_mask, *, capacity_m3, channel_capacity_m3s=None,
                  storage_source_kind="provided_storage", naturalized=None, quantile_method="hazen",
                  source_gates=()):
    """Return a recipe receipt, possibly incomplete, without inventing observations.

    Physical ordering against dead storage/initial states is checked by the
    calling workflow using the existing ReservoirParameters contract.
    """
    if method not in METHODS:
        raise ValueError("Unknown parameterization method")
    if not np.isfinite(capacity_m3) or capacity_m3 <= 0:
        raise ValueError("Adopted capacity must be finite and positive")
    mask = checked_mask(frame.index, fit_mask)
    if mask.sum() / len(mask) + 1e-12 < .65:
        raise ValueError("At least 65 percent of the record must be assigned to calibration")
    cap = capabilities(frame, fit_mask, storage_source_kind=storage_source_kind, source_gates=source_gates)
    reasons, assumptions = list(source_gates), []
    storage = release = None
    counts = {}
    paired = method in {"observed_monthly", "level_geometry_monthly"}
    needs_storage = paired or method == "storage_only"
    needs_release = paired or method == "flow_only"
    if method == "observed_monthly" and storage_source_kind != "provided_storage":
        reasons.append("observed_monthly_requires_provided_storage")
    if method == "level_geometry_monthly" and storage_source_kind != "level_geometry":
        reasons.append("level_geometry_monthly_requires_level_conversion")
    if needs_storage:
        if not cap["storage_monthly_available"]:
            reasons.append("storage_observations_missing_in_calibration_months")
        else:
            storage, counts["storage"] = empirical_targets(frame.storage_m3, fit_mask, monthly=True,
                                                          percentiles=(10, 45, 85), method=quantile_method)
    else:
        storage = np.tile(np.array([.20, .30, .97]) * capacity_m3, (12, 1))
        assumptions.append("Storage targets fixed to 0.20/0.30/0.97 of adopted capacity")
    release_source = None
    if needs_release:
        if not cap["release_monthly_available"]:
            reasons.append("release_observations_missing_in_calibration_months")
        else:
            release, counts["release"] = empirical_targets(frame.release_m3s, fit_mask, monthly=True,
                                                          percentiles=(10, 45, 85), method=quantile_method)
            release_source = frame.release_m3s
    else:
        if naturalized is None:
            reasons.append("needs_release_prior" if method == "storage_only" else "needs_simulated_naturalized_outflow")
        else:
            if not naturalized.values.index.equals(frame.index):
                raise ValueError("Naturalized outflow must use the exact original calendar")
            release, counts["release_prior"] = empirical_targets(naturalized.values, fit_mask, monthly=False,
                                                                 percentiles=(5, 30, 97), method=quantile_method)
            if not np.isfinite(release).all():
                reasons.append("naturalized_outflow_missing_in_fitting_period")
            release_source = naturalized.values
            assumptions.append("Annual naturalized-outflow P5/P30/P97 defines release targets")
    if method == "storage_only":
        assumptions.append("Storage-only naturalized-release hybrid is an exploratory extension")
    channel = None
    channel_source = "source_defined"
    if channel_capacity_m3s is not None:
        if not np.isfinite(channel_capacity_m3s) or channel_capacity_m3s < 0:
            raise ValueError("Channel target must be finite and nonnegative")
        channel = float(channel_capacity_m3s)
    elif release_source is not None:
        sample = _sample(release_source, mask)
        if len(sample):
            channel = float(np.percentile(sample, 99, method=quantile_method))
            channel_source = "fitting_release_source_p99_proxy"
            assumptions.append("Channel P99 is an empirical release proxy, not engineering certification")
    if channel is None:
        reasons.append("needs_channel_target")
    targets = None
    if storage is not None and release is not None:
        targets = pd.DataFrame(np.column_stack([storage, release]), columns=MONTHLY_TARGET_COLUMNS,
                               index=pd.Index(range(1, 13), name="month"))
        if not np.isfinite(targets.to_numpy()).all():
            reasons.append("targets_contain_missing_estimates")
    objective = ("storage", "release") if paired else ("release",) if method == "flow_only" else ("storage",) if method == "storage_only" else ()
    return {"method": method, "targets": targets, "channel_capacity_m3s": channel,
            "channel_target_source": channel_source, "capabilities": cap,
            "target_estimation_ready": targets is not None and not reasons,
            "objective_variables": objective, "assumptions": assumptions,
            "reasons": reasons, "sample_counts": counts, "quantile_method": quantile_method,
            "storage_target_source": storage_source_kind if needs_storage else "capacity_fractions",
            "release_target_source": "observed_release" if needs_release else "simulated_naturalized_outflow",
            "naturalized_source": None if naturalized is None else {
                "evidence": naturalized.evidence, "signature": series_signature(naturalized.values),
                "independent_of_target_responses": naturalized.independent_of_target_responses}}
