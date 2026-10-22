"""Translate empirical quantiles into DZTR's six monthly generalized targets."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from dztr.schemas import GeneralizedParameters, ReservoirObservations

from .constants import TARGET_PERCENTILES
from .quantiles import estimate_percentiles, validate_percentiles


def estimate_generalized_parameters(
    observations: ReservoirObservations,
    target_percentiles: Iterable[float] = TARGET_PERCENTILES,
    *,
    channel_capacity_m3s: float | None = None,
    quantile_method: str = "linear",
) -> GeneralizedParameters:
    """Estimate ordered critical, normal and flood targets for each month.

    Missing storage/release months raise rather than substituting overall
    quantiles. Inflow is optional here. Channel capacity defaults to overall
    release P99; an observed channel capacity can override this fallback.
    """
    targets = validate_percentiles(target_percentiles)
    if len(targets) != 3 or tuple(sorted(targets)) != targets:
        raise ValueError("Three increasing target percentiles are required.")
    required = tuple(sorted(set(targets) | {99.0}))
    estimates = estimate_percentiles(observations, required, quantile_method=quantile_method)
    lookup = estimates.frame.set_index(["variable", "month", "percentile"])
    monthly = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
    for variable, stem, suffix in (("storage_m3", "storage", "m3"), ("release_m3s", "release", "m3s")):
        for label, percentile in zip(("critical", "normal", "flood"), targets):
            monthly[f"{stem}_{label}_{suffix}"] = [lookup.loc[(variable, month, percentile), "value"] for month in range(1, 13)]
    missing = monthly.isna()
    if missing.any().any():
        months = monthly.index[missing.any(axis=1)].tolist()
        raise ValueError(f"Storage and release observations are required in every month; missing targets in months {months}.")
    capacity = float(lookup.loc[("release_m3s", 0, 99.0), "value"]) if channel_capacity_m3s is None else float(channel_capacity_m3s)
    if not np.isfinite(capacity) or capacity < 0:
        raise ValueError("Channel capacity must be finite and nonnegative.")
    issues = []
    for stem, suffix in (("storage", "m3"), ("release", "m3s")):
        values = monthly[[f"{stem}_{level}_{suffix}" for level in ("critical", "normal", "flood")]].to_numpy()
        if (np.diff(values, axis=1) < 0).any():
            raise ValueError(f"Monthly {stem} targets must be ordered.")
        if stem == "storage" and (np.diff(values, axis=1) == 0).any():
            issues.append("Equal storage targets occur; the future release solver must handle degenerate zones explicitly.")
    return GeneralizedParameters(monthly, capacity, targets, observations.source, quantile_method=quantile_method, issues=tuple(issues))
