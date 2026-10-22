"""Empirical non-exceedance percentiles of normalized reservoir observations."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from dztr.schemas import QuantileEstimates, ReservoirObservations

from .constants import DEFAULT_PERCENTILES, OBSERVATION_VARIABLES


def validate_percentiles(percentiles: Iterable[float]) -> tuple[float, ...]:
    """Validate a percentage sequence, retaining the caller's ordering."""
    values = tuple(float(p) for p in percentiles)
    if not values or not all(np.isfinite(p) and 0 <= p <= 100 for p in values):
        raise ValueError("Percentiles must be a nonempty sequence between 0 and 100.")
    if len(set(values)) != len(values):
        raise ValueError("Percentiles must be unique.")
    return values


def estimate_percentiles(
    observations: ReservoirObservations,
    percentiles: Iterable[float] = DEFAULT_PERCENTILES,
    *,
    quantile_method: str = "linear",
) -> QuantileEstimates:
    """Calculate monthly and overall linear or MATLAB-compatible Hazen quantiles.

    Missing values are excluded per variable; empty samples return NaN with
    sample_count=0. Placeholder and unavailable data cannot be parameterized.
    """
    if not isinstance(observations, ReservoirObservations):
        raise TypeError("Normalize observations before estimating percentiles.")
    if quantile_method not in ("linear", "hazen"):
        raise ValueError("quantile_method must be 'linear' or 'hazen'.")
    if observations.quality.status != "available":
        raise ValueError(f"Cannot parameterize {observations.quality.status} observations.")
    probabilities = validate_percentiles(percentiles)
    frame = observations.frame
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Observations require a daily DatetimeIndex.")
    if set(OBSERVATION_VARIABLES) - set(frame.columns):
        raise ValueError("Observations must contain the canonical SI columns.")
    rows = []
    for variable in OBSERVATION_VARIABLES:
        for month in range(13):
            sample = frame[variable] if month == 0 else frame.loc[frame.index.month == month, variable]
            sample = sample.dropna().to_numpy(dtype=float)
            if sample.size and (not np.isfinite(sample).all() or (sample < 0).any()):
                raise ValueError(f"{variable} must contain finite, nonnegative observations.")
            values = np.percentile(sample, probabilities, method=quantile_method) if sample.size else np.full(len(probabilities), np.nan)
            rows.extend(
                {"variable": variable, "month": month, "percentile": p, "value": float(value), "sample_count": int(sample.size)}
                for p, value in zip(probabilities, values)
            )
    return QuantileEstimates(pd.DataFrame(rows), source=observations.source, quantile_method=quantile_method)
