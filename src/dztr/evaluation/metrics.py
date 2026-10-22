"""Port of the published MATLAB workflow's ``obj_fun.m`` criteria."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from dztr.schemas import ReservoirObservations

if TYPE_CHECKING:
    from dztr.schemas import SimulationResult


SUPPORTED_METRICS = frozenset({"nse", "lnse", "kge", "pbias", "rmse", "nrmse", "kges", "kgea", "kgeb", "kger"})


def _paired_vectors(observed, simulated) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(observed, pd.Series) and isinstance(simulated, pd.Series):
        if not observed.index.equals(simulated.index):
            raise ValueError("Observed and simulated Series must have exactly the same index.")
    try:
        x = np.asarray(observed, dtype=float)
        y = np.asarray(simulated, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Observed and simulated values must be numeric vectors.") from error
    if x.shape != y.shape:
        raise ValueError("Observed and simulated vectors must have equal shapes.")
    if x.ndim != 1:
        raise ValueError("Observed and simulated values must be one-dimensional vectors.")
    if np.isinf(x).any() or np.isinf(y).any():
        raise ValueError("Observed and simulated values cannot contain infinities.")
    valid = ~np.isnan(x) & ~np.isnan(y)
    return x[valid], y[valid]


def _nse(x: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(x) == 0:
        return float("nan")
    variance_sum = float(np.sum((x - np.mean(x)) ** 2))
    return float(1 - np.sum((x - y) ** 2) / variance_sum) if variance_sum > 0 else float("nan")


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    dx, dy = x - np.mean(x), y - np.mean(y)
    denominator = float(np.sqrt(np.sum(dx ** 2) * np.sum(dy ** 2)))
    return float(np.sum(dx * dy) / denominator) if denominator > 0 else float("nan")


def objective_metric(observed, simulated, metric: str = "nse") -> float:
    """Return a MATLAB-compatible objective for matched observation pairs.

    Inputs are equally shaped, one-dimensional numeric vectors. NaN pairs are
    removed independently for each call; infinities raise an error. Empty pairs,
    zero denominators, undefined variances, and undefined correlations return
    NaN. KGE uses sample standard deviations (ddof=1); ``kges`` uses MATLAB's
    unit weights and therefore equals ``kge``. PBIAS is positive for overprediction.
    LNSE applies ``log(value + 0.01)`` and rejects paired values <= -0.01.
    """
    if not isinstance(metric, str) or metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose one of {', '.join(sorted(SUPPORTED_METRICS))}.")
    x, y = _paired_vectors(observed, simulated)
    if not len(x):
        return float("nan")
    if metric == "nse":
        return _nse(x, y)
    if metric == "lnse":
        if (x <= -0.01).any() or (y <= -0.01).any():
            raise ValueError("LNSE requires paired observed and simulated values greater than -0.01.")
        return _nse(np.log(x + 0.01), np.log(y + 0.01))
    if metric == "pbias":
        denominator = float(np.sum(x))
        return float(100 * np.sum(y - x) / denominator) if denominator != 0 else float("nan")
    if metric in ("rmse", "nrmse"):
        rmse = float(np.sqrt(np.mean((x - y) ** 2)))
        if metric == "rmse":
            return rmse
        observed_range = float(np.max(x) - np.min(x))
        return rmse / observed_range if observed_range > 0 else float("nan")
    if metric == "kger":
        return _correlation(x, y)
    observed_mean = float(np.mean(x))
    beta = float(np.mean(y) / observed_mean) if observed_mean != 0 else float("nan")
    if metric == "kgeb":
        return beta
    alpha = float("nan")
    if len(x) >= 2 and np.ptp(x) != 0:
        observed_std = float(np.std(x, ddof=1))
        if observed_std > 0:
            alpha = float(np.std(y, ddof=1) / observed_std)
    if metric == "kgea":
        return alpha
    correlation = _correlation(x, y)
    return float(1 - np.sqrt((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def evaluate_simulation(
    observations: ReservoirObservations,
    result: SimulationResult,
    *,
    warmup_steps: int = 0,
) -> dict[str, float]:
    """Evaluate storage/release on exactly aligned dates after skipping rows.

    ``warmup_steps`` is a count of excluded rows: MATLAB's ``wrmp=366``
    corresponds to ``warmup_steps=365``. Gaps are not filled, and missing pairs
    are excluded separately for storage and release. Skipping every row returns
    NaN for every criterion. Undefined criteria follow ``objective_metric``.
    """
    if not isinstance(observations, ReservoirObservations):
        raise TypeError("observations must be normalized ReservoirObservations.")
    if not hasattr(result, "frame") or not isinstance(result.frame, pd.DataFrame):
        raise TypeError("result must contain a simulation DataFrame.")
    if not isinstance(warmup_steps, Integral) or isinstance(warmup_steps, bool) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a nonnegative integer count of excluded rows.")
    observed_frame, simulated_frame = observations.frame, result.frame
    if not isinstance(observed_frame.index, pd.DatetimeIndex) or not isinstance(simulated_frame.index, pd.DatetimeIndex):
        raise ValueError("Observed and simulated frames must have daily DatetimeIndexes.")
    if not observed_frame.index.equals(simulated_frame.index):
        raise ValueError("Observed and simulated frames must have exactly the same date index.")
    if not observed_frame.index.is_unique:
        raise ValueError("Observed and simulated dates must be unique.")
    required = {"storage_m3", "release_m3s"}
    if not required.issubset(observed_frame.columns) or not required.issubset(simulated_frame.columns):
        raise ValueError("Observed and simulated frames require storage_m3 and release_m3s columns.")
    # Validate the full vectors before warmup exclusion, so invalid numerical
    # simulation output is never silently hidden by the warmup slice.
    for column in required:
        _paired_vectors(observed_frame[column], simulated_frame[column])
    observed_frame = observed_frame.iloc[int(warmup_steps):]
    simulated_frame = simulated_frame.iloc[int(warmup_steps):]
    metrics = {}
    for label, column, criteria in (
        ("release", "release_m3s", ("nse", "kge", "pbias", "rmse")),
        ("storage", "storage_m3", ("nse", "kge")),
    ):
        for criterion in criteria:
            metrics[f"{label}_{criterion}"] = objective_metric(observed_frame[column], simulated_frame[column], criterion)
    return metrics
