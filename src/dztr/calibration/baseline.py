"""Estimate generalized targets on calibration data and score a held-out period."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from dztr.evaluation import evaluate_simulation
from dztr.parameterization import estimate_generalized_parameters, estimate_percentiles
from dztr.schemas import GeneralizedParameters, QuantileEstimates, ReservoirObservations, ReservoirParameters, SimulationResult
from dztr.simulation import simulate_reservoir

from .periods import CalibrationValidationSplit, DatePeriod, select_period


@dataclass(frozen=True)
class BaselineResult:
    """A single continuous run, with parameters fitted only on calibration data."""

    split: CalibrationValidationSplit
    observations: ReservoirObservations
    estimates: GeneralizedParameters
    quantiles: QuantileEstimates
    simulation: SimulationResult
    metrics: dict[str, dict[str, float]]
    paired_counts: dict[str, dict[str, int]]

    def daily_table(self) -> pd.DataFrame:
        """Return observed/simulated values and explicit fitting/scoring labels."""
        frame = self.simulation.frame.copy()
        for column in ("storage_m3", "release_m3s"):
            frame[f"observed_{column}"] = self.observations.frame[column]
        frame["period"] = "unscored"
        frame.loc[frame.index < self.split.calibration.start, "period"] = "spinup"
        for label, period in (("calibration", self.split.calibration), ("validation", self.split.validation)):
            frame.loc[period.start:period.end, "period"] = label
        return frame

    def summary(self) -> dict:
        """Describe fitting, scoring, initial conditions and balance diagnostics."""
        split = self.split
        parameters = self.simulation.parameters
        periods = {
            "simulation": {**split.simulation_period.as_dict(), "daily_rows": split.simulation_period.day_count},
            "spinup": None,
            "calibration": {**split.calibration.as_dict(), "daily_rows": split.calibration.day_count},
            "validation": {**split.validation.as_dict(), "daily_rows": split.validation.day_count},
        }
        if split.warmup_steps:
            spinup = DatePeriod(split.simulation_period.start, split.calibration.start - pd.Timedelta(days=1))
            periods["spinup"] = {**spinup.as_dict(), "daily_rows": split.warmup_steps}
        return {
            "experiment": "generalized_parameter_baseline",
            "periods": periods,
            "warmup_steps": split.warmup_steps,
            "parameter_estimation_period": split.calibration.as_dict(),
            "quantile_method": self.estimates.quantile_method,
            "target_percentiles": list(self.estimates.target_percentiles),
            "model": {key: getattr(parameters, key) for key in (
                "capacity_m3", "initial_storage_m3", "initial_release_m3s", "channel_capacity_m3s",
                "dead_storage_fraction", "inflow_correction", "flood_mode",
            )},
            "solver": self.simulation.solver,
            "overflow_mode": self.simulation.overflow_mode,
            "metrics": self.metrics,
            "paired_counts": self.paired_counts,
            "diagnostics": self.balance_diagnostics(),
            "notes": [
                "Percentile targets and any inferred release-P99 channel capacity use calibration observations only.",
                "Spin-up observations are excluded from parameter estimation and metric scoring.",
                "Validation continues the simulated state; it is not initialized from validation observations.",
                "Warmup is applied before calibration; no additional validation days are skipped.",
                "Capacity, initial conditions and inflow correction are fixed caller-supplied configuration.",
            ],
        }

    def balance_diagnostics(self) -> dict[str, dict]:
        frames = {
            "simulation": self.simulation.frame,
            "calibration": self.simulation.frame.loc[self.split.calibration.start:self.split.calibration.end],
            "validation": self.simulation.frame.loc[self.split.validation.start:self.split.validation.end],
        }
        diagnostics = {}
        for label, frame in frames.items():
            diagnostics[label] = {
                "max_abs_balance_residual_m3": float(frame.balance_residual_m3.abs().max()),
                "overflow_days": int(frame.overflow_volume_m3.gt(0).sum()),
                "minimum_storage_m3": float(frame.storage_m3.min()),
                "maximum_storage_m3": float(frame.storage_m3.max()),
            }
        diagnostics["simulation"]["issues"] = list(self.simulation.issues)
        return diagnostics


def run_baseline(
    observations: ReservoirObservations,
    split: CalibrationValidationSplit,
    *,
    capacity_m3: float,
    initial_storage_m3: float,
    initial_release_m3s: float,
    channel_capacity_m3s: float | None = None,
    dead_storage_fraction: float = 0.1,
    inflow_correction: float = 1.0,
    flood_mode: str = "storage",
    target_percentiles: tuple[float, float, float] = (10, 45, 85),
    quantile_method: str = "hazen",
    solver: str = "implicit",
    overflow_mode: str = "conservative",
    atol_storage_m3: float = 1e-4,
) -> BaselineResult:
    """Fit on calibration dates, run through validation, and score separately.

    The initial state applies at ``split.simulation_period.start``. Warmup is
    a prefix before calibration, not a count to discard from calibration or
    validation. Dates between non-adjacent periods are simulated but unscored.
    Missing storage/release pairs are excluded from metrics independently;
    every simulated day must have a finite inflow.
    """
    if not isinstance(split, CalibrationValidationSplit):
        raise TypeError("split must be a CalibrationValidationSplit.")
    fitting = select_period(observations, split.calibration)
    continuous = select_period(observations, split.simulation_period)
    estimates = estimate_generalized_parameters(
        fitting, target_percentiles, channel_capacity_m3s=channel_capacity_m3s,
        quantile_method=quantile_method,
    )
    quantiles = estimate_percentiles(
        fitting, tuple(sorted(set(estimates.target_percentiles) | {99.0})),
        quantile_method=quantile_method,
    )
    parameters = ReservoirParameters.from_generalized(
        estimates, capacity_m3=capacity_m3, initial_storage_m3=initial_storage_m3,
        initial_release_m3s=initial_release_m3s, dead_storage_fraction=dead_storage_fraction,
        inflow_correction=inflow_correction, flood_mode=flood_mode,
    )
    result = simulate_reservoir(continuous, parameters, solver=solver,
                                overflow_mode=overflow_mode, atol_storage_m3=atol_storage_m3)
    metrics, counts = {}, {}
    for label, period in (("calibration", split.calibration), ("validation", split.validation)):
        observed = select_period(continuous, period)
        simulated = replace(result, frame=result.frame.loc[period.start:period.end].copy())
        metrics[label] = evaluate_simulation(observed, simulated)
        counts[label] = {
            name: int((observed.frame[column].notna() & simulated.frame[column].notna()).sum())
            for name, column in (("release", "release_m3s"), ("storage", "storage_m3"))
        }
    return BaselineResult(split, continuous, estimates, quantiles, result, metrics, counts)


def json_ready(value):
    """Represent undefined metrics as JSON null, without nonstandard NaN tokens."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value
