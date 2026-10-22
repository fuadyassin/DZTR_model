"""Calibration-only experiments for native daily and monthly observations.

This adapter leaves the established daily checkpoint workflow unchanged.
Warmup counts observations, not days; monthly series are never upsampled.
"""

from dataclasses import asdict, dataclass, replace
from numbers import Integral

import numpy as np
import pandas as pd

from dztr.evaluation import evaluate_simulation
from dztr.parameterization import estimate_generalized_parameters, estimate_percentiles
from dztr.schemas import DataQuality, ReservoirObservations, ReservoirParameters
from dztr.simulation.sampled import SamplingSettings, simulate_sampled_reservoir, validate_sampling_index
from .bounds import CalibrationBounds, DEFAULT_PERCENTILE_BOUNDS
from .candidates import parameters_from_vector
from .objectives import CalibrationObjective, CandidateEvaluation
from .periods import DatePeriod


def _observations(frame, source=None):
    """Keep SI measurements unchanged, including missing or signed source data."""
    frame = frame.copy(deep=True)
    missing = {name: int(frame[name].isna().sum()) for name in frame.columns}
    quality = DataQuality("available", len(frame), missing, 0,
                          tuple(f"{name} has {count} missing samples." for name, count in missing.items() if count))
    return ReservoirObservations(frame, quality, source)


def target_observations(observations):
    """Fit storage/release targets independently of the inflow forcing gate.

    The shared quantile estimator validates every supplied measurement column.
    Inflow is not used in these six targets or their percentile bounds, so omit
    it from this fitting view; original observations remain unchanged.
    """
    frame = observations.frame.copy(deep=True)
    frame["inflow_m3s"] = np.nan
    return _observations(frame, observations.source)


def select_sampled_period(observations, period, sampling):
    """Require exact boundaries and complete native sampling coverage."""
    if not isinstance(observations, ReservoirObservations) or observations.quality.status != "available":
        raise ValueError("Available SI observations are required.")
    if not isinstance(period, DatePeriod) or not isinstance(sampling, SamplingSettings):
        raise TypeError("Expected DatePeriod and SamplingSettings.")
    validate_sampling_index(observations.frame.index, sampling.frequency)
    frame = observations.frame.loc[period.start:period.end]
    expected = pd.date_range(period.start, period.end, freq="D" if sampling.frequency == "daily" else "MS")
    if (not len(expected) or expected[0] != period.start or expected[-1] != period.end
            or len(frame) != len(expected) or not frame.index.equals(expected.rename(frame.index.name))):
        raise ValueError("Period boundaries and consecutive samples must match the configured frequency.")
    return _observations(frame, observations.source)


@dataclass(frozen=True)
class SampledSplit:
    simulation_start: pd.Timestamp
    calibration: DatePeriod
    validation: DatePeriod
    warmup_rows: int

    def __post_init__(self):
        object.__setattr__(self, "simulation_start", DatePeriod(self.simulation_start, self.simulation_start).start)
        if not isinstance(self.calibration, DatePeriod) or not isinstance(self.validation, DatePeriod):
            raise TypeError("Scoring periods must be DatePeriod objects.")
        if self.validation.start <= self.calibration.end or self.simulation_start > self.calibration.start:
            raise ValueError("Warmup, calibration and validation must be chronological and nonoverlapping.")
        if isinstance(self.warmup_rows, bool) or not isinstance(self.warmup_rows, Integral) or self.warmup_rows < 0:
            raise ValueError("warmup_rows must be a nonnegative integer.")

    @property
    def simulation_period(self):
        return DatePeriod(self.simulation_start, self.validation.end)

    def as_dict(self):
        return {"simulation_start": self.simulation_start.date().isoformat(),
                "calibration": self.calibration.as_dict(), "validation": self.validation.as_dict(),
                "warmup_rows": int(self.warmup_rows)}


def default_sampled_split(observations, sampling):
    """First calendar year warmup, then a chronological division by year.

    This is an explicit new experiment protocol, not a recovered paper split.
    A final partial year is retained in validation, rather than discarded.
    """
    validate_sampling_index(observations.frame.index, sampling.frequency)
    index = observations.frame.index
    if index[0].month != 1 or index[0].day != 1:
        raise ValueError("Default split requires a January 1 start; supply explicit periods otherwise.")
    years = np.unique(index.year)
    if len(years) < 3:
        raise ValueError("Default split requires at least three calendar years.")
    fitting_years = (len(years) - 1) // 2
    final_cal_year = years[fitting_years]
    cal_index = index[(index.year > years[0]) & (index.year <= final_cal_year)]
    val_index = index[index.year > final_cal_year]
    return SampledSplit(index[0], DatePeriod(cal_index[0], cal_index[-1]),
                        DatePeriod(val_index[0], val_index[-1]), int((index.year == years[0]).sum()))


def validate_split(observations, split, sampling):
    continuous = select_sampled_period(observations, split.simulation_period, sampling)
    for period in (split.calibration, split.validation):
        select_sampled_period(continuous, period, sampling)
    actual_warmup = int((continuous.frame.index < split.calibration.start).sum())
    if actual_warmup != split.warmup_rows:
        raise ValueError("warmup_rows disagrees with the configured dates and sampling frequency.")
    return continuous


def estimate_sampled_bounds(observations, period, template, sampling, *, quantile_method="hazen"):
    fitting = target_observations(select_sampled_period(observations, period, sampling))
    probabilities = tuple(sorted({value for pair in DEFAULT_PERCENTILE_BOUNDS for value in pair}))
    quantiles = estimate_percentiles(fitting, probabilities, quantile_method=quantile_method)
    lookup = quantiles.frame.set_index(["variable", "month", "percentile"])
    monthly = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
    for variable, stem, suffix in (("storage_m3", "storage", "m3"), ("release_m3s", "release", "m3s")):
        for level, pair in zip(("critical", "normal", "flood"), DEFAULT_PERCENTILE_BOUNDS):
            for side, percentile in zip(("lower", "upper"), pair):
                monthly[f"{side}_{stem}_{level}_{suffix}"] = [
                    float(lookup.loc[(variable, month, percentile), "value"]) for month in range(1, 13)]
    return CalibrationBounds(monthly, template.capacity_m3, template.dead_storage_fraction, period,
                             quantile_method=quantile_method, source=fitting.source)


class SampledCalibrationObjective:
    """The same 72 coordinates and negative NSE objectives on native samples."""

    def __init__(self, observations, split, template, *, sampling, bounds=None, atol_storage_m3=1e-4):
        if not isinstance(template, ReservoirParameters):
            raise TypeError("Optimization requires physically valid ReservoirParameters.")
        if not np.isfinite(atol_storage_m3) or atol_storage_m3 <= 0:
            raise ValueError("atol_storage_m3 must be finite and positive.")
        if bounds is not None and (not isinstance(bounds, CalibrationBounds)
                or bounds.period != split.calibration or bounds.capacity_m3 != template.capacity_m3
                or bounds.dead_storage_fraction != template.dead_storage_fraction):
            raise ValueError("Bounds must match the template and calibration period.")
        self.split, self.sampling = split, sampling
        self.calibration_period = split.calibration
        self.template, self.bounds = replace(template), replace(bounds) if bounds is not None else None
        self.atol_storage_m3 = float(atol_storage_m3)
        # Constructor checks calendar coverage; forcing checks for evaluation
        # touch only the warmup/calibration prefix.
        validate_split(observations, split, sampling)
        self._source = _observations(observations.frame, observations.source)
        self._continuous = select_sampled_period(self._source,
                DatePeriod(split.simulation_start, split.calibration.end), sampling)
        self._scored = select_sampled_period(self._source, split.calibration, sampling)
        self._require_forcing(self._continuous)
        CalibrationObjective._require_defined_objectives(self._scored)

    def _require_forcing(self, observations):
        inflow = observations.frame.inflow_m3s.to_numpy(dtype=float)
        if not np.isfinite(inflow).all() or (not self.sampling.allow_signed_inflow and (inflow < 0).any()):
            raise ValueError("Optimization requires finite forcing; signed inflows need a reviewed explicit policy.")

    def __call__(self, vector):
        return np.asarray(self.evaluate(vector).objectives)

    def evaluate(self, vector):
        return self._evaluate(vector, self._continuous, self._scored, self.split.calibration)

    def evaluate_validation(self, vector):
        continuous = select_sampled_period(self._source, self.split.simulation_period, self.sampling)
        observed = select_sampled_period(self._source, self.split.validation, self.sampling)
        self._require_forcing(continuous)
        CalibrationObjective._require_defined_objectives(observed)
        return self._evaluate(vector, continuous, observed, self.split.validation)

    def _evaluate(self, vector, continuous, observed, period):
        counts = {label: int(observed.frame[column].notna().sum())
                  for label, column in (("release", "release_m3s"), ("storage", "storage_m3"))}
        parameters = simulation = None
        try:
            parameters = (parameters_from_vector(vector, self.template) if self.bounds is None
                          else self.bounds.validate(vector, self.template))
            simulation = simulate_sampled_reservoir(continuous, parameters, sampling=self.sampling,
                                                    atol_storage_m3=self.atol_storage_m3)
            if simulation.frame.storage_m3.lt(0).any():
                raise ValueError("Candidate produced negative storage.")
            if simulation.frame.balance_residual_m3.abs().max() > self.atol_storage_m3:
                raise ValueError("Candidate water-balance residual exceeds the configured tolerance.")
            scored = replace(simulation, frame=simulation.frame.loc[period.start:period.end].copy())
            metrics = evaluate_simulation(observed, scored)
            costs = (-metrics["release_nse"], -metrics["storage_nse"])
            if not np.isfinite(costs).all():
                raise ValueError("Candidate release/storage NSE is undefined.")
        except (TypeError, ValueError) as error:
            return CandidateEvaluation(False, (float("inf"), float("inf")), {}, counts, str(error),
                                       parameters, simulation, period)
        return CandidateEvaluation(True, costs, metrics, counts, None, parameters, simulation, period)


@dataclass(frozen=True)
class SampledBaselineResult:
    split: SampledSplit
    sampling: SamplingSettings
    observations: ReservoirObservations
    estimates: object
    simulation: object
    metrics: dict
    paired_counts: dict

    def table(self):
        frame = self.simulation.frame.copy()
        for column in ("storage_m3", "release_m3s"):
            frame[f"observed_{column}"] = self.observations.frame[column]
        frame["period"] = "unscored"
        frame.loc[frame.index < self.split.calibration.start, "period"] = "spinup"
        for label, period in (("calibration", self.split.calibration), ("validation", self.split.validation)):
            frame.loc[period.start:period.end, "period"] = label
        return frame

    def summary(self):
        result = self.simulation
        return {"experiment": "sampled_generalized_baseline", "split": self.split.as_dict(),
                "sampling": asdict(self.sampling), "parameter_estimation_period": self.split.calibration.as_dict(),
                "target_percentiles": list(self.estimates.target_percentiles),
                "quantile_method": self.estimates.quantile_method, "metrics": self.metrics,
                "paired_counts": self.paired_counts, "solver": result.solver, "overflow_mode": result.overflow_mode,
                "max_abs_balance_residual_m3": float(result.frame.balance_residual_m3.abs().max()),
                "minimum_storage_m3": float(result.frame.storage_m3.min()), "issues": list(result.issues),
                "notes": ["Targets and bounds use calibration observations only.",
                          "Warmup is excluded from fitting and scoring; validation carries the continuous state.",
                          "Fixed monthly duration and target-month selection are explicit independent settings."]}


def run_sampled_baseline(observations, split, *, sampling, model, quantile_method="hazen",
                         target_percentiles=(10, 45, 85), atol_storage_m3=1e-4):
    continuous = validate_split(observations, split, sampling)
    fitting = target_observations(select_sampled_period(continuous, split.calibration, sampling))
    estimates = estimate_generalized_parameters(fitting, target_percentiles,
                   channel_capacity_m3s=model["channel_capacity_m3s"], quantile_method=quantile_method)
    parameters = ReservoirParameters.from_generalized(estimates, **{
        name: model[name] for name in ("capacity_m3", "initial_storage_m3", "initial_release_m3s",
                                       "dead_storage_fraction", "inflow_correction", "flood_mode")})
    result = simulate_sampled_reservoir(continuous, parameters, sampling=sampling, atol_storage_m3=atol_storage_m3)
    if result.frame.storage_m3.lt(0).any() or result.frame.balance_residual_m3.abs().max() > atol_storage_m3:
        raise ValueError("Baseline trajectory failed storage or water-balance checks.")
    metrics, counts = {}, {}
    for label, period in (("calibration", split.calibration), ("validation", split.validation)):
        observed = select_sampled_period(continuous, period, sampling)
        simulated = replace(result, frame=result.frame.loc[period.start:period.end].copy())
        metrics[label] = evaluate_simulation(observed, simulated)
        counts[label] = {name: int(observed.frame[column].notna().sum())
                        for name, column in (("release", "release_m3s"), ("storage", "storage_m3"))}
    return SampledBaselineResult(split, sampling, continuous, estimates, result, metrics, counts)
