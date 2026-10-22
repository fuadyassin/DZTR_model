"""Calibration-only negative-NSE objectives, separate from held-out evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
import pandas as pd

from dztr.evaluation import evaluate_simulation
from dztr.schemas import ReservoirObservations, ReservoirParameters, SimulationResult
from dztr.simulation import simulate_reservoir

from .bounds import CalibrationBounds
from .candidates import parameters_from_vector
from .periods import DatePeriod, select_period


@dataclass(frozen=True)
class CandidateEvaluation:
    """Two minimization costs and an explicit candidate-feasibility result.

    An infeasible candidate has infinite costs and a reason, never a fabricated
    NSE. Successful evaluations retain the continuous trajectory for inspection.
    """

    feasible: bool
    objectives: tuple[float, float]
    metrics: dict[str, float]
    paired_counts: dict[str, int]
    reason: str | None
    parameters: ReservoirParameters | None
    simulation: SimulationResult | None
    period: DatePeriod


class CalibrationObjective:
    """Minimize ``[-NSE(release), -NSE(storage)]`` on calibration dates only.

    Pass calibration-derived bounds for optimizer use. Omitting bounds permits
    physical-only evaluation of historical reference candidates; it does not
    waive physical storage-zone constraints. The configured initial state
    applies ``warmup_steps`` calendar days before the calibration start.

    The callable never reads validation inflows or observations. Held-out
    evaluation requires an explicit separate ``evaluate_validation`` call.
    """

    def __init__(
        self,
        observations: ReservoirObservations,
        calibration_period: DatePeriod,
        template: ReservoirParameters,
        *,
        warmup_steps: int = 365,
        bounds: CalibrationBounds | None = None,
        solver: str = "implicit",
        overflow_mode: str = "conservative",
        atol_storage_m3: float = 1e-4,
    ) -> None:
        if not isinstance(calibration_period, DatePeriod):
            raise TypeError("calibration_period must be a DatePeriod.")
        if not isinstance(template, ReservoirParameters):
            raise TypeError("template must be ReservoirParameters.")
        if not isinstance(warmup_steps, Integral) or isinstance(warmup_steps, bool) or warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer count before calibration.")
        if solver not in ("implicit", "matlab_fixed_point"):
            raise ValueError("solver must be 'implicit' or 'matlab_fixed_point'.")
        if overflow_mode not in ("conservative", "matlab"):
            raise ValueError("overflow_mode must be 'conservative' or 'matlab'.")
        if not np.isfinite(atol_storage_m3) or atol_storage_m3 <= 0:
            raise ValueError("atol_storage_m3 must be positive and finite.")
        if bounds is not None:
            if not isinstance(bounds, CalibrationBounds):
                raise TypeError("bounds must be CalibrationBounds or None.")
            if bounds.capacity_m3 != template.capacity_m3 or bounds.dead_storage_fraction != template.dead_storage_fraction:
                raise ValueError("Bounds and template must use the same capacity and dead-storage fraction.")
            if bounds.period != calibration_period:
                raise ValueError("Optimization bounds must be estimated from the selected calibration period.")
        self.calibration_period = calibration_period
        self.template = replace(template)
        self.bounds = replace(bounds) if bounds is not None else None
        self.warmup_steps = int(warmup_steps)
        self.solver = solver
        self.overflow_mode = overflow_mode
        self.atol_storage_m3 = float(atol_storage_m3)
        self.simulation_period = DatePeriod(calibration_period.start - pd.Timedelta(days=self.warmup_steps),
                                             calibration_period.end)
        self._calibration_observations = select_period(observations, calibration_period)
        self._simulation_observations = select_period(observations, self.simulation_period)
        if self._simulation_observations.frame.inflow_m3s.isna().any():
            raise ValueError("Calibration simulation requires finite inflow on every spin-up and calibration day.")
        if self.simulation_period.day_count < 2:
            raise ValueError("Calibration simulation requires at least two daily rows.")
        self._require_defined_objectives(self._calibration_observations)
        # Freeze input data for repeated candidate evaluations; callers may edit
        # their own frames without changing an already constructed problem.
        self._source = ReservoirObservations(observations.frame.copy(deep=True), observations.quality, observations.source)

    @staticmethod
    def _require_defined_objectives(observations: ReservoirObservations) -> None:
        for label, column in (("release", "release_m3s"), ("storage", "storage_m3")):
            values = observations.frame[column].dropna().to_numpy(dtype=float)
            if len(values) < 2 or np.ptp(values) == 0:
                raise ValueError(f"NSE objective requires at least two nonmissing, nonconstant {label} observations in the scored period.")

    def __call__(self, vector) -> np.ndarray:
        """Return calibration costs only; rejected candidates receive infinity."""
        return np.asarray(self.evaluate(vector).objectives, dtype=float)

    def evaluate(self, vector) -> CandidateEvaluation:
        return self._evaluate(vector, self._simulation_observations, self._calibration_observations, self.calibration_period)

    def evaluate_validation(self, vector, validation_period: DatePeriod) -> CandidateEvaluation:
        """Score fixed parameters on validation after carrying the full state.

        No validation days are skipped, and no observation is used to reset the
        state. Bad period/forcing configuration raises instead of being treated
        as a rejected optimization candidate.
        """
        if not isinstance(validation_period, DatePeriod):
            raise TypeError("validation_period must be a DatePeriod.")
        if validation_period.start <= self.calibration_period.end:
            raise ValueError("Validation must start after the calibration period ends.")
        continuous = select_period(self._source, DatePeriod(self.simulation_period.start, validation_period.end))
        observed = select_period(self._source, validation_period)
        if continuous.frame.inflow_m3s.isna().any():
            raise ValueError("Validation simulation requires finite inflow across the complete continuous run.")
        self._require_defined_objectives(observed)
        return self._evaluate(vector, continuous, observed, validation_period)

    def _evaluate(self, vector, continuous: ReservoirObservations, scored: ReservoirObservations,
                  period: DatePeriod) -> CandidateEvaluation:
        counts = {label: int(scored.frame[column].notna().sum())
                  for label, column in (("release", "release_m3s"), ("storage", "storage_m3"))}

        def rejected(reason, parameters=None, simulation=None):
            return CandidateEvaluation(False, (float("inf"), float("inf")), {}, counts,
                                       str(reason), parameters, simulation, period)

        try:
            parameters = (parameters_from_vector(vector, self.template) if self.bounds is None
                          else self.bounds.validate(vector, self.template))
        except (TypeError, ValueError) as exc:
            return rejected(exc)
        try:
            simulation = simulate_reservoir(continuous, parameters, solver=self.solver,
                                            overflow_mode=self.overflow_mode, atol_storage_m3=self.atol_storage_m3)
        except ValueError as exc:
            return rejected(exc, parameters)
        if simulation.frame.storage_m3.lt(0).any():
            return rejected("Candidate simulation produced negative storage.", parameters, simulation)
        balance = simulation.frame.balance_residual_m3
        if self.overflow_mode == "matlab":
            # Remove only the documented spill-accounting discrepancy, so the
            # compatibility option still rejects unconverged fixed iterations.
            balance = balance + simulation.frame.overflow_volume_m3 / 2
        residual = float(balance.abs().max())
        if residual > self.atol_storage_m3:
            return rejected("Candidate water-balance residual exceeds the configured tolerance.", parameters, simulation)
        scored_simulation = replace(simulation, frame=simulation.frame.loc[period.start:period.end].copy())
        metrics = evaluate_simulation(scored, scored_simulation)
        costs = (-metrics["release_nse"], -metrics["storage_nse"])
        if not np.isfinite(costs).all():
            return rejected("Candidate release/storage NSE is undefined.", parameters, simulation)
        return CandidateEvaluation(True, costs, metrics, counts, None, parameters, simulation, period)
