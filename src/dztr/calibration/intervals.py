"""Calibration-only targets and pooled objectives for declared daily means.

These APIs are opt-in. They use the interval solver and the prepared source
quality masks without changing the existing MATLAB-compatible workflow.
Observed storage initializes clean blocks, so all results are explicitly
labelled conditional hindcasts rather than uninterrupted forecasts.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from dztr.data import normalize_observations
from dztr.data.intervals import DailyIntervalPolicy, IntervalPreparation
from dztr.evaluation.metrics import objective_metric
from dztr.parameterization import estimate_generalized_parameters
from dztr.schemas import GeneralizedParameters, ReservoirParameters
from dztr.simulation.intervals import (
    IntervalSimulationSettings,
    simulate_interval_reservoir,
)

from .bounds import CalibrationBounds, estimate_calibration_bounds
from .candidates import parameters_from_vector
from .objectives import CandidateEvaluation
from .periods import DatePeriod


_CONDITIONAL_ISSUE = (
    "conditional_hindcast: clean blocks initialize from provisional observed "
    "storage; initialization rows are excluded from scores."
)


@dataclass(frozen=True)
class IntervalTargetFit:
    """Generalized targets and independent monthly/overall sample support."""

    estimates: GeneralizedParameters
    support: pd.DataFrame
    period: DatePeriod


def _selected(frame: pd.DataFrame, period: DatePeriod) -> pd.DataFrame:
    if not isinstance(period, DatePeriod):
        raise TypeError("period must be a DatePeriod.")
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Prepared inputs require a daily DatetimeIndex.")
    if frame.index.tz is not None:
        raise ValueError("Prepared calendar labels must be timezone-naive.")
    selected = frame.loc[(frame.index >= period.start) & (frame.index <= period.end)]
    expected = pd.date_range(period.start, period.end, freq="D")
    if len(selected) != len(expected) or not (selected.index == expected).all():
        raise ValueError(
            f"Period {period.start.date().isoformat()} through {period.end.date().isoformat()} "
            "requires complete consecutive daily coverage, including both boundaries."
        )
    return selected


def _mask(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"Prepared inputs require {column}.")
    values = frame[column]
    if not pd.api.types.is_bool_dtype(values.dtype) or values.isna().any():
        raise ValueError(f"{column} must be a complete boolean eligibility mask.")
    return values.to_numpy(dtype=bool, copy=True)


def _validated_selected(preparation: IntervalPreparation, period: DatePeriod) -> pd.DataFrame:
    """Check canonical quality masks locally without inspecting future values."""
    if not isinstance(preparation.policy, DailyIntervalPolicy):
        raise TypeError("Prepared policy must be a DailyIntervalPolicy.")
    selected = _selected(preparation.frame, period)
    dates = preparation.policy.quarantined_interval_end_dates
    quarantined_labels = set(dates) | {value - pd.Timedelta(days=1) for value in dates}
    clean = ~selected.index.isin(quarantined_labels)
    previous_labels = selected.index - pd.Timedelta(days=1)
    previous_clean = (~previous_labels.isin(quarantined_labels)
                      & previous_labels.isin(preparation.frame.index))
    interval = clean & previous_clean
    expected = {
        "source_transition_flagged": selected.index.isin(dates),
        "storage_endpoint_quarantined": ~clean,
        "segment_initialization": clean & ~previous_clean,
        "storage_target_eligible": clean,
        "simulation_interval_eligible": interval,
        "release_score_eligible": interval,
        "storage_score_eligible": interval,
    }
    for column, values in expected.items():
        if not np.array_equal(_mask(selected, column), values):
            raise ValueError(f"Prepared {column} does not match the declared source-quality graph.")
    return selected


def _target_values(frame: pd.DataFrame, column: str, mask: np.ndarray) -> np.ndarray:
    """Inspect only eligible observations; missing target pairs remain missing."""
    if column not in frame:
        raise ValueError(f"Prepared inputs require {column}.")
    try:
        numeric = pd.to_numeric(frame[column].where(mask), errors="raise")
        if np.iscomplexobj(numeric.to_numpy()):
            raise ValueError("Complex observations are unsupported.")
        values = numeric.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Eligible {column} observations must be real numeric SI values.") from exc
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError(f"Eligible {column} observations must be finite nonnegative values or NaN.")
    return values


def _fitting_view(preparation: IntervalPreparation, calibration_period: DatePeriod, source):
    selected = _validated_selected(preparation, calibration_period)
    storage = _target_values(selected, "storage_m3", _mask(selected, "storage_target_eligible"))
    release = _target_values(selected, "release_m3s", _mask(selected, "simulation_interval_eligible"))
    fitting = pd.DataFrame({
        "storage_m3": storage,
        "inflow_m3s": np.full(len(selected), np.nan),
        "release_m3s": release,
    }, index=selected.index.copy())
    if source is None:
        source = preparation.frame.attrs.get("source")
    return selected, storage, release, normalize_observations(fitting, source=source)


def fit_interval_targets(
    preparation: IntervalPreparation,
    calibration_period: DatePeriod,
    *,
    channel_capacity_m3s: float | None = None,
    target_percentiles=(10.0, 45.0, 85.0),
    quantile_method: str = "hazen",
    source: str | None = None,
) -> IntervalTargetFit:
    """Fit monthly targets on independently eligible calibration observations.

    Storage uses ``storage_target_eligible``. Observed daily mean release uses
    ``simulation_interval_eligible``, which excludes block initialization labels
    and all transitions touching quarantined states. Missing S/Q are excluded
    independently. Inflow is set to NaN in the fitting view because it is not a
    target variable and signed computed net inflow is not gross river inflow.

    The existing estimator handles quantiles, missing months and the optional
    release-P99 channel-capacity fallback. This helper does not repair targets,
    impose physical parameter ordering, or estimate initial conditions.
    """
    if not isinstance(preparation, IntervalPreparation):
        raise TypeError("preparation must be an IntervalPreparation.")
    selected, storage, release, observations = _fitting_view(preparation, calibration_period, source)
    estimates = estimate_generalized_parameters(
        observations, target_percentiles,
        channel_capacity_m3s=channel_capacity_m3s,
        quantile_method=quantile_method,
    )
    estimates = replace(estimates, issues=estimates.issues + (
        "Targets use only source-quality-eligible calibration storage and daily mean total-release observations; inflow is excluded from fitting.",
    ))
    rows = []
    for month in range(13):
        include = np.ones(len(selected), dtype=bool) if month == 0 else selected.index.month == month
        count = int(include.sum())
        storage_count = int(np.isfinite(storage[include]).sum())
        release_count = int(np.isfinite(release[include]).sum())
        rows.append({
            "month": month,
            "row_count": count,
            "storage_sample_count": storage_count,
            "release_sample_count": release_count,
            "storage_excluded_rows": count - storage_count,
            "release_excluded_rows": count - release_count,
        })
    support = pd.DataFrame(rows).set_index("month")
    return IntervalTargetFit(estimates, support, calibration_period)


def estimate_interval_bounds(
    preparation: IntervalPreparation,
    calibration_period: DatePeriod,
    template: ReservoirParameters,
    *,
    quantile_method: str = "hazen",
    source: str | None = None,
) -> CalibrationBounds:
    """Fit calibration-only bounds using the same independent target masks.

    The existing bounds estimator validates matching physical constants and
    reports empty empirical feasible boxes. No quarantine, warmup or validation
    observation enters the sample, and no percentile limits are repaired.
    """
    if not isinstance(preparation, IntervalPreparation):
        raise TypeError("preparation must be an IntervalPreparation.")
    _, _, _, observations = _fitting_view(preparation, calibration_period, source)
    return estimate_calibration_bounds(
        observations, calibration_period, template, quantile_method=quantile_method,
    )


class IntervalCalibrationObjective:
    """Minimize pooled ``[-NSE(mean total outflow), -NSE(storage)]``.

    Calibration simulates only the declared warmup/calibration prefix. Future
    observations and forcing may be copied for an explicit later validation
    call but are not inspected by the constructor or ``evaluate``. Clean period
    boundaries carry simulated state; only source quarantine or an explicitly
    selected simulation start creates a provisional observed-storage anchor.
    """

    def __init__(
        self,
        preparation: IntervalPreparation,
        calibration_period: DatePeriod,
        template: ReservoirParameters,
        *,
        simulation_start=None,
        settings: IntervalSimulationSettings | None = None,
        bounds: CalibrationBounds | None = None,
        atol_storage_m3: float = 1e-4,
    ) -> None:
        if not isinstance(preparation, IntervalPreparation):
            raise TypeError("preparation must be an IntervalPreparation.")
        if not isinstance(calibration_period, DatePeriod):
            raise TypeError("calibration_period must be a DatePeriod.")
        if not isinstance(template, ReservoirParameters):
            raise TypeError("template must be ReservoirParameters.")
        if settings is None:
            settings = IntervalSimulationSettings()
        if not isinstance(settings, IntervalSimulationSettings):
            raise TypeError("settings must be IntervalSimulationSettings.")
        if not np.isscalar(atol_storage_m3) or not np.isfinite(atol_storage_m3) or atol_storage_m3 <= 0:
            raise ValueError("atol_storage_m3 must be positive and finite.")
        if bounds is not None:
            if not isinstance(bounds, CalibrationBounds):
                raise TypeError("bounds must be CalibrationBounds or None.")
            if bounds.capacity_m3 != template.capacity_m3 or bounds.dead_storage_fraction != template.dead_storage_fraction:
                raise ValueError("Bounds and template must use the same capacity and dead-storage fraction.")
            if bounds.period != calibration_period:
                raise ValueError("Optimization bounds must match the calibration period.")
        if not isinstance(preparation.frame.index, pd.DatetimeIndex) or len(preparation.frame) < 2:
            raise ValueError("Prepared inputs require at least two daily calendar rows.")
        if simulation_start is None:
            simulation_start = preparation.frame.index[0]
        simulation_period = DatePeriod(simulation_start, calibration_period.end)
        if simulation_period.start > calibration_period.start:
            raise ValueError("simulation_start must be on or before the calibration start.")
        if simulation_period.day_count < 2:
            raise ValueError("Calibration simulation requires at least two daily rows.")
        self._preparation = IntervalPreparation(
            deepcopy(preparation.frame), deepcopy(preparation.segments), preparation.policy,
        )
        self.calibration_period = calibration_period
        self.simulation_period = simulation_period
        self.template = replace(template)
        self.settings = replace(settings)
        self.bounds = replace(bounds) if bounds is not None else None
        self.atol_storage_m3 = float(atol_storage_m3)
        _selected(self._preparation.frame, simulation_period)
        self._validate_forcing(simulation_period)
        self._require_defined_objectives(calibration_period)

    def _effective_scoring_masks(self, period: DatePeriod) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        frame = _validated_selected(self._preparation, period)
        masks = {
            "release": _mask(frame, "release_score_eligible"),
            "storage": _mask(frame, "storage_score_eligible"),
        }
        # An explicit start inside a clean block is a new computational anchor.
        # Quarantined starts already have false masks. No phase boundary resets.
        if self.simulation_period.start in frame.index:
            position = frame.index.get_loc(self.simulation_period.start)
            for values in masks.values():
                values[position] = False
        return frame, masks

    def _observed_targets(self, period: DatePeriod):
        frame, masks = self._effective_scoring_masks(period)
        observed = {
            "release": _target_values(frame, "release_m3s", masks["release"]),
            "storage": _target_values(frame, "storage_m3", masks["storage"]),
        }
        return frame, masks, observed

    def _require_defined_objectives(self, period: DatePeriod) -> None:
        _, _, observed = self._observed_targets(period)
        for label, values in observed.items():
            present = values[np.isfinite(values)]
            if len(present) < 2 or np.ptp(present) == 0:
                raise ValueError(
                    f"NSE objective requires at least two eligible nonmissing, nonconstant {label} observations in the scored period."
                )

    def _validate_forcing(self, period: DatePeriod) -> None:
        frame = _validated_selected(self._preparation, period)
        try:
            numeric = pd.to_numeric(frame.inflow_m3s, errors="raise")
            if np.iscomplexobj(numeric.to_numpy()):
                raise ValueError("Complex inflow is unsupported.")
            inflow = numeric.to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Simulation requires finite numeric inflow throughout its selected prefix.") from exc
        if not np.isfinite(inflow).all():
            raise ValueError("Simulation requires finite inflow throughout its selected prefix.")
        if self._preparation.policy.flow_semantics == "physical_inflow" and (inflow < 0).any():
            raise ValueError("Physical inflow cannot be negative.")

    def __call__(self, vector) -> np.ndarray:
        return np.asarray(self.evaluate(vector).objectives, dtype=float)

    def evaluate(self, vector) -> CandidateEvaluation:
        """Evaluate a candidate using only warmup and calibration forcing."""
        return self._evaluate(vector, self.calibration_period)

    def evaluate_validation(self, vector, validation_period: DatePeriod) -> CandidateEvaluation:
        """Explicitly evaluate a fixed candidate over the continuous prefix.

        Bad validation configuration raises a clear error. Candidate root or
        physical failures receive infinite costs with no partial trajectory
        scoring, just as in calibration. Source quarantines still initialize
        separate clean blocks, so this remains a conditional hindcast.
        """
        if not isinstance(validation_period, DatePeriod):
            raise TypeError("validation_period must be a DatePeriod.")
        if validation_period.start <= self.calibration_period.end:
            raise ValueError("Validation must start after the calibration period ends.")
        continuous = DatePeriod(self.simulation_period.start, validation_period.end)
        self._validate_forcing(continuous)
        self._require_defined_objectives(validation_period)
        return self._evaluate(vector, validation_period)

    @staticmethod
    def _label_simulation(simulation):
        if simulation is None:
            return None
        return replace(simulation, issues=simulation.issues + (_CONDITIONAL_ISSUE,))

    def _evaluate(self, vector, period: DatePeriod) -> CandidateEvaluation:
        frame, masks, observed = self._observed_targets(period)
        counts = {label: int(np.isfinite(values).sum()) for label, values in observed.items()}

        def rejected(reason, parameters=None, simulation=None):
            return CandidateEvaluation(
                False, (float("inf"), float("inf")), {}, counts,
                str(reason), parameters, self._label_simulation(simulation), period,
            )

        try:
            parameters = (parameters_from_vector(vector, self.template) if self.bounds is None
                          else self.bounds.validate(vector, self.template))
        except (TypeError, ValueError) as exc:
            return rejected(exc)
        try:
            simulation = simulate_interval_reservoir(
                self._preparation, parameters,
                settings=self.settings,
                atol_storage_m3=self.atol_storage_m3,
                start_date=self.simulation_period.start,
                end_date=period.end,
            )
        except ValueError as exc:
            return rejected(exc, parameters, getattr(exc, "partial_result", None))
        if not simulation.frame.index.equals(self._preparation.frame.index):
            return rejected("Candidate simulation changed the prepared calendar.", parameters, simulation)
        scored = simulation.frame.loc[frame.index]
        metrics = {"conditional_hindcast": True}
        for label, column, criteria in (
            ("release", "interval_mean_total_outflow_m3s", ("nse", "kge", "pbias", "rmse")),
            ("storage", "storage_m3", ("nse", "kge")),
        ):
            if column not in scored:
                return rejected(f"Candidate simulation lacks {column}.", parameters, simulation)
            try:
                predicted = pd.to_numeric(scored[column], errors="raise").to_numpy(dtype=float, na_value=np.nan)
            except (TypeError, ValueError, OverflowError) as exc:
                return rejected(exc, parameters, simulation)
            active = masks[label] & np.isfinite(observed[label])
            if not np.isfinite(predicted[active]).all():
                return rejected(f"Candidate {label} output is missing or nonfinite on an eligible observed pair.", parameters, simulation)
            for criterion in criteria:
                metrics[f"{label}_{criterion}"] = objective_metric(observed[label][active], predicted[active], criterion)
        costs = (-metrics["release_nse"], -metrics["storage_nse"])
        if not np.isfinite(costs).all():
            return rejected("Candidate release/storage NSE is undefined.", parameters, simulation)
        return CandidateEvaluation(
            True, costs, metrics, counts, None, parameters,
            self._label_simulation(simulation), period,
        )
