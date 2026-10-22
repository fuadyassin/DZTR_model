"""Calibration-only empirical bounds, separate from physical feasibility."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dztr.parameterization import estimate_percentiles
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirObservations, ReservoirParameters

from .candidates import _validated_vector, parameters_from_vector
from .periods import DatePeriod, select_period


DEFAULT_PERCENTILE_BOUNDS = ((5.0, 35.0), (35.0, 75.0), (75.0, 95.0))
BOUND_COLUMNS = tuple(f"{side}_{column}" for side in ("lower", "upper") for column in MONTHLY_TARGET_COLUMNS)


@dataclass(frozen=True)
class CalibrationBounds:
    """Raw monthly SI percentile limits and their fitting provenance.

    The empirical box is intentionally not clipped or reordered to enforce
    reservoir physics. Some points inside it can violate strict storage order,
    dead storage, capacity, or release order. ``contains`` checks only the box;
    ``validate`` additionally enforces the template's physical constraints.
    """

    monthly_bounds: pd.DataFrame
    capacity_m3: float
    dead_storage_fraction: float
    period: DatePeriod
    quantile_method: str = "hazen"
    percentile_bounds: tuple[tuple[float, float], ...] = DEFAULT_PERCENTILE_BOUNDS
    source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.period, DatePeriod):
            raise TypeError("period must be a DatePeriod.")
        if not np.isfinite(self.capacity_m3) or self.capacity_m3 <= 0:
            raise ValueError("Bounds capacity_m3 must be positive and finite.")
        if not np.isfinite(self.dead_storage_fraction) or not 0 <= self.dead_storage_fraction < 1:
            raise ValueError("Bounds dead_storage_fraction must be finite and in [0, 1).")
        if self.quantile_method not in ("linear", "hazen"):
            raise ValueError("quantile_method must be 'linear' or 'hazen'.")
        try:
            percentiles = tuple(tuple(float(value) for value in pair) for pair in self.percentile_bounds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("percentile_bounds must contain three finite lower/upper percentile pairs.") from exc
        if len(percentiles) != 3 or any(len(pair) != 2 or not all(np.isfinite(pair)) or not 0 <= pair[0] <= pair[1] <= 100 for pair in percentiles):
            raise ValueError("percentile_bounds must contain three ordered pairs within [0, 100].")
        frame = self.monthly_bounds
        if not isinstance(frame, pd.DataFrame) or len(frame) != 12 or not frame.index.is_unique or set(frame.index) != set(range(1, 13)):
            raise ValueError("Monthly bounds must contain exactly months 1 through 12.")
        if frame.columns.has_duplicates or set(BOUND_COLUMNS) - set(frame.columns):
            raise ValueError("Monthly bounds require twelve unique lower_/upper_ SI target columns.")
        try:
            frame = frame.loc[list(range(1, 13)), list(BOUND_COLUMNS)].astype(float).copy()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Monthly bounds must be finite nonnegative numbers.") from exc
        frame.index = pd.Index(range(1, 13), name="month")
        values = frame.to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Monthly bounds must be finite and nonnegative; every storage/release month requires observations.")
        lower, upper = values[:, :6], values[:, 6:]
        if (lower > upper).any():
            raise ValueError("Each monthly lower bound must be less than or equal to its upper bound.")
        dead = self.dead_storage_fraction * self.capacity_m3
        for i in range(12):
            storage = np.empty(3)
            preceding = dead
            for target in range(3):
                storage[target] = max(lower[i, target], np.nextafter(preceding, np.inf))
                preceding = storage[target]
            release = np.maximum.accumulate(lower[i, 3:])
            if (storage > upper[i, :3]).any() or storage[-1] > self.capacity_m3 or (release > upper[i, 3:]).any():
                raise ValueError(
                    f"Calibration bounds contain no physically feasible candidate in month {i + 1}; "
                    "check dead storage, capacity, strict storage ordering, and release ordering."
                )
        object.__setattr__(self, "monthly_bounds", frame)
        object.__setattr__(self, "percentile_bounds", percentiles)

    def _native_vector(self, side: str) -> np.ndarray:
        values = self.monthly_bounds.loc[list(range(1, 13)), [f"{side}_{column}" for column in MONTHLY_TARGET_COLUMNS]].to_numpy(copy=True)
        values[:, :3] /= self.capacity_m3
        return values.T.reshape(72).copy()

    @property
    def lower_vector(self) -> np.ndarray:
        """Fresh lower limits in native six-block mixed-unit order."""
        return self._native_vector("lower")

    @property
    def upper_vector(self) -> np.ndarray:
        """Fresh upper limits in native six-block mixed-unit order."""
        return self._native_vector("upper")

    def contains(self, vector) -> bool:
        """Check finite shape and box membership only, with inclusive limits."""
        try:
            values = _validated_vector(vector)
        except ValueError:
            return False
        return bool(((values >= self.lower_vector) & (values <= self.upper_vector)).all())

    def validate(self, vector, template: ReservoirParameters) -> ReservoirParameters:
        """Check matching configuration, empirical box, and reservoir physics."""
        if not isinstance(template, ReservoirParameters):
            raise TypeError("template must be ReservoirParameters.")
        if template.capacity_m3 != self.capacity_m3 or template.dead_storage_fraction != self.dead_storage_fraction:
            raise ValueError("Template capacity_m3 and dead_storage_fraction must match the calibration bounds.")
        values = _validated_vector(vector)
        if not self.contains(values):
            raise ValueError("Candidate vector lies outside the calibration percentile bounds.")
        return parameters_from_vector(values, template)

    def summary(self) -> dict:
        """Portable provenance, without claiming the entire box is feasible."""
        return {
            "estimation_period": self.period.as_dict(),
            "source": self.source,
            "quantile_method": self.quantile_method,
            "percentile_bounds": {
                level: {"lower": lower, "upper": upper}
                for level, (lower, upper) in zip(("critical", "normal", "flood"), self.percentile_bounds)
            },
            "capacity_m3": float(self.capacity_m3),
            "dead_storage_fraction": float(self.dead_storage_fraction),
            "vector_order": ["Sc/capacity (Jan-Dec)", "Sn/capacity (Jan-Dec)", "Sf/capacity (Jan-Dec)", "Qc (Jan-Dec)", "Qn (Jan-Dec)", "Qf (Jan-Dec)"],
            "storage_bounds_unit": "m3",
            "release_bounds_unit": "m3/s",
            "notes": ["Raw percentile limits are unchanged; candidate validation separately enforces reservoir physics.", "Release bounds are not capped by channel capacity; DZTR applies that cap only in the emergency zone."],
        }


def estimate_calibration_bounds(
    observations: ReservoirObservations,
    period: DatePeriod,
    template: ReservoirParameters,
    *,
    quantile_method: str = "hazen",
) -> CalibrationBounds:
    """Fit P5-35/P35-75/P75-95 storage and release bounds on selected dates.

    Period selection is performed here, rather than delegated to callers, so
    spinup and validation observations never enter empirical bounds fitting.
    Missing months and physically empty empirical boxes raise clear errors.
    """
    if not isinstance(template, ReservoirParameters):
        raise TypeError("template must be ReservoirParameters.")
    fitting = select_period(observations, period)
    probabilities = tuple(sorted({value for pair in DEFAULT_PERCENTILE_BOUNDS for value in pair}))
    quantiles = estimate_percentiles(fitting, probabilities, quantile_method=quantile_method)
    lookup = quantiles.frame.set_index(["variable", "month", "percentile"])
    monthly = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
    for variable, stem, suffix in (("storage_m3", "storage", "m3"), ("release_m3s", "release", "m3s")):
        for level, pair in zip(("critical", "normal", "flood"), DEFAULT_PERCENTILE_BOUNDS):
            for side, percentile in zip(("lower", "upper"), pair):
                monthly[f"{side}_{stem}_{level}_{suffix}"] = [float(lookup.loc[(variable, month, percentile), "value"]) for month in range(1, 13)]
    missing = monthly.isna().any(axis=1)
    if missing.any():
        raise ValueError(f"Storage and release observations are required in every calibration month; missing bounds in months {monthly.index[missing].tolist()}.")
    return CalibrationBounds(
        monthly, template.capacity_m3, template.dead_storage_fraction, period,
        quantile_method=quantile_method, source=fitting.source,
    )
