"""Shared observations, parameters and reservoir simulation contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


MONTHLY_TARGET_COLUMNS = (
    "storage_critical_m3", "storage_normal_m3", "storage_flood_m3",
    "release_critical_m3s", "release_normal_m3s", "release_flood_m3s",
)


@dataclass(frozen=True)
class DataQuality:
    """Daily input quality; gaps are reported, never interpolated."""

    status: str
    row_count: int
    missing_counts: dict[str, int]
    missing_days: int
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReservoirObservations:
    """Daily observations indexed by date, with storage in m³ and flows in m³/s."""

    frame: pd.DataFrame
    quality: DataQuality
    source: str | None = None

    @property
    def is_placeholder(self) -> bool:
        return self.quality.status == "placeholder"


@dataclass(frozen=True)
class QuantileEstimates:
    """Tidy SI quantiles; month 0 denotes the full supplied record."""

    frame: pd.DataFrame
    source: str | None = None
    quantile_method: str = "linear"
    storage_unit: str = "m3"
    flow_unit: str = "m3/s"


@dataclass(frozen=True)
class GeneralizedParameters:
    """Six monthly targets, separate from future simulation initial conditions."""

    monthly_targets: pd.DataFrame
    channel_capacity_m3s: float
    target_percentiles: tuple[float, float, float] = (10, 45, 85)
    source: str | None = None
    quantile_method: str = "linear"
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReservoirParameters:
    """Physical reservoir configuration with six targets for each calendar month.

    This accepts estimated or imported targets in SI units. Initial conditions,
    capacity and inflow adjustments are explicit, independent of estimation.
    """

    monthly_targets: pd.DataFrame
    capacity_m3: float
    initial_storage_m3: float
    initial_release_m3s: float
    channel_capacity_m3s: float
    dead_storage_fraction: float = 0.1
    inflow_correction: float = 1.0
    flood_mode: str = "storage"

    def __post_init__(self) -> None:
        for name in ("capacity_m3", "initial_storage_m3", "initial_release_m3s", "channel_capacity_m3s", "dead_storage_fraction", "inflow_correction"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        if self.capacity_m3 <= 0:
            raise ValueError("capacity_m3 must be positive.")
        if not 0 <= self.initial_storage_m3 <= self.capacity_m3:
            raise ValueError("Initial storage must be between zero and reservoir capacity.")
        if self.initial_release_m3s < 0 or self.channel_capacity_m3s < 0:
            raise ValueError("Initial release and channel capacity must be nonnegative.")
        if not 0 <= self.dead_storage_fraction < 1:
            raise ValueError("dead_storage_fraction must be in [0, 1).")
        if self.inflow_correction < 0:
            raise ValueError("inflow_correction must be nonnegative.")
        if self.flood_mode not in ("storage", "inflow"):
            raise ValueError("flood_mode must be 'storage' or 'inflow'.")
        frame = self.monthly_targets
        if not isinstance(frame, pd.DataFrame) or not frame.index.is_unique or len(frame) != 12 or set(frame.index) != set(range(1, 13)):
            raise ValueError("Monthly targets must contain exactly months 1 through 12.")
        if frame.columns.has_duplicates:
            raise ValueError("Monthly target column names must be unique.")
        if set(MONTHLY_TARGET_COLUMNS) - set(frame.columns):
            raise ValueError("Monthly targets must contain all six SI target columns.")
        frame = frame.loc[list(range(1, 13)), list(MONTHLY_TARGET_COLUMNS)].astype(float).copy()
        frame.index = pd.Index(range(1, 13), name="month")
        values = frame.to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Monthly targets must be finite and nonnegative.")
        storage = values[:, :3]
        if (storage[:, 0] <= self.dead_storage_fraction * self.capacity_m3).any() or (np.diff(storage, axis=1) <= 0).any() or (storage[:, 2] > self.capacity_m3).any():
            raise ValueError("Storage targets must satisfy dead storage < critical < normal < flood <= capacity.")
        if (np.diff(values[:, 3:], axis=1) < 0).any():
            raise ValueError("Release targets must satisfy critical <= normal <= flood.")
        object.__setattr__(self, "monthly_targets", frame)

    @classmethod
    def from_generalized(
        cls,
        estimates: GeneralizedParameters,
        *,
        capacity_m3: float,
        initial_storage_m3: float,
        initial_release_m3s: float,
        dead_storage_fraction: float = 0.1,
        inflow_correction: float = 1.0,
        flood_mode: str = "storage",
    ) -> ReservoirParameters:
        return cls(
            estimates.monthly_targets, capacity_m3, initial_storage_m3,
            initial_release_m3s, estimates.channel_capacity_m3s,
            dead_storage_fraction, inflow_correction, flood_mode,
        )


@dataclass(frozen=True)
class SimulationResult:
    """Daily states, release and numerical diagnostics aligned to the inputs."""

    frame: pd.DataFrame
    parameters: ReservoirParameters
    solver: str
    overflow_mode: str
    issues: tuple[str, ...] = ()
