"""Versioned, vectorized level-to-volume preparation with explicit references.

This module never fetches data, extends a curve, rescales volume or changes a
reservoir capacity. The legacy dztr_param API remains unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_index(index):
    if not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.tz is not None:
        raise ValueError("Use a timezone-naive DatetimeIndex without missing dates")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and increasing")


def checked_mask(index, mask):
    validate_index(index)
    if not isinstance(mask, pd.Series) or not mask.index.equals(index) or mask.dtype != bool:
        raise ValueError("Fitting mask must be a boolean Series on the exact input calendar")
    if not mask.any():
        raise ValueError("Fitting mask has no observations")
    return mask.to_numpy()


@dataclass(frozen=True)
class GeometryCurve:
    depth_m: tuple[float, ...]
    storage_m3: tuple[float, ...]
    area_m2: tuple[float, ...] | None
    source: str
    source_sha256: str
    kind: str = "surveyed_si"
    grand_id: int | None = None
    header_height_m: float | None = None
    shape: str | None = None

    def __post_init__(self):
        for name in ("depth_m", "storage_m3", "area_m2"):
            values = getattr(self, name)
            if values is not None:
                object.__setattr__(self, name, tuple(float(v) for v in values))
        if len(self.depth_m) < 2 or len(self.storage_m3) != len(self.depth_m):
            raise ValueError("Geometry needs at least two paired depth/storage coordinates")
        for name in ("depth_m", "storage_m3", "area_m2"):
            values = getattr(self, name)
            if values is None:
                continue
            if len(values) != len(self.depth_m) or not np.isfinite(values).all() or (np.array(values) < 0).any():
                raise ValueError("Curve coordinates must be paired, finite and nonnegative")
            if (np.diff(values) < 0).any():
                raise ValueError("Curve coordinates must be monotone")
        if (np.diff(self.depth_m) <= 0).any():
            raise ValueError("Geometry depth must be strictly increasing; duplicates are ambiguous")
        if self.storage_m3[-1] <= self.storage_m3[0]:
            raise ValueError("Geometry needs a positive storage range")
        if self.kind not in {"regeom", "surveyed_si"}:
            raise ValueError("Declare ReGeom or surveyed SI geometry")
        if not self.source or len(self.source_sha256) != 64:
            raise ValueError("Geometry needs source provenance and SHA256")
        if self.grand_id is not None and (isinstance(self.grand_id, bool) or int(self.grand_id) != self.grand_id or self.grand_id <= 0):
            raise ValueError("Geometry GRanD ID must be a positive integer or absent")
        if self.header_height_m is not None and (not np.isfinite(self.header_height_m) or self.header_height_m <= 0):
            raise ValueError("Missing/negative height metadata must be represented as None")

    @property
    def signature(self):
        payload = asdict(self)
        payload.pop("source")  # relocation does not change scientific geometry
        return digest(payload)

    def table(self):
        frame = pd.DataFrame({"depth_m": self.depth_m, "storage_m3": self.storage_m3})
        if self.area_m2 is not None:
            frame["area_m2"] = self.area_m2
        return frame

    def diagnostics(self, capacity_m3=None):
        area_integral = float(np.trapz(self.area_m2, self.depth_m)) if self.area_m2 is not None else None
        closure = (self.storage_m3[-1] - self.storage_m3[0]) / area_integral - 1 if area_integral else None
        if capacity_m3 is not None and (not np.isfinite(capacity_m3) or capacity_m3 <= 0):
            raise ValueError("Capacity must be positive and finite")
        return {"kind": self.kind, "geometry_signature": self.signature,
                "supported_depth_min_m": self.depth_m[0], "supported_depth_max_m": self.depth_m[-1],
                "curve_top_storage_m3": self.storage_m3[-1], "header_height_m": self.header_height_m,
                "height_metadata_missing": self.header_height_m is None,
                "area_integral_storage_relative_difference": closure,
                "area_flux_coupling_verified": False,
                "capacity_m3": capacity_m3,
                "curve_top_relative_capacity_difference": self.storage_m3[-1] / capacity_m3 - 1 if capacity_m3 else None,
                "approximate_geometry": self.kind == "regeom"}


def read_regeom(path, *, expected_grand_id=None):
    path = Path(path).resolve(strict=True)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    metadata = {}
    for line in lines:
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
    gid = int(metadata["GRanD ID"])
    if expected_grand_id is not None and gid != expected_grand_id:
        raise ValueError("Geometry ID does not match the verified mapping")
    start = next((i for i, line in enumerate(lines) if line.lstrip().startswith("Depth(m)")), None)
    if start is None:
        raise ValueError("ReGeom depth/area/storage table is missing")
    table = pd.read_csv(StringIO("\n".join(lines[start + 1:])), header=None)
    if table.shape[1] != 3:
        raise ValueError("ReGeom table must contain exactly depth, area and storage")
    values = table.to_numpy(dtype=float)
    height = float(metadata["Dam Height (m)"])
    return GeometryCurve(tuple(values[:, 0]), tuple(values[:, 2] * 1e6), tuple(values[:, 1] * 1e6),
                         str(path), file_sha(path), "regeom", gid,
                         height if np.isfinite(height) and height > 0 else None,
                         metadata.get("Approximated Geometry"))


def read_si_curve(path, *, source_kind, grand_id=None):
    path = Path(path).resolve(strict=True)
    frame = pd.read_csv(path, float_precision="round_trip")
    if list(frame.columns) not in (["depth_m", "storage_m3"], ["depth_m", "storage_m3", "area_m2"]):
        raise ValueError("Geometry CSV must declare its SI columns in depth/storage/optional-area order")
    return GeometryCurve(tuple(frame.depth_m), tuple(frame.storage_m3),
                         tuple(frame.area_m2) if "area_m2" in frame else None,
                         str(path), file_sha(path), source_kind, grand_id)


@dataclass(frozen=True)
class LevelReference:
    level_m: float
    depth_m: float
    basis: str
    evidence: str
    fitting_signature: str | None = None

    def __post_init__(self):
        if not np.isfinite([self.level_m, self.depth_m]).all() or self.depth_m < 0:
            raise ValueError("Reference coordinates must be finite with nonnegative curve depth")
        if self.basis not in {"source_reference", "calibration_maximum_assumption"} or not self.evidence:
            raise ValueError("Reference requires an explicit source or assumption")
        if self.basis == "calibration_maximum_assumption" and not self.fitting_signature:
            raise ValueError("An estimated reference must be tied to fitting years")


def calibration_maximum_reference(levels, curve, fit_mask, *, fitting_signature, allow_exploratory=False):
    if not allow_exploratory:
        raise ValueError("Calibration-maximum alignment requires explicit exploratory opt-in")
    values = pd.to_numeric(levels, errors="raise").to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Levels must be finite or missing")
    selected = values[checked_mask(levels.index, fit_mask)]
    if not np.isfinite(selected).any():
        raise ValueError("No calibration levels are available to estimate a reference")
    return LevelReference(float(np.nanmax(selected)), curve.depth_m[-1],
                          "calibration_maximum_assumption", "Calibration maximum assumed to be curve top; not a surveyed datum", fitting_signature)


def convert_levels(levels, curve, reference, *, time_basis, allow_mean_level_proxy=False):
    """Return SI estimates and fixed source-quality flags without extrapolation."""
    validate_index(levels.index)
    if time_basis not in {"point_level", "end_of_day_level", "daily_mean_level", "monthly_mean_level"}:
        raise ValueError("Declare the observed level time basis")
    proxy = time_basis in {"daily_mean_level", "monthly_mean_level"}
    if proxy and not allow_mean_level_proxy:
        raise ValueError("Transforming mean level into a storage proxy requires explicit opt-in")
    if not curve.depth_m[0] <= reference.depth_m <= curve.depth_m[-1]:
        raise ValueError("Reference depth is outside the supported geometry")
    values = pd.to_numeric(levels, errors="raise").to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Levels must be finite or missing")
    depth = reference.depth_m + (values - reference.level_m)
    supported = np.isfinite(depth) & (depth >= curve.depth_m[0]) & (depth <= curve.depth_m[-1])
    storage = np.full(len(depth), np.nan)
    area = np.full(len(depth), np.nan)
    storage[supported] = np.interp(depth[supported], curve.depth_m, curve.storage_m3)
    if curve.area_m2 is not None:
        area[supported] = np.interp(depth[supported], curve.depth_m, curve.area_m2)
    status = np.where(~np.isfinite(values), "missing_level", np.where(supported, "converted", "outside_geometry_domain"))
    frame = pd.DataFrame({"level_m": values, "depth_m": depth, "storage_m3": storage,
                          "area_m2": area, "conversion_status": status}, index=levels.index)
    receipt = {"geometry_signature": curve.signature, "reference": asdict(reference),
               "level_time_basis": time_basis, "storage_is_mean_level_proxy": proxy,
               "approximate_geometry": curve.kind == "regeom", "converted_rows": int(supported.sum()),
               "missing_level_rows": int(np.isnan(values).sum()),
               "out_of_domain_rows": int((np.isfinite(values) & ~supported).sum()),
               "extrapolation_applied": False, "capacity_rescaled": False,
               "source_certified": False}
    return frame, receipt
