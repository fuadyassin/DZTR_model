"""Explicit datum/depth alignment for approximate reservoir geometry.

An effective geometry zero fitted to storage is not a surveyed bed elevation.
No volume scaling, clipping, gap filling or extrapolation is performed here.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from workflows.geometry_v1 import (LevelReference, checked_mask, convert_levels,
                                   digest, validate_index)


@dataclass(frozen=True)
class DepthReference:
    zero_level_m: float
    level_datum: str
    reference_datum: str
    basis: str
    evidence: str
    level_to_reference_offset_m: float = 0.0
    datum_conversion_evidence: str | None = None
    fitting_signature: str | None = None
    uses_storage_observations: bool = False

    def __post_init__(self):
        if not np.isfinite([self.zero_level_m, self.level_to_reference_offset_m]).all():
            raise ValueError("Reference and datum offset must be finite")
        if not all(isinstance(x, str) and x.strip() for x in
                   (self.level_datum, self.reference_datum, self.evidence)):
            raise ValueError("Name both datums and provide reference evidence")
        bases = {"documented_bottom", "documented_level_depth", "assumed_full_supply_top",
                 "assumed_full_supply_capacity", "calibration_maximum", "calibration_storage_alignment"}
        if self.basis not in bases:
            raise ValueError("Unknown depth-reference method")
        if self.level_datum != self.reference_datum:
            if not self.datum_conversion_evidence:
                raise ValueError("Different datums require an explicit conversion and evidence")
        elif self.level_to_reference_offset_m != 0:
            raise ValueError("A conversion within the same named datum is ambiguous")
        fitted = self.basis.startswith("calibration_")
        if fitted != bool(self.fitting_signature):
            raise ValueError("Only calibration-estimated references require a fitting signature")
        if self.uses_storage_observations != (self.basis == "calibration_storage_alignment"):
            raise ValueError("Declare whether reference fitting uses storage observations")

    @property
    def signature(self):
        return digest(asdict(self))


def anchor_reference(level_m, depth_m, *, level_datum, anchor_datum, basis, evidence,
                     datum_offset_m=0., datum_evidence=None, fitting_signature=None):
    """Anchor level is in anchor_datum; offset converts reported levels to it."""
    if not np.isfinite([level_m, depth_m]).all() or depth_m < 0:
        raise ValueError("Anchor level/depth must be finite with nonnegative depth")
    return DepthReference(float(level_m - depth_m), level_datum, anchor_datum, basis, evidence,
                          float(datum_offset_m), datum_evidence, fitting_signature)


def _levels(levels):
    validate_index(levels.index)
    values = pd.to_numeric(levels, errors="raise").to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Levels must be finite or missing")
    return values


def inverse_depth(storage_values, curve):
    """Invert only uniquely identifiable volumes; rounded plateaus stay missing."""
    v = np.asarray(storage_values, dtype=float)
    volumes = np.asarray(curve.storage_m3)
    lower = np.searchsorted(volumes, v, side="left")
    upper = np.searchsorted(volumes, v, side="right")
    ambiguous = (upper - lower) > 1
    valid = np.isfinite(v) & (v >= volumes[0]) & (v <= volumes[-1]) & ~ambiguous
    result = np.full(v.shape, np.nan)
    result[valid] = np.interp(v[valid], volumes, curve.depth_m)
    return result, ambiguous


def fit_storage_reference(levels, storage, curve, fit_mask, *, fitting_signature,
                          level_datum, min_pairs=30):
    """Robust constant effective zero: median(h - inverse_geometry(V)), fit only.

    Rounded flat-volume segments are ambiguous only at their exact volume.
    Outside/ambiguous volumes are counted, never clamped into the inverse range.
    """
    h = _levels(levels)
    mask = checked_mask(levels.index, fit_mask)
    if not storage.index.equals(levels.index):
        raise ValueError("Storage and level require the exact same calendar")
    v = pd.to_numeric(storage, errors="raise").to_numpy(dtype=float)
    if np.isinf(v).any() or (v[np.isfinite(v)] < 0).any():
        raise ValueError("Storage must be nonnegative or missing")
    if not isinstance(min_pairs, int) or isinstance(min_pairs, bool) or min_pairs < 2:
        raise ValueError("At least two fitting pairs are required")
    paired = mask & np.isfinite(h) & np.isfinite(v)
    inverse, ambiguous = inverse_depth(v, curve)
    in_range = (v >= curve.storage_m3[0]) & (v <= curve.storage_m3[-1])
    valid = paired & np.isfinite(inverse)
    if valid.sum() < min_pairs:
        raise ValueError("Insufficient supported calibration level/storage pairs")
    depth = inverse[valid]
    zeros = h[valid] - depth
    reference = DepthReference(float(np.median(zeros)), level_datum, level_datum,
        "calibration_storage_alignment",
        "Median reported level minus inverse-geometry depth on calibration pairs; effective geometry zero, not surveyed bed",
        fitting_signature=fitting_signature, uses_storage_observations=True)
    receipt = {"fitting_method": "median_inverse_storage_offset", "fitting_signature": fitting_signature,
        "fitting_pairs": int(valid.sum()), "excluded_outside_curve_pairs": int((paired & ~in_range).sum()),
        "excluded_ambiguous_plateau_pairs": int((paired & ambiguous).sum()),
        "uses_storage_observations": True, "assessment_used_for_fitting": False,
        "effective_zero_quantiles_m": dict(zip(("p05", "p50", "p95"),
                                               np.quantile(zeros, [.05, .50, .95]).tolist())),
        "not_a_surveyed_bed_elevation": True, "geometry_shape_fitted": False}
    return reference, receipt


def convert(levels, curve, reference, *, time_basis, allow_mean_level_proxy=False):
    values = _levels(levels)
    translated = pd.Series(values + reference.level_to_reference_offset_m,
                           index=levels.index, name=levels.name)
    # Existing converter handles bounded SI interpolation and mean-level opt-in.
    anchor = LevelReference(reference.zero_level_m + curve.depth_m[0], curve.depth_m[0],
                            "source_reference", reference.evidence)
    frame, receipt = convert_levels(translated, curve, anchor, time_basis=time_basis,
                                    allow_mean_level_proxy=allow_mean_level_proxy)
    frame["reported_level_m"] = values
    receipt["reference"] = asdict(reference)
    receipt["reference_signature"] = reference.signature
    receipt["depth_equation"] = "reported_level + level_to_reference_offset - zero_level"
    receipt["depth_is_effective_geometry_coordinate"] = curve.kind == "regeom" or reference.basis != "documented_bottom"
    receipt["source_certified"] = False
    return frame, receipt
