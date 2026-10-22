"""Shared, calibration-only datum/reference and storage-uncertainty options."""
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from workflows.geometry_v1 import checked_mask
from workflows.level_reference_v1 import (DepthReference, anchor_reference, convert,
                                          fit_storage_reference, inverse_depth)


def prepare_storage(levels, supplied_storage, curve, fit_mask, selection_signature,
                    capacity_m3, reference_spec, uncertainty_spec, *, time_basis,
                    allow_mean_level_proxy):
    """Return center estimates and explicit scenario bounds without extrapolation.

    Offset quantiles describe calibration residual spread, not a confidence
    interval for a surveyed bottom or the whole geometry product.
    """
    mask = checked_mask(levels.index, fit_mask)
    spec = dict(reference_spec)
    method = spec.pop("method")
    datum = spec.pop("level_datum")
    fitting = {"assessment_used_for_fitting": False}
    if method == "calibration_storage_alignment":
        if set(spec) - {"min_pairs"}:
            raise ValueError("Unknown calibration-reference option")
        reference, fitting = fit_storage_reference(levels, supplied_storage, curve, fit_mask,
            fitting_signature=selection_signature, level_datum=datum, min_pairs=spec.get("min_pairs", 30))
    elif method == "documented_bottom":
        allowed = {"zero_level_m", "reference_datum", "evidence", "level_to_reference_offset_m", "datum_conversion_evidence"}
        if set(spec) - allowed:
            raise ValueError("Unknown documented-bottom option")
        reference = DepthReference(level_datum=datum, basis=method, **spec)
    else:
        allowed = {"anchor_level_m", "anchor_depth_m", "reference_datum", "datum_offset_m", "datum_evidence", "evidence", "allow_exploratory"}
        if set(spec) - allowed:
            raise ValueError("Unknown anchor-reference option")
        exploratory = method in {"full_supply_capacity", "full_supply_top", "calibration_maximum"}
        if exploratory and spec.get("allow_exploratory") is not True:
            raise ValueError("Assumed depth references require explicit exploratory opt-in")
        basis = method
        signature = None
        if method == "documented_level_depth":
            level, depth = spec["anchor_level_m"], spec["anchor_depth_m"]
        elif method in {"full_supply_capacity", "full_supply_top"}:
            level = spec["anchor_level_m"]
            if method == "full_supply_capacity":
                depths, _ = inverse_depth([capacity_m3], curve)
                if not np.isfinite(depths[0]):
                    raise ValueError("Capacity has no unique supported geometry depth")
                depth, basis = float(depths[0]), "assumed_full_supply_capacity"
            else:
                depth, basis = curve.depth_m[-1], "assumed_full_supply_top"
        elif method == "calibration_maximum":
            level = float(levels.iloc[np.flatnonzero(mask)].max())
            depth, signature = curve.depth_m[-1], selection_signature
        else:
            raise ValueError("Unknown level-reference method")
        reference = anchor_reference(level, depth, level_datum=datum,
            anchor_datum=spec.get("reference_datum", datum), basis=basis, evidence=spec["evidence"],
            datum_offset_m=spec.get("datum_offset_m", 0.), datum_evidence=spec.get("datum_evidence"),
            fitting_signature=signature)
        fitting.update(anchor_level_m=level, anchor_depth_m=depth)
    center, receipt = convert(levels, curve, reference, time_basis=time_basis,
                              allow_mean_level_proxy=allow_mean_level_proxy)
    uncertainty = dict(uncertainty_spec)
    kind = uncertainty.pop("kind")
    zeros = [reference.zero_level_m, reference.zero_level_m]
    evidence = "No uncertainty adjustment; storage proxy treated as a point target"
    if kind == "none":
        if uncertainty: raise ValueError("Point-proxy uncertainty has no additional options")
    elif kind == "fixed_offset_band":
        if set(uncertainty) != {"half_width_m", "evidence"} or not uncertainty["evidence"]:
            raise ValueError("Fixed offset band needs half_width_m and evidence")
        width = uncertainty["half_width_m"]
        if isinstance(width, bool) or not np.isfinite(width) or width < 0:
            raise ValueError("Reference half-width must be finite and nonnegative")
        zeros = [reference.zero_level_m - width, reference.zero_level_m + width]
        evidence = uncertainty["evidence"]
    elif kind == "calibration_offset_quantiles":
        if set(uncertainty) != {"probabilities"} or not reference.uses_storage_observations:
            raise ValueError("Offset quantiles need a calibration storage-alignment reference")
        probabilities = uncertainty["probabilities"]
        if len(probabilities) != 2 or not 0 < probabilities[0] < .5 < probabilities[1] < 1:
            raise ValueError("Declare two quantiles straddling the median")
        depth, _ = inverse_depth(supplied_storage.to_numpy(dtype=float), curve)
        offsets = levels.to_numpy(dtype=float) - depth
        selected = offsets[mask & np.isfinite(offsets)]
        zeros = np.quantile(selected, probabilities).tolist()
        evidence = "Calibration equivalent-zero residual quantiles; includes geometry/timing disagreement, not a surveyed confidence interval"
    else:
        raise ValueError("Unknown storage-uncertainty option")
    lower, _ = convert(levels, curve, replace(reference, zero_level_m=zeros[1]),
                        time_basis=time_basis, allow_mean_level_proxy=allow_mean_level_proxy)
    upper, _ = convert(levels, curve, replace(reference, zero_level_m=zeros[0]),
                        time_basis=time_basis, allow_mean_level_proxy=allow_mean_level_proxy)
    bands = pd.DataFrame({"storage_lower_m3": lower.storage_m3, "storage_upper_m3": upper.storage_m3}, index=levels.index)
    receipt.update(reference_method=method, fitting=fitting,
        uncertainty={"kind": kind, **uncertainty, "effective_zero_bounds_m": zeros, "evidence": evidence,
                     "confidence_interval": False, "assessment_used_for_fitting": False,
                     "selection_signature": selection_signature if kind == "calibration_offset_quantiles" else None},
        reference_is_surveyed_bed=method == "documented_bottom" and curve.kind == "surveyed_si")
    return center, receipt, bands
