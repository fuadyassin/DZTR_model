"""Exploratory release priors from a mean-flow scale and external donor shapes."""
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.data_availability_v1 import RELEASE_COLUMNS
from workflows import reservoir_v2 as native
from workflows.atomic_io_v1 import write
from workflows.geometry_v1 import digest


def prepare(spec, index, selection, *, quantile_method, folder, pin, base, frame, metadata, reservoir_id, grand_id):
    required = {"source_kind", "donor_table", "grand_table", "mean_policy", "minimum_donors", "evidence", "release_target_limit"}
    if set(spec) != required or spec["source_kind"] != "mean_flow_scaled_donor_prior" or not spec["evidence"]:
        raise ValueError("Declare the complete mean-flow donor prior")
    if spec["mean_policy"] not in {"grand_only", "grand_then_calibration_inflow"}: raise ValueError("Declare mean-flow source/fallback policy")
    if spec["release_target_limit"] not in {"none", "source_channel"}: raise ValueError("Declare release-target limiting policy")
    if isinstance(spec["minimum_donors"], bool) or not isinstance(spec["minimum_donors"], int) or spec["minimum_donors"] < 3:
        raise ValueError("Use at least three external donors")
    donor_path, grand_path = [pin((Path(base) / spec[k]).resolve(strict=True)) for k in ("donor_table", "grand_table")]
    donors = pd.read_csv(donor_path, float_precision="round_trip")
    donors = donors[donors.quantile_method == quantile_method].copy()
    # One reservoir's own external simulation must not teach its shape prior.
    donors = donors[donors.reservoir_id != reservoir_id]
    if grand_id is not None: donors = donors[donors.GRAND_ID != grand_id]
    if donors.reservoir_id.duplicated().any() or len(donors) < spec["minimum_donors"]: raise ValueError("Insufficient unique donors after target exclusion")
    columns = ["p05_ratio", "p30_ratio", "p97_ratio", "p99_ratio"]
    ratios = donors[columns].to_numpy(float)
    if not np.isfinite(ratios).all() or (ratios < 0).any() or (np.diff(ratios, axis=1) < 0).any(): raise ValueError("Invalid ordered donor flow ratios")
    grand = pd.read_csv(grand_path, float_precision="round_trip")
    if grand.GRAND_ID.duplicated().any(): raise ValueError("Duplicate GRAND discharge rows")
    found = grand[grand.GRAND_ID == grand_id] if grand_id is not None else grand.iloc[:0]
    mean = None; mean_source = None; mean_receipt = None
    if len(found) == 1:
        row = found.iloc[0]
        raw = float(row.DIS_AVG_LS)
        if np.isfinite(raw) and raw > 0:
            mean = raw / 1000.; mean_source = "GRAND_DIS_AVG_LS"
            if not np.isclose(mean, float(row.mean_discharge_m3s), rtol=0, atol=1e-12): raise ValueError("GRAND L/s to m3/s conversion disagrees")
            mean_receipt = {"GRAND_ID": int(grand_id), "source_field": "DIS_AVG_LS", "source_value": raw, "source_units": "L/s", "conversion_factor": .001}
    if mean is None:
        if spec["mean_policy"] != "grand_then_calibration_inflow": raise ValueError("No positive mean discharge for a verified GRAND ID")
        # Volume-weighted mean over intervals wholly within calibration blocks.
        mask = selection.mask(index, "calibration"); valid = mask[1:] & mask[:-1]
        fixed = metadata["sampling"].get("fixed_step_days")
        dt = np.full(len(index) - 1, fixed * 86400.) if fixed is not None else np.diff(index.asi8) / 1e9
        rates = frame.inflow_m3s.to_numpy(float) * metadata["physical_constants"]["inflow_correction"]
        if not valid.any() or not np.isfinite(rates).all() or not (dt > 0).all(): raise ValueError("Invalid calibration inflow scale support")
        mean = float(np.sum((rates[1:] + rates[:-1])[valid] * .5 * dt[valid]) / np.sum(dt[valid]))
        if not np.isfinite(mean) or mean <= 0: raise ValueError("Mean-flow proxy must be positive; signed rates are not rectified")
        mean_source = "calibration_effective_inflow_proxy"
        mean_receipt = {"calibration_intervals": int(valid.sum()), "integrated_seconds": float(dt[valid].sum()),
            "uses_calibration_inflow_only": True, "signed_rates_retained": True, "is_GRAND_discharge": False}
    shape = np.median(ratios, axis=0); raw_values = shape * mean; values = raw_values.copy()
    channel = metadata["physical_constants"]["channel_capacity_m3s"]
    if spec["release_target_limit"] == "source_channel": values[:3] = np.minimum(values[:3], channel)
    targets = pd.DataFrame(np.tile(values[:3], (12, 1)), columns=RELEASE_COLUMNS, index=pd.Index(range(1, 13), name="month"))
    receipt = {"kind": "mean_flow_release_prior_v1", **spec, "scope": "external_donor_reference", "mean_source": mean_source,
        "mean_discharge_m3s": mean, "mean_source_receipt": mean_receipt, "donor_count": len(donors),
        "donor_reservoir_ids": donors.reservoir_id.tolist(), "target_excluded_from_donors": True,
        "shape_rule": "unweighted median of donor percentile/mean ratios", "release_percentiles": [5, 30, 97],
        "median_shape_ratios": shape.tolist(), "shape_ratio_p10": np.percentile(ratios, 10, axis=0).tolist(),
        "shape_ratio_p90": np.percentile(ratios, 90, axis=0).tolist(), "spread_is_confidence_interval": False,
        "raw_release_targets_m3s": raw_values[:3].tolist(), "release_targets_m3s": values[:3].tolist(),
        "target_levels_truncated": int((raw_values[:3] != values[:3]).sum()), "source_channel_capacity_m3s": channel,
        "channel_p99_proxy_m3s": float(raw_values[3]), "quantile_method": quantile_method, "units": "m3/s",
        "fitting_years": sorted(set(y for text in donors.reference_years for y in map(int, str(text).split(",")))),
        "observed_release_column_used": False, "independent_of_target_responses": False, "source_certified": False,
        "retrospective_external_climatology": True, "selection_signature": selection.signature,
        "donor_table_sha256": native.sha(donor_path), "grand_table_sha256": native.sha(grand_path)}
    receipt["signature"] = digest(receipt)
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    donors.to_csv(folder / "external_donors.csv", index=False, float_format="%.17g")
    targets.to_csv(folder / "monthly_release_prior_si.csv", float_format="%.17g")
    write(folder / "receipt.json", receipt)
    return targets, receipt
