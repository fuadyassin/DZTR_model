"""Explicit naturalized-outflow priors shared by weak-data DZTR routes.

Never substitute observed regulated release or raw inflow for simulated
naturalized outflow. Etienne's crude-format observed column is ignored.
"""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from strategies.data_availability_v1 import RELEASE_COLUMNS, series_signature
from workflows import data_availability_v2 as availability
from workflows.atomic_io_v1 import write
from workflows.geometry_v1 import digest, validate_index


def read_source(path, format):
    path = Path(path)
    if format == "si_csv":
        frame = availability.read_frame(path)
        if list(frame.columns) != ["naturalized_outflow_m3s"]: raise ValueError("Naturalized CSV requires one explicitly named SI column")
        values = frame.iloc[:, 0]
    elif format == "etienne_crude":
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        if "sim flow" not in lines[0].lower() or "obs" not in lines[0].lower(): raise ValueError("Expected Etienne date/simulated/observed header")
        dates, data = [], []
        for line in lines[1:]:
            if not line.strip(): continue
            tokens = line.split()
            if len(tokens) != 3: raise ValueError("Expected three columns in Etienne naturalized source")
            dates.append(pd.to_datetime(tokens[0], format="%Y%m%d"))
            data.append(float(tokens[1]))  # Never parse the observed-response column.
        values = pd.Series(data, index=pd.DatetimeIndex(dates, name="date"), name="naturalized_outflow_m3s")
    else: raise ValueError("Choose si_csv or etienne_crude source format")
    validate_index(values.index)
    if len(values) < 2: raise ValueError("Naturalized source needs at least two dates")
    frequency = next((name for name, freq in (("daily", "D"), ("monthly", "MS"))
        if values.index.equals(pd.date_range(values.index[0], values.index[-1], freq=freq, name=values.index.name))), None)
    if frequency is None: raise ValueError("Naturalized source must have consecutive daily or month-start dates; do not interpolate missing dates")
    values = pd.to_numeric(values, errors="raise")
    if np.isinf(values.to_numpy()).any() or (values.dropna() < 0).any(): raise ValueError("Naturalized values must be nonnegative or missing")
    values.attrs["source_frequency"] = frequency
    return values


def prepare(spec, index, selection, *, quantile_method, folder, pin, base, frame=None, metadata=None, reservoir_id=None, grand_id=None):
    if spec.get("source_kind") == "mean_flow_scaled_donor_prior":
        from workflows.mean_flow_prior_v1 import prepare as mean_prior
        if frame is None or metadata is None or reservoir_id is None: raise ValueError("Mean-flow prior needs reservoir context")
        return mean_prior(spec, index, selection, quantile_method=quantile_method, folder=folder, pin=pin, base=base,
            frame=frame[["inflow_m3s"]], metadata=metadata, reservoir_id=reservoir_id, grand_id=grand_id)
    required = {"source_kind", "csv", "format", "evidence", "independent_of_target_responses", "scope", "multiplier"}
    optional = {"reference_period", "external_reference_rationale"}
    if set(spec) - required - optional or not required <= set(spec): raise ValueError("Declare the complete naturalized prior specification")
    if spec["source_kind"] != "simulated_naturalized_outflow" or not isinstance(spec["evidence"], str) or not spec["evidence"].strip():
        raise ValueError("Declare simulated naturalized outflow with source evidence")
    if not isinstance(spec["independent_of_target_responses"], bool): raise ValueError("Declare source dependency explicitly")
    multiplier = spec["multiplier"]
    if isinstance(multiplier, bool) or not np.isfinite(multiplier) or multiplier <= 0: raise ValueError("Prior multiplier must be finite and positive")
    if quantile_method not in {"hazen", "linear"}: raise ValueError("Declare percentile estimator")
    source = pin((Path(base) / spec["csv"]).resolve(strict=True))
    raw = read_source(source, spec["format"]); frequency = raw.attrs["source_frequency"]
    values = raw * multiplier
    mask = selection.mask(index, "calibration")
    if spec["scope"] == "calibration_overlap":
        if set(spec) & optional: raise ValueError("Reference-period fields require explicit external_reference scope")
        if index.to_period("M").is_unique or frequency == "monthly":
            lookup = pd.Series(mask, index=index.to_period("M")).groupby(level=0).all()
            fitted = lookup.reindex(values.index.to_period("M"), fill_value=False).to_numpy()
        else:
            fitted = pd.Series(mask, index=index).reindex(values.index, fill_value=False).to_numpy()
    elif spec["scope"] == "external_reference":
        period = spec.get("reference_period")
        if not isinstance(period, dict) or set(period) != {"start", "end"} or not spec.get("external_reference_rationale"):
            raise ValueError("External reference requires an explicit period and exploratory rationale")
        start, end = pd.Timestamp(period["start"]), pd.Timestamp(period["end"])
        if start > end or start < values.index[0] or end > values.index[-1]: raise ValueError("External reference period is not covered by the source")
        fitted = (values.index >= start) & (values.index <= end)
    else: raise ValueError("Choose calibration_overlap or explicit external_reference scope")
    sample = values.loc[fitted].dropna()
    minimum = 30 if frequency == "daily" else 12
    if len(sample) < minimum or set(sample.index.month) != set(range(1, 13)):
        raise ValueError(f"Naturalized prior needs at least {minimum} fitting samples covering all 12 months in the declared scope")
    quantiles = np.percentile(sample, [5, 30, 97, 99], method=quantile_method)
    targets = pd.DataFrame(np.tile(quantiles[:3], (12, 1)), columns=RELEASE_COLUMNS, index=pd.Index(range(1, 13), name="month"))
    receipt = {"kind": "naturalized_release_prior_v1", **spec, "source_csv": str(source),
        "source_sha256": availability.w.sha(source), "source_frequency": frequency, "units": "m3/s",
        "observed_release_column_used": False, "valid_zero_retained": True, "quantile_method": quantile_method,
        "release_percentiles": [5, 30, 97], "channel_p99_proxy_m3s": float(quantiles[3]),
        "fitting_samples": len(sample), "fitting_start": str(sample.index[0].date()), "fitting_end": str(sample.index[-1].date()),
        "fitting_years": sorted(set(map(int, sample.index.year))), "fitting_signature": series_signature(sample),
        "selection_signature": selection.signature, "source_certified": False,
        "retrospective_external_climatology": spec["scope"] == "external_reference"}
    receipt["signature"] = digest(receipt)
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, folder / ("source" + source.suffix))
    values.to_frame().to_csv(folder / "naturalized_outflow_si.csv", index_label="date", float_format="%.17g")
    targets.to_csv(folder / "monthly_release_prior_si.csv", float_format="%.17g")
    write(folder / "receipt.json", receipt)
    return targets, receipt
