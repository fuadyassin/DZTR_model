"""Configured, checkpointed datum/geometry pilot using the existing DZTR solver.

This is a conditional parameterization experiment, not a new optimizer campaign.
Frozen observations remain the assessment reference and initial state is fixed.
"""
from dataclasses import asdict, replace
from pathlib import Path
import os
import re

import numpy as np
import pandas as pd

from dztr.evaluation.metrics import objective_metric
from dztr.schemas import ReservoirParameters
from strategies.data_availability_v1 import build_targets
from strategies.representative_years import YearSelection
from workflows import profile_application_v1 as app
from workflows import reservoir_v2 as w
from workflows.geometry_v1 import digest, read_regeom, validate_index
from workflows.level_reference_v1 import (DepthReference, anchor_reference, convert,
                                          fit_storage_reference)
from workflows.observation_operators_v1 import score_native

ROOT = w.ROOT
METHODS = ("full_supply_top", "full_supply_capacity", "calibration_maximum", "calibration_storage_alignment")


def read_frame(path):
    frame = pd.read_csv(path, index_col="date", parse_dates=True, float_precision="round_trip")
    validate_index(frame.index)
    return frame


def plan(config_path):
    path = Path(config_path).resolve(strict=True)
    cfg = w.read(path)
    fields = {"kind", "schema_version", "reservoir_id", "GRAND_ID", "inputs", "level_datum",
              "station_number", "full_supply_level_m", "full_supply_evidence", "level_time_basis",
              "allow_mean_level_proxy", "preferred_reference", "datum_metadata", "sensitivity_offsets_m"}
    if set(cfg) != fields or cfg["kind"] != "dztr_level_storage_pilot_v1" or cfg["schema_version"] != 1:
        raise ValueError("Invalid level-storage pilot configuration")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", cfg["reservoir_id"]):
        raise ValueError("Unsafe reservoir identifier")
    if cfg["preferred_reference"] not in METHODS:
        raise ValueError("Choose the reference policy before assessment")
    required = {"observations", "selection", "model", "geometry", "levels", "acquisition_provenance",
                "metadata_receipt", "station_metadata", "station_datum", "operator_full_supply",
                "cached_baseline_receipt", "cached_baseline_targets"}
    if set(cfg["inputs"]) != required:
        raise ValueError("Declare every input, metadata and cached-baseline source")
    paths = {k: app.reference(v) for k, v in cfg["inputs"].items()}
    pinned = {str(p): w.sha(p) for p in [path, *paths.values()]}
    def pin(p, expected=None):
        p = Path(p).resolve(strict=True)
        value = w.sha(p)
        if expected is not None and value != expected:
            raise ValueError("Cached provider/benchmark input changed: " + str(p))
        pinned[str(p)] = value
    provenance = w.read(paths["acquisition_provenance"])
    if provenance["configuration"]["level_stations"] != [cfg["station_number"]] or provenance["time_basis"] != "daily_mean":
        raise ValueError("Level acquisition station/time basis changed")
    if cfg["level_time_basis"] != "daily_mean_level" or cfg["allow_mean_level_proxy"] is not True:
        raise ValueError("Historical daily means need explicit mean-level-proxy consent")
    for source in provenance["sources"].values():
        for name, expected in source["page_sha256"].items():
            matches = list((paths["acquisition_provenance"].parent / "raw").glob("*/" + name))
            if len(matches) != 1:
                raise ValueError("Provider raw-page path is ambiguous")
            pin(matches[0], expected)
    meta = w.read(paths["station_metadata"])
    props = meta["features"][0]["properties"]
    if len(meta["features"]) != 1 or props["STATION_NUMBER"] != cfg["station_number"] or props["VERTICAL_DATUM"] != cfg["level_datum"]:
        raise ValueError("Station datum/identity disagrees with archived metadata")
    for name, record in w.read(paths["metadata_receipt"])["files"].items():
        pin(paths["metadata_receipt"].parent / name, record["sha256"])
    baseline = w.read(paths["cached_baseline_receipt"])
    if baseline["status"] != "completed":
        raise ValueError("Cached observed baseline did not complete")
    for name, expected in baseline["artifact_sha256"].items():
        pin(paths["cached_baseline_receipt"].parent / name, expected)
    old_common = paths["cached_baseline_targets"].parent.parent
    for p in (paths["cached_baseline_targets"].with_name("profile.json"),
              old_common / "observations_si.csv", old_common / "selection.json",
              old_common / "numerical_profile.json", old_common / "preprocessing.json"):
        pin(p)
    frame = read_frame(paths["observations"])
    selection = YearSelection(**w.read(paths["selection"]))
    selection.mask(frame.index, "calibration")
    levels = read_frame(paths["levels"])
    if list(levels.columns) != ["level_m"] or not levels.index.equals(frame.index):
        raise ValueError("Level record must match the frozen observation calendar")
    curve = read_regeom(paths["geometry"], expected_grand_id=cfg["GRAND_ID"])
    offsets = np.array(cfg["sensitivity_offsets_m"], dtype=float)
    if offsets.ndim != 1 or not np.isfinite(offsets).all() or len(set(offsets)) != len(offsets):
        raise ValueError("Sensitivity offsets must be unique finite values")
    code = {**w.source_hashes(), **{name: w.sha(ROOT / name) for name in (
        "workflows/level_reference_v1.py", "workflows/level_storage_pilot_v1.py",
        "workflows/geometry_v1.py", "strategies/data_availability_v1.py",
        "workflows/observation_operators_v1.py", "workflows/profile_application_v1.py",
        "tools/dztr_run_v4.py")}}
    identity = {"config": cfg, "input_sha256": pinned, "source_sha256": code}
    return {"kind": "dztr_level_storage_plan_v1", **identity, "signature": digest(identity),
            "selection_signature": selection.signature, "geometry_signature": curve.signature,
            "stages": ["verify_inputs_and_datum", "align_and_convert", "estimate_monthly_targets",
                       "fixed_profile_simulations", "assess_and_report"],
            "new_optimizer_evaluations": 0, "assessment_selects_reference": False}


def references(cfg, frame, levels, curve, selection, capacity):
    datum = cfg["level_datum"]
    fit_mask = pd.Series(selection.mask(frame.index, "calibration"), index=frame.index)
    hmax = float(levels.loc[fit_mask].max())
    if not np.isfinite(hmax):
        raise ValueError("No calibration level observations")
    if not curve.storage_m3[0] <= capacity <= curve.storage_m3[-1]:
        raise ValueError("Adopted capacity is outside the supported geometry volume range")
    depth_at_capacity = float(np.interp(capacity, curve.storage_m3, curve.depth_m))
    anchors = {}
    for method, h, depth, basis, evidence, signature in (
        ("full_supply_top", cfg["full_supply_level_m"], curve.depth_m[-1], "assumed_full_supply_top",
         cfg["full_supply_evidence"] + "; assume published FSL uses gauge datum and equals actual table top, not surveyed", None),
        ("full_supply_capacity", cfg["full_supply_level_m"], depth_at_capacity, "assumed_full_supply_capacity",
         cfg["full_supply_evidence"] + "; assume FSL uses gauge datum and corresponds to adopted model capacity; no curve rescaling", None),
        ("calibration_maximum", hmax, curve.depth_m[-1], "calibration_maximum",
         "Calibration maximum assumed to equal actual table top; not full-record maximum or surveyed datum", selection.signature)):
        ref = anchor_reference(h, depth, level_datum=datum, anchor_datum=datum, basis=basis,
                               evidence=evidence, fitting_signature=signature)
        anchors[method] = ref, {"anchor_level_m": h, "anchor_depth_m": depth,
                              "assessment_used_for_fitting": False}
    anchors["calibration_storage_alignment"] = fit_storage_reference(levels, frame.storage_m3, curve,
        fit_mask, fitting_signature=selection.signature, level_datum=datum)
    return anchors


def metrics(observed, estimated, mask, *, capacity):
    x, y = np.asarray(observed, dtype=float), np.asarray(estimated, dtype=float)
    support = np.asarray(mask, dtype=bool) & np.isfinite(x)
    supported = support & np.isfinite(y)
    xs, ys = x[supported], y[supported]
    complete = bool(np.isfinite(y[support]).all()) and bool(support.any())
    result = {"reference_rows": int(support.sum()), "converted_pairs": int(supported.sum()),
              "missing_estimates_on_fixed_support": int((support & ~supported).sum()),
              "fixed_support_complete": complete,
              "conversion_coverage": float(supported.sum() / support.sum()) if support.any() else None,
              "bias_m3": float(np.mean(ys - xs)) if len(xs) else None,
              "above_capacity_rows": int((np.asarray(mask) & np.isfinite(y) & (y > capacity)).sum())}
    for metric in ("nse", "rmse", "pbias"):
        value = objective_metric(xs, ys, metric)
        result["supported_" + metric] = value
        result[metric] = value if complete else None
    return result


def _csv(frame, path):
    frame.to_csv(path, index_label=frame.index.name or "date", float_format="%.17g")


def run(config_path, output, *, resume=False):
    design = plan(config_path)
    out = Path(output).resolve()
    if out.exists():
        if not resume or w.read(out / "plan.json")["signature"] != design["signature"]:
            raise ValueError("Use a fresh output or resume the identical pilot")
        if (out / "manifest.json").exists():
            verify(out)
            return {"status": "reused", "output": str(out), "new_fixed_profile_simulations": 0,
                    "new_optimizer_evaluations": 0}
    else:
        out = w.fresh_output(out)
        w.write(out / "plan.json", design)
    lock = out / "run.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    try:
        return _run(design, out)
    finally:
        lock.unlink()


def _run(design, out):
    cfg = design["config"]
    paths = {k: app.reference(v) for k, v in cfg["inputs"].items()}
    frame = read_frame(paths["observations"])
    levels = read_frame(paths["levels"]).level_m
    selection = YearSelection(**w.read(paths["selection"]))
    curve = read_regeom(paths["geometry"], expected_grand_id=cfg["GRAND_ID"])
    model = w.read(paths["model"])
    physical = model["model"]; capacity = physical["capacity_m3"]
    metadata = {"physical_constants": physical, "sampling": model["sampling"],
        "root_policy": model["root_policy"], "atol_storage_m3": model["atol_storage_m3"],
        "initial_date": frame.index[0].date().isoformat(), "route": "standard_native_v1"}
    roles = pd.Series(selection.roles, index=frame.index)
    fit = roles.eq("calibration")
    stop = np.flatnonzero(roles.isin(["calibration", "assessment"]))[-1] + 1
    refs = references(cfg, frame, levels, curve, selection, capacity)
    conversion_rows, seasonal, annual, quality_rows, baseline_rows, profiles, new_calls = [], [], [], [], [], [], 0
    monthly_estimates = pd.DataFrame({"provided_storage_m3": frame.storage_m3})
    preferred_conversion = None
    # Raw provider symbols are preserved in a convenient daily table as well as raw JSON.
    raw = []
    for page in sorted((paths["acquisition_provenance"].parent / "raw").glob("*/page_*.json")):
        raw.extend(item["properties"] for item in w.read(page)["features"])
    flags = pd.DataFrame(raw)
    flags.to_csv(out / "provider_daily_records.csv", index=False, float_format="%.17g")
    quality = pd.Series(flags.LEVEL_SYMBOL_EN.fillna("unflagged").to_numpy(),
                        index=pd.DatetimeIndex(pd.to_datetime(flags.DATE)), name="level_quality").reindex(frame.index)
    if not quality.index.is_unique or quality.isna().any():
        raise ValueError("Provider quality calendar is incomplete/ambiguous")
    for method in ("provided_storage", *METHODS):
        folder = out / method; folder.mkdir(exist_ok=True)
        working = frame.copy()
        reference = None
        if method != "provided_storage":
            reference, fitting = refs[method]
            conversion, receipt = convert(levels, curve, reference, time_basis=cfg["level_time_basis"],
                                         allow_mean_level_proxy=cfg["allow_mean_level_proxy"])
            _csv(conversion, folder / "level_conversion_si.csv")
            w.write(folder / "reference_and_conversion.json", {**receipt, "fitting": fitting})
            working.storage_m3 = conversion.storage_m3
            monthly_estimates[method] = conversion.storage_m3
            if method == cfg["preferred_reference"]:
                preferred_conversion = conversion
            for phase in ("calibration", "assessment"):
                conversion_rows.append({"method": method, "phase": phase,
                    **metrics(frame.storage_m3, conversion.storage_m3, roles.eq(phase), capacity=capacity)})
                for symbol in sorted(quality.unique()):
                    quality_rows.append({"method": method, "phase": phase, "provider_symbol": symbol,
                        **metrics(frame.storage_m3, conversion.storage_m3,
                                  roles.eq(phase) & quality.eq(symbol), capacity=capacity)})
                for month in range(1, 13):
                    seasonal.append({"method": method, "phase": phase, "month": month,
                        **metrics(frame.storage_m3, conversion.storage_m3,
                                  roles.eq(phase) & (frame.index.month == month), capacity=capacity)})
            for year in sorted(set(frame.index.year)):
                mask = frame.index.year == year
                annual.append({"method": method, "year": int(year), "role": roles.loc[mask].iloc[0],
                    **metrics(frame.storage_m3, conversion.storage_m3, mask, capacity=capacity)})
        recipe = build_targets("observed_monthly" if reference is None else "level_geometry_monthly",
            working, fit, capacity_m3=capacity, channel_capacity_m3s=physical["channel_capacity_m3s"],
            storage_source_kind="provided_storage" if reference is None else "level_geometry", quantile_method="hazen")
        targets = recipe.pop("targets")
        record = {"method": method, "reference": asdict(reference) if reference else None,
                  "selection_signature": selection.signature, "recipe": recipe,
                  "calibration_ready": False, "new_optimizer_evaluations": 0,
                  "storage_is_mean_level_proxy": reference is not None,
                  "source_certified": False, "parameters_valid": False}
        if targets is not None:
            targets.to_csv(folder / "monthly_targets_si.csv", float_format="%.17g")
        parameters = None
        try:
            if not recipe["target_estimation_ready"]:
                raise ValueError("; ".join(recipe["reasons"]))
            parameters = ReservoirParameters(targets, **physical)
            record["parameters_valid"] = True
        except ValueError as exc:
            record["parameter_error"] = str(exc)
        w.write(folder / "profile.json", record)
        profiles.append(record)
        receipt_path = folder / "simulation_receipt.json"
        identity = digest({"pilot_signature": design["signature"], "method": method})
        if receipt_path.exists():
            job = w.read(receipt_path)
            if job["identity"] != identity:
                raise ValueError("Cached pilot job has a different identity")
            for name, expected in job.get("artifact_sha256", {}).items():
                if w.sha(folder / name) != expected:
                    raise ValueError("Cached pilot simulation changed")
        else:
            job = {"identity": identity, "status": "parameter_invalid", "artifact_sha256": {},
                   "new_optimizer_evaluations": 0, "new_simulation_calls": 0}
            if method == "provided_storage":
                cached_targets = pd.read_csv(paths["cached_baseline_targets"], index_col="month", float_precision="round_trip")
                pd.testing.assert_frame_equal(targets, cached_targets)
                source = paths["cached_baseline_receipt"].parent
                simulation = read_frame(source / "simulation.csv")
                # Recheck current prepared forcing, metadata and targets against the cache's pinned recipe.
                old_profile = w.read(paths["cached_baseline_targets"].with_name("profile.json"))
                if old_profile["identity"]["physical_constants"] != physical or old_profile["identity"]["numerical_sampling"] != model["sampling"] or old_profile["identity"]["root_policy"] != model["root_policy"]:
                    raise ValueError("Cached baseline physical/numerical settings changed")
                old_common = paths["cached_baseline_targets"].parent.parent
                if (w.read(old_common / "preprocessing.json") != model["inflow_preprocessing"] or
                    w.read(old_common / "numerical_profile.json")["atol_storage_m3"] != model["atol_storage_m3"] or
                    old_profile["identity"]["quantile_method"] != "hazen" or
                    old_profile["reservoir_id"] != cfg["reservoir_id"]):
                    raise ValueError("Cached baseline source/percentile/tolerance settings changed")
                pd.testing.assert_frame_equal(frame, read_frame(old_common / "observations_si.csv"))
                if YearSelection(**w.read(old_common / "selection.json")).signature != selection.signature:
                    raise ValueError("Cached baseline selection changed")
                balance = w.independent_balance(simulation, physical["inflow_correction"])
                if balance > model["atol_storage_m3"]:
                    raise ValueError("Cached baseline balance check failed")
                _csv(simulation, folder / "simulation.csv")
                scores = score_native(frame.loc[simulation.index], simulation, roles.loc[simulation.index])
                w.write(folder / "metrics.json", scores)
                job.update(status="completed", original_receipt=str(paths["cached_baseline_receipt"]),
                           reused_verified_baseline=True, maximum_balance_residual_m3=balance)
            elif parameters is not None:
                job["new_simulation_calls"] = 1
                new_calls += 1
                try:
                    result, balance = app.run_frame(frame.iloc[:stop], metadata, parameters,
                        forcing_stage="prepared", preprocessing=model["inflow_preprocessing"])
                    _csv(result.frame, folder / "simulation.csv")
                    scores = score_native(frame.iloc[:stop], result.frame, roles.iloc[:stop])
                    w.write(folder / "metrics.json", scores)
                    job.update(status="completed", maximum_balance_residual_m3=balance)
                except (ValueError, RuntimeError) as exc:
                    job.update(status="solver_failed", reason=str(exc))
            if job["status"] == "completed":
                job["artifact_sha256"] = {name: w.sha(folder / name) for name in ("simulation.csv", "metrics.json")}
            w.write(receipt_path, job)
        if job["status"] == "completed":
            for row in w.read(folder / "metrics.json"):
                baseline_rows.append({"method": method, **row, "simulation_status": job["status"],
                                      "maximum_balance_residual_m3": job["maximum_balance_residual_m3"]})
        else:
            baseline_rows.append({"method": method, "simulation_status": job["status"], "reason": job.get("reason", record.get("parameter_error"))})
    pd.DataFrame(conversion_rows).to_csv(out / "conversion_metrics.csv", index=False)
    pd.DataFrame(seasonal).to_csv(out / "seasonal_conversion_bias.csv", index=False)
    pd.DataFrame(annual).to_csv(out / "annual_conversion_diagnostics.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(out / "quality_flag_conversion_diagnostics.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(out / "baseline_comparison.csv", index=False)
    _csv(monthly_estimates.resample("MS").mean(), out / "monthly_storage_comparison.csv")
    # Export conversion as a separate usable response, never overwrite supplied storage.
    derived = frame.copy(); derived.storage_m3 = preferred_conversion.storage_m3
    _csv(derived, out / "geometry_observations_si.csv")
    zero = refs[cfg["preferred_reference"]][0]
    sensitivity = []
    for delta in cfg["sensitivity_offsets_m"]:
        perturbed, _ = convert(levels, curve, replace(zero, zero_level_m=zero.zero_level_m + delta),
                               time_basis=cfg["level_time_basis"], allow_mean_level_proxy=True)
        for phase in ("calibration", "assessment"):
            sensitivity.append({"effective_zero_change_m": delta, "phase": phase,
                **metrics(frame.storage_m3, perturbed.storage_m3, roles.eq(phase), capacity=capacity)})
    pd.DataFrame(sensitivity).to_csv(out / "reference_sensitivity.csv", index=False)
    reference_rows = [{"method": method, **asdict(ref), **fitting} for method, (ref, fitting) in refs.items()]
    w.write(out / "references.json", reference_rows)
    summary = {"reservoir_id": cfg["reservoir_id"], "GRAND_ID": cfg["GRAND_ID"], "rows": len(frame),
        "period_start": str(frame.index[0].date()), "period_end": str(frame.index[-1].date()),
        "level_rows": int(levels.notna().sum()), "roles": roles.value_counts().to_dict(),
        "level_quality_counts": quality.value_counts().to_dict(), "quality_policy": "retain published values and report symbol strata separately",
        "calibration_fraction": float(fit.mean()), "selection_signature": selection.signature,
        "preferred_reference": cfg["preferred_reference"], "reference_selected_using_assessment": False,
        "preferred_effective_zero_m": zero.zero_level_m, "level_datum": cfg["level_datum"],
        "datum_metadata": cfg["datum_metadata"], "geometry_diagnostics": curve.diagnostics(capacity),
        "new_optimizer_evaluations": 0, "fixed_profile_attempts": sum(w.read(out / m / "simulation_receipt.json")["new_simulation_calls"] for m in METHODS),
        "cached_observed_baseline_reused": True, "source_certified": False, "geometry_calibration_ready": False,
        "readiness": "conditional_geometry_parameterization_pilot_complete; explicit proxy-aware calibration adapter still required",
        "independent_ungauged_test": False, "fitted_reference_requires_observed_storage": True,
        "storage_time_basis": "volume_of_daily_mean_level_proxy", "exact_mean_volume_operator": False,
        "capacity_rescaled": False, "geometry_extrapolated": False, "reference_is_surveyed_bed": False}
    w.write(out / "summary.json", summary)
    plot(out, monthly_estimates, pd.DataFrame(seasonal), frame, roles, preferred_conversion, cfg)
    write_report(out, summary, pd.DataFrame(conversion_rows), pd.DataFrame(baseline_rows))
    # Recheck input/code integrity before sealing the completed pilot.
    for path, expected in design["input_sha256"].items():
        if w.sha(path) != expected: raise ValueError("Pilot input changed during execution")
    for path, expected in design["source_sha256"].items():
        if w.sha(ROOT / path) != expected: raise ValueError("Pilot source changed during execution")
    w.write(out / "status.json", {"status": "completed", "signature": design["signature"],
                                 "new_optimizer_evaluations": 0})
    artifacts = {p.relative_to(out).as_posix(): w.sha(p) for p in out.rglob("*")
                 if p.is_file() and p.name not in {"manifest.json", "run.lock"}}
    w.write(out / "manifest.json", {"signature": design["signature"], "artifact_sha256": artifacts})
    return {"status": "completed", "output": str(out), "new_fixed_profile_simulations": new_calls,
            "new_optimizer_evaluations": 0}


def plot(out, series, seasonal, original, roles, converted, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    monthly = series.resample("MS").mean() / 1e9
    monthly.plot(ax=axes[0, 0], linewidth=.8)
    axes[0, 0].set(title="Monthly mean storage estimates", ylabel="Storage (billion m³)", xlabel="")
    axes[0, 0].legend(fontsize=6)
    for method, group in seasonal[seasonal.phase == "assessment"].groupby("method"):
        axes[0, 1].plot(group.month, group.bias_m3 / 1e6, label=method)
    axes[0, 1].axhline(0, color="black", linewidth=.6)
    axes[0, 1].set(title="Held-out seasonal conversion bias", xlabel="Month", ylabel="Estimated − provided (million m³)")
    axes[0, 1].legend(fontsize=7)
    mask = roles.eq("assessment")
    axes[1, 0].scatter(original.loc[mask, "storage_m3"] / 1e9, converted.loc[mask, "storage_m3"] / 1e9,
                       s=4, alpha=.3)
    limits = [original.storage_m3.min() / 1e9, original.storage_m3.max() / 1e9]
    axes[1, 0].plot(limits, limits, color="black", linewidth=.7)
    axes[1, 0].set(title="Held-out conversion: calibration-fitted reference", xlabel="Provided storage (billion m³)", ylabel="Geometry storage (billion m³)")
    errors = (converted.storage_m3 - original.storage_m3).resample("YS").mean() / 1e6
    axes[1, 1].bar(errors.index.year, errors.values)
    axes[1, 1].axhline(0, color="black", linewidth=.7)
    axes[1, 1].set(title="Annual conversion bias: all roles for diagnosis", xlabel="Year", ylabel="Mean error (million m³)")
    fig.suptitle(f"{cfg['reservoir_id']} · daily-mean-level proxy · approximate geometry", fontsize=13)
    fig.savefig(out / "storage_diagnostics.png", dpi=170)
    plt.close(fig)


def write_report(out, summary, conversion, baseline):
    def number(value, places=3):
        return "unavailable" if pd.isna(value) else f"{value:.{places}f}"
    lines = ["# Gardiner datum and level-to-storage pilot", "",
        f"Historical gauge levels cover {summary['level_rows']:,} of {summary['rows']:,} days (1980–2011). "
        f"The unchanged mixed-year split assigns {summary['calibration_fraction']:.2%} of the full calendar to calibration. "
        "Calibration, assessment, warmup and context roles are retained exactly.", "",
        "## Datum and depth", "",
        "The archived ECCC station metadata declares PFRA datum. Its datum page gives PFRA → CGVD2013:EPOCH2010 as −0.313 m "
        "(published November 2022; GNSS occupation 2021). That is metadata about a coordinate conversion, not proof of a constant "
        "historical 1980–2011 datum or of the reservoir bottom. This pilot keeps levels and fitted references in PFRA coordinates; "
        "the modern conversion is recorded, not applied retrospectively.", "",
        "Depth is `reported level + documented datum conversion − geometry-zero level`. A datum identifies a height coordinate; "
        "it does not supply the bed elevation. The reusable adapter requires an explicit evidenced transformation when datums differ.", "",
        f"The predeclared primary exploratory method is `{summary['preferred_reference']}`. It estimates an effective geometry zero "
        f"of **{summary['preferred_effective_zero_m']:.6f} m PFRA** using only calibration storage/level pairs. "
        "This is not a surveyed bed elevation, and this method cannot be described as storage-free or fully ungauged.", "",
        "The other predeclared checks are: operator FSL 556.87 m aligned to actual curve top; FSL aligned to the inverse curve at "
        "the existing 9.4 billion m³ capacity; and calibration maximum aligned to curve top. FSL datum equivalence and its connection "
        "to the approximate geometry are explicit assumptions, not documented survey ties. No choice uses held-out performance.", "",
        "## Conversion assessment", "",
        "All scores use existing supplied storage as the comparison reference. Daily mean level transformed through a nonlinear "
        "curve is an approximate storage proxy, not an exact daily mean volume or an end-of-day stock. Timing/shape errors remain combined.", "",
        "| Reference method | Held-out NSE | Held-out bias (million m³) | Held-out RMSE (million m³) | Missing held-out estimates |",
        "|---|---:|---:|---:|---:|"]
    for _, row in conversion[conversion.phase == "assessment"].iterrows():
        lines.append(f"| {row.method} | {number(row.nse)} | {number(row.bias_m3 / 1e6, 1)} | {number(row.rmse / 1e6, 1)} | {int(row.missing_estimates_on_fixed_support)} |")
    lines += ["", "Strict NSE/RMSE/PBIAS are unavailable when any estimate on fixed observed support is missing. "
        "`supported_*` columns show explicitly labelled subset diagnostics. No row is silently removed to improve a score.", "",
        "## Parameterization and fixed simulations", "",
        "Monthly P10/P45/P85 targets use calibration years only and Hazen percentiles. The observed baseline is reused after "
        "target, observation, mask, physical-setting and mass-balance checks. Four geometry scenarios use the same inflow, "
        "initial states, capacity, source channel limit and strict root policy. Physical validation and solver failures remain visible. "
        "The published level record includes estimated and partial-day symbols; these are retained explicitly and assessed separately "
        "in quality_flag_conversion_diagnostics.csv. No quality flag was silently discarded.", "",
        "| Method | Status | Held-out storage NSE | Held-out release NSE |", "|---|---|---:|---:|"]
    for method, group in baseline.groupby("method", sort=False):
        held = group[group.get("phase", pd.Series(index=group.index, dtype=object)) == "assessment"]
        row = held.iloc[0] if len(held) else group.iloc[0]
        lines.append(f"| {method} | {row.simulation_status} | {number(row.get('storage_nse', np.nan))} | {number(row.get('release_nse', np.nan))} |")
    lines += ["", "## Readiness decision", "",
        "The pilot supports exploratory use of a calibration-anchored geometry proxy where overlapping usable storage exists. "
        "It does not certify a surveyed reference or justify universal full-supply/maximum-level alignment. The curve top is "
        "9.97639 billion m³, versus model capacity 9.4 billion m³; neither curve nor capacity was rescaled or clipped.", "",
        "A proxy-aware calibration handoff is still needed before optimizing geometry profiles. Existing selected monthly profiles "
        "and all-37 exports remain unchanged. Inflow/initial states remain conditional on the archived source workflow. "
        "No new optimizer evaluations were performed. Sensitivity offsets are scenarios, not confidence intervals.", "",
        "![Storage diagnostics](storage_diagnostics.png)", "",
        "Inputs, code and generated artifacts are SHA256-pinned in plan.json and manifest.json. A completed resume verifies these "
        "receipts and performs no further simulations. Raw daily provider symbols and station metadata are retained on D:.", "",
        "Sources: [ECCC station metadata](https://api.weather.gc.ca/collections/hydrometric-stations/items?STATION_NUMBER=05HF003&f=json), "
        "[ECCC datum information](https://wateroffice.ec.gc.ca/report/datum_e.html?stn=05HF003), "
        "[WSA full supply level](https://wsask.ca/infrastructure/infrastructure-l1-1/lake-diefenbaker/), "
        "[ECCC explanation of gauge datum](https://wateroffice.ec.gc.ca/contactus/faq_e.html), "
        "[ReGeom dataset](https://doi.org/10.5281/zenodo.1322884).", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def verify(output):
    out = Path(output).resolve(strict=True)
    manifest = w.read(out / "manifest.json")
    for name, expected in manifest["artifact_sha256"].items():
        if w.sha(out / name) != expected:
            raise ValueError("Pilot artifact changed: " + name)
    design = w.read(out / "plan.json")
    if manifest["signature"] != design["signature"]:
        raise ValueError("Pilot manifest identity mismatch")
    for path, expected in design["input_sha256"].items():
        if w.sha(path) != expected: raise ValueError("Pilot source input changed: " + path)
    for path, expected in design["source_sha256"].items():
        if w.sha(ROOT / path) != expected: raise ValueError("Pilot implementation changed: " + path)
    return {"status": "verified", "artifacts": len(manifest["artifact_sha256"]), "new_optimizer_evaluations": 0}
