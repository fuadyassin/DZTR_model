"""Explain portable sample gates without modifying data or operating policy."""

from __future__ import annotations

from collections import Counter
from math import isfinite, ulp
import re

import numpy as np
import pandas as pd

from dztr.calibration.bounds import DEFAULT_PERCENTILE_BOUNDS
from dztr.calibration.periods import DatePeriod
from dztr.calibration.sampled import (SampledSplit, estimate_sampled_bounds,
                                     select_sampled_period, target_observations, validate_split)
from dztr.model.balance import _implicit_solution
from dztr.model.release import _monthly_values, _release_for_values, _zone
from dztr.parameterization import estimate_generalized_parameters
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirObservations, ReservoirParameters
from dztr.simulation.sampled import SamplingSettings


def _json(value):
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _field_stats(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {"count": len(values), "missing_count": int(np.isnan(values).sum()),
            "infinite_count": int(np.isinf(values).sum()), "negative_count": int((values < 0).sum()),
            "zero_count": int((values == 0).sum()),
            "minimum": float(finite.min()) if len(finite) else None,
            "maximum": float(finite.max()) if len(finite) else None}


def _root_kind(message):
    text = message.casefold()
    if "multiple" in text and "root" in text:
        return "implicit_multiple_roots"
    if "no feasible" in text and "root" in text:
        return "implicit_no_root"
    if "negative storage" in text or "negative predictor" in text:
        return "implicit_negative_storage"
    return None


def inspect_implicit_branches(predictor, inflow, dt_seconds, month, parameters, tolerance=1e-4):
    """Report the solver's algebraic candidates; never choose among roots.

    This mirrors the branch equations solely for diagnostics and checks each
    candidate against the unchanged release rule and solver tolerance.
    """
    values = _monthly_values(month, parameters)
    dt, half = float(dt_seconds), float(dt_seconds) / 2
    normal = (values.qnormal - values.qcritical) / (values.normal - values.critical)
    flood_rise = max(inflow - values.qnormal, values.qflood - values.qnormal) if values.inflow_enabled else values.qflood - values.qnormal
    flood = flood_rise / (values.flood - values.normal)
    branches = (
        ("dead", 0., 0.), ("critical_available_water", 1 / dt, -values.dead / dt),
        ("critical_release", 0., values.qcritical),
        ("normal", normal, values.qcritical - normal * values.critical),
        ("flood", flood, values.qnormal - flood * values.normal),
        ("emergency_target", 0., values.qflood),
        ("emergency_storage", 1 / dt, -values.flood / dt),
        ("emergency_channel", 0., values.channel),
    )
    residual_tolerance = max(tolerance, 64 * ulp(max(abs(predictor), 1.)))
    candidates, roots = [], []
    for name, slope, intercept in branches:
        storage = (predictor - half * intercept) / (1 + half * slope)
        valid = isfinite(storage) and storage >= 0
        release = _release_for_values(storage, inflow, values, dt) if valid else None
        residual = storage - predictor + half * release if valid else None
        accepted = valid and abs(residual) <= residual_tolerance
        candidates.append({"branch": name, "slope": slope, "intercept": intercept,
                           "storage_m3": storage, "release_m3s": release,
                           "zone": _zone(storage, values) if valid else None,
                           "balance_residual_m3": residual, "accepted": accepted})
        if accepted and not any(abs(storage - old[0]) <= residual_tolerance and half * abs(release - old[1]) <= residual_tolerance for old in roots):
            roots.append((storage, release))
    left = values.qnormal + flood_rise
    right = min(values.qflood, values.channel)
    return _json({"predictor_m3": predictor, "inflow_m3s": inflow, "dt_seconds": dt,
                  "month": month, "residual_tolerance_m3": residual_tolerance,
                  "targets": dict(zip(("dead_m3", "critical_m3", "normal_m3", "flood_m3", "critical_release_m3s", "normal_release_m3s", "flood_release_m3s", "channel_capacity_m3s", "inflow_enabled"), values)),
                  "candidate_branches": candidates,
                  "distinct_roots": [{"storage_m3": s, "release_m3s": q} for s, q in roots],
                  "distinct_root_count": len(roots),
                  "flood_boundary": {"left_release_m3s": left, "right_release_m3s": right,
                                     "release_jump_m3s": right - left,
                                     "left_balance_predictor_m3": values.flood + half * left,
                                     "right_balance_predictor_m3": values.flood + half * right},
                  "policy_selected": False})


def trace_implicit_failure(observations, parameters, sampling, *, tolerance=1e-4, stop_date=None):
    """Replay a baseline prefix to explain its first failure; no optimization.

    Only already valid calibration-derived parameters are accepted. The
    observations supplied here contain the explicitly transformed forcing.
    """
    frame = observations.frame
    inflow = frame.inflow_m3s.to_numpy(dtype=float)
    storage, release = parameters.initial_storage_m3, parameters.initial_release_m3s
    months = frame.index.month.to_numpy()
    if sampling.month_clock == "legacy":
        months = (pd.Timestamp(sampling.legacy_start) + pd.to_timedelta(np.arange(len(frame)) * sampling.fixed_step_days, unit="D")).month.to_numpy()
    for index in range(1, len(frame)):
        date = frame.index[index]
        if stop_date is not None and date > pd.Timestamp(stop_date):
            break
        dt = sampling.fixed_step_days * 86400 if sampling.fixed_step_days is not None else (date - frame.index[index - 1]).total_seconds()
        predictor = storage + dt / 2 * (parameters.inflow_correction * (inflow[index] + inflow[index - 1]) - release)
        month = int(months[index])
        previous = {"storage_m3": storage, "release_m3s": release, "inflow_m3s": inflow[index - 1]}
        try:
            new_storage, new_release = _implicit_solution(predictor, inflow[index], dt, _monthly_values(month, parameters), tolerance)
        except (ValueError, ZeroDivisionError) as error:
            detail = inspect_implicit_branches(predictor, inflow[index], dt, month, parameters, tolerance)
            detail.update({"date": date.isoformat(), "sample_row_1based": index + 1, "error": str(error), "failure_code": _root_kind(str(error)), "previous_state": _json(previous)})
            return detail
        excess = max(new_storage - parameters.capacity_m3, 0.)
        storage = min(new_storage, parameters.capacity_m3)
        release = new_release + 2 * excess / dt
    return None


def _diagnostic_targets(fitting, percentiles, method):
    rows = []
    for month in range(1, 13):
        row = []
        for column in ("storage_m3", "release_m3s"):
            values = fitting.frame.loc[fitting.frame.index.month == month, column].dropna().to_numpy(dtype=float)
            row.extend(np.percentile(values, percentiles, method=method) if len(values) and np.isfinite(values).all() else [np.nan] * 3)
        rows.append(row)
    return pd.DataFrame(rows, columns=MONTHLY_TARGET_COLUMNS, index=pd.Index(range(1, 13), name="month"))


def diagnose_sample(observations, config, report=None):
    """Return input/model/experiment issues without editing files or settings.

    Calibration observations alone determine diagnostic targets and bounds.
    A supplied runtime report enables a prefix trace of an implicit root
    failure. The result describes readiness, never optimization convergence.
    """
    if not isinstance(observations, ReservoirObservations) or not isinstance(config, dict):
        raise TypeError("Expected ReservoirObservations and a portable sample configuration dictionary.")
    report = {} if report is None else report
    issues = []
    def add(code, severity, phase, message, evidence=None, next_action="Review the source evidence before changing the experiment."):
        issues.append({"code": code, "severity": severity, "phase": phase, "message": message,
                       "evidence": evidence or {}, "next_action": next_action})
    summary = {"reservoir_id": config.get("reservoir", {}).get("id"), "optimizer_executed": False,
               "baseline_status": report.get("baseline_status", "not_run"),
               "optimization_status": report.get("optimization_status", "not_run")}
    if not config.get("simulation_supported", True):
        add("external_forcing_required", "error", "forcing", config.get("simulation_unsupported_reason") or "Source simulation uses unsupported external forcing.", next_action="Provide and validate the external forcing adapter; do not substitute the raw zero inflow record.")
    frame = observations.frame
    for column in ("storage_m3", "inflow_m3s", "release_m3s"):
        if column not in frame:
            add("missing_observation_field", "error", "observations", f"Required field {column} is absent.", {"field": column})
            continue
        stats = _field_stats(frame[column])
        for key, code in (("negative_count", "negative_observations"), ("missing_count", "missing_observations"), ("infinite_count", "nonfinite_observations")):
            if stats[key]:
                add(code, "warning", "observations", f"{column} contains {stats[key]} {key.split('_')[0]} values; measurements are retained.", {"field": column, **stats}, "Review measurement units and provenance; use an explicit forcing policy for signed inflow.")
        if stats["zero_count"] == len(frame):
            add("all_zero_observations", "warning", "observations", f"{column} is entirely zero.", {"field": column, **stats}, "Confirm that zeros represent available measurements rather than missing forcing or a placeholder.")
    if any(column not in frame for column in ("storage_m3", "inflow_m3s", "release_m3s")):
        return _finish(issues, summary)
    try:
        sampling = SamplingSettings(**config["sampling"])
        periods = config["periods"]
        split = SampledSplit(periods["simulation_start"], DatePeriod(**periods["calibration"]), DatePeriod(**periods["validation"]), periods["warmup_rows"])
        continuous = validate_split(observations, split, sampling)
        fitting = target_observations(select_sampled_period(observations, split.calibration, sampling))
        summary.update({"sampling": config["sampling"], "parameter_estimation_period": split.calibration.as_dict(), "warmup_rows": split.warmup_rows})
    except (ValueError, TypeError, KeyError) as error:
        add("invalid_sampling_or_periods", "error", "calendar", str(error), next_action="Correct explicit native sampling and period boundaries; do not fill missing intervals implicitly.")
        return _finish(issues, summary)
    preprocessing = config.get("inflow_preprocessing")
    forced = None
    try:
        if not isinstance(preprocessing, dict) or set(preprocessing) != {"factor", "offset_m3s"} or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) for value in preprocessing.values()):
            raise ValueError("An explicit finite inflow factor and offset are required.")
        forced_frame = continuous.frame.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            forced_frame["inflow_m3s"] = forced_frame.inflow_m3s * preprocessing["factor"] + preprocessing["offset_m3s"]
        forced = ReservoirObservations(forced_frame, continuous.quality, continuous.source)
        stats = _field_stats(forced_frame.inflow_m3s)
        if stats["missing_count"] or stats["infinite_count"] or (stats["negative_count"] and not sampling.allow_signed_inflow):
            add("transformed_inflow_gate", "error", "forcing", "Transformed forcing violates the finite/nonnegative inflow policy.", {"preprocessing": preprocessing, "transformed": stats, "allow_signed_inflow": sampling.allow_signed_inflow}, "Review the raw forcing and its explicit transformation; do not clip signed or missing values.")
        elif stats["negative_count"]:
            add("signed_forcing_policy", "warning", "forcing", "Signed transformed inflow is accepted by an explicit policy.", {"transformed": stats, "allow_signed_inflow": True})
    except (ValueError, TypeError) as error:
        add("invalid_inflow_preprocessing", "error", "forcing", str(error), {"preprocessing": preprocessing})
    model = config.get("model", {})
    expression = report.get("source_dead_storage_expression", config.get("source_dead_storage_expression"))
    add("inherited_dead_storage", "info", "protocol", "Dead storage is a fixed inherited model setting; calibration-only target estimation does not re-estimate it.", {"dead_storage_fraction": model.get("dead_storage_fraction"), "source_expression": expression, "full_record_dependency": "min(Lc)" in str(expression) if expression else "not established by portable config alone"}, "Document any dependence on full-record legacy targets separately from this calibration-only experiment.")
    parameters = None
    targets = None
    try:
        estimates = estimate_generalized_parameters(fitting, config.get("target_percentiles", (10, 45, 85)), channel_capacity_m3s=model["channel_capacity_m3s"], quantile_method=config.get("quantile_method", "hazen"))
        targets = estimates.monthly_targets
    except (ValueError, TypeError, KeyError) as error:
        add("target_estimation_failed", "error", "parameterization", str(error), {"period": split.calibration.as_dict()})
        try:
            targets = _diagnostic_targets(fitting, config.get("target_percentiles", (10, 45, 85)), config.get("quantile_method", "hazen"))
            summary["targets_are_diagnostic_only"] = True
        except (ValueError, TypeError):
            pass
    if targets is not None:
        summary["calibration_targets"] = [{"month": int(month), **row.to_dict()} for month, row in targets.iterrows()]
        capacity, dead_fraction = model.get("capacity_m3"), model.get("dead_storage_fraction")
        if isinstance(capacity, (int, float)) and isinstance(dead_fraction, (int, float)) and np.isfinite(capacity) and np.isfinite(dead_fraction):
            dead = capacity * dead_fraction
            for month, row in targets.iterrows():
                values = row.to_numpy(float)
                evidence = {"month": int(month), "capacity_m3": capacity, "dead_storage_m3": dead, **row.to_dict()}
                if not np.isfinite(values).all():
                    add("nonfinite_monthly_targets", "error", "parameterization", "A month lacks finite storage/release target estimates.", evidence)
                for condition, code, message in (
                    (dead >= values[0], "dead_storage_not_below_critical", "Dead storage is at or above critical storage."),
                    (values[0] >= values[1], "critical_not_below_normal", "Critical storage is not strictly below normal storage."),
                    (values[1] >= values[2], "normal_not_below_flood", "Normal storage is not strictly below flood storage."),
                    (values[2] > capacity, "flood_target_above_capacity", "Flood storage target exceeds physical capacity."),
                    ((values[3:] < 0).any(), "negative_release_targets", "Release targets contain negative rates."),
                    ((np.diff(values[3:]) < 0).any(), "release_targets_not_ordered", "Release targets are not nondecreasing."),
                ):
                    if condition:
                        add(code, "error", "model", message, evidence, "Review physical constants, units, and calibration observations; do not clip or reorder targets automatically.")
        initial = model.get("initial_storage_m3")
        if not isinstance(initial, (int, float)) or not np.isfinite(initial) or not isinstance(capacity, (int, float)) or not 0 <= initial <= capacity:
            add("invalid_initial_storage", "error", "model", "Initial storage must be finite and within physical capacity.", {"initial_storage_m3": initial, "capacity_m3": capacity})
        try:
            parameters = ReservoirParameters(targets, **model)
        except (ValueError, TypeError) as error:
            add("physical_parameters_invalid", "error", "model", str(error))
        if parameters is not None:
            try:
                bounds = estimate_sampled_bounds(observations, split.calibration, parameters, sampling, quantile_method=config.get("quantile_method", "hazen"))
                summary["calibration_bounds_status"] = "feasible_box"
                summary["calibration_bounds"] = [{"month": int(month), **row.to_dict()} for month, row in bounds.monthly_bounds.iterrows()]
            except (ValueError, TypeError) as error:
                add("calibration_bounds_invalid", "error", "calibration", str(error), {"period": split.calibration.as_dict(), "percentile_bounds": DEFAULT_PERCENTILE_BOUNDS})
                summary["calibration_bounds_status"] = "blocked"
    traced = False
    for key in ("parameterization_reason", "baseline_reason", "calibration_reason", "calibration_baseline_reason", "optimization_reason"):
        message = report.get(key)
        if not message:
            continue
        kind = _root_kind(message)
        evidence = {"report_field": key}
        if kind and not traced and parameters is not None and forced is not None and config.get("simulation_supported", True) and not any(item["code"] == "transformed_inflow_gate" for item in issues):
            date = re.search(r"\d{4}-\d{2}-\d{2}", message)
            try:
                detail = trace_implicit_failure(forced, parameters, sampling, tolerance=config.get("atol_storage_m3", 1e-4), stop_date=date.group() if date else split.validation.end)
                if detail:
                    evidence["step_diagnostic"] = detail
                    traced = True
            except (ValueError, TypeError, ZeroDivisionError) as error:
                evidence["trace_unavailable_reason"] = str(error)
        add(kind or "reported_runtime_failure", "error", "baseline" if key == "baseline_reason" else "calibration", message, evidence,
            "Review discontinuous release branches and an explicit root-selection policy before changing solver behavior; capacity clipping does not resolve root ambiguity." if kind == "implicit_multiple_roots" else "Review the release discontinuity, timestep, and prior state; do not assume this is a target/capacity violation." if kind else "Review the specific runtime gate before running optimization.")
    return _finish(issues, summary)


def _finish(issues, summary):
    counts = dict(Counter(issue["severity"] for issue in issues))
    summary.update({"status": "blocked" if counts.get("error") else "review_required" if counts.get("warning") else "ready_for_baseline", "severity_counts": counts, "issue_count": len(issues), "convergence_assessed": False})
    return _json({"schema_version": 1, "issues": issues, "summary": summary})
