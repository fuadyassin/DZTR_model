"""Explicit daily mean volume integration with separate physical spill.

Each prepared mean supplies only its own interval volume. The operating release
uses one daily endpoint trapezoid; spill is a separate nonnegative interval
volume and is never carried as an operating release endpoint. This is an
experimental segmented conditional hindcast, not the MATLAB replay solver.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite, ulp
from numbers import Real

import numpy as np
import pandas as pd

from dztr.data.intervals import (
    DailyIntervalPolicy, IntervalPreparation, _copied_frame, prepare_daily_intervals,
)
from dztr.model.release import _monthly_values, _release_for_values, _zone
from dztr.schemas import ReservoirParameters
from .root_policy import POLICIES, _candidates


@dataclass(frozen=True)
class IntervalSimulationSettings:
    """Declared root selection and a common provisional-anchor stress offset.

    The capacity fraction is a robustness perturbation, not a confidence bound.
    No offset is clipped into the physical storage range.
    """

    root_policy: str = "reject"
    initial_storage_offset_capacity_fraction: float = 0.0

    def __post_init__(self):
        if not isinstance(self.root_policy, str) or self.root_policy not in POLICIES:
            raise ValueError(f"root_policy must be one of {POLICIES}.")
        value = self.initial_storage_offset_capacity_fraction
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError("initial_storage_offset_capacity_fraction must be finite and real.")


@dataclass(frozen=True)
class IntervalSimulationResult:
    """Copied inputs and parameters with modeled states on the full input index.

    ``interval_mean_total_outflow_m3s`` includes spill and matches the temporal
    meaning of observed daily mean total discharge. ``release_endpoint_m3s``
    and ``interval_mean_release_m3s`` describe only the operating release.
    DataFrames remain mutable, independently of the caller's preparation.
    """

    frame: pd.DataFrame
    parameters: ReservoirParameters
    settings: IntervalSimulationSettings
    policy: DailyIntervalPolicy
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    issues: tuple[str, ...] = ()


class IntervalSimulationError(ValueError):
    """An explicit failure with the completed prefix on the unchanged calendar."""

    def __init__(self, detail: dict, partial_result: IntervalSimulationResult):
        self.detail = deepcopy(detail)
        self.partial_result = partial_result
        super().__init__(f"Interval simulation failed on {detail['date']}: {detail['reason']}")


class _StepError(ValueError):
    def __init__(self, code, reason, **detail):
        self.code, self.detail = code, detail
        super().__init__(reason)


def _calendar_date(value, default, index, label):
    if value is None:
        return default
    if isinstance(value, (bool, int, float, np.number)):
        raise ValueError(f"{label} must be an explicit calendar date.")
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid {label}.") from error
    if (pd.isna(stamp) or stamp.tz is not None or stamp != stamp.normalize()
            or stamp not in index):
        raise ValueError(f"{label} must be a timezone-naive daily label in the preparation.")
    return stamp


def validate_interval_preparation(preparation: IntervalPreparation) -> IntervalPreparation:
    """Verify structural metadata without reading any future numeric data.

    A later quality flag can quarantine the preceding state in a selected
    prefix. Rebuilding on the full calendar with dummy observations preserves
    that edge, while held-out numeric observations and forcing remain unread.
    Return the independently rebuilt graph with dummy numeric observations;
    callers can use its masks to verify their own phase-restricted fitting views.
    """
    if not isinstance(preparation, IntervalPreparation):
        raise TypeError("Expected an IntervalPreparation.")
    if not isinstance(preparation.frame, pd.DataFrame):
        raise ValueError("Preparation frame must be a DataFrame.")
    frame = preparation.frame
    columns = ("storage_m3", "inflow_m3s", "release_m3s")
    if frame.columns.has_duplicates or not set(columns).issubset(frame.columns):
        raise ValueError("Preparation requires unique observation columns.")
    dummy = pd.DataFrame(0.0, index=frame.index, columns=columns)
    periods = frame.period if "period" in frame else None
    expected = prepare_daily_intervals(dummy, policy=preparation.policy, period_labels=periods)
    for column in expected.frame.columns.difference(columns):
        if column in ("reported_inflow_volume_m3", "usable_inflow_volume_m3"):
            continue
        if column not in frame:
            raise ValueError(f"Preparation is missing derived field {column}.")
        try:
            if pd.api.types.is_datetime64_any_dtype(expected.frame[column].dtype):
                left, right = frame[column].to_numpy(), expected.frame[column].to_numpy()
                if not pd.api.types.is_datetime64_any_dtype(frame[column].dtype):
                    raise AssertionError("Date metadata must retain datetime values.")
                if not np.all((left == right) | (pd.isna(left) & pd.isna(right))):
                    raise AssertionError("Date metadata differs.")
            else:
                pd.testing.assert_series_equal(frame[column], expected.frame[column], check_exact=True)
        except AssertionError as error:
            raise ValueError(f"Preparation derived field {column} has changed.") from error
    for column in ("reported_inflow_volume_m3", "usable_inflow_volume_m3"):
        if column not in frame:
            raise ValueError(f"Preparation is missing derived field {column}.")
    if not np.array_equal(frame.usable_inflow_volume_m3.isna().to_numpy(),
                          ~expected.frame.simulation_interval_eligible.to_numpy()):
        raise ValueError("Preparation usable-volume support has changed.")
    if not isinstance(preparation.segments, pd.DataFrame):
        raise ValueError("Preparation segments must be a DataFrame.")
    segment_columns = expected.segments.columns.difference(["initial_storage_m3", "start", "end"])
    if not set(expected.segments.columns).issubset(preparation.segments.columns):
        raise ValueError("Preparation segment metadata is incomplete.")
    try:
        pd.testing.assert_frame_equal(preparation.segments.loc[:, segment_columns],
                                      expected.segments.loc[:, segment_columns], check_exact=True)
        for column in ("start", "end"):
            left, right = preparation.segments[column].to_numpy(), expected.segments[column].to_numpy()
            if not np.array_equal(left, right):
                raise AssertionError("Segment calendar differs.")
    except AssertionError as error:
        raise ValueError("Preparation segment metadata has changed.") from error
    return expected


def _parameters_copy(parameters):
    if not isinstance(parameters, ReservoirParameters):
        raise TypeError("Interval simulation requires ReservoirParameters.")
    # Reconstruct to validate mutable tables again, and detach the returned copy.
    return ReservoirParameters(
        parameters.monthly_targets.copy(deep=True), parameters.capacity_m3,
        parameters.initial_storage_m3, parameters.initial_release_m3s,
        parameters.channel_capacity_m3s, parameters.dead_storage_fraction,
        parameters.inflow_correction, parameters.flood_mode,
    )


def _number(value, name, *, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite real data.")
    result = float(value)
    if not isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite{' and nonnegative' if nonnegative else ''}.")
    return result


def _solve_interval_step(predictor, driver, dt, values, capacity, tolerance,
                         previous_storage, policy):
    if not isfinite(predictor):
        raise _StepError("nonfinite_predictor", "The interval balance predictor is not finite.")
    if predictor < 0:
        raise _StepError("negative_predictor", "The interval balance requires negative storage.")
    roots, admissibility = _candidates(predictor, driver, dt, values, tolerance)
    candidates = [(storage, release, 0.0) for storage, release in roots
                  if 0 <= storage <= capacity and isfinite(release) and release >= 0]
    cap_release = _release_for_values(capacity, driver, values, dt)
    cap_spill = predictor - dt / 2.0 * cap_release - capacity
    if not isfinite(cap_release) or cap_release < 0 or not isfinite(cap_spill):
        raise _StepError("nonfinite_release", "The capacity release or spill volume is not finite.")
    if cap_spill >= 0:
        cap = (float(capacity), float(cap_release), float(cap_spill))
        if not any(abs(cap[0] - s) <= admissibility
                   and dt / 2.0 * abs(cap[1] - q) <= admissibility
                   and abs(cap[2] - spill) <= admissibility for s, q, spill in candidates):
            candidates.append(cap)
    candidates.sort()
    details = {
        "root_count": len(candidates),
        "candidate_storage_m3": tuple(s for s, _, _ in candidates),
        "candidate_release_m3s": tuple(q for _, q, _ in candidates),
        "candidate_spill_volume_m3": tuple(spill for _, _, spill in candidates),
        "root_admissibility_tolerance_m3": admissibility,
    }
    if not candidates:
        raise _StepError("no_root", "No feasible in-bounds storage/release/spill root exists.", **details)
    if len(candidates) > 1 and policy == "reject":
        raise _StepError("multiple_roots", "Multiple feasible interval roots exist.", **details)
    tie = False
    if len(candidates) == 1 or policy == "minimum_storage":
        selected = candidates[0]
    elif policy == "maximum_storage":
        selected = candidates[-1]
    else:
        distances = [abs(s - previous_storage) for s, _, _ in candidates]
        nearest = [candidate for candidate, distance in zip(candidates, distances)
                   if distance - min(distances) <= admissibility]
        tie = len(nearest) > 1
        selected = nearest[0]
    return selected, details, len(candidates) > 1, tie


def simulate_interval_reservoir(
    preparation: IntervalPreparation,
    parameters: ReservoirParameters,
    *,
    settings: IntervalSimulationSettings | None = None,
    atol_storage_m3: float = 1e-4,
    start_date=None,
    end_date=None,
) -> IntervalSimulationResult:
    """Integrate prepared current-interval volumes on contiguous clean blocks.

    Only selected-range forcing and actually used storage anchors are inspected
    numerically. Noninitial observed storage and observed mean discharge are
    targets, never model forcing. A selected prefix still honors quarantine
    implied by later quality metadata. Results retain the full preparation
    index; unsimulated, quarantined, and unfinished rows have NaN model values.
    A selected start within a block adds an explicit unscored initialization.

    Computed net flux requires storage-only release and unit correction. The
    physical-inflow mode uses the current daily mean as the release-rule driver;
    this is a declared representative driver, not a recovered endpoint sample.
    Failures raise IntervalSimulationError with dated details and a partial
    result; no alternative root, skipped block, or repaired observation is used.
    """
    settings = IntervalSimulationSettings() if settings is None else settings
    if not isinstance(settings, IntervalSimulationSettings):
        raise TypeError("settings must be IntervalSimulationSettings.")
    tolerance = _number(atol_storage_m3, "atol_storage_m3", nonnegative=True)
    if tolerance <= 0:
        raise ValueError("atol_storage_m3 must be positive.")
    graph = validate_interval_preparation(preparation)
    params = _parameters_copy(parameters)
    net_mode = preparation.policy.flow_semantics == "computed_net_inflow"
    if net_mode and (params.flood_mode != "storage" or params.inflow_correction != 1.0):
        raise ValueError("Computed net flux requires flood_mode='storage' and inflow_correction=1.")
    index, source = preparation.frame.index, preparation.frame
    start = _calendar_date(start_date, index[0], index, "start_date")
    end = _calendar_date(end_date, index[-1], index, "end_date")
    if end < start:
        raise ValueError("end_date must not precede start_date.")
    selected = np.asarray((index >= start) & (index <= end))
    clean = ~graph.frame.storage_endpoint_quarantined.to_numpy()
    initialization = graph.frame.segment_initialization.to_numpy().copy() & selected
    selected_clean = np.flatnonzero(clean & selected)
    if len(selected_clean):
        initialization[selected_clean[0]] = True
    eligible = graph.frame.simulation_interval_eligible.to_numpy().copy() & selected & ~initialization
    metadata = _copied_frame(source.drop(columns=["storage_m3", "inflow_m3s", "release_m3s"]))
    metadata.attrs = deepcopy(source.attrs)
    metadata["source_simulation_interval_eligible"] = graph.frame.simulation_interval_eligible
    metadata["source_segment_initialization"] = graph.frame.segment_initialization
    metadata["segment_initialization"] = initialization
    metadata["simulation_interval_eligible"] = eligible
    metadata["storage_score_eligible"] = eligible
    metadata["release_score_eligible"] = eligible
    metadata["source_audit_interval_eligible"] = graph.frame.source_audit_interval_eligible & eligible
    metadata["reported_inflow_mean_m3s"] = _copied_frame(source[["inflow_m3s"]]).inflow_m3s
    floats = (
        "storage_m3", "release_endpoint_m3s", "interval_mean_release_m3s",
        "interval_mean_total_outflow_m3s", "integrated_release_volume_m3",
        "integrated_total_outflow_volume_m3", "integrated_inflow_volume_m3",
        "spill_volume_m3", "balance_residual_m3", "release_rule_residual_m3s",
        "root_admissibility_tolerance_m3", "rule_driver_m3s", "rule_month", "zone",
    )
    for column in floats:
        metadata[column] = np.nan
    metadata["interval_completed"] = False
    metadata["root_count"] = 0
    metadata["root_policy_applied"] = False
    metadata["root_policy_tie_break"] = False
    metadata["root_policy"] = settings.root_policy
    for column in ("candidate_storage_m3", "candidate_release_m3s", "candidate_spill_volume_m3"):
        metadata[column] = pd.Series([()] * len(index), index=index, dtype=object)
    issues = ["Observed storage anchors condition the blocks; initialization rows are unscored.",
              "Operating release uses one daily trapezoid; spill is a separate interval volume."]
    if net_mode:
        issues.append("Computed net boundary flux retains its reported sign without loss attribution.")
    months = {}
    dt = preparation.policy.nominal_step_seconds
    storage = release = None

    def result():
        if metadata.root_policy_applied.any() and not any("root convention" in issue for issue in issues):
            issues.append(f"Experimental {settings.root_policy} root convention selected multiple roots.")
        return IntervalSimulationResult(deepcopy(metadata), _parameters_copy(params), settings,
                                        preparation.policy, start, end, tuple(issues))

    def fail(position, error, code, **detail):
        information = {"date": index[position].date().isoformat(), "row": int(position),
                       "segment_id": int(graph.frame.segment_id.iloc[position]),
                       "failure_code": code, "reason": str(error),
                       "root_policy": settings.root_policy, **detail}
        issues.append(f"Simulation stopped on {information['date']}: {information['reason']}")
        raise IntervalSimulationError(information, result()) from error

    for position in np.flatnonzero(selected & clean):
        stamp = index[position]
        details = {}
        try:
            month = int(stamp.month)
            if month not in months:
                months[month] = _monthly_values(month, params)
            values = months[month]
            if initialization[position]:
                observed = _number(source.storage_m3.iloc[position], "initial observed storage", nonnegative=True)
                storage = observed + settings.initial_storage_offset_capacity_fraction * params.capacity_m3
                if not isfinite(storage) or not 0 <= storage <= params.capacity_m3:
                    raise _StepError("initial_storage_out_of_bounds", "Provisional initial storage is outside [0, capacity].")
                driver = 0.0 if net_mode else _number(source.inflow_m3s.iloc[position], "initial inflow", nonnegative=True)
                release = _release_for_values(storage, driver, values, dt)
                if not isfinite(release) or release < 0:
                    raise _StepError("nonfinite_release", "The initial operating release is not finite and nonnegative.")
                metadata.at[stamp, "storage_m3"] = storage
                metadata.at[stamp, "release_endpoint_m3s"] = release
                metadata.at[stamp, "release_rule_residual_m3s"] = 0.0
                metadata.at[stamp, "rule_driver_m3s"] = driver
                metadata.at[stamp, "rule_month"] = month
                metadata.at[stamp, "zone"] = _zone(storage, values)
                continue
            mean = _number(source.inflow_m3s.iloc[position], "inflow mean", nonnegative=not net_mode)
            reported_volume = mean * dt
            if not isfinite(reported_volume):
                raise _StepError("nonfinite_forcing", "The current interval inflow volume is not finite.")
            for column in ("reported_inflow_volume_m3", "usable_inflow_volume_m3"):
                if _number(source[column].iloc[position], column) != reported_volume:
                    raise _StepError("mutated_preparation", f"Preparation volume {column} disagrees with the current mean.")
            volume_in = reported_volume * params.inflow_correction
            driver = 0.0 if net_mode else mean
            predictor = storage + volume_in - dt / 2.0 * release
            details = {"previous_storage_m3": storage, "previous_release_endpoint_m3s": release,
                       "inflow_mean_m3s": mean, "integrated_inflow_volume_m3": volume_in,
                       "predictor_m3": predictor, "rule_month": month}
            chosen, roots, applied, tie = _solve_interval_step(
                predictor, driver, dt, values, params.capacity_m3, tolerance, storage, settings.root_policy)
            next_storage, next_release, spill = chosen
            volume_release = dt / 2.0 * (release + next_release)
            volume_total = volume_release + spill
            residual = next_storage - storage - volume_in + volume_total
            rule_residual = next_release - _release_for_values(next_storage, driver, values, dt)
            numbers = (next_storage, next_release, volume_in, volume_release, volume_total, spill,
                       residual, rule_residual)
            if not all(isfinite(value) for value in numbers):
                raise _StepError("nonfinite_solution", "An interval state, rate, volume, or residual is not finite.")
            if (not 0 <= next_storage <= params.capacity_m3 or next_release < 0 or spill < 0
                    or abs(residual) > roots["root_admissibility_tolerance_m3"]
                    or dt / 2.0 * abs(rule_residual) > roots["root_admissibility_tolerance_m3"]):
                raise _StepError("invalid_solution", "Physical bounds, conservation, or release-rule checks failed.")
            updates = {
                "storage_m3": next_storage, "release_endpoint_m3s": next_release,
                "interval_mean_release_m3s": volume_release / dt,
                "interval_mean_total_outflow_m3s": volume_total / dt,
                "integrated_release_volume_m3": volume_release,
                "integrated_total_outflow_volume_m3": volume_total,
                "integrated_inflow_volume_m3": volume_in, "spill_volume_m3": spill,
                "balance_residual_m3": residual, "release_rule_residual_m3s": rule_residual,
                "rule_driver_m3s": driver, "rule_month": month, "zone": _zone(next_storage, values),
                "interval_completed": True, "root_policy_applied": applied,
                "root_policy_tie_break": tie, **roots,
            }
            for column, value in updates.items():
                metadata.at[stamp, column] = value
            storage, release = next_storage, next_release
        except (_StepError, ValueError, ZeroDivisionError, OverflowError) as error:
            code = error.code if isinstance(error, _StepError) else ("invalid_initialization" if initialization[position] else "invalid_forcing")
            if isinstance(error, _StepError):
                details.update(error.detail)
            fail(position, error, code, **details)
    return result()
