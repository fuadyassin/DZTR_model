"""Provisional, calibration-only stopping diagnostics for negative-NSE search.

Sliding objective windows and hard evaluation caps are useful resource controls:
https://pymoo.org/interface/termination.html . Hypervolume measures dominated
area relative to a fixed reference: https://pymoo.org/misc/indicators.html .

The thresholds here are illustrative, not validated budget recommendations.
Small recent changes cannot establish global convergence or identify reservoir
parameters. Even nonoverlapping windows come from a correlated search trajectory.
This module assesses existing trial records; it never runs an optimizer, scores
validation data, or modifies the candidate stream.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np

from .convergence import (
    _feasible_pairs,
    _ordered_prefix,
    _pareto_from_pairs,
    hypervolume_2d,
)


_NSE_IDEAL_COST = -1.0
_NSE_ROUNDOFF_TOLERANCE = 1e-12
_STATUSES = frozenset({
    "insufficient_history", "continue_search", "stopping_candidate", "budget_exhausted",
})


def _integer(value, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}.")
    return int(value)


def _real(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number.") from exc
    if not isfinite(result) or result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return result


@dataclass(frozen=True)
class StoppingPolicy:
    """Explicit provisional tolerances and a hard cap for one seeded search.

    Checks occur at fixed evaluation-count multiples of ``check_interval``.
    Each compares the full feasible Pareto archive with its archive one window
    earlier. All count settings are population aligned; windows may overlap
    when ``window_evaluations`` exceeds ``check_interval``.

    ``normalized_hv_tolerance`` uses the fixed theoretical negative-NSE area
    ``(reference[0] + 1) * (reference[1] + 1)``. At the default reference this
    area is four, independently of the observed scores or a particular seed.
    Movement and extrema tolerances are in raw NSE-coordinate units. Parameter
    coordinates do not enter this policy because equifinality can persist.
    """

    population_size: int = 16
    check_interval: int = 128
    window_evaluations: int = 128
    minimum_evaluations: int = 512
    patience: int = 3
    max_evaluations: int = 4096
    hypervolume_reference: tuple[float, float] = (1.0, 1.0)
    normalized_hv_tolerance: float = 0.002
    front_tolerance: float = 0.005
    extrema_tolerance: float = 0.005

    def __post_init__(self) -> None:
        for name in ("population_size", "check_interval", "window_evaluations",
                     "minimum_evaluations", "patience", "max_evaluations"):
            minimum = 2 if name == "population_size" else 1
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=minimum))
        for name in ("check_interval", "window_evaluations", "minimum_evaluations", "max_evaluations"):
            if getattr(self, name) % self.population_size:
                raise ValueError(f"{name} must be a multiple of population_size.")
        if self.window_evaluations % self.check_interval:
            raise ValueError("window_evaluations must be a positive multiple of check_interval.")
        if self.minimum_evaluations < self.population_size + self.window_evaluations:
            raise ValueError("minimum_evaluations must allow a full window after the initial population.")
        if self.max_evaluations < self.minimum_evaluations:
            raise ValueError("max_evaluations must be at least minimum_evaluations.")
        try:
            reference = tuple(self.hypervolume_reference)
        except TypeError as exc:
            raise ValueError("hypervolume_reference must contain two finite real coordinates greater than -1.") from exc
        if len(reference) != 2 or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in reference
        ):
            raise ValueError("hypervolume_reference must contain two finite real coordinates greater than -1.")
        try:
            reference = tuple(float(value) for value in reference)
        except (ValueError, OverflowError) as exc:
            raise ValueError("hypervolume_reference must contain two finite real coordinates greater than -1.") from exc
        if any(not isfinite(value) or value <= _NSE_IDEAL_COST for value in reference):
            raise ValueError("hypervolume_reference must contain two finite real coordinates greater than -1.")
        area = (reference[0] - _NSE_IDEAL_COST) * (reference[1] - _NSE_IDEAL_COST)
        if not isfinite(area) or area <= 0:
            raise ValueError("The fixed theoretical hypervolume normalization area must be finite and positive.")
        object.__setattr__(self, "hypervolume_reference", reference)
        for name in ("normalized_hv_tolerance", "front_tolerance", "extrema_tolerance"):
            value = _real(getattr(self, name), name, positive=name == "normalized_hv_tolerance")
            object.__setattr__(self, name, value)

    @property
    def hypervolume_normalization_area(self) -> float:
        """Fixed theoretical NSE area, never an observed or seed-specific range."""
        return (self.hypervolume_reference[0] + 1) * (self.hypervolume_reference[1] + 1)


@dataclass(frozen=True)
class StoppingAssessment:
    """A current diagnostic and the earliest provisional suggestion, if any.

    Later failures reset the current status and are recorded separately from
    the earliest suggestion. ``candidate_invalidated`` records any subsequent
    failed comparison window, including missing feasible/reference coverage.
    History consists of detached, JSON-compatible dictionaries.
    """

    policy: StoppingPolicy
    status: str
    reason: str
    evaluations: int
    suggested_stop_evaluations: int | None
    history: tuple[dict, ...]
    candidate_invalidated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.policy, StoppingPolicy):
            raise ValueError("policy must be a StoppingPolicy.")
        if self.status not in _STATUSES:
            raise ValueError(f"status must be one of {sorted(_STATUSES)}.")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a nonempty string.")
        object.__setattr__(self, "evaluations", _integer(self.evaluations, "evaluations", minimum=0))
        if self.suggested_stop_evaluations is not None:
            count = _integer(self.suggested_stop_evaluations, "suggested_stop_evaluations", minimum=1)
            if count > min(self.evaluations, self.policy.max_evaluations):
                raise ValueError("suggested_stop_evaluations cannot exceed the available evaluations or policy cap.")
            if count < self.policy.minimum_evaluations or count % self.policy.check_interval:
                raise ValueError("suggested_stop_evaluations must be an eligible policy checkpoint.")
            object.__setattr__(self, "suggested_stop_evaluations", count)
        if not isinstance(self.candidate_invalidated, (bool, np.bool_)):
            raise ValueError("candidate_invalidated must be boolean.")
        object.__setattr__(self, "candidate_invalidated", bool(self.candidate_invalidated))
        if self.candidate_invalidated and self.suggested_stop_evaluations is None:
            raise ValueError("candidate_invalidated requires an earlier stopping suggestion.")
        try:
            history = tuple(self.history)
        except TypeError as exc:
            raise ValueError("history must contain diagnostic dictionaries.") from exc
        if any(not isinstance(row, dict) for row in history):
            raise ValueError("history must contain diagnostic dictionaries.")
        object.__setattr__(self, "history", tuple(deepcopy(row) for row in history))


def _front_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Symmetric Hausdorff distance using raw-coordinate L-infinity distances."""
    # Tied costs describe one geometric point, however many vectors attain it.
    first, second = np.unique(first, axis=0), np.unique(second, axis=0)

    def directed(left, right):
        largest = 0.0
        # Bound the temporary pair-distance memory for large Pareto archives.
        for start in range(0, len(left), 256):
            distances = np.max(np.abs(left[start:start + 256, None, :] - right[None, :, :]), axis=2)
            largest = max(largest, float(np.max(np.min(distances, axis=1))))
        return largest

    return max(directed(first, second), directed(second, first))


def assess_stopping(evaluated, policy: StoppingPolicy | None = None) -> StoppingAssessment:
    """Assess honest trial prefixes without making a global-convergence claim.

    Records may be shuffled, but evaluation IDs must be unique and contiguous
    from zero. Rejected and nonfinite trials count toward the budget. A finite
    feasible negative-NSE cost cannot be below -1 beyond rounding tolerance.

    A comparison needs nonempty feasible archives at both ends, complete fixed
    reference coverage, and at least one new feasible trial. It requires all
    three small-change signals for ``patience`` consecutive eligible checks.
    HV gain is strictly less than its tolerance; movement and extrema drift
    may equal their tolerances. Before-minimum checks never accumulate patience.

    Unaligned trailing trials await the next scheduled check. If a supplied
    stream extends beyond an earlier suggestion or the policy cap, those later
    complete checks still audit that suggestion; no first suggestion is made
    after the cap. Such an audit may invalidate an earlier stopping candidate.
    """
    if policy is None:
        policy = StoppingPolicy()
    if not isinstance(policy, StoppingPolicy):
        raise ValueError("policy must be a StoppingPolicy.")
    records = tuple(evaluated)
    if records:
        records = _ordered_prefix(records, len(records))
        feasible_records, feasible_costs = _feasible_pairs(records)
        if np.any(feasible_costs < _NSE_IDEAL_COST - _NSE_ROUNDOFF_TOLERANCE):
            raise ValueError("Stopping diagnostics require negative-NSE costs at least -1 for finite feasible candidates.")
        feasible_ids = {point.evaluation_id for point in feasible_records}
        cumulative_feasible = np.cumsum([point.evaluation_id in feasible_ids for point in records])
    else:
        cumulative_feasible = np.empty(0, dtype=int)

    reference = np.asarray(policy.hypervolume_reference)
    snapshots = {}

    def snapshot(count):
        if count not in snapshots:
            finite_records, costs = _feasible_pairs(records[:count])
            front = _pareto_from_pairs(finite_records, costs)
            pairs = np.asarray([point.objectives for point in front], dtype=float).reshape(-1, 2)
            snapshots[count] = {
                "costs": pairs,
                "feasible_count": int(cumulative_feasible[count - 1]),
                "front_size": len(front),
                "hypervolume": hypervolume_2d(pairs, reference),
                "front_points_outside_reference": int(np.any(pairs > reference, axis=1).sum()),
            }
        return snapshots[count]

    history = []
    consecutive = 0
    earliest = None
    candidate_invalidated = False
    comparable_history = False
    for count in range(policy.check_interval, len(records) + 1, policy.check_interval):
        current = snapshot(count)
        start = count - policy.window_evaluations
        comparison_available = start >= policy.population_size
        reasons = []
        hv_gain = normalized_gain = movement = extrema = None
        new_feasible = None
        start_outside = None
        signal_reasons = []
        if not comparison_available:
            signal_reasons.append("insufficient_window_history")
        else:
            comparable_history = True
            previous = snapshot(start)
            start_outside = previous["front_points_outside_reference"]
            new_feasible = current["feasible_count"] - previous["feasible_count"]
            # A cumulative feasible archive cannot lose dominated area. Clip
            # negligible subtraction roundoff, never moving the reference.
            hv_gain = max(0.0, current["hypervolume"] - previous["hypervolume"])
            normalized_gain = hv_gain / policy.hypervolume_normalization_area
            if not current["front_size"]:
                signal_reasons.append("no_finite_feasible_front")
            if not previous["front_size"]:
                signal_reasons.append("no_finite_feasible_window_start_front")
            if not new_feasible:
                signal_reasons.append("no_new_feasible_evaluations")
            if current["front_points_outside_reference"]:
                signal_reasons.append("reference_truncated_front")
            if start_outside:
                signal_reasons.append("reference_truncated_window_start_front")
            if normalized_gain >= policy.normalized_hv_tolerance:
                signal_reasons.append("hypervolume_gain_exceeds_tolerance")
            if current["front_size"] and previous["front_size"]:
                movement = _front_distance(previous["costs"], current["costs"])
                extrema = float(np.max(np.abs(np.min(current["costs"], axis=0) - np.min(previous["costs"], axis=0))))
                if movement > policy.front_tolerance:
                    signal_reasons.append("front_movement_exceeds_tolerance")
                if extrema > policy.extrema_tolerance:
                    signal_reasons.append("extrema_drift_exceeds_tolerance")

        stable_window = not signal_reasons
        within_limits = policy.minimum_evaluations <= count <= policy.max_evaluations
        eligible = comparison_available and within_limits
        reasons.extend(signal_reasons)
        if count < policy.minimum_evaluations:
            reasons.append("below_minimum_evaluations")
        if count > policy.max_evaluations:
            reasons.append("maximum_evaluations_exceeded")
        # Extensions past the hard cap audit an existing suggestion without
        # being allowed to create a first one after resource exhaustion.
        if comparison_available and count >= policy.minimum_evaluations and stable_window:
            consecutive += 1
        else:
            consecutive = 0
        if earliest is None and eligible and consecutive >= policy.patience:
            earliest = count
        invalidates_candidate = earliest is not None and count > earliest and not stable_window
        candidate_invalidated = candidate_invalidated or invalidates_candidate
        history.append({
            "evaluations": count,
            "window_start_evaluations": start if comparison_available else None,
            "eligible": eligible,
            "stable_window": stable_window,
            "consecutive_stable_windows": consecutive,
            "reasons": reasons,
            "hypervolume": current["hypervolume"],
            "hypervolume_gain": hv_gain,
            "normalized_hv_gain": normalized_gain,
            "hypervolume_normalization_area": policy.hypervolume_normalization_area,
            "front_movement": movement,
            "extrema_drift": extrema,
            "newly_feasible_count": new_feasible,
            "feasible_count": current["feasible_count"],
            "rejected_count": count - current["feasible_count"],
            "front_size": current["front_size"],
            "front_points_outside_reference": current["front_points_outside_reference"],
            "window_start_front_points_outside_reference": start_outside,
            "candidate_invalidated": invalidates_candidate,
        })

    if earliest is not None and consecutive >= policy.patience:
        status = "stopping_candidate"
        reason = "Recent eligible windows satisfy all provisional objective-change tolerances; this is not evidence of global convergence."
    elif len(records) >= policy.max_evaluations:
        status = "budget_exhausted"
        reason = "The policy evaluation cap was reached without a currently sustained stopping candidate."
    elif not comparable_history:
        status = "insufficient_history"
        reason = "A complete scheduled comparison window after the initial population is not yet available."
    else:
        status = "continue_search"
        reason = "The minimum budget, feasibility/reference guards, or consecutive-window tolerances are not currently satisfied."
    if candidate_invalidated:
        reason += " A later comparison invalidated the earliest provisional suggestion; retain that extension evidence."
    return StoppingAssessment(policy, status, reason, len(records), earliest, tuple(history), candidate_invalidated)
