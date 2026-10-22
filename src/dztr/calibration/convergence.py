"""Descriptive calibration search indicators with a fixed hypervolume reference.

Hypervolume is dominated area relative to a reference chosen before inspecting
results: https://pymoo.org/misc/indicators.html#hypervolume
No helper moves the reference, rescales objectives, or declares convergence.
"""

from __future__ import annotations

from math import fsum, isfinite
from numbers import Integral

import numpy as np

from .candidates import _validated_vector
from .optimizer import EvaluatedCandidate
from .selection import select_pareto_representatives


DEFAULT_HYPERVOLUME_REFERENCE = (1.0, 1.0)


def _reference_pair(reference) -> np.ndarray:
    raw = np.asarray(reference)
    if np.iscomplexobj(raw):
        raise ValueError("Hypervolume reference must contain two finite real costs.")
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Hypervolume reference must contain two finite real costs.") from exc
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("Hypervolume reference must contain two finite real costs.")
    return values.copy()


def _cost_pairs(costs) -> np.ndarray:
    raw = np.asarray(costs)
    if np.iscomplexobj(raw):
        raise ValueError("Objective costs must be real numbers.")
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Objective costs must be numeric pairs.") from exc
    if values.size == 0 and values.ndim == 1:
        values = values.reshape(0, 2)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Objective costs must have shape (candidate_count, 2).")
    return values


def hypervolume_2d(costs, reference=DEFAULT_HYPERVOLUME_REFERENCE) -> float:
    """Return the union area dominated by two minimization costs.

    Coordinates are used unchanged. The default reference [1, 1] corresponds
    to NSE [-1, -1] for negative-NSE objectives. Callers must use the same fixed
    reference across all compared checkpoints and seeds. Nonfinite cost pairs
    and points exceeding the reference in either coordinate are excluded.
    Boundary points contribute zero area, and identical pairs are deduplicated.
    """
    ref = _reference_pair(reference)
    values = _cost_pairs(costs)
    eligible = np.isfinite(values).all(axis=1) & np.all(values <= ref, axis=1)
    points = np.unique(values[eligible], axis=0)
    if not len(points):
        return 0.0
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    best_y = float(ref[1])
    rectangles = []
    for x, y in points:
        x, y = float(x), float(y)
        if y < best_y:
            # Only the newly covered horizontal strip is added. Dominated
            # points cannot lower best_y after sorting on ascending x.
            rectangles.append((float(ref[0]) - x) * (best_y - y))
            best_y = y
    try:
        area = fsum(rectangles)
    except OverflowError as exc:
        raise ValueError("Hypervolume area exceeds finite floating-point range.") from exc
    if not isfinite(area):
        raise ValueError("Hypervolume area exceeds finite floating-point range.")
    return float(area)


def _ordered_prefix(evaluated, checkpoint: int) -> tuple[EvaluatedCandidate, ...]:
    records = tuple(evaluated)
    if isinstance(checkpoint, bool) or not isinstance(checkpoint, Integral) or not 1 <= checkpoint <= len(records):
        raise ValueError("checkpoint must be an integer count between one and the available evaluations.")
    ids = [point.evaluation_id for point in records]
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in ids):
        raise ValueError("Evaluation IDs must be unique contiguous integers starting at zero.")
    if sorted(ids) != list(range(len(records))):
        raise ValueError("Evaluation IDs must be unique and contiguous starting at zero; gaps or duplicates are not allowed.")
    return tuple(sorted(records, key=lambda point: point.evaluation_id)[:int(checkpoint)])


def _feasible_pairs(prefix) -> tuple[list[EvaluatedCandidate], np.ndarray]:
    pairs = _cost_pairs([point.objectives for point in prefix])
    if any(not isinstance(point.feasible, (bool, np.bool_)) for point in prefix):
        raise ValueError("Candidate feasibility flags must be booleans.")
    eligible = np.asarray([point.feasible for point in prefix], dtype=bool) & np.isfinite(pairs).all(axis=1)
    return [point for point, keep in zip(prefix, eligible) if keep], pairs[eligible]


def _pareto_from_pairs(records, costs) -> tuple[EvaluatedCandidate, ...]:
    """Independent O(n log n) 2D sweep retaining different tied-cost records."""
    if not records:
        return ()
    ids = np.asarray([point.evaluation_id for point in records])
    order = np.lexsort((ids, costs[:, 1], costs[:, 0]))
    selected = []
    best_y, best_x = float("inf"), None
    for index in order:
        x, y = costs[index]
        if y < best_y:
            selected.append(records[index])
            best_y, best_x = y, x
        elif y == best_y and x == best_x:
            selected.append(records[index])
    return tuple(sorted(selected, key=lambda point: point.evaluation_id))


def prefix_pareto_front(evaluated, checkpoint: int) -> tuple[EvaluatedCandidate, ...]:
    """Reconstruct the feasible archive after exactly ``checkpoint`` trials.

    Records may be shuffled, but IDs must cover every evaluation from zero.
    Infeasible or nonfinite trials still count toward the prefix. They never
    enter its Pareto front. No validation scoring or generation assumptions are
    made; a caller can require generation-aligned checkpoint counts separately.
    """
    prefix = _ordered_prefix(evaluated, checkpoint)
    feasible, costs = _feasible_pairs(prefix)
    return _pareto_from_pairs(feasible, costs)


def checkpoint_metrics(evaluated, checkpoint: int, *, reference=DEFAULT_HYPERVOLUME_REFERENCE) -> dict:
    """Describe calibration-only progress without asserting convergence.

    The existing calibration Pareto selection rule supplies the balanced
    representative. The same fixed-reference coordinates are used for area
    comparisons, independently of that representative-selection normalization.
    """
    ref = _reference_pair(reference)
    prefix = _ordered_prefix(evaluated, checkpoint)
    feasible, costs = _feasible_pairs(prefix)
    front = _pareto_from_pairs(feasible, costs)
    front_costs = np.asarray([point.objectives for point in front], dtype=float).reshape(-1, 2)
    result = {
        "evaluations": int(checkpoint),
        "feasible_count": len(feasible),
        "rejected_count": int(checkpoint) - len(feasible),
        "front_size": len(front),
        "hypervolume": hypervolume_2d(front_costs, ref),
        "reference": ref.tolist(),
        "front_points_outside_reference": int(np.any(front_costs > ref, axis=1).sum()),
        "max_release_nse": None,
        "max_storage_nse": None,
        "representative_ids": None,
        "balanced_release_nse": None,
        "balanced_storage_nse": None,
    }
    if front:
        selected = select_pareto_representatives(front)
        balanced = next(point for point in front if point.evaluation_id == selected["balanced"])
        result.update({
            "max_release_nse": float(-np.min(front_costs[:, 0])),
            "max_storage_nse": float(-np.min(front_costs[:, 1])),
            "representative_ids": selected,
            "balanced_release_nse": float(-balanced.objectives[0]),
            "balanced_storage_nse": float(-balanced.objectives[1]),
        })
    return result


def normalize_parameter_vectors(vectors, bounds) -> np.ndarray:
    """Normalize native (n, 72) candidates by one fixed empirical bounds box.

    Fixed-width coordinates become zero. Candidates must already lie inside
    the box; no clipping or physical repair occurs. Storage coordinates are
    native capacity fractions and release coordinates are m3/s before this
    normalization. Output is detached from caller-owned arrays.
    """
    lower, upper = _validated_vector(bounds.lower_vector), _validated_vector(bounds.upper_vector)
    width = upper - lower
    if (width < 0).any() or not np.isfinite(width).all():
        raise ValueError("Parameter bounds must have finite nonnegative widths.")
    raw = np.asarray(vectors)
    if np.iscomplexobj(raw):
        raise ValueError("Parameter vectors must contain finite real values.")
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Parameter vectors must contain finite real values.") from exc
    if values.size == 0 and values.ndim == 1:
        values = values.reshape(0, 72)
    if values.ndim != 2 or values.shape[1] != 72 or not np.isfinite(values).all():
        raise ValueError("Parameter vectors must have shape (candidate_count, 72) and finite values.")
    if not ((values >= lower) & (values <= upper)).all():
        raise ValueError("Parameter vectors must lie inside the supplied bounds.")
    return np.divide(values - lower, width, out=np.zeros_like(values), where=width > 0)


def normalized_parameter_spread(vectors, bounds) -> np.ndarray:
    """Return 72 peak-to-peak spreads; this describes variability, not uncertainty."""
    values = normalize_parameter_vectors(vectors, bounds)
    if not len(values):
        raise ValueError("Parameter spread requires at least one candidate vector.")
    return np.ptp(values, axis=0)
