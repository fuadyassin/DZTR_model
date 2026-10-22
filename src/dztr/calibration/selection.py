"""Choose calibration Pareto representatives before inspecting validation."""

from __future__ import annotations

import numpy as np

from .optimizer import EvaluatedCandidate


def select_pareto_representatives(front: tuple[EvaluatedCandidate, ...]) -> dict[str, int]:
    """Return release, storage and balanced evaluation IDs, allowing shared IDs.

    The balanced representative minimizes Euclidean distance to the calibration
    ideal after scaling each objective by the observed Pareto-front range.
    A constant objective contributes zero. This descriptive selection rule does
    not change the two search objectives. Evaluation IDs break remaining ties.
    """
    candidates = tuple(front)
    if not candidates or any(not point.feasible for point in candidates):
        raise ValueError("Selection requires a nonempty feasible Pareto front.")
    costs = np.asarray([point.objectives for point in candidates], dtype=float)
    if costs.shape != (len(candidates), 2) or not np.isfinite(costs).all():
        raise ValueError("Pareto representatives require two finite costs per candidate.")
    if len({point.evaluation_id for point in candidates}) != len(candidates):
        raise ValueError("Pareto evaluation IDs must be unique.")
    dominates = (costs[:, None, :] <= costs[None, :, :]).all(axis=2) & (costs[:, None, :] < costs[None, :, :]).any(axis=2)
    if dominates.any():
        raise ValueError("Representative selection requires a nondominated front.")
    release = min(candidates, key=lambda point: (*point.objectives, point.evaluation_id))
    storage = min(candidates, key=lambda point: (point.objectives[1], point.objectives[0], point.evaluation_id))
    ranges = np.ptp(costs, axis=0)
    normalized = np.divide(costs - costs.min(axis=0), ranges,
                           out=np.zeros_like(costs), where=ranges > 0)
    distances = np.linalg.norm(normalized, axis=1)
    balanced_index = min(range(len(candidates)), key=lambda index: (
        distances[index], *candidates[index].objectives, candidates[index].evaluation_id,
    ))
    return {"best_release": release.evaluation_id, "best_storage": storage.evaluation_id,
            "balanced": candidates[balanced_index].evaluation_id}
