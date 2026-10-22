"""Seeded single-objective DDS with a fixed horizon and exact continuation.

Tolson & Shoemaker (2007), doi:10.1029/2005WR004723, Figure 1.
One supplied initial solution counts toward the total horizon. Checkpoint
budgets may increase within that horizon without changing the cooling schedule.
Infeasible hydrological candidates cost infinity, never a repaired simulation.
"""
from copy import deepcopy
import math

import numpy as np


def reflect(value, lower, upper):
    if value < lower:
        value = lower + (lower - value)
        if value > upper: value = lower
    elif value > upper:
        value = upper - (value - upper)
        if value < lower: value = upper
    return value


def search(objective, initial, lower, upper, *, seed, horizon, budget, perturbation=.2, state=None, checkpoint=None):
    initial, lower, upper = (np.asarray(v, float) for v in (initial, lower, upper))
    if initial.ndim != 1 or not len(initial) or initial.shape != lower.shape or initial.shape != upper.shape:
        raise ValueError("DDS vectors must have matching nonempty dimensions")
    if not all(np.isfinite(v).all() for v in (initial, lower, upper)) or (lower > upper).any() or (initial < lower).any() or (initial > upper).any():
        raise ValueError("Invalid DDS bounds or initial vector")
    for value in (seed, horizon, budget):
        if isinstance(value, bool) or not isinstance(value, int): raise ValueError("DDS counts and seed must be integers")
    if seed < 0 or not 2 <= horizon or not 1 <= budget <= horizon or not 0 < perturbation <= 1:
        raise ValueError("Invalid DDS budget, horizon, seed or perturbation")
    identity = dict(seed=seed, horizon=horizon, perturbation=perturbation,
                    initial=initial.tolist(), lower=lower.tolist(), upper=upper.tolist())
    rng = np.random.default_rng(seed)
    if state is None:
        state = dict(settings=identity, trials=[], rng_state=deepcopy(rng.bit_generator.state))
    else:
        state = deepcopy(state)
        if state["settings"] != identity: raise ValueError("DDS checkpoint settings changed")
        if len(state["trials"]) > budget: raise ValueError("DDS budget cannot shrink")
        rng.bit_generator.state = state["rng_state"]
        for i, row in enumerate(state["trials"]):
            if row["evaluation_id"] != i: raise ValueError("DDS checkpoint trial order changed")
    active = np.flatnonzero(upper > lower)
    while len(state["trials"]) < budget:
        trials = state["trials"]; i = len(trials)
        finite = [t for t in trials if t["cost"] is not None]
        best = min(finite, key=lambda t: (t["cost"], t["evaluation_id"])) if finite else None
        vector = initial.copy() if best is None else np.array(best["vector"])
        if i and len(active):
            probability = 1 - math.log(i + 1) / math.log(horizon)
            chosen = active[rng.random(len(active)) < probability]
            if not len(chosen): chosen = np.array([rng.choice(active)])
            for j in chosen:
                vector[j] = reflect(vector[j] + perturbation * (upper[j] - lower[j]) * rng.normal(), lower[j], upper[j])
        result = objective(vector)
        cost = float(result.cost) if result.feasible and np.isfinite(result.cost) else None
        trials.append(dict(evaluation_id=i, vector=vector.tolist(), cost=cost, feasible=cost is not None, reason=result.reason))
        state["rng_state"] = deepcopy(rng.bit_generator.state)
        if checkpoint is not None: checkpoint(deepcopy(state))
    return state
