"""Elitist scalar GA using the frozen NSGA-II bounded variation operators.

Parent-plus-offspring survival minimizes one cost, with evaluation ID breaking
ties. Initialization combines the supplied baseline, local normal draws and a
Latin hypercube. Queued proposals and PCG64 state make partial generations
exactly resumable. Every objective call, including a rejection, uses budget.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from dztr.calibration.optimizer import _sbx, _mutate, _latin_hypercube, _vector_key

DEFAULTS = dict(population_size=32, crossover_probability=.9, crossover_eta=15., mutation_eta=20.,
                local_initialization_fraction=.5, local_initialization_scale=.05)


def settings(values=None):
    if values is not None and not isinstance(values, dict): raise ValueError("GA settings must be a JSON object")
    supplied = {} if values is None else dict(values)
    if set(supplied) - set(DEFAULTS): raise ValueError("Unknown GA setting")
    result = {**DEFAULTS, **supplied}
    size = result["population_size"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError("GA population_size must be an integer at least two")
    for key in set(result) - {"population_size"}:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError("GA settings must be finite numbers")
        result[key] = float(value)
    if not 0 <= result["crossover_probability"] <= 1 or not 0 <= result["local_initialization_fraction"] <= 1:
        raise ValueError("GA probabilities/fractions must be in [0, 1]")
    if result["crossover_eta"] <= 0 or result["mutation_eta"] <= 0 or result["local_initialization_scale"] < 0:
        raise ValueError("Invalid GA distribution/initialization setting")
    return result


def search(objective, initial, lower, upper, *, seed, horizon, budget, ga=None, state=None, checkpoint=None):
    options = settings(ga)
    initial, lower, upper = (np.asarray(v, float) for v in (initial, lower, upper))
    if initial.ndim != 1 or not len(initial) or initial.shape != lower.shape or initial.shape != upper.shape:
        raise ValueError("GA vectors need matching nonempty dimensions")
    if not all(np.isfinite(v).all() for v in (initial, lower, upper)) or (lower > upper).any() or not np.isfinite(upper - lower).all() or (initial < lower).any() or (initial > upper).any():
        raise ValueError("Invalid GA bounds or initial vector")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (seed, horizon, budget)):
        raise ValueError("GA seed, horizon and budget must be integers")
    if seed < 0 or horizon < 2 or not 1 <= budget <= horizon: raise ValueError("Invalid GA seed/horizon/budget")
    identity = dict(seed=seed, horizon=horizon, initial=initial.tolist(), lower=lower.tolist(), upper=upper.tolist(), ga=options)
    if state is not None:
        state = deepcopy(state)
        if state.get("kind") != "single_objective_ga_v1" or state.get("settings") != identity:
            raise ValueError("GA checkpoint settings changed")
        trials = state.get("trials")
        if not isinstance(trials, list) or not trials: raise ValueError("GA checkpoint needs evaluated trials")
        if len(trials) > budget: raise ValueError("GA budget cannot shrink")
        cursor = 0
        def replay(vector):
            nonlocal cursor
            if cursor >= len(trials): raise ValueError("GA checkpoint termination lineage changed")
            row = trials[cursor]
            if row.get("evaluation_id") != cursor or not np.array_equal(vector, np.asarray(row.get("vector"), float)):
                raise ValueError("GA checkpoint proposal lineage changed")
            cost = row.get("cost"); feasible = row.get("feasible"); reason = row.get("reason")
            if type(feasible) is not bool or feasible != (cost is not None): raise ValueError("Invalid GA checkpoint feasibility")
            if feasible and (isinstance(cost, bool) or not isinstance(cost, (int, float)) or not np.isfinite(cost) or reason is not None):
                raise ValueError("Invalid GA checkpoint cost")
            if not feasible and (not isinstance(reason, str) or not reason): raise ValueError("Rejected GA trials need a reason")
            cursor += 1
            return SimpleNamespace(feasible=feasible, cost=cost if feasible else float("inf"), reason=reason)
        replay_budget = min(horizon, len(trials) + 1) if state.get("terminal_reason") == "duplicate_exhaustion" else len(trials)
        reproduced = search(replay, initial, lower, upper, seed=seed, horizon=horizon, budget=replay_budget, ga=options)
        if reproduced != state: raise ValueError("GA checkpoint population, proposals or RNG lineage changed")
    rng = np.random.default_rng(seed)
    active = upper > lower; width = upper - lower; size = options["population_size"]
    variation = SimpleNamespace(**options)
    def normalize(vector):
        result = np.zeros(len(initial)); result[active] = (vector[active] - lower[active]) / width[active]; return result
    def decode(vector): return np.clip(lower + vector * width, lower, upper)
    if state is None:
        state = dict(kind="single_objective_ga_v1", settings=identity, trials=[], population_ids=[], batch_ids=[],
                     initialized=False, generation=0, pending_vectors=[], terminal_reason=None, rng_state=None)
        proposals = [initial.copy()]
        if not active.any():
            state["terminal_reason"] = "fixed_bounds"
        else:
            count = size - 1; local_count = int(np.floor(options["local_initialization_fraction"] * count))
            unit = normalize(initial); local = []
            for _ in range(local_count):
                point = unit.copy(); point[active] += rng.normal(0, options["local_initialization_scale"], int(active.sum()))
                local.append(np.clip(point, 0, 1))
            for point in [*local, *_latin_hypercube(count - local_count, active, rng)]:
                candidate = decode(point); seen = {_vector_key(p) for p in proposals}
                for _ in range(256):
                    if _vector_key(candidate) not in seen: break
                    candidate = decode(np.where(active, rng.random(len(initial)), 0.))
                else:
                    state["terminal_reason"] = "duplicate_exhaustion"; break
                proposals.append(candidate)
        state["pending_vectors"] = [v.tolist() for v in proposals]
        state["rng_state"] = deepcopy(rng.bit_generator.state)
    else:
        rng.bit_generator.state = deepcopy(state["rng_state"])
    def quality(identifier):
        row = state["trials"][identifier]
        return (row["cost"] if row["cost"] is not None else float("inf"), identifier)
    def choose():
        first, second = rng.choice(state["population_ids"], size=2)
        return state["trials"][min(int(first), int(second), key=quality)]["vector"]
    while len(state["trials"]) < budget:
        if not state["pending_vectors"]:
            if state["terminal_reason"] is not None: break
            state["generation"] += 1
            seen = {_vector_key(np.asarray(t["vector"])) for t in state["trials"]}
            pending = []; queued = []
            for _ in range(size):
                for _ in range(256):
                    if not queued:
                        children = _sbx(normalize(np.asarray(choose())), normalize(np.asarray(choose())), active, variation, rng)
                        queued.extend(_mutate(child, active, options["mutation_eta"], rng) for child in children)
                    candidate = decode(queued.pop(0)); key = _vector_key(candidate)
                    if key not in seen: seen.add(key); pending.append(candidate.tolist()); break
                else:
                    state["terminal_reason"] = "duplicate_exhaustion"; break
            state["pending_vectors"] = pending
            state["rng_state"] = deepcopy(rng.bit_generator.state)
            if not pending: break
        vector = np.asarray(state["pending_vectors"].pop(0), float)
        evaluation = objective(vector.copy())
        cost = float(evaluation.cost) if evaluation.feasible and np.isfinite(evaluation.cost) else None
        reason = None if cost is not None else (evaluation.reason or "Objective returned an infeasible or nonfinite cost")
        identifier = len(state["trials"])
        state["trials"].append(dict(evaluation_id=identifier, generation=state["generation"], vector=vector.tolist(),
                                    cost=cost, feasible=cost is not None, reason=reason))
        state["batch_ids"].append(identifier)
        if not state["pending_vectors"]:
            state["population_ids"] = sorted(state["population_ids"] + state["batch_ids"], key=quality)[:size]
            state["batch_ids"] = []; state["initialized"] = True
        state["rng_state"] = deepcopy(rng.bit_generator.state)
        if checkpoint is not None: checkpoint(deepcopy(state))
    return deepcopy(state)
