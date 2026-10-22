"""Explicit DDS/GA selection without changing objectives or physical solvers."""
from copy import deepcopy
import numpy as np

from strategies import dds_v1, ga_v1


def options(values):
    result = deepcopy(values)
    if not isinstance(result, dict): raise ValueError("Declare optimization settings")
    algorithm = result.get("optimizer", "dds")
    if not isinstance(algorithm, str) or algorithm not in {"dds", "ga"}: raise ValueError("Choose optimizer dds or ga")
    required = {"seeds", "horizon", "evaluation_budget", "workers"}
    allowed = required | {"optimizer", "perturbation"} if algorithm == "dds" else required | {"optimizer", "ga"}
    if not required <= set(result) or set(result) - allowed: raise ValueError("Invalid optimizer-specific settings")
    result["optimizer"] = algorithm
    seeds = result["seeds"]
    if not isinstance(seeds, list) or not seeds or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("Declare unique nonnegative integer seeds")
    if any(isinstance(result[k], bool) or not isinstance(result[k], int) for k in ("horizon", "evaluation_budget", "workers")):
        raise ValueError("Optimizer budgets and workers must be integers")
    if result["horizon"] < 2 or not 1 <= result["evaluation_budget"] <= result["horizon"] or result["workers"] < 1:
        raise ValueError("Invalid optimizer budget/horizon/workers")
    if algorithm == "dds":
        perturbation = result.setdefault("perturbation", .2)
        if isinstance(perturbation, bool) or not isinstance(perturbation, (float, int)) or not np.isfinite(perturbation) or not 0 < perturbation <= 1:
            raise ValueError("Invalid DDS perturbation")
    else:
        result["ga"] = ga_v1.settings(result.get("ga"))
    return result


def search(objective, initial, lower, upper, *, seed, budget, optimization, state=None, checkpoint=None):
    cfg = options(optimization)
    common = dict(seed=seed, horizon=cfg["horizon"], budget=budget, state=state, checkpoint=checkpoint)
    if cfg["optimizer"] == "dds": return dds_v1.search(objective, initial, lower, upper, perturbation=cfg["perturbation"], **common)
    return ga_v1.search(objective, initial, lower, upper, ga=cfg["ga"], **common)
