"""Scalar selection, shared bounded variation, exact continuation and DDS parity."""
from copy import deepcopy
import json

import numpy as np
import pytest

from strategies import ga_v1 as ga, single_objective_v1 as dispatch, dds_v1
from strategies.flow_only_v1 import Evaluation


def sphere(vector): return Evaluation(True, float(np.sum((vector - .25) ** 2)))


@pytest.mark.parametrize("first_budget", [1, 5, 8, 9, 13, 17])
def test_exact_continuation_including_partial_populations_and_generations(first_budget):
    calls = []; checkpoints = []
    def counted(v): calls.append(v.copy()); return sphere(v)
    args = dict(seed=42, horizon=64, ga={"population_size": 8})
    first = ga.search(counted, [.9, .8], [0, 0], [1, 1], budget=first_budget, checkpoint=checkpoints.append, **args)
    saved = json.loads(json.dumps(first)); original = deepcopy(saved)
    resumed = ga.search(counted, [.9, .8], [0, 0], [1, 1], budget=32, state=saved, **args)
    fresh = ga.search(sphere, [.9, .8], [0, 0], [1, 1], budget=32, **args)
    assert len(calls) == 32 and resumed == fresh and saved == original
    assert resumed["trials"][:first_budget] == first["trials"]
    assert checkpoints[-1] == first
    assert min(t["cost"] for t in resumed["trials"]) < resumed["trials"][0]["cost"]


def test_feasible_first_elitism_bounds_fixed_coordinates_and_rejected_trials():
    def constrained(v):
        if v[0] > .8: return Evaluation(False, float("inf"), "Synthetic constraint")
        return sphere(v)
    result = ga.search(constrained, [.9, .9, .5], [0, 0, .5], [1, 1, .5], seed=4, horizon=64, budget=64, ga={"population_size": 8})
    assert result["trials"][0]["cost"] is None
    assert all(t["vector"][2] == .5 for t in result["trials"])
    assert len({tuple(t["vector"]) for t in result["trials"]}) == 64
    costs = [result["trials"][i]["cost"] for i in result["population_ids"]]
    assert min(costs) == min(t["cost"] for t in result["trials"] if t["cost"] is not None)
    assert costs == sorted(costs)


def test_fixed_bounds_and_duplicate_exhaustion_do_not_waste_evaluations():
    fixed = ga.search(sphere, [.5], [.5], [.5], seed=0, horizon=32, budget=16)
    assert len(fixed["trials"]) == 1 and fixed["terminal_reason"] == "fixed_bounds"
    assert ga.search(sphere, [.5], [.5], [.5], seed=0, horizon=32, budget=32, state=fixed) == fixed
    upper = np.nextafter(1., 2.)
    result = ga.search(sphere, [1.], [1.], [upper], seed=0, horizon=32, budget=16, ga={"population_size": 2})
    assert len(result["trials"]) == 2 and result["terminal_reason"] == "duplicate_exhaustion"
    assert ga.search(sphere, [1.], [1.], [upper], seed=0, horizon=32, budget=32, ga={"population_size": 2}, state=result) == result


def test_no_feasible_search_keeps_explicit_rejections():
    result = ga.search(lambda v: Evaluation(False, float("inf"), "No root"), [.5], [0], [1], seed=7, horizon=16, budget=16, ga={"population_size": 4})
    assert len(result["trials"]) == 16 and all(t["cost"] is None and t["reason"] == "No root" for t in result["trials"])


@pytest.mark.parametrize("field", ["population_ids", "pending_vectors", "rng_state"])
def test_state_tampering_and_decision_changes_are_rejected(field):
    args = dict(seed=1, horizon=32, ga={"population_size": 4})
    result = ga.search(sphere, [.9, .8], [0, 0], [1, 1], budget=7, **args)
    bad = deepcopy(result)
    if field == "population_ids": bad[field] = list(reversed(bad[field]))
    elif field == "pending_vectors": bad[field][0][0] = -1
    else: bad[field]["has_uint32"] = 1 - bad[field]["has_uint32"]
    with pytest.raises(ValueError, match="lineage changed"):
        ga.search(sphere, [.9, .8], [0, 0], [1, 1], budget=16, state=bad, **args)
    with pytest.raises(ValueError, match="settings changed"):
        ga.search(sphere, [.9, .8], [0, 0], [1, 1], budget=16, state=result, **{**args, "horizon": 64})
    with pytest.raises(ValueError, match="shrink"):
        ga.search(sphere, [.9, .8], [0, 0], [1, 1], budget=4, state=result, **args)


@pytest.mark.parametrize("optimizer", [None, "dds"])
def test_optional_optimizer_defaults_to_bitwise_identical_dds(optimizer):
    options = dict(seeds=[42], horizon=32, evaluation_budget=16, workers=1, perturbation=.2)
    if optimizer is not None: options["optimizer"] = optimizer
    actual = dispatch.search(sphere, [.9, .8], [0, 0], [1, 1], seed=42, budget=16, optimization=options)
    expected = dds_v1.search(sphere, [.9, .8], [0, 0], [1, 1], seed=42, horizon=32, budget=16, perturbation=.2)
    assert actual == expected


@pytest.mark.parametrize("ga_options", [{"population_size": True}, {"population_size": 1}, {"mutation_eta": 0}, {"crossover_probability": 2}, {"unknown": 1}])
def test_invalid_ga_controls_are_rejected(ga_options):
    with pytest.raises(ValueError): ga.settings(ga_options)


def test_optimizer_specific_options_are_not_silently_ignored():
    common = dict(seeds=[42], horizon=32, evaluation_budget=16, workers=1)
    with pytest.raises(ValueError): dispatch.options({**common, "optimizer": "ga", "perturbation": .2})
    with pytest.raises(ValueError): dispatch.options({**common, "ga": {"population_size": 4}})
    with pytest.raises(ValueError): dispatch.options({**common, "optimizer": {}})
    with pytest.raises(ValueError): ga.settings(4)
    assert dispatch.options(common)["optimizer"] == "dds"
