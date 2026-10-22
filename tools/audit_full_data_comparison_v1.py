"""Audit cached comparison results without rerunning hydrological searches."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import pandas as pd

from strategies.single_objective_v1 import search
from workflows import full_data_comparison_v1 as workflow, reservoir_v2 as native
from workflows.geometry_v1 import digest


def audit(campaign):
    out, cfg = workflow.verify(campaign)
    budget = cfg["optimization"]["evaluation_budget"]
    release = out / "releases" / f"budget_{budget}"
    workflow.checked(release)
    table = pd.read_csv(release / "baseline_comparison.csv", float_precision="round_trip")
    choices = native.read(release / "selection.json")
    assert choices["selection_uses_assessment"] is False
    count, feasible = 0, 0
    shared = {}
    def no_new_evaluation(_): raise AssertionError("Audit attempted an additional optimizer evaluation")
    for request in cfg["profiles"]:
        rid = request["reservoir_id"]
        for objective in cfg["objectives"]:
            obj = workflow.objective_for(out, rid, objective["id"])
            assert obj.mask.mean() >= .65
            assert obj.frame.loc[~obj.mask, ["storage_m3", "release_m3s"]].isna().all().all()
            for alg in cfg["optimization"]["algorithms"]:
                optimizer = alg["optimizer"]
                for seed in cfg["optimization"]["seeds"]:
                    folder = workflow.job_folder(out, rid, objective["id"], optimizer, seed)
                    receipt = native.read(folder / "checkpoint.json")
                    state = native.read(folder / f"budget_{budget}" / "result.json")
                    assert receipt["state"] == state and digest(state) == receipt["state_sha256"]
                    assert receipt["identity"]["weights"] == obj.weights
                    assert receipt["identity"]["protocol_sha256"] == native.sha(out / "protocol.json")
                    signature = digest({k: state["settings"][k] for k in ("initial", "lower", "upper", "horizon")})
                    if rid in shared: assert shared[rid] == signature
                    shared[rid] = signature
                    assert len(state["trials"]) == budget
                    for trial in state["trials"]:
                        vector = np.asarray(trial["vector"])
                        assert vector.shape == (72,) and (vector >= obj.lower).all() and (vector <= obj.upper).all()
                        if trial["cost"] is not None:
                            obj.parameters(vector)  # physical schema checks only, no simulation
                            feasible += 1
                    replay = search(no_new_evaluation, state["settings"]["initial"], obj.lower, obj.upper, seed=seed,
                        budget=budget, optimization=workflow.algorithm_options(cfg, optimizer), state=state)
                    assert replay == state
                    chosen = next(c for c in choices["records"] if c["reservoir_id"] == rid and c["objective"] == objective["id"] and c["optimizer"] == optimizer and c["seed"] == seed)
                    expected = min((t for t in state["trials"] if t["cost"] is not None), key=lambda t: (t["cost"], t["evaluation_id"]))
                    assert chosen["vector"] == expected["vector"] and chosen["cost"] == expected["cost"]
                    row = table[(table.reservoir_id == rid) & (table.objective == objective["id"]) & (table.optimizer == optimizer) & (table.seed == seed) & (table.phase == "calibration")].iloc[0]
                    cost = sum(obj.weights[v] * (1-row["calibrated_"+v+"_nse"]) for v in obj.weights if obj.weights[v])
                    if cfg["scoring"] == "pooled": np.testing.assert_allclose(cost, expected["cost"], rtol=0, atol=1e-10)
                    count += len(state["trials"])
    max_residual = 0.; trajectories = 0
    for path in list((release / "trajectories").rglob("simulation.csv")) + list((release / "baselines").rglob("simulation.csv")):
        frame = pd.read_csv(path, index_col="date", parse_dates=True, float_precision="round_trip")
        rid = next(r for r in shared if r in path.parts)
        model = native.read(out / "handoff/prepared" / rid / "model_config.json")
        residual = native.independent_balance(frame, model["physical_constants"]["inflow_correction"])
        assert residual <= model["atol_storage_m3"]
        assert frame.storage_m3.between(0, model["physical_constants"]["capacity_m3"]).all()
        max_residual = max(max_residual, residual); trajectories += 1
    prior_files = 0
    previous = ROOT / "outputs/optional_ga_gardiner_v1_20261020/completion_manifest.json"
    proof = native.read(previous)
    for relative, expected in proof["files_sha256"].items():
        assert native.sha(ROOT / relative) == expected, relative
        prior_files += 1
    result = {"status": "passed", "optimizer_evaluations_audited": count, "feasible_vectors_physically_validated": feasible,
        "cached_trajectories_checked": trajectories, "maximum_independent_balance_residual_m3": max_residual,
        "common_initial_vectors_bounds_horizon_verified": True, "calibration_only_selection_verified": True,
        "checkpoint_replay_without_additional_model_evaluations": True, "new_optimizer_evaluations": 0,
        "previous_completion_files_unchanged": prior_files}
    native.write(out / "scientific_verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--campaign", required=True)
    print(audit(parser.parse_args().campaign))
