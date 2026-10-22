"""Audit completed fleet artifacts without replaying a model or optimizer."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from policy_fleet_job import ROOT, read, sha, write
from policy_fleet import checked
from dztr.calibration.checkpoint import load_optimizer_checkpoint


def nse(observed, simulated):
    x, y = np.asarray(observed), np.asarray(simulated)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    denominator = np.sum((x - np.mean(x)) ** 2)
    if not len(x) or denominator <= 0:
        raise ValueError("Undefined paired NSE.")
    return float(1 - np.sum((x - y) ** 2) / denominator)


def nondominated_ids(costs, identifiers):
    return {int(identifiers[i]) for i, point in enumerate(costs)
            if not np.any(np.all(costs <= point, axis=1) & np.any(costs < point, axis=1))}


def representatives(front):
    costs = front[["release_cost", "storage_cost"]].to_numpy()
    identifiers = front.index.to_numpy(dtype=int)
    normalized = np.divide(costs - costs.min(axis=0), np.ptp(costs, axis=0), out=np.zeros_like(costs), where=np.ptp(costs, axis=0) != 0)
    distances = np.linalg.norm(normalized, axis=1)
    choose = lambda key: int(identifiers[min(range(len(costs)), key=key)])
    return {"best_release": choose(lambda i: (*costs[i], identifiers[i])),
            "best_storage": choose(lambda i: (costs[i, 1], costs[i, 0], identifiers[i])),
            "balanced": choose(lambda i: (distances[i], *costs[i], identifiers[i]))}


def balance(frame, inflow_correction=1.):
    s, q, i, dt = [frame[name].to_numpy() for name in ("storage_m3", "release_m3s", "inflow_m3s", "step_seconds")]
    reconstructed = np.diff(s) - dt[1:] / 2 * (inflow_correction * (i[1:] + i[:-1]) - q[1:] - q[:-1])
    return reconstructed


def audit_job(out, job):
    folder = out / "pilots" / job["reservoir_id"] / f"seed_{job['seed']}"
    receipt = read(folder / "fleet_completion.json")
    assert receipt["status"] == "passed"
    for name, digest in receipt["artifact_sha256"].items():
        assert sha(folder / name) == digest, name
    identity = read(folder / "checkpoint_identity.json")
    assert identity["root_policy"] == job["root_policy"] and identity["seed"] == job["seed"]
    assert identity["protocol_identity"] == checked(out)
    config, result = read(folder / "resolved_config.json"), read(folder / "result.json")
    assert identity["sampling"] == config["sampling"] and identity["periods"] == config["periods"]
    state = load_optimizer_checkpoint(folder / "checkpoint.npz", identity)
    trials = pd.read_csv(folder / "trials.csv", index_col="evaluation_id", float_precision="round_trip")
    vectors = pd.read_csv(folder / "vectors.csv", index_col="evaluation_id", float_precision="round_trip")
    assert len(trials) == len(state.evaluated) == result["search_evaluations"]
    np.testing.assert_array_equal(vectors.to_numpy(), [p.vector for p in state.evaluated])
    np.testing.assert_array_equal(trials[["release_cost", "storage_cost"]].to_numpy(), [p.objectives for p in state.evaluated])
    assert np.array_equal(trials.feasible, [p.feasible for p in state.evaluated])
    bounds = pd.read_csv(folder / "calibration_bounds_si.csv", index_col="month", float_precision="round_trip")
    lower, upper = bounds.iloc[:, :6].to_numpy(), bounds.iloc[:, 6:].to_numpy()
    capacity = config["model"]["capacity_m3"]
    native_lower, native_upper = lower.copy(), upper.copy()
    native_lower[:, :3] /= capacity; native_upper[:, :3] /= capacity
    assert (vectors.to_numpy() >= native_lower.T.reshape(72)).all()
    assert (vectors.to_numpy() <= native_upper.T.reshape(72)).all()
    feasible = trials[trials.feasible]
    expected_front = nondominated_ids(feasible[["release_cost", "storage_cost"]].to_numpy(), feasible.index)
    front = pd.read_csv(folder / "pareto_front.csv", index_col="evaluation_id", float_precision="round_trip")
    assert set(front.index) == expected_front
    np.testing.assert_array_equal(front[["release_cost", "storage_cost"]].to_numpy(), trials.loc[front.index, ["release_cost", "storage_cost"]].to_numpy())
    selection = read(folder / "selection.json")
    assert selection["representatives"] == representatives(front)
    assert not selection["validation_used"] and selection["written_before_validation"]
    rows, maximum_residual, spills, multi = 0, 0., 0, 0
    for role in result["representatives"]:
        index = role["evaluation_id"]
        frame = pd.read_csv(folder / f"selected_{index}_simulation.csv", index_col="date", parse_dates=True, float_precision="round_trip")
        targets = vectors.loc[index].to_numpy().reshape(6, 12).T.copy()
        targets[:, :3] *= capacity
        dead = config["model"]["dead_storage_fraction"] * capacity
        assert (targets[:, 0] > dead).all() and (np.diff(targets[:, :3], axis=1) > 0).all() and (targets[:, 2] <= capacity).all()
        assert (np.diff(targets[:, 3:], axis=1) >= 0).all()
        residual = balance(frame, config["model"]["inflow_correction"])
        np.testing.assert_array_equal(residual, frame.balance_residual_m3.to_numpy()[1:])
        assert (frame.storage_m3 >= 0).all() and (frame.storage_m3 <= capacity).all()
        assert np.abs(residual).max() <= config["atol_storage_m3"]
        step = frame.step_seconds.to_numpy()[1:]
        raw_s = frame.selected_root_storage_m3.to_numpy()[1:]
        raw_q = frame.selected_root_release_m3s.to_numpy()[1:]
        excess = np.maximum(raw_s - capacity, 0)
        np.testing.assert_array_equal(excess, frame.overflow_volume_m3.to_numpy()[1:])
        np.testing.assert_array_equal(np.minimum(raw_s, capacity), frame.storage_m3.to_numpy()[1:])
        np.testing.assert_array_equal(raw_q + 2 * excess / step, frame.release_m3s.to_numpy()[1:])
        if job["root_policy"] == "reject":
            assert not frame.root_policy_applied.any()
        for label in ("calibration", "validation"):
            if label == "validation" and not role["validation_feasible"]:
                assert "validation" not in set(frame.period)
                continue
            period = config["periods"][label]
            scored = frame.loc[period["start"]:period["end"]]
            for variable in ("release", "storage"):
                value = nse(scored["observed_" + variable + "_m3" + ("s" if variable == "release" else "")], scored[variable + "_m3" + ("s" if variable == "release" else "")])
                np.testing.assert_allclose(value, role[label + "_" + variable + "_nse"], rtol=0, atol=1e-12)
        rows += len(frame)
        maximum_residual = max(maximum_residual, float(np.abs(residual).max()))
        spills += int((excess > 0).sum())
        multi += int((frame.root_count > 1).sum())
    return {"reservoir_id": job["reservoir_id"], "seed": job["seed"], "root_policy": job["root_policy"], "search_evaluations": len(trials), "feasible_trials": len(feasible), "front_size": len(front), "role_trajectory_rows": rows, "max_balance_residual_m3": maximum_residual, "role_spill_rows": spills, "role_multiple_root_rows": multi, "initialization_calls": result["initialization"]["simulation_calls"], "calibration_replays": result["representative_calibration_replays"], "validation_attempts": result["validation_attempts"], "held_out_failed_roles": sum(not r["validation_feasible"] for r in result["representatives"])}


def verify(out):
    destination = out / "scientific_verification.json"
    if destination.exists():
        raise ValueError("Existing scientific verification is preserved.")
    summary = read(out / "execution_summary.json")
    checked(out)
    audited = [audit_job(out, job) for job in summary["jobs"] if job["status"] == "completed"]
    failures = [job for job in summary["jobs"] if job["status"] != "completed"]
    pd.DataFrame(audited).to_csv(out / "scientific_checks.csv", index=False, float_format="%.17g")
    write(destination, {"status": "passed", "completed_jobs_audited": len(audited), "failed_jobs": failures,
        "checks": "Checkpoint/trial/vector equality, empirical bounds, physical targets, independent nondominance and representative selection, direct trapezoidal balance and spill reconstruction, calibration/validation NSE from saved trajectories.",
        "new_model_evaluations": 0, "new_search_evaluations": 0,
        "total_search_evaluations": sum(r["search_evaluations"] for r in audited),
        "max_balance_residual_m3": max((r["max_balance_residual_m3"] for r in audited), default=0),
        "verifier_sha256": sha(Path(__file__)), "scientific_checks_sha256": sha(out / "scientific_checks.csv")})
    print(f"Scientific checks passed for {len(audited)} completed jobs; {len(failures)} failed jobs retained separately.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/policy_fleet_20261012")
    verify(parser.parse_args().output.resolve())
