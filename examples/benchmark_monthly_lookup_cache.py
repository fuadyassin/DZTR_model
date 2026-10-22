"""Research prototype: compare a local monthly lookup cache with the original.

Only this process's release/balance module bindings are temporarily patched.
Each candidate evaluation gets a fresh cache; the bindings are restored before
the next evaluation. This does not change production code or search identity.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dztr.calibration.adaptive import _code_hashes
from dztr.calibration.bounds import estimate_calibration_bounds
from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.objectives import CalibrationObjective
from dztr.calibration.periods import DatePeriod, select_period
from dztr.calibration.population_study import _expected_identity, _verify_run
from dztr.calibration.stopping import StoppingPolicy
from dztr.data import read_reservoir_observations
import dztr.model.balance as balance_module
import dztr.model.release as release_module
from dztr.parameterization import estimate_generalized_parameters
from dztr.schemas import ReservoirParameters


ROLES = ("generalized_baseline", "best_release", "best_storage", "balanced")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def evaluation_monthly_cache():
    """Cache immutable lookup values for one evaluation, then restore bindings."""
    original_release = release_module._monthly_values
    original_balance = balance_module._monthly_values
    if original_release is not original_balance:
        raise RuntimeError("The original release/balance lookup bindings differ; refusing nested patches.")
    cache, references = {}, {}
    statistics = {"lookup_requests": 0, "lookup_misses": 0, "lookup_hits": 0,
                  "parameter_objects": 0, "cached_months": 0, "bindings_restored": False}

    def cached(month, parameters):
        statistics["lookup_requests"] += 1
        # Preserve the original validation even when an invalid value could
        # otherwise compare equal to a cached integer month (for example True).
        if isinstance(month, bool) or int(month) != month or not 1 <= month <= 12:
            return original_release(month, parameters)
        identity = id(parameters)
        references[identity] = parameters  # Prevent object-ID reuse within the evaluation.
        key = (identity, int(month))
        if key in cache:
            statistics["lookup_hits"] += 1
            return cache[key]
        statistics["lookup_misses"] += 1
        cache[key] = original_release(month, parameters)
        return cache[key]

    release_module._monthly_values = cached
    balance_module._monthly_values = cached
    try:
        yield statistics
    finally:
        release_module._monthly_values = original_release
        balance_module._monthly_values = original_balance
        statistics["parameter_objects"] = len(references)
        statistics["cached_months"] = len(cache)
        statistics["bindings_restored"] = (
            release_module._monthly_values is original_release
            and balance_module._monthly_values is original_balance
        )
        cache.clear()
        references.clear()


def _measure(problem, vector, *, cached):
    """Make exactly one unprofiled calibration evaluation and record timing."""
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    if cached:
        with evaluation_monthly_cache() as statistics:
            evaluation = problem.evaluate(vector)
    else:
        statistics = None
        evaluation = problem.evaluate(vector)
    elapsed = {"cpu_seconds": time.process_time() - cpu_start,
               "wall_seconds": time.perf_counter() - wall_start}
    if not evaluation.feasible:
        raise RuntimeError(f"A previously feasible selected candidate was rejected: {evaluation.reason}")
    return evaluation, elapsed, statistics


def _frame_sha(frame):
    descriptor = json.dumps({"columns": list(frame.columns), "dtypes": [str(value) for value in frame.dtypes],
                             "index_dtype": str(frame.index.dtype), "index_name": frame.index.name},
                            sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(descriptor)
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def run_benchmark(repo_root, output_dir):
    root = Path(repo_root).resolve()
    output = (root / output_dir).resolve()
    config_path = root / "configs/trinity_adaptive.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = (root / config["observations_file"]).resolve()
    archive = root / "outputs/trinity_adaptive/seed_42"
    if (not output.is_relative_to(root) or output == archive or output.is_relative_to(archive)
            or any(path == output or path.is_relative_to(output) for path in (source, config_path, archive))):
        raise ValueError("Prototype output must stay in the repository and protect input/archive paths.")
    if output.exists():
        raise ValueError("Use a new prototype output directory; previous replays must retain their accounting.")
    policy_path = (root / config["adaptive"]["stopping_policy_file"]).resolve()
    policy = StoppingPolicy(**json.loads(policy_path.read_text(encoding="utf-8")))
    identity = _expected_identity(config, policy, 16, 42, _sha(source), 2048)
    state, report, proof = _verify_run(archive, identity, 2048)
    source_hashes = {str(path): _sha(path) for path in (config_path, policy_path, source)}
    code_hashes = _code_hashes()
    physical = dict(config["model"])
    channel = physical.pop("channel_capacity_m3s", None)
    period = DatePeriod(**config["periods"]["calibration"])
    observed = read_reservoir_observations(source)
    estimates = estimate_generalized_parameters(select_period(observed, period),
        tuple(config["target_percentiles"]), channel_capacity_m3s=channel,
        quantile_method=config["quantile_method"])
    template = ReservoirParameters.from_generalized(estimates, **physical)
    bounds = estimate_calibration_bounds(observed, period, template, quantile_method=config["quantile_method"])
    problem = CalibrationObjective(observed, period, template, warmup_steps=config["periods"]["warmup_steps"],
                                   bounds=bounds, **config["simulation"])
    if any(not np.array_equal(actual, expected) for actual, expected in (
            (state.baseline_vector, parameters_to_vector(template)),
            (state.lower_vector, bounds.lower_vector), (state.upper_vector, bounds.upper_vector))):
        raise ValueError("The archived baseline or bounds differ from current calibration-derived inputs.")
    selected = report["selected_candidate_ids"]
    candidates = {}
    for role in ROLES:
        identifier = selected[role]
        if isinstance(identifier, bool) or not isinstance(identifier, int) or not 0 <= identifier < state.evaluations:
            raise ValueError("The archive's selected candidate IDs are invalid.")
        candidates.setdefault(identifier, []).append(role)
    output.mkdir(parents=True)
    attempts, rows = [], []
    summary = {"experiment": "isolated_per_evaluation_monthly_lookup_cache_prototype",
        "status": "running", "production_code_modified": False, "checkpoint_identity_rebound": False,
        "validation_replays": 0, "optimizer_trials": 0, "calibration_replay_attempts": 0,
        "calibration_replays_completed": 0, "candidate_count": len(candidates), "source_proof": proof,
        "source_sha256": source_hashes, "search_code_sha256": code_hashes,
        "prototype_sha256": _sha(Path(__file__)), "cache_scope": "One candidate evaluation; parameter-object identity and calendar month",
        "calibration_period": period.as_dict(), "simulation_period": problem.simulation_period.as_dict(),
        "solver": problem.solver, "overflow_mode": problem.overflow_mode,
        "qualification": "Research prototype in its own process. Every candidate runs original then cached once, without cProfile. Concurrent live searches, ordering and the small sample limit timing interpretation. Ratios describe only these evaluations, not complete optimizer throughput. No production change or checkpoint compatibility claim."}

    def publish():
        summary["calibration_replay_attempts"] = len(attempts)
        summary["calibration_replays_completed"] = sum(row["status"] == "complete" for row in attempts)
        summary["replay_ledger"] = attempts
        summary["candidates"] = rows
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if rows:
            pd.DataFrame(rows).to_csv(output / "candidate_comparison.csv", index=False)

    try:
        for identifier, roles in candidates.items():
            candidate = state.evaluated[identifier]
            measured = {}
            for mode in ("original", "cached"):
                attempt = {"evaluation_id": identifier, "roles": roles, "mode": mode, "status": "started"}
                attempts.append(attempt)
                publish()
                measured[mode] = _measure(problem, candidate.vector, cached=mode == "cached")
                attempt["status"] = "complete"
                publish()
            original, original_time, _ = measured["original"]
            cached, cached_time, cache = measured["cached"]
            if original.objectives != candidate.objectives or cached.objectives != original.objectives:
                raise AssertionError("Baseline/selected objective costs differ from the archive or cached evaluation.")
            pd.testing.assert_frame_equal(original.simulation.frame, cached.simulation.frame,
                                          check_exact=True, check_dtype=True, check_index_type=True,
                                          check_column_type=True, check_names=True)
            if original.metrics != cached.metrics or original.paired_counts != cached.paired_counts:
                raise AssertionError("Cached and original calibration metrics or paired counts differ.")
            if not cache["bindings_restored"] or cache["parameter_objects"] != 1 or cache["cached_months"] != 12:
                raise AssertionError("The lookup cache was not confined to one evaluation's parameter object/months.")
            original.simulation.frame.to_csv(output / f"candidate_{identifier}_reference_simulation.csv")
            row = {"evaluation_id": identifier, "roles": ";".join(roles),
                "release_cost": original.objectives[0], "storage_cost": original.objectives[1],
                "simulation_rows": len(original.simulation.frame), "frames_exactly_equal": True,
                "objectives_exactly_equal": True, "metrics_exactly_equal": True,
                "original_cpu_seconds": original_time["cpu_seconds"], "cached_cpu_seconds": cached_time["cpu_seconds"],
                "original_wall_seconds": original_time["wall_seconds"], "cached_wall_seconds": cached_time["wall_seconds"],
                "cpu_speedup": original_time["cpu_seconds"] / cached_time["cpu_seconds"] if cached_time["cpu_seconds"] else None,
                "wall_speedup": original_time["wall_seconds"] / cached_time["wall_seconds"] if cached_time["wall_seconds"] else None,
                "reference_frame_sha256": _frame_sha(original.simulation.frame),
                "cached_frame_sha256": _frame_sha(cached.simulation.frame), **cache}
            rows.append(row)
            publish()
            print(f"Candidate {identifier} ({', '.join(roles)}): exact trajectory/cost parity; "
                  f"original {original_time['cpu_seconds']:.3f} CPU s, cached {cached_time['cpu_seconds']:.3f} CPU s", flush=True)
        if (_code_hashes() != code_hashes or any(_sha(path) != digest for path, digest in source_hashes.items())
                or _verify_run(archive, identity, 2048)[2] != proof):
            raise RuntimeError("Prototype sources changed during the benchmark; results require review.")
        summary["status"] = "complete"
        summary["all_frames_and_costs_exactly_equal"] = True
        summary["unprofiled_original_cpu_seconds_total"] = sum(row["original_cpu_seconds"] for row in rows)
        summary["unprofiled_cached_cpu_seconds_total"] = sum(row["cached_cpu_seconds"] for row in rows)
        summary["unprofiled_original_wall_seconds_total"] = sum(row["original_wall_seconds"] for row in rows)
        summary["unprofiled_cached_wall_seconds_total"] = sum(row["cached_wall_seconds"] for row in rows)
        publish()
    except BaseException as error:
        summary["status"] = "error"
        summary["error"] = str(error)
        publish()
        raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("outputs/population_runtime_profile/cache_prototype"))
    args = parser.parse_args(argv)
    report = run_benchmark(args.repo, args.output)
    print(f"Completed {report['calibration_replays_completed']} separate calibration replays; "
          "zero validation replays and zero optimizer trials.")
    print((args.repo / args.output).resolve())


if __name__ == "__main__":
    main()
