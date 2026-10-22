"""Compare native monthly preparation with separately loaded historical code.

This performs calibration replays only. Historical checkpoints keep their
recorded identities; neither optimizer continuation nor validation is invoked.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dztr import __version__
from dztr.calibration.adaptive import _SEARCH_MODULES, _code_hashes, _write_json
from dztr.calibration.bounds import estimate_calibration_bounds
from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.checkpoint import _replace_checkpoint
from dztr.calibration.objectives import CalibrationObjective
from dztr.calibration.optimizer import NSGA2Settings
from dztr.calibration.periods import DatePeriod, select_period
from dztr.calibration.population_study import _verify_run
from dztr.calibration.stopping import StoppingPolicy
from dztr.data import read_reservoir_observations
from dztr.parameterization import estimate_generalized_parameters
from dztr.schemas import ReservoirParameters


ROLES = ("generalized_baseline", "best_release", "best_storage", "balanced")
ARCHIVES = (
    ("Trinity", 42, 2048, "configs/trinity_adaptive.json", "outputs/trinity_adaptive/seed_42"),
    ("Trinity", 43, 2048, "configs/trinity_adaptive.json", "outputs/trinity_adaptive/seed_43"),
    ("Trinity", 44, 2048, "configs/trinity_adaptive.json", "outputs/trinity_adaptive/seed_44"),
    ("Gardiner", 42, 1024, "configs/gardiner_adaptive.json", "outputs/gardiner_adaptive/seed_42"),
)
CHANGED_MODULES = {"model/balance.py", "simulation/runner.py"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reference_hashes(root):
    return {name: _sha(root / "src/dztr" / name) for name in _SEARCH_MODULES}


def _module(name, path):
    """Load a reference alias without replacing any production module."""
    if name in sys.modules:
        raise RuntimeError(f"Reference module alias already exists: {name}")
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module  # Dataclass construction needs its module registered.
    try:
        specification.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _reference_objective(root):
    release = _module("dztr.model._prepared_benchmark_reference_release", root / "src/dztr/model/release.py")
    balance = _module("dztr.model._prepared_benchmark_reference_balance", root / "src/dztr/model/balance.py")
    for name in ("_finite", "_monthly_values", "_release_for_values", "_timestep"):
        setattr(balance, name, getattr(release, name))
    runner = _module("dztr.simulation._prepared_benchmark_reference_runner", root / "src/dztr/simulation/runner.py")
    runner.solve_implicit_step = balance.solve_implicit_step
    runner.fixed_point_step = balance.fixed_point_step
    runner.dztr_release = release.dztr_release
    runner.dztr_zone = release.dztr_zone
    objectives = _module("dztr.calibration._prepared_benchmark_reference_objectives", root / "src/dztr/calibration/objectives.py")
    objectives.simulate_reservoir = runner.simulate_reservoir
    return objectives.CalibrationObjective


def _frame_sha(frame):
    descriptor = json.dumps({"columns": list(frame.columns), "dtypes": [str(value) for value in frame.dtypes],
                             "index_dtype": str(frame.index.dtype), "index_name": frame.index.name},
                            sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(descriptor)
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _write_table(path, rows):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
                                         delete=False, mode="w", encoding="utf-8", newline="") as stream:
            temporary = Path(stream.name)
            pd.DataFrame(rows).to_csv(stream, index=False)
        _replace_checkpoint(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _check_configuration(config, policy, state, identity):
    """Check historical configuration independently of current source identity."""
    resolved = json.loads(json.dumps(config))
    options = dict(resolved["optimization"])
    options.update(seed=state.settings.seed, evaluation_budget=state.evaluations)
    resolved["optimization"] = asdict(NSGA2Settings(**options))
    resolved["optimization"].pop("evaluation_budget")
    resolved["adaptive"] = {"checkpoint_interval": config["adaptive"]["checkpoint_interval"],
                            "stopping_policy": asdict(policy)}
    resolved.pop("study", None)
    resolved.pop("preflight", None)
    # JSON normalization also matches tuple/list representation in recorded identities.
    if json.loads(json.dumps(resolved)) != identity["resolved_search_configuration"]:
        raise ValueError("Reference configuration differs from its historical checkpoint identity.")


def _problems(config, root, original_objective):
    source = root / config["observations_file"]
    observed = read_reservoir_observations(source)
    period = DatePeriod(**config["periods"]["calibration"])
    physical = dict(config["model"])
    channel = physical.pop("channel_capacity_m3s", None)
    estimates = estimate_generalized_parameters(select_period(observed, period),
        tuple(config["target_percentiles"]), channel_capacity_m3s=channel, quantile_method=config["quantile_method"])
    template = ReservoirParameters.from_generalized(estimates, **physical)
    bounds = estimate_calibration_bounds(observed, period, template, quantile_method=config["quantile_method"])
    arguments = dict(warmup_steps=config["periods"]["warmup_steps"], bounds=bounds, **config["simulation"])
    return {"reference": original_objective(observed, period, template, **arguments),
            "production": CalibrationObjective(observed, period, template, **arguments)}, template, bounds


def _assert_parity(reference, production, point, saved_scores):
    if reference.feasible != production.feasible or reference.reason != production.reason:
        raise AssertionError("Production feasibility or rejection reason differs from the reference.")
    if not reference.feasible or reference.simulation is None or production.simulation is None:
        raise AssertionError("An archived selected candidate did not reproduce a feasible trajectory.")
    if reference.objectives != point.objectives or production.objectives != reference.objectives:
        raise AssertionError("Reference, production or archived objective costs differ.")
    if (reference.metrics != production.metrics or reference.paired_counts != production.paired_counts
            or reference.period != production.period):
        raise AssertionError("Production metrics, pair counts or scored period differ from the reference.")
    for evaluation in (reference, production):
        if (evaluation.objectives != tuple(saved_scores["objectives"])
                or evaluation.metrics != saved_scores["metrics"]
                or evaluation.paired_counts != saved_scores["paired_counts"]
                or evaluation.feasible != saved_scores["feasible"]
                or evaluation.reason != saved_scores["reason"]):
            raise AssertionError("A replay differs from the archive's recorded calibration scores.")
    pd.testing.assert_frame_equal(reference.simulation.frame, production.simulation.frame,
                                  check_exact=True, check_dtype=True, check_index_type=True,
                                  check_column_type=True, check_names=True)
    if (reference.simulation.issues != production.simulation.issues
            or reference.simulation.solver != production.simulation.solver
            or reference.simulation.overflow_mode != production.simulation.overflow_mode):
        raise AssertionError("Production simulation diagnostics or numerical settings differ from the reference.")


def run_benchmark(repo_root, reference_repo, output_dir, *, reference_code_root=None):
    root, reference_root = Path(repo_root).resolve(), Path(reference_repo).resolve()
    code_root = reference_root if reference_code_root is None else (root / reference_code_root).resolve()
    output = (root / output_dir).resolve()
    if root != PROJECT_ROOT or (reference_root == root and (reference_code_root is None or code_root == root)):
        raise ValueError("Run this script from its production repository; shared archive/data roots require an explicit distinct reference-code-root.")
    protected = [root / "src", root / "configs", root / "examples", root / "tests", code_root]
    if reference_root != root:
        protected.append(reference_root)
    else:
        protected.extend([root / "ResDataInflowStorageRelease", *(root / archive for *_, archive in ARCHIVES)])
    if (not output.is_relative_to(root) or output == root
            or any(path == output or path.is_relative_to(output) or output.is_relative_to(path) for path in protected)):
        raise ValueError("Benchmark output must stay within this repository and protect source/reference directories.")
    if output.exists():
        raise ValueError("Use a new benchmark output directory to preserve previous replay accounting.")
    production_code, reference_code = _code_hashes(), _reference_hashes(code_root)
    changed = {name for name in production_code if production_code[name] != reference_code[name]}
    if changed != CHANGED_MODULES:
        raise ValueError(f"Expected only the prepared balance/runner change; actual changed modules: {sorted(changed)}")
    original_objective = _reference_objective(code_root)
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "dztr": __version__}
    input_hashes, plans, proofs = {}, [], []
    for reservoir, seed, budget, config_name, archive_name in ARCHIVES:
        config_path, archive = reference_root / config_name, reference_root / archive_name
        config = json.loads(config_path.read_text(encoding="utf-8"))
        policy_path = reference_root / config["adaptive"]["stopping_policy_file"]
        policy = StoppingPolicy(**json.loads(policy_path.read_text(encoding="utf-8")))
        state, report, proof = _verify_run(archive, budget=budget)
        identity = proof["identity"]
        if identity["search_code_sha256"] != reference_code or identity["software_versions"] != versions:
            raise ValueError("Historical code or software differs from the original reference environment.")
        if state.settings.seed != seed or state.settings.population_size != 16:
            raise ValueError("Historical seed or population differs from the planned source archive.")
        _check_configuration(config, policy, state, identity)
        source = reference_root / config["observations_file"]
        if _sha(source) != identity["observations_sha256"]:
            raise ValueError("Reference observations differ from the historical checkpoint.")
        for path in (config_path, policy_path, source):
            input_hashes[str(path)] = _sha(path)
        problems, template, bounds = _problems(config, reference_root, original_objective)
        if any(not np.array_equal(actual, expected) for actual, expected in (
                (state.baseline_vector, parameters_to_vector(template)),
                (state.lower_vector, bounds.lower_vector), (state.upper_vector, bounds.upper_vector))):
            raise ValueError("Historical baseline or bounds differ from current calibration-only reconstruction.")
        selected = report["selected_candidate_ids"]
        candidates = {}
        for role in ROLES:
            identifier = selected[role]
            if isinstance(identifier, bool) or not isinstance(identifier, int) or not 0 <= identifier < state.evaluations:
                raise ValueError("Historical representative evaluation ID is invalid.")
            candidates.setdefault(identifier, []).append(role)
        proofs.append(proof)
        for identifier, roles in candidates.items():
            point = state.evaluated[identifier]
            saved = report["selected_scores"][roles[0]]["calibration"]
            if any(report["selected_scores"][role]["calibration"] != saved for role in roles):
                raise ValueError("Shared representative roles have inconsistent archived calibration scores.")
            plans.append({"reservoir": reservoir, "seed": seed, "archived_search_evaluations": budget,
                          "evaluation_id": identifier, "roles": roles, "point": point,
                          "saved_scores": saved, "problems": problems})
    if len(plans) != 16:
        raise ValueError("The planned four archives must supply exactly sixteen distinct archive/candidate pairs.")
    benchmark_digest = _sha(Path(__file__))

    def unchanged():
        if (_code_hashes() != production_code or _reference_hashes(code_root) != reference_code
                or _sha(Path(__file__)) != benchmark_digest
                or any(_sha(path) != digest for path, digest in input_hashes.items())):
            raise RuntimeError("Benchmark sources or observations changed; results require review.")

    output.mkdir(parents=True)
    attempts, rows = [], []
    summary = {"experiment": "native_per_simulation_monthly_preparation", "status": "running",
        "production_repo": str(root), "reference_repo": str(reference_root), "reference_code_root": str(code_root),
        "production_search_code_sha256": production_code, "reference_search_code_sha256": reference_code,
        "changed_modules": sorted(changed), "benchmark_sha256": benchmark_digest,
        "input_sha256": input_hashes, "source_proofs": proofs, "software_versions": versions,
        "optimizer_trials": 0, "validation_replays": 0, "optimizer_resume_attempts": 0,
        "checkpoint_identity_rebound": False, "candidate_pairs_planned": len(plans),
        "measurement_scope": "Complete calibration objective replay, including candidate validation, daily simulation and metrics; construction and source verification excluded.",
        "mode_order": "Reference first for even candidate-pair indices; production first for odd indices.",
        "qualification": "One unprofiled replay per mode for each archived candidate, with alternating mode order. This small, selected sample and concurrent machine load limit timing interpretation. Ratios describe these complete calibration evaluations, not optimizer throughput or universal fleet performance. Historical checkpoint identities remain unchanged."}

    def publish():
        summary["calibration_replay_attempts"] = len(attempts)
        summary["calibration_replays_completed"] = sum(attempt["status"] == "complete" for attempt in attempts)
        summary["candidate_pairs_verified"] = len(rows)
        summary["replay_ledger"] = attempts
        summary["candidate_comparisons"] = rows
        if attempts:
            _write_table(output / "replay_attempts.csv", attempts)
        if rows:
            _write_table(output / "candidate_comparison.csv", rows)
        _write_json(output / "summary.json", summary)

    try:
        publish()
        for ordinal, plan in enumerate(plans):
            measured = {}
            order = ("reference", "production") if ordinal % 2 == 0 else ("production", "reference")
            for mode in order:
                unchanged()
                attempt = {name: plan[name] for name in ("reservoir", "seed", "evaluation_id")}
                attempt.update(pair_index=ordinal, mode=mode, status="started")
                attempts.append(attempt)
                publish()
                wall_start, cpu_start = time.perf_counter(), time.process_time()
                evaluation = plan["problems"][mode].evaluate(plan["point"].vector)
                elapsed = {"cpu_seconds": time.process_time() - cpu_start,
                           "wall_seconds": time.perf_counter() - wall_start}
                measured[mode] = evaluation, elapsed
                attempt.update(status="complete", **elapsed)
                publish()
            reference, reference_time = measured["reference"]
            production, production_time = measured["production"]
            _assert_parity(reference, production, plan["point"], plan["saved_scores"])
            row = {name: plan[name] for name in ("reservoir", "seed", "archived_search_evaluations", "evaluation_id")}
            row.update(pair_index=ordinal, roles=";".join(plan["roles"]), mode_order=";".join(order),
                release_cost=reference.objectives[0], storage_cost=reference.objectives[1],
                simulation_rows=len(reference.simulation.frame), frames_exactly_equal=True,
                objectives_exactly_equal=True, metrics_exactly_equal=True, feasibility_exactly_equal=True,
                counts_exactly_equal=True, issues_exactly_equal=True,
                reference_frame_sha256=_frame_sha(reference.simulation.frame),
                production_frame_sha256=_frame_sha(production.simulation.frame),
                reference_cpu_seconds=reference_time["cpu_seconds"], production_cpu_seconds=production_time["cpu_seconds"],
                reference_wall_seconds=reference_time["wall_seconds"], production_wall_seconds=production_time["wall_seconds"],
                cpu_speedup=reference_time["cpu_seconds"] / production_time["cpu_seconds"] if production_time["cpu_seconds"] else None,
                wall_speedup=reference_time["wall_seconds"] / production_time["wall_seconds"] if production_time["wall_seconds"] else None)
            rows.append(row)
            publish()
            print(f"{plan['reservoir']} seed {plan['seed']}, candidate {plan['evaluation_id']} "
                  f"({', '.join(plan['roles'])}): exact parity; reference {reference_time['cpu_seconds']:.3f} CPU s, "
                  f"production {production_time['cpu_seconds']:.3f} CPU s", flush=True)
        unchanged()
        for proof in proofs:
            if _verify_run(proof["directory"], proof["identity"])[2] != proof:
                raise RuntimeError("A historical source archive changed during the benchmark.")
        summary.update(status="complete", all_frames_and_scores_exactly_equal=True)
        for mode in ("reference", "production"):
            for timing in ("cpu", "wall"):
                summary[f"{mode}_{timing}_seconds_total"] = sum(row[f"{mode}_{timing}_seconds"] for row in rows)
        for timing in ("cpu", "wall"):
            denominator = summary[f"production_{timing}_seconds_total"]
            summary[f"aggregate_{timing}_speedup"] = summary[f"reference_{timing}_seconds_total"] / denominator if denominator else None
        summary["reservoir_totals"] = []
        for reservoir in dict.fromkeys(row["reservoir"] for row in rows):
            selected = [row for row in rows if row["reservoir"] == reservoir]
            totals = {"reservoir": reservoir, "candidate_pairs": len(selected)}
            for mode in ("reference", "production"):
                for timing in ("cpu", "wall"):
                    totals[f"{mode}_{timing}_seconds_total"] = sum(row[f"{mode}_{timing}_seconds"] for row in selected)
            for timing in ("cpu", "wall"):
                denominator = totals[f"production_{timing}_seconds_total"]
                totals[f"aggregate_{timing}_speedup"] = totals[f"reference_{timing}_seconds_total"] / denominator if denominator else None
            summary["reservoir_totals"].append(totals)
        publish()
        _write_json(output / "benchmark_manifest.json", {"schema_version": 1,
            "artifact_sha256": {name: _sha(output / name) for name in ("summary.json", "candidate_comparison.csv", "replay_attempts.csv")},
            "benchmark_sha256": benchmark_digest, "historical_sources_verified": True,
            "production_sources_unchanged": True, "reference_sources_unchanged": True})
    except BaseException as error:
        if attempts and attempts[-1]["status"] == "started":
            attempts[-1].update(status="error", error=str(error))
        summary.update(status="error", error=str(error))
        publish()
        raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--reference-code-root", type=Path,
                        help="Original source tree with src/dztr, separate from reference archives/config/data.")
    parser.add_argument("--output", type=Path, default=Path("outputs/prepared_simulation_benchmark"))
    arguments = parser.parse_args(argv)
    report = run_benchmark(arguments.repo, arguments.reference_repo, arguments.output,
                           reference_code_root=arguments.reference_code_root)
    print(f"Verified {report['candidate_pairs_verified']} exact candidate pairs in "
          f"{report['calibration_replays_completed']} calibration replays; zero validation replays and optimizer trials.")
    print(f"Aggregate CPU ratio: {report['aggregate_cpu_speedup']:.3f}x; "
          f"wall ratio: {report['aggregate_wall_speedup']:.3f}x.")
    print((arguments.repo / arguments.output).resolve())


if __name__ == "__main__":
    main()
