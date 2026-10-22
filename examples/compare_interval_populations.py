"""Run a reviewed interval population comparison with logs and checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dztr import __version__
import numpy as np
import pandas as pd
from dztr.calibration import interval_search as search
from dztr.calibration.checkpoint import save_optimizer_checkpoint, load_optimizer_checkpoint
from dztr.calibration.interval_resume import resume_interval_seed
from dztr.calibration.optimizer import NSGA2Settings
from dztr.calibration.stopping import StoppingPolicy

DEFAULT_OUT = ROOT / "outputs/bull_lake_interval_population_20261009"
DEFAULT_CONFIG = ROOT / "configs/reviews/bull_lake_interval_population.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Preserve existing evidence: {path}")
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(config_path, out):
    config = read(config_path)
    preserved = read(out / "preservation_inputs.json")
    changed = "src/dztr/calibration/interval_resume.py"
    for category, base in (("historical_files_sha256", ROOT), ("production_modules_sha256", ROOT), ("external_source_sha256", Path())):
        for name, expected in preserved[category].items():
            if name != changed:
                assert sha(base / name) == expected, name
    assert sha(out / "historical_code/interval_resume_stage8.py") == preserved["production_modules_sha256"][changed]
    old_modules = {Path(k).relative_to("src/dztr").as_posix(): v for k, v in preserved["production_modules_sha256"].items()}
    current = search._module_hashes()
    assert set(current) == set(old_modules) and all(current[k] == v for k, v in old_modules.items() if k != "calibration/interval_resume.py")
    assert current["calibration/interval_resume.py"] != old_modules["calibration/interval_resume.py"]
    cached = (config_path.parent / config["cached_population16"]).resolve()
    parent = read(cached / "verification.json")
    assert parent["status"] == "passed"
    for name, expected in parent["artifact_sha256"].items():
        assert sha(cached / name) == expected, name
    receipt = {"status": "passed", "config_file": str(config_path), "config_sha256": sha(config_path),
               "runner_sha256": sha(Path(__file__)), "cached_parent": str(cached), "cached_verification_sha256": sha(cached / "verification.json"),
               "code_sha256": current, "approved_adapter_change": {"old_sha256": old_modules["calibration/interval_resume.py"],
                                                                    "new_sha256": current["calibration/interval_resume.py"]},
               "cached_attempts": 1536, "declared_new_population_attempts": 3072, "declared_comparison_attempts": 4608,
               "separate_compatibility_new_calls": 16, "validation_calls": 0}
    write(out / "preflight.json", receipt)
    print(json.dumps(receipt, indent=2))


def checked(config_path, out):
    config, receipt = read(config_path), read(out / "preflight.json")
    assert sha(config_path) == receipt["config_sha256"] and sha(Path(__file__)) == receipt["runner_sha256"]
    assert search._module_hashes() == receipt["code_sha256"]
    return config, receipt


def run_job(config_path, out, population, seed):
    config, preflight = checked(config_path, out)
    assert population in (32, 64) and seed in config["seeds"]
    parent = out / "runs" / f"population_{population}"
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / f"seed_{seed}"
    live = out / "work" / f"population_{population}_seed_{seed}"
    live.mkdir(parents=True, exist_ok=False)
    settings = NSGA2Settings(population_size=population, seed=seed, evaluation_budget=config["evaluation_budget"], **config["optimization"])
    stopping = StoppingPolicy(population_size=population, **config["common_stopping"])
    policy_path = (config_path.parent / config["policy_file"]).resolve()
    loaded = search.load_interval_problem(policy_path)
    context = loaded[3]
    identity = {"policy_id": context["policy"]["policy_id"], "consumed_files_sha256": context["consumed_files_sha256"],
                "code_sha256": search._module_hashes(), "software_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "dztr": __version__},
                "nonbudget_settings": {k: v for k, v in asdict(settings).items() if k != "evaluation_budget"},
                "root_policy": "reject", "initial_storage_offset_capacity_fraction": 0.0, "periods": context["periods"], "bounds": context["bounds"].summary()}
    write(live / "identity.json", identity)
    optimizer, loader = search.optimize_nsga2, search.load_interval_problem

    def persisted_optimizer(*args, checkpoint_callback=None, **kwargs):
        def persist(state):
            if checkpoint_callback is not None:
                checkpoint_callback(state)
            save_optimizer_checkpoint(live / "optimizer_checkpoint.npz", state, identity)
        return optimizer(*args, checkpoint_callback=persist, **kwargs)

    # Process-local wrappers only add durable state writes. Optimizer arguments,
    # objectives, random stream, survivor selection and equations stay intact.
    search.optimize_nsga2 = persisted_optimizer
    search.load_interval_problem = lambda path: loaded
    try:
        report = search.run_interval_seed(policy_path, output, settings=settings,
            checkpoints=config["evaluation_checkpoints"], stopping_policy=stopping,
            progress_callback=lambda r: print(json.dumps(asdict(r)), flush=True))
    finally:
        search.optimize_nsga2, search.load_interval_problem = optimizer, loader
    assert report["checkpoint_identity"] == identity
    saved = load_optimizer_checkpoint(live / "optimizer_checkpoint.npz", identity)
    final = load_optimizer_checkpoint(output / "optimizer_checkpoint.npz", identity)
    assert saved.rng_state == final.rng_state and saved.population_ids == final.population_ids
    np.testing.assert_array_equal([p.vector for p in saved.evaluated], [p.vector for p in final.evaluated])
    write(live / "completion.json", {"status": "passed", "population_size": population, "seed": seed,
        "search_evaluations": report["search_evaluations"], "validation_calls": 0,
        "live_checkpoint_matches_final": True, "source_files_sha256": report["consumed_files_sha256"],
        "artifact_sha256": {name: sha(output / name) for name in search.ARTIFACT_NAMES}})


def run_all(config_path, out):
    config, _ = checked(config_path, out)
    logs = out / "logs"
    logs.mkdir(exist_ok=False)
    jobs = [(population, seed) for population in (32, 64) for seed in config["seeds"]]
    def execute(job):
        population, seed = job
        log = logs / f"population_{population}_seed_{seed}.log"
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--config", str(config_path), "--output", str(out), "--job", str(population), str(seed)]
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return {"population_size": population, "seed": seed, "exit_code": result.returncode,
                "log_file": log.relative_to(out).as_posix(), "log_sha256": sha(log)}
    statuses = []
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures = {pool.submit(execute, job): job for job in jobs}
        for future in as_completed(futures):
            status = future.result()
            statuses.append(status)
            print(json.dumps(status), flush=True)
    write(out / "job_status.json", sorted(statuses, key=lambda r: (r["population_size"], r["seed"])))
    if any(row["exit_code"] for row in statuses):
        raise RuntimeError("A population job failed; preserve its log and checkpoint for diagnosis.")


def compatibility(config_path, out):
    config, preflight = checked(config_path, out)
    parent = Path(preflight["cached_parent"])
    output = out / "compatibility" / "seed_44"
    output.parent.mkdir(exist_ok=False)
    approved = {**preflight["approved_adapter_change"], "prior_verification_path": str((parent / "verification.json").resolve()),
                "prior_verification_sha256": preflight["cached_verification_sha256"]}
    report = resume_interval_seed(parent / "seed_44", config_path.parent / config["policy_file"], output,
                                  total_evaluation_budget=528, checkpoints=(512, 528), approved_adapter_migration=approved)
    assert report["inherited_evaluations"] == 512 and report["new_search_evaluations"] == 16
    assert report["reporting_validation_evaluations"] == 0 and not report["baseline_reevaluated"]
    from dztr.calibration.interval_resume import _extended_identity
    continued_identity = _extended_identity(report["checkpoint_identity"], search._module_hashes())
    assert continued_identity["identity_extension"]["origin_identity_sha256"] == report["checkpoint_identity"]["identity_extension"]["origin_identity_sha256"]
    write(output.parent / "verification.json", {"status": "passed", "scope": "Separate16-call native historical-checkpoint demonstration; excluded from population comparison.",
        "artifact_sha256": {f"seed_44/{name}": sha(output / name) for name in search.ARTIFACT_NAMES}})
    write(out / "compatibility_check.json", {"status": "passed", "inherited_evaluations": 512, "new_evaluations": 16,
        "total_evaluations": 528, "validation_calls": 0, "included_in_population_comparison": False,
        "inherited_prefix_preserved": True, "subsequent_chain_identity_accepted": True, "approved_adapter_migration": approved})
    print("Historical fresh checkpoint continued512->528 with16 new calibration calls.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--job", nargs=2, type=int, metavar=("POPULATION", "SEED"))
    group.add_argument("--compatibility", action="store_true")
    args = parser.parse_args()
    config_path, out = args.config.resolve(), args.output.resolve()
    if args.prepare:
        prepare(config_path, out)
    elif args.run:
        run_all(config_path, out)
    elif args.job:
        run_job(config_path, out, *args.job)
    else:
        compatibility(config_path, out)


if __name__ == "__main__":
    main()
