"""Prepare, execute, and monitor reviewed fleet jobs with integrated reporting.

Preparation and archive reports make zero model calls. Execution is a separate
explicit command; it never promotes reservoirs through failed source gates.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/fleet_workflow_20261010"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def code_hashes():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / "src/dztr").rglob("*.py"))}


def verify(root, proof, name):
    hashes = {k.replace("\\", "/"): v for k, v in proof["artifact_sha256"].items()}
    if sha(root / name) != hashes.get(name):
        raise ValueError(f"Changed historical evidence: {name}")


def prepare(out):
    if out.exists() or not out.is_relative_to(ROOT):
        raise ValueError("Choose a fresh fleet output inside the repository.")
    old = ROOT / "outputs/ready_cohort_20261008"
    proof = read(old / "final_verification.json")
    if proof["status"] != "passed":
        raise ValueError("Source-review verification did not pass.")
    names = ["cohort.json", "run_pilots.py", "protocol.json"]
    for name in names:
        verify(ROOT, proof, f"outputs/ready_cohort_20261008/{name}")
    cohort = read(old / "cohort.json")
    manifest_path = ROOT / "data/reservoirs/manifest.json"
    manifest = read(manifest_path)
    if len(manifest["reservoirs"]) != cohort["active_reservoirs"]:
        raise ValueError("Collection membership has changed; review it before execution.")
    reviews = {r["reservoir_id"]: r for r in cohort["reservoirs"]}
    selected = []
    hashes = {str(manifest_path.relative_to(ROOT)): sha(manifest_path)}
    statuses = []
    for entry in manifest["reservoirs"]:
        identifier = entry["reservoir_id"]
        record = reviews[identifier]
        for name, digest in entry["files_sha256"].items():
            if sha(ROOT / name) != digest:
                raise ValueError(f"Changed input: {name}")
            hashes[name] = digest
        eligible = record["eligible_for_exploratory_calibration"]
        # Eligibility uses recorded calibration evidence, never held-out fit.
        if eligible:
            for name, digest in record["input_sha256"].items():
                if sha(ROOT / name) != digest:
                    raise ValueError("Reviewed calibration inputs changed.")
            selected.append(identifier)
        status = "planned_awaiting_execution_authorization" if eligible else "deferred_source_or_parameter_issue"
        reasons = record["reasons"]
        if identifier == "ex35_Bull_Lake":
            status = "verified_population32_results_available"
            reasons = ["Reuse three512-evaluation interval archives under the separate computed-net policy."]
        statuses.append({"reservoir_id": identifier, "name": entry["name"], "frequency": entry["frequency"],
                         "status": status, "reason": " | ".join(reasons), "physical_source_certified": False})
    prior = ROOT / "outputs/multi_reservoir_archive_20261010"
    prior_proof = read(prior / "workflow_verification.json")
    if prior_proof["status"] != "passed":
        raise ValueError("Prior result exports did not pass verification.")
    for name, digest in prior_proof["artifact_sha256"].items():
        if sha(prior / name) != digest:
            raise ValueError("Prior result export has changed.")
    out.mkdir(parents=True)
    cohort["selected_for_pilot"] = selected
    cohort["pilot_seeds"] = [42, 43, 44]
    for r in cohort["reservoirs"]:
        r["pilot_selected"] = r["reservoir_id"] in selected
    cohort["selection_reason"] = "All hash-verified reservoirs passing the inherited calibration-only exploratory gates. No validation-based selection."
    write(out / "cohort.json", cohort)
    jobs = [{"reservoir_id": identifier, "seed": seed, "population_size": 32, "evaluation_budget": 512}
            for identifier in selected for seed in (42, 43, 44)]
    protocol = {"schema_version": 1, "kind": "reviewed_fleet_protocol", "execution_authorization": "required_separately",
                "pilot": {"population_size": 32, "evaluation_budget_per_job": 512, "seeds": [42, 43, 44],
                          "solver": "strict implicit", "overflow": "conservative", "root_policy": "reject"},
                "deferred_reservoirs": {r["reservoir_id"]: r["reason"] for r in statuses if r["status"] == "deferred_source_or_parameter_issue"},
                "excluded_reservoirs": manifest["excluded_reservoirs"], "code_sha256": code_hashes(),
                "inputs_sha256": hashes, "fleet_driver_sha256": sha(Path(__file__)),
                "inherited_job_runner": "outputs/ready_cohort_20261008/run_pilots.py",
                "inherited_job_runner_sha256": sha(old / "run_pilots.py"),
                "review_sha256": sha(old / "final_verification.json"), "prior_export_sha256": sha(prior / "workflow_verification.json"),
                "counts": {"active_reservoirs": len(statuses), "eligible_new_reservoirs": len(selected), "new_jobs": len(jobs),
                           "maximum_search_evaluations": len(jobs) * 512, "maximum_extra_reporting_calls_per_job": 8},
                "reporting": "Inherited runner performs one baseline check, up to3 selected calibration replays and up to4 held-out reports, separately counted. IDs are frozen before held-out reporting.",
                "limitations": "Exploratory inherited source assumptions; cap512 does not establish convergence. Monthly30.5-day integration and all constants remain unchanged."}
    write(out / "protocol.json", protocol)
    write(out / "job_plan.json", {"jobs": jobs, "worker_processes": 3, "new_evaluations_started": 0})
    import pandas as pd
    pd.DataFrame(statuses).to_csv(out / "reservoir_status.csv", index=False)
    print(json.dumps(protocol["counts"], indent=2))


def checked(out):
    protocol = read(out / "protocol.json")
    if code_hashes() != protocol["code_sha256"] or sha(Path(__file__)) != protocol["fleet_driver_sha256"]:
        raise ValueError("Fleet numerical code or driver changed; prepare a new worklist.")
    for name, digest in protocol["inputs_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Fleet input changed: {name}")
    if sha(ROOT / protocol["inherited_job_runner"]) != protocol["inherited_job_runner_sha256"]:
        raise ValueError("Inherited job runner changed.")
    return protocol


def run_job(out, identifier, seed):
    checked(out)
    jobs = read(out / "job_plan.json")["jobs"]
    if not any(j["reservoir_id"] == identifier and j["seed"] == seed for j in jobs):
        raise ValueError("Job is outside the reviewed plan.")
    folder = out / "pilots" / identifier / f"seed_{seed}"
    if folder.exists():
        raise FileExistsError("Existing checkpoints/results are preserved; do not restart over them.")
    folder.mkdir(parents=True)
    sys.path.insert(0, str(ROOT / "src"))
    source = ROOT / "outputs/ready_cohort_20261008/run_pilots.py"
    spec = importlib.util.spec_from_file_location("inherited_fleet_runner", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.OUT = ROOT, out
    try:
        result = module.run_one(identifier, seed)
        checked(out)
        write(folder / "fleet_completion.json", {"status": "passed", "search_evaluations": result["search_evaluations"],
                                                  "validation_replays": result["validation_replays"],
                                                  "artifact_sha256": {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}})
    except Exception as error:
        write(folder / "fleet_failure.json", {"status": "failed", "reason": str(error), "checkpoint_preserved": (folder / "checkpoint.npz").exists()})
        raise


def run_all(out):
    checked(out)
    logs = out / "logs"
    logs.mkdir(exist_ok=False)
    jobs = read(out / "job_plan.json")["jobs"]
    def execute(job):
        identifier, seed = job["reservoir_id"], job["seed"]
        log = logs / f"{identifier}_seed_{seed}.log"
        cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--output", str(out), "--job", identifier, str(seed)]
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return {**job, "status": "completed" if result.returncode == 0 else "failed", "exit_code": result.returncode,
                "log_file": log.relative_to(out).as_posix(), "log_sha256": sha(log)}
    completed = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(execute, j) for j in jobs]
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            # Persist only complete-job status; generation checkpoints are
            # written atomically inside each independent process.
            write(out / "job_status.json", completed)
            print(json.dumps(result), flush=True)
    write(out / "execution_summary.json", {"status": "completed" if all(r["exit_code"] == 0 for r in completed) else "completed_with_failures", "jobs": completed})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run-authorized", action="store_true", help="Explicitly launch the prepared calibration and reporting jobs.")
    mode.add_argument("--job", nargs=2, metavar=("RESERVOIR", "SEED"))
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Fleet outputs must stay inside the repository.")
    if args.prepare:
        prepare(out)
    elif args.job:
        run_job(out, args.job[0], int(args.job[1]))
    else:
        run_all(out)


if __name__ == "__main__":
    main()
