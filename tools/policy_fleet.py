"""Prepare or explicitly execute the expanded policy-aware reservoir fleet."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys

from policy_fleet_job import read, sha, write

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/policy_fleet_20261012"


def prepare(out):
    if out.exists() or not out.resolve().is_relative_to(ROOT):
        raise ValueError("Choose a fresh output inside the repository.")
    source = ROOT / "outputs/fleet_admission_verified_20261012"
    final = read(source / "final_verification.json")
    for name, digest in final["artifact_sha256"].items():
        if sha(source / name) != digest:
            raise ValueError("Admission evidence changed.")
    evidence = read(source / "verification.json")
    for name, digest in {**evidence["inputs_sha256"], **evidence["code_sha256"]}.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Admission inputs/code changed.")
    plan = read(source / "calibration_plan.json")
    records = {r["reservoir_id"]: r for r in read(source / "admission.json")["reservoirs"]}
    for job in plan["jobs"]:
        record = records[job["reservoir_id"]]
        if not record["calibration_schedulable"] or job["root_policy"] != record["root_policy"]:
            raise ValueError("Job is inconsistent with reviewed admission.")
    out.mkdir(parents=True)
    for name in ("admission.json", "reservoir_status.csv", "work_queue.json"):
        shutil.copyfile(source / name, out / name)
    write(out / "job_plan.json", {"jobs": plan["jobs"], "worker_processes": 3, "execution_started": False})
    protocol = {"kind": "policy_aware_fleet", "code_sha256": evidence["code_sha256"], "inputs_sha256": evidence["inputs_sha256"],
        "guarded_artifacts_sha256": {name: sha(out / name) for name in ("admission.json", "job_plan.json")},
        "tools_sha256": {name: sha(ROOT / name) for name in ("tools/policy_fleet.py", "tools/policy_fleet_job.py", "tools/fleet_reports.py")},
        "admission_evidence_sha256": sha(source / "final_verification.json"), "search_budget_per_job": 512,
        "max_initialization_checks_per_fresh_job": 2, "max_selected_calibration_replays": 3,
        "max_validation_attempts": 4, "extra_resume_baseline_checks_per_invocation": 1,
        "initialization": "Generalized candidate followed, when available, by the saved critical-gap witness. Each must pass bounds, physical constraints and calibration simulation.",
        "selection_uses_validation": False, "initialization_uses_validation": False, "execution_authorization": "separate explicit invocation",
        "checkpoint_continuation": "Incomplete jobs may resume at the same declared budget; changing policy, inputs, versions, code or plan is rejected."}
    write(out / "protocol.json", protocol)
    write(out / "preparation_identity.json", {"protocol_sha256": sha(out / "protocol.json")})
    (out / "README.md").write_text("# Executable policy-aware fleet\n\nAll 41 reservoirs remain represented. The reviewed subset has 60 jobs across 20 reservoirs; Bull Lake results are reused separately. Each job declares its root policy. No jobs have started.\n\nFrom the repository, after execution is authorized:\n\n```powershell\npython -B tools/policy_fleet.py --run-authorized\n```\n\nThe command runs three independent processes, preserves generation checkpoints/logs and reports completed jobs. Failed initialization stops only that job. Held-out failure preserves the calibration result and its calibration trajectory.\n\nIncomplete jobs with checkpoints can be resumed explicitly with `--resume-job RESERVOIR SEED`. Completed jobs and existing fresh-job directories cannot be overwritten. The declared budget stays fixed.\n\n[All reservoir actions](reservoir_status.csv), [declared jobs](job_plan.json), [checkpoint/scientific protocol](protocol.json). Fixed budgets do not establish convergence. Source definitions remain provisional.\n", encoding="utf-8")
    print(f"Prepared {len(plan['jobs'])} executable jobs; no evaluations started.")


def checked(out):
    if sha(out / "protocol.json") != read(out / "preparation_identity.json")["protocol_sha256"]:
        raise ValueError("Reviewed protocol changed.")
    protocol = read(out / "protocol.json")
    for name, digest in {**protocol["code_sha256"], **protocol["inputs_sha256"], **protocol["tools_sha256"]}.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Checkpoint/job dependency changed: {name}")
    for name, digest in protocol["guarded_artifacts_sha256"].items():
        if sha(out / name) != digest:
            raise ValueError(f"Reviewed plan changed: {name}")
    return protocol


def run_job(out, identifier, seed, *, resume=False):
    checked(out)
    job = next((j for j in read(out / "job_plan.json")["jobs"] if j["reservoir_id"] == identifier and j["seed"] == seed), None)
    if job is None:
        raise ValueError("Job is outside the reviewed admission plan.")
    if not resume and (out / "pilots" / identifier / f"seed_{seed}").exists():
        raise FileExistsError("Existing jobs are preserved; use explicit checkpoint continuation for an incomplete job.")
    from policy_fleet_job import run
    try:
        return run(out, job, resume=resume)
    except Exception as error:
        folder = out / "pilots" / identifier / f"seed_{seed}"
        # Refusals must not mutate an existing completed job.
        if folder.exists() and not (folder / "fleet_completion.json").exists():
            write(folder / "fleet_failure.json", {"status": "failed", "reason": str(error), "checkpoint_preserved": (folder / "checkpoint.npz").exists()})
        raise


def run_all(out):
    checked(out)
    logs = out / "logs"
    logs.mkdir(exist_ok=False)
    results = []
    def execute(job):
        path = logs / f"{job['reservoir_id']}_seed_{job['seed']}.log"
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--output", str(out), "--job", job["reservoir_id"], str(job["seed"])]
        with path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return {**job, "status": "completed" if completed.returncode == 0 else "failed", "exit_code": completed.returncode, "log_file": path.relative_to(out).as_posix()}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(execute, j) for j in read(out / "job_plan.json")["jobs"]]):
            result = future.result()
            results.append(result)
            write(out / "job_status.json", results)
            print(json.dumps(result), flush=True)
    write(out / "execution_summary.json", {"status": "completed" if all(r["exit_code"] == 0 for r in results) else "completed_with_failures", "jobs": results})
    subprocess.run([sys.executable, "-B", str(ROOT / "tools/fleet_reports.py"), "--completed-fleet", str(out), "--output", str(out / "reports/completed_fleet")], check=True)
    if any(r["exit_code"] for r in results):
        raise SystemExit("Some jobs failed; successful jobs were retained and reported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run-authorized", action="store_true")
    mode.add_argument("--job", nargs=2)
    mode.add_argument("--resume-job", nargs=2)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Output must stay inside the repository.")
    if args.prepare:
        prepare(out)
    elif args.job or args.resume_job:
        identifier, seed = args.job or args.resume_job
        run_job(out, identifier, int(seed), resume=bool(args.resume_job))
    else:
        run_all(out)
