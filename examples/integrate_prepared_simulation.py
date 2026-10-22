"""Apply a pinned local simulation change after the population study publishes.

The plan explicitly names every installed file and original/prepared hash. This
helper preserves historical optimizer artifacts and leaves a recovery journal
if a per-file atomic installation is interrupted. It never commits or pushes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dztr.calibration.checkpoint import _replace_checkpoint
from dztr.calibration.population_study import verify_population_study


ADDITIONS = {
    "configs/trinity_population_study_prepared.json",
    "tests/test_prepared_simulation.py", "tests/test_prepared_integration.py",
    "docs/simulation_performance.md", "examples/benchmark_prepared_simulation.py",
    "examples/integrate_prepared_simulation.py",
    *("tests/reference_cases/uncached_simulation/" + name
      for name in ("release.py", "balance.py", "runner.py", "provenance.json")),
}
PROOF_REPORTS = {"summary.json", "candidate_comparison.csv", "replay_attempts.csv",
                 "benchmark_manifest.json", "benchmark_source.py"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inside(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or PurePosixPath(relative).as_posix() != relative
            or PureWindowsPath(relative).drive
            or PureWindowsPath(relative).is_absolute()
            or PurePosixPath(relative).is_absolute()
            or ".." in PureWindowsPath(relative).parts):
        raise ValueError(f"Unsafe relative file path: {relative!r}")
    unresolved = root / relative
    path = unresolved.resolve()
    if not path.is_relative_to(root) or path != unresolved.absolute():
        raise ValueError(f"File escapes its declared root: {relative}")
    return path


def _atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".prepared_install_", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_checkpoint(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _journal(path, state):
    _atomic_bytes(path, (json.dumps(state, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _validate_plan(plan_path, expected_sha):
    if _sha(plan_path) != expected_sha:
        raise ValueError("Integration plan changed after it was pinned.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or _sha(__file__) != plan["helper_sha256"]:
        raise ValueError("Unsupported plan or changed integration helper.")
    workspace = Path(plan["workspace_root"]).resolve()
    source = Path(plan["source_root"]).resolve()
    target = Path(plan["target_root"]).resolve()
    if (source.is_relative_to(target) or target.is_relative_to(source)
            or not source.is_relative_to(workspace)
            or not target.is_relative_to(workspace)):
        raise ValueError("Integration roots must be distinct folders inside the declared workspace.")
    entries = plan["files"]
    names = [entry["target"] for entry in entries]
    if not entries or len(set(names)) != len(names):
        raise ValueError("Integration file list is empty or has duplicate targets.")
    production = [name for name in names if name.startswith("src/")]
    if production != ["src/dztr/model/balance.py", "src/dztr/simulation/runner.py"]:
        raise ValueError("Only the two reviewed production modules may be installed, in dependency order.")
    allowed = ADDITIONS | set(production)
    allowed.update("outputs/simulation_performance/reference_code/src/dztr/" + name
                   for name in plan["original_code_sha256"])
    allowed.update("outputs/simulation_performance/" + name
                   for name in ("reference_provenance.json", "verification.json", "simulation_change.patch"))
    allowed.update("outputs/simulation_performance/" + run + "/" + name
                   for run in ("benchmark_original", "benchmark_frozen") for name in PROOF_REPORTS)
    if any(name not in allowed for name in names):
        raise ValueError("Integration contains a target outside the reviewed file allowlist.")
    if plan["receipt"] != "outputs/simulation_performance/integration_receipt.json":
        raise ValueError("Integration receipt must use its dedicated output folder.")
    for entry in entries:
        _inside(source, entry["source"])
        _inside(target, entry["target"])
    if set(plan["original_code_sha256"]) != set(plan["prepared_code_sha256"]):
        raise ValueError("Original and prepared consumed-code file sets differ.")
    changed = {name for name, digest in plan["original_code_sha256"].items()
               if digest != plan["prepared_code_sha256"][name]}
    if changed != {"model/balance.py", "simulation/runner.py"}:
        raise ValueError("The plan changes more than the two reviewed consumed modules.")
    return plan, source, target


def _preflight_files(plan, source, target, recovery=False):
    for name, digest in plan["prepared_code_sha256"].items():
        if _sha(_inside(source, "src/dztr/" + name)) != digest:
            raise ValueError(f"Prepared source changed: {name}")
        current = _sha(_inside(target, "src/dztr/" + name))
        allowed = {plan["original_code_sha256"][name]}
        if recovery:
            allowed.add(digest)
        if current not in allowed:
            raise ValueError(f"Target source changed: {name}")
    for entry in plan["files"]:
        if _sha(_inside(source, entry["source"])) != entry["sha256"]:
            raise ValueError(f"Staged install file changed: {entry['source']}")
        destination = _inside(target, entry["target"])
        current = _sha(destination) if destination.exists() else None
        allowed = {entry["original_sha256"]}
        if recovery:
            allowed.add(entry["sha256"])
        if current not in allowed:
            raise ValueError(f"Target install file changed: {entry['target']}")
    for relative, digest in plan["proof_file_sha256"].items():
        if _sha(_inside(source, relative)) != digest:
            raise ValueError(f"Prepared benchmark proof changed: {relative}")


def _publication(plan, target):
    folder = _inside(target, plan["study_directory"])
    marker = folder / "population_identity.json"
    if _sha(marker) != plan["study_identity_sha256"]:
        raise ValueError("Population study identity changed.")
    summary_path = folder / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("failed_job_count", 0):
            raise RuntimeError("Population jobs failed; integration requires a completed study.")
    required = ("study_manifest.json", "population_results.md", "population_comparison.png")
    if not all((folder / name).is_file() for name in required):
        return None
    verified = verify_population_study(folder)
    settings = verified["settings"]
    for name, expected in plan["study_settings"].items():
        if settings.get(name) != expected:
            raise ValueError(f"Completed population study has different {name}.")
    manifest_sha = _sha(folder / "study_manifest.json")
    if f"`{manifest_sha}`" not in (folder / "population_results.md").read_text(encoding="utf-8"):
        raise ValueError("Published results note does not identify the verified study manifest.")
    if not (folder / "population_comparison.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Published population figure is not a PNG.")
    return {"study_manifest_sha256": manifest_sha,
            "results_note_sha256": _sha(folder / "population_results.md"),
            "figure_sha256": _sha(folder / "population_comparison.png"),
            "jobs": verified["jobs"]}


def integrate(plan_path, expected_sha, *, wait=False, timeout_hours=24):
    """Return a complete receipt, or wait without editing active model files."""
    if not isinstance(wait, bool):
        raise TypeError("wait must be a boolean.")
    if (isinstance(timeout_hours, bool) or not isinstance(timeout_hours, Real)
            or not math.isfinite(float(timeout_hours)) or not 0 < timeout_hours <= 168):
        raise ValueError("timeout_hours must be finite, greater than zero and at most 168.")
    plan_path = Path(plan_path).resolve()
    plan, source, target = _validate_plan(plan_path, expected_sha)
    receipt = _inside(target, plan["receipt"])
    recovery = receipt.exists()
    state = {"schema_version": 1, "plan_sha256": expected_sha, "status": "prepared",
             "source_root": str(source), "target_root": str(target), "installed_files": []}
    if recovery:
        state = json.loads(receipt.read_text(encoding="utf-8"))
        if state.get("plan_sha256") != expected_sha:
            raise ValueError("Existing integration receipt belongs to a different plan.")
    deadline = time.monotonic() + timeout_hours * 3600
    while True:
        _validate_plan(plan_path, expected_sha)
        _preflight_files(plan, source, target, recovery=recovery)
        publication = _publication(plan, target)
        if publication is not None:
            break
        if not wait:
            raise RuntimeError("Population study and final report are still running; model files were not changed.")
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for the completed population study and final report.")
        print("Waiting for verified population results before local integration.", flush=True)
        time.sleep(min(30.0, max(0.0, deadline - time.monotonic())))
    if recovery and state.get("publication") != publication:
        raise ValueError("Population publication changed after the first installation attempt.")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    lock_path = receipt.with_suffix(".lock")
    try:
        lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError("An installation lock already exists; no model files were changed.") from error
    os.close(lock)
    state["publication"] = publication
    try:
        _preflight_files(plan, source, target, recovery=recovery)
        if _publication(plan, target) != publication:
            raise ValueError("Population publication changed before installation.")
        state["status"] = "installing"
        _journal(receipt, state)
        for entry in plan["files"]:
            _validate_plan(plan_path, expected_sha)
            _preflight_files(plan, source, target, recovery=True)
            destination = _inside(target, entry["target"])
            state["pending_file"] = entry["target"]
            _journal(receipt, state)
            if not destination.exists() or _sha(destination) != entry["sha256"]:
                data = _inside(source, entry["source"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    raise ValueError("Prepared file changed immediately before installation.")
                current = _sha(destination) if destination.exists() else None
                if current != entry["original_sha256"]:
                    raise ValueError("Target file changed immediately before installation.")
                _atomic_bytes(destination, data)
            if _sha(destination) != entry["sha256"]:
                raise ValueError(f"Installed file verification failed: {entry['target']}")
            if entry["target"] not in state["installed_files"]:
                state["installed_files"].append(entry["target"])
            state.pop("pending_file", None)
            _journal(receipt, state)
        if _publication(plan, target) != publication:
            raise ValueError("Population artifacts changed during installation.")
        for name, digest in plan["prepared_code_sha256"].items():
            if _sha(_inside(target, "src/dztr/" + name)) != digest:
                raise ValueError(f"Final source verification failed: {name}")
        state["status"] = "complete"
        state.pop("error", None)
        state["historical_checkpoint_identities_rewritten"] = False
        _journal(receipt, state)
        print(f"Installed verified simulation improvement; receipt: {receipt}", flush=True)
        return state
    except BaseException as error:
        state["status"] = "partial_failure"
        state["error"] = f"{type(error).__name__}: {error}"
        _journal(receipt, state)
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    args = parser.parse_args(argv)
    if not 0 < args.timeout_hours <= 168:
        parser.error("--timeout-hours must be greater than zero and at most 168")
    integrate(args.plan, args.plan_sha256, wait=args.wait, timeout_hours=args.timeout_hours)


if __name__ == "__main__":
    main()
