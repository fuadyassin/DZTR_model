"""Persistent storage-only run-manager job and verified cached reporting."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from workflows import run_manager_v5 as manager
from workflows import reservoir_v2 as native
from workflows.atomic_io_v1 import write
from tools.report_storage_only_v1 import report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--config", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); out = Path(a.output).resolve()
    if not out.is_relative_to(ROOT / "outputs") or out == ROOT / "outputs": raise ValueError("Use project outputs")
    receipt = out.with_name(out.name + "_job.json")
    write(receipt, {"status": "running", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(a.config).resolve()), "output": str(out), "runner_sha256": native.sha(__file__)})
    try:
        manager.run(a.config, out, resume=out.exists())
        destination = out / "report"
        if not destination.exists(): report(out, destination)
        else:
            proof = native.read(destination / "manifest.json")
            for name, expected in proof["artifact_sha256"].items():
                if native.sha(destination / name) != expected: raise ValueError("Existing report artifact changed")
        write(receipt, {**native.read(receipt), "status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(), "report": str(destination / "README.md")})
    except BaseException as exc:
        write(receipt, {**native.read(receipt), "status": "failed", "reason": str(exc)})
        traceback.print_exc(); raise
