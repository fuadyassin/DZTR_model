"""Persistent four-route run-manager job, without new optimizer searches."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from workflows import run_manager_v6 as manager
from workflows import reservoir_v2 as native
from workflows.atomic_io_v1 import write


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); out = Path(args.output).resolve()
    if not out.is_relative_to(ROOT / "outputs") or out == ROOT / "outputs": raise ValueError("Use project outputs")
    receipt = out.with_name(out.name + "_job.json")
    write(receipt, {"status": "running", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config).resolve()), "output": str(out), "runner_sha256": native.sha(__file__), "optimizer_evaluations": 0})
    try:
        state = manager.run(args.config, out, resume=out.exists())
        write(receipt, {**native.read(receipt), "status": state["status"], "completed_utc": datetime.now(timezone.utc).isoformat(),
            "report": str(out / "stages/comparison/campaign/README.md")})
    except BaseException as exc:
        write(receipt, {**native.read(receipt), "status": "failed", "reason": str(exc)})
        traceback.print_exc(); raise
