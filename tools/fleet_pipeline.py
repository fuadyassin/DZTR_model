"""One explicit fleet execution command, followed by common saved-result reports."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fleet_workflow_20261010")
    parser.add_argument("--run-authorized", action="store_true", required=True,
                        help="Explicit authorization for the declared calibration and reporting calls.")
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Output must stay inside the repository.")
    execution = subprocess.run([sys.executable, "-B", str(ROOT / "tools/fleet_workflow.py"), "--output", str(out), "--run-authorized"])
    if not (out / "execution_summary.json").exists():
        raise SystemExit(f"Fleet execution stopped before a summary was written (exit {execution.returncode}); inspect logs.")
    subprocess.run([sys.executable, "-B", str(ROOT / "tools/fleet_reports.py"), "--completed-fleet", str(out), "--output", str(out / "reports/completed_fleet")], check=True)
    result = json.loads((out / "execution_summary.json").read_text(encoding="utf-8"))
    if execution.returncode != 0 or result["status"] != "completed":
        raise SystemExit("Fleet finished with isolated failures; inspect execution_summary.json and retained checkpoints.")


if __name__ == "__main__":
    main()
