"""DZTR gateway: shared estimator choices and datum-aware geometry calibration."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from workflows import run_manager_v7 as manager
from workflows import reservoir_v2 as w
from tools import dztr_run_v8 as previous


def main():
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("command", nargs="?"); probe.add_argument("--config")
    known, _ = probe.parse_known_args()
    if known.config and w.read(known.config).get("kind") == "dztr_run_manager_v7":
        parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=("plan", "run"))
        parser.add_argument("--config", required=True); parser.add_argument("--output"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--until")
        args = parser.parse_args()
        if args.command == "plan":
            design = manager.plan(args.config); result = {k: design[k] for k in ("kind", "signature", "stages")}
        else:
            if not args.output: parser.error("run requires --output")
            result = manager.run(args.config, args.output, resume=args.resume, until=args.until)
        print(json.dumps(result, indent=2))
    else: previous.main()


if __name__ == "__main__": main()
