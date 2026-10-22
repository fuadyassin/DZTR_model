"""DZTR gateway with a configured datum/level-to-storage pilot.

Existing run/batch/simulation arguments retain the v3 gateway behaviour.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from workflows import level_storage_pilot_v1 as pilot
from workflows import reservoir_v2 as w
from tools import dztr_run_v3 as previous


def main():
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("command", nargs="?"); probe.add_argument("--config"); probe.add_argument("--output")
    known, _ = probe.parse_known_args()
    is_pilot = known.config and w.read(known.config).get("kind") == "dztr_level_storage_pilot_v1"
    if known.command == "verify-level-storage":
        if not known.output: probe.error("--output is required")
        result = pilot.verify(known.output)
    elif is_pilot and known.command in ("plan", "run"):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("command", choices=("plan", "run")); parser.add_argument("--config", required=True)
        parser.add_argument("--output"); parser.add_argument("--resume", action="store_true")
        args = parser.parse_args()
        if args.command == "plan":
            design = pilot.plan(args.config)
            result = {k: design[k] for k in ("kind", "signature", "stages", "new_optimizer_evaluations", "assessment_selects_reference")}
        else:
            if not args.output: parser.error("run requires --output")
            result = pilot.run(args.config, args.output, resume=args.resume)
    else:
        return previous.main()
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
