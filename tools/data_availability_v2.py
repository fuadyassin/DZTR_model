"""Shared data-availability estimators with datum-aware storage preparation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from workflows import data_availability_v2 as workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--config", required=True); p.add_argument("--output", required=True)
    for name in ("run-baselines", "report"):
        p = sub.add_parser(name); p.add_argument("--campaign", required=True)
    args = parser.parse_args()
    if args.command == "prepare": result = str(workflow.prepare(args.config, args.output))
    elif args.command == "run-baselines": result = workflow.run_baselines(args.campaign)
    else: result = workflow.report(args.campaign)
    print(json.dumps(result, indent=2))
