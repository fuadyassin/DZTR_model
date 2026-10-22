"""Prepare geometry/target recipes and run persistent fixed-profile baselines."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from workflows.data_availability_v1 import package_geometry, prepare, report, run_baselines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("package-geometry")
    p.add_argument("--source", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--destination", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    for name in ("run-baselines", "report"):
        p = sub.add_parser(name)
        p.add_argument("--campaign", required=True)
        if name == "run-baselines":
            p.add_argument("--reservoir-id")
    args = parser.parse_args()
    if args.command == "package-geometry":
        result = str(package_geometry(args.source, args.catalog, args.destination))
    elif args.command == "prepare":
        result = str(prepare(args.config, args.output))
    elif args.command == "run-baselines":
        result = run_baselines(args.campaign, reservoir_id=args.reservoir_id)
    else:
        result = report(args.campaign)
    print(json.dumps(result, indent=2))
