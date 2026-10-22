"""Plan, run, resume or inspect a configured DZTR workflow."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from workflows import run_manager_v1 as manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, type=Path)
        if name == "run":
            p.add_argument("--output", required=True, type=Path)
            p.add_argument("--until", help="Stop after this configured stage ID")
            p.add_argument("--resume", action="store_true", help="Reuse verified stages and native checkpoints")
    p = sub.add_parser("status")
    p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        result = manager.plan(args.config)
        result = {"description": result["description"], "signature": result["signature"], "notes": result["notes"],
                  "stages": result["stages"]}
    elif args.command == "status":
        result = manager.w.read(args.output / "status.json")
    else:
        result = manager.run(args.config, args.output, until=args.until, resume=args.resume)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
