"""Calibrate a frozen observed-monthly profile with native checkpointed jobs."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from workflows import prepared_calibration_v1 as adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        p = sub.add_parser(name); p.add_argument("--config", required=True); p.add_argument("--output", required=True)
        if name == "run": p.add_argument("--evaluation-budget", type=int)
    p = sub.add_parser("job"); p.add_argument("--output", required=True); p.add_argument("--reservoir-id", required=True)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--evaluation-budget", type=int)
    args = parser.parse_args()
    if args.command == "prepare": result = adapter.prepare(args.config, args.output)
    elif args.command == "run": result = adapter.run(args.config, args.output, args.evaluation_budget)
    else:
        adapter.verify(args.output)
        result = adapter.native.calibrate(args.output, args.reservoir_id, args.seed, args.evaluation_budget)
    print(result)


if __name__ == "__main__": main()
