"""Run matched full-data scalar GA/DDS calibration with configurable weights."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from workflows import full_data_comparison_v1 as adapter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "finish", "job", "simulate"):
        p = sub.add_parser(command); p.add_argument("--output", required=True)
        if command in {"prepare", "run"}: p.add_argument("--config", required=True)
        if command in {"run", "finish", "job"}: p.add_argument("--evaluation-budget", type=int)
        if command == "job":
            p.add_argument("--reservoir-id", required=True); p.add_argument("--objective", required=True)
            p.add_argument("--optimizer", required=True, choices=("dds", "ga")); p.add_argument("--seed", type=int, required=True)
        if command == "simulate":
            p.add_argument("--registry", required=True); p.add_argument("--forcing-csv", required=True)
            p.add_argument("--reservoir-id"); p.add_argument("--grand-id", type=int); p.add_argument("--initial-state")
    args = parser.parse_args()
    if args.command == "prepare": result = adapter.prepare(args.config, args.output)
    elif args.command == "run": result = adapter.run(args.config, args.output, args.evaluation_budget)
    elif args.command == "finish": result = adapter.finish(args.output, args.evaluation_budget)
    elif args.command == "job": result = adapter.calibrate(args.output, args.reservoir_id, args.objective, args.optimizer, args.seed, args.evaluation_budget)
    else:
        result = adapter.simulate(args.registry, args.forcing_csv, args.output, identifier=args.reservoir_id,
            grand_id=args.grand_id, initial=adapter.native.read(args.initial_state) if args.initial_state else None)
    print(result)
