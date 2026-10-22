"""Run or apply fixed DZTR profiles without target storage/release observations."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from workflows import no_observations_v1 as workflow
from workflows import reservoir_v2 as native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run"); run.add_argument("--config", required=True); run.add_argument("--output", required=True)
    sim = sub.add_parser("simulate")
    for key in ("registry", "forcing-csv", "output"): sim.add_argument("--" + key, required=True)
    group = sim.add_mutually_exclusive_group(required=True)
    group.add_argument("--reservoir-id"); group.add_argument("--grand-id", type=int)
    sim.add_argument("--initial-state")
    sim.add_argument("--method")
    args = parser.parse_args()
    if args.command == "run": result = workflow.run(args.config, args.output)
    else:
        result = workflow.simulate(args.registry, args.forcing_csv, args.output, identifier=args.reservoir_id,
            grand_id=args.grand_id, initial=native.read(args.initial_state) if args.initial_state else None, method=args.method)
    print(result)


if __name__ == "__main__": main()
