"""Run a calibration-only seed under an explicitly reviewed daily policy."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dztr.calibration.interval_search import run_interval_seed
from dztr.calibration.optimizer import NSGA2Settings
from dztr.calibration.stopping import StoppingPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/reviews/bull_lake_interval_policy.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/bull_lake_interval_search")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluations", type=int, default=128)
    parser.add_argument("--population", type=int, default=16)
    args = parser.parse_args()
    if args.population < 2 or args.evaluations < args.population or args.evaluations % args.population:
        parser.error("Use a population of at least two and a budget at least that large, in complete populations.")
    if args.seed < 0:
        parser.error("Seed must be nonnegative.")
    if args.output.exists():
        parser.error("Choose a fresh output directory so existing results are preserved.")
    checkpoints = tuple(sorted({min(multiplier * args.population, args.evaluations) for multiplier in (1, 2, 4, 6, 8)} | {args.evaluations}))
    stopping = None
    if args.evaluations >= 6 * args.population:
        stopping = StoppingPolicy(population_size=args.population, check_interval=args.population,
            window_evaluations=4 * args.population, minimum_evaluations=6 * args.population,
            patience=2, max_evaluations=args.evaluations, normalized_hv_tolerance=.0025,
            front_tolerance=.01, extrema_tolerance=.005)

    def progress(record):
        print(f"{record.evaluations}/{args.evaluations} attempted; "
              f"{record.feasible_count} feasible; {record.front_count} nondominated", flush=True)

    report = run_interval_seed(args.policy, args.output,
        settings=NSGA2Settings(seed=args.seed, population_size=args.population, evaluation_budget=args.evaluations),
        checkpoints=checkpoints, stopping_policy=stopping, progress_callback=progress)
    print(json.dumps({key: report[key] for key in ("search_evaluations", "feasible_evaluations",
                     "rejected_evaluations", "termination", "stopping_assessment")}, indent=2))


if __name__ == "__main__":
    main()
