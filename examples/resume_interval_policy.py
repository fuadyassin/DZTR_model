"""Continue a verified calibration checkpoint into a fresh output directory."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dztr.calibration.interval_resume import resume_interval_seed
from dztr.calibration.stopping import StoppingPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-seed", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/reviews/bull_lake_interval_policy.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-evaluations", type=int, default=256)
    args = parser.parse_args()
    summary_path = args.previous_seed / "summary.json"
    if not summary_path.is_file():
        parser.error("Previous seed must contain a verified summary and optimizer checkpoint.")
    previous = json.loads(summary_path.read_text(encoding="utf-8"))
    population = previous["optimization"]["population_size"]
    inherited = previous["search_evaluations"]
    if args.total_evaluations <= inherited or args.total_evaluations % population:
        parser.error("The new total must exceed the inherited count and finish a complete population.")
    if args.output.exists():
        parser.error("Choose a fresh output directory; previous evidence is preserved.")
    stopping = None
    if previous.get("stopping_policy") is not None:
        stopping_settings = dict(previous["stopping_policy"])
        stopping_settings["max_evaluations"] = args.total_evaluations
        stopping = StoppingPolicy(**stopping_settings)
    checkpoints = tuple(sorted(set(previous["requested_checkpoints"]) | {inherited, args.total_evaluations}))

    def progress(record):
        print(f"{record.evaluations}/{args.total_evaluations} total; "
              f"{record.evaluations - inherited} new attempts", flush=True)

    report = resume_interval_seed(args.previous_seed, args.policy, args.output,
        total_evaluation_budget=args.total_evaluations, checkpoints=checkpoints,
        stopping_policy=stopping, progress_callback=progress)
    print(json.dumps({key: report[key] for key in ("inherited_evaluations", "new_search_evaluations",
                    "search_evaluations", "feasible_evaluations", "rejected_evaluations", "termination")}, indent=2))


if __name__ == "__main__":
    main()
