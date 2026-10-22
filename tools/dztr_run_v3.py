"""DZTR gateway: verified batch import and existing configurable workflows."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from workflows import reservoir_v2 as w
from workflows import run_manager_v2 as manager
from workflows import verified_batch_v1 as batch
from workflows import profile_application_v1 as app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name); p.add_argument("--config", required=True)
        if name == "run":
            p.add_argument("--output", required=True); p.add_argument("--resume", action="store_true"); p.add_argument("--until")
    p = sub.add_parser("status"); p.add_argument("--output", required=True)
    p = sub.add_parser("init-batch"); p.add_argument("--catalog", required=True); p.add_argument("--assessment", required=True)
    p.add_argument("--config", required=True); p.add_argument("--reservoir-id", action="append")
    p = sub.add_parser("simulate"); p.add_argument("--catalog", required=True); p.add_argument("--output", required=True)
    choice = p.add_mutually_exclusive_group(required=True); choice.add_argument("--reservoir-id"); choice.add_argument("--grand-id", type=int)
    p.add_argument("--forcing-csv"); p.add_argument("--forcing-stage", choices=("canonical", "prepared")); p.add_argument("--initial-state")
    args = parser.parse_args()
    if args.command == "status": result = w.read(Path(args.output) / "status.json")
    elif args.command == "simulate":
        result = str(app.simulate(args.catalog, args.output, reservoir_id=args.reservoir_id, grand_id=args.grand_id,
          forcing_csv=args.forcing_csv, forcing_stage=args.forcing_stage, initial=w.read(args.initial_state) if args.initial_state else None))
    elif args.command == "init-batch":
        target = Path(args.config).resolve()
        if not target.is_relative_to(ROOT / "configs") or target.exists(): raise ValueError("Use a fresh batch config inside project configs")
        cfg = batch.make_config(args.catalog, args.assessment, args.reservoir_id)
        w.write(target, cfg); result = str(target)
    else:
        workflow = batch if w.read(args.config).get("kind") == "dztr_verified_batch_v1" else manager
        if args.command == "plan":
            design = workflow.plan(args.config)
            result = {k: design[k] for k in ("kind", "signature", "stages")}
        else: result = workflow.run(args.config, args.output, resume=args.resume, until=args.until)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
