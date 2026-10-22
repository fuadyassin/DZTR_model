"""Add new archived reservoir observations and descriptive target estimates."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from workflows import etienne_collection_v1 as workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("import"); p.add_argument("--config", required=True)
    p.add_argument("--collection", required=True); p.add_argument("--report", required=True)
    p = sub.add_parser("verify"); p.add_argument("--collection", required=True)
    p = sub.add_parser("report"); p.add_argument("--collection", required=True); p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "import": result = workflow.build(args.config, args.collection, args.report)
    elif args.command == "report": result = workflow.write_report(args.collection, args.output)
    else:
        _, result = workflow.verify(args.collection)
        result = {k: result[k] for k in ("existing_reservoirs", "added_reservoirs", "combined_reservoirs", "new_complete_target_tables", "new_calibration_ready_reservoirs")}
    print(result)
