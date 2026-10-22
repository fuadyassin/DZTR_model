"""Assess all Etienne additions, reconstruct conditional flux and probe baselines."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from workflows import extension_assessment_v1 as workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build"); p.add_argument("--config", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("verify"); p.add_argument("--output", required=True)
    args = parser.parse_args()
    result = workflow.build(args.config, args.output) if args.command == "build" else workflow.verify(args.output)
    print({k: result[k] for k in ("combined_reservoirs", "assessed_additions", "conditional_inflow_diagnostics",
        "supplementary_flow_sources", "new_verified_GRAND_mappings", "new_calibration_ready_reservoirs", "optimizer_evaluations")})
