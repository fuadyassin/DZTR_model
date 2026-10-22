"""Small persisted optional-GA verification; reuse frozen DDS reference searches."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import pandas as pd
from workflows import flow_calibration_v1 as previous_flow, storage_calibration_v1 as previous_storage
from workflows import flow_calibration_v2 as flow, storage_calibration_v2 as storage
from workflows import reservoir_v2 as native
from workflows.atomic_io_v1 import write


def run(output):
    out = Path(output).resolve()
    if not out.is_relative_to(ROOT / "outputs") or out == ROOT / "outputs": raise ValueError("Use project outputs")
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / "job.json"
    write(receipt_path, {"status": "running", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": native.sha(__file__), "description": "Two 64-evaluation GA checks, with exact 32-to-64 continuation; cached DDS references"})
    rows, checks, inputs = [], [], {}
    try:
        for role, module, previous, objective, old_rel in (
            ("flow", flow, previous_flow, "sqrt_release_nse", "managed_flow_only_paper_v1_20261020/stages/flow_calibration/campaign"),
            ("storage", storage, previous_storage, "storage_nse", "managed_storage_only_paper_v1_20261020/stages/mean_flow_calibration/campaign")):
            rid = "ex14_Gardiner"; cfg = ROOT / "configs" / f"{role}_ga_gardiner_v2.json"
            inputs[str(cfg)] = native.sha(cfg)
            campaign = out / role
            if campaign.exists(): module.verify(campaign)
            else: module.prepare(cfg, campaign)
            first = module.calibrate(campaign, rid, objective, 42, 32)
            prefix = native.read(first / "result.json")
            final = module.calibrate(campaign, rid, objective, 42, 64)
            result = native.read(final / "result.json")
            assert result["trials"][:32] == prefix["trials"]
            release = module.finish(campaign, 64)
            old, _ = previous.verify(ROOT / "outputs" / old_rel)
            old_release = old / "releases/budget_64"; previous.checked(old_release)
            inputs[str(old_release / "manifest.json")] = native.sha(old_release / "manifest.json")
            previous.checked(old / "jobs" / rid / objective / "seed_42/budget_64")
            earlier = native.read(old / "jobs" / rid / objective / "seed_42/budget_64/result.json")
            assert len(earlier["trials"]) == len(result["trials"]) == 64
            a = module.objective_for(campaign, rid, objective); b = previous.objective_for(old, rid, objective)
            np.testing.assert_array_equal(a.template.monthly_targets, b.template.monthly_targets)
            np.testing.assert_array_equal(a.lower, b.lower); np.testing.assert_array_equal(a.upper, b.upper)
            pd.testing.assert_frame_equal(a.frame, b.frame)
            assert a.selection.signature == b.selection.signature
            assert a.metadata["physical_constants"] == b.metadata["physical_constants"]
            for algorithm, source in (("GA", release), ("DDS", old_release)):
                table = pd.read_csv(source / "baseline_comparison.csv", float_precision="round_trip")
                selected = table[(table.reservoir_id == rid) & (table.objective == objective) & (table.seed == 42)]
                for _, row in selected.iterrows():
                    rows.append({"reservoir_id": rid, "route": role + "_only", "optimizer": algorithm, "seed": 42,
                        "objective": objective, "evaluations": 64, "phase": row.phase, "calibration_cost": row.calibration_cost,
                        "storage_nse": row.calibrated_storage_nse, "release_nse": row.calibrated_release_nse,
                        "sqrt_release_nse": row.calibrated_sqrt_release_nse, "assessment_status": row.assessment_status,
                        "maximum_balance_residual_m3": row.maximum_independent_balance_residual_m3})
            profile, parameters = native.load_profile(release / objective / "registry.json", identifier=rid)
            assert profile["optimizer"] == "GA" and not profile["selection_uses_assessment"]
            diagnostic = release / "diagnostics" / rid / objective / "simulation.csv"
            simulation = flow.availability.read_frame(diagnostic)
            residual = native.independent_balance(simulation, parameters.inflow_correction)
            assert residual <= a.metadata["atol_storage_m3"]
            checks.append({"route": role + "_only", "exact_checkpoint_prefix_retained": True,
                "identical_objective_inputs_bounds_and_initialization": True, "DDS_reference_reused": True,
                "maximum_independent_balance_residual_m3": residual, "GA_optimizer_evaluations": 64,
                "GA_rejected_candidates": sum(not r["feasible"] for r in result["trials"])})
        frame = pd.DataFrame(rows); frame.to_csv(out / "comparison.csv", index=False, float_format="%.17g")
        pd.DataFrame(checks).to_csv(out / "scientific_checks.csv", index=False, float_format="%.17g")
        lines = ["# Optional scalar GA verification", "",
            "GA is an optional alternative to default DDS for release-only and storage-only calibration. This check uses Gardiner, seed 42, the same objective-specific inputs/bounds/initial states and 64 total evaluations per search. GA population is 32 and the fixed continuation horizon is 512. Each GA search continued from its 32-evaluation checkpoint. Frozen DDS searches were reused without new DDS evaluations.", "",
            "This one-reservoir, one-seed check verifies execution, exact continuation and physical checks. It does not establish convergence or rank optimizers. Calibration cost alone chooses each parameter vector; held-out scores are assessments.", "",
            "| Route | Optimizer | Phase | Calibration cost | Storage NSE | Release NSE | sqrt-release NSE |",
            "|---|---|---|---:|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['route']} | {row['optimizer']} | {row['phase']} | {row['calibration_cost']:.3f} | {row['storage_nse']:.3f} | {row['release_nse']:.3f} | {row['sqrt_release_nse']:.3f} |")
        lines += ["", "- [Scientific checks](scientific_checks.csv)", "- [Machine-readable scores](comparison.csv)",
                  "- [Workflow and configuration instructions](../../docs/optional_ga_v1.md)", "",
                  "Each route's `releases/budget_64` contains a selected profile registry and a 37-row GRAND inventory. Other paper reservoirs were not requested in this check. Earlier paper results are unchanged. All inputs remain conditional on their original forcing and physical assumptions.", ""]
        (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
        write(receipt_path, {**native.read(receipt_path), "status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "GA_optimizer_evaluations": 128, "DDS_optimizer_evaluations": 0, "new_model_calls_for_reporting_are_separate": True,
            "checks": checks, "convergence_established": False})
        for p, expected in inputs.items():
            if native.sha(p) != expected: raise ValueError("Pilot input changed")
        write(out / "manifest.json", {"input_sha256": inputs, "source_sha256": {**flow.sources(), **storage.sources(),
            "tools/run_optional_ga_pilot_v1.py": native.sha(__file__)}, "artifact_sha256": {
                p.relative_to(out).as_posix(): native.sha(p) for p in out.rglob("*") if p.is_file() and not p.name.endswith(".log") and p != out / "manifest.json"}})
        return out
    except BaseException as exc:
        write(receipt_path, {**native.read(receipt_path), "status": "failed", "reason": str(exc)})
        traceback.print_exc(); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", required=True)
    print(run(parser.parse_args().output))
