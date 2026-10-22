"""Independently check cached recipe estimates, scored support and conservation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from workflows import data_availability_v1 as fleet
from workflows import reservoir_v2 as w


def independent_nse(observed, simulated):
    if not len(observed) or np.ptp(observed) == 0:
        return None
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    return float(1 - np.sum((observed - simulated) ** 2) / denominator)


def verify_campaign(path):
    out, profiles = fleet.verify(path)
    report = out / "report"
    for name, expected in w.read(report / "manifest.json")["artifact_sha256"].items():
        assert w.sha(report / name) == expected, name
    estimates_checked = jobs_checked = nse_checked = 0
    max_error = max_balance = 0.
    for record in profiles:
        if "prepared_folder" not in record:
            assert not record["simulation_ready"] and not record["parameters_ready"]
            continue
        pdir = out / record["prepared_folder"]
        common = pdir.parent
        if not (common / record["source_frame"]).exists():
            assert not record["parameters_ready"] and not record["simulation_ready"]
            continue
        frame = fleet.read_frame(common / record["source_frame"])
        selection = w.read(common / "selection.json")
        mask = np.array(selection["roles"]) == "calibration"
        assert mask.mean() >= .65
        physical = w.read(common / "numerical_profile.json")["physical_constants"]
        target = pdir / "monthly_targets_si.csv"
        if target.exists() and record["method"] in {"observed_monthly", "flow_only"}:
            values = pd.read_csv(target, index_col="month", float_precision="round_trip")
            for month in range(1, 13):
                selected = mask & (frame.index.month == month)
                storage = (np.percentile(frame.loc[selected, "storage_m3"].dropna(), [10, 45, 85], method=record["quantile_method"])
                           if record["method"] == "observed_monthly" else np.array([.2, .3, .97]) * physical["capacity_m3"])
                release = np.percentile(frame.loc[selected, "release_m3s"].dropna(), [10, 45, 85], method=record["quantile_method"])
                np.testing.assert_allclose(values.loc[month].to_numpy(), np.r_[storage, release], rtol=1e-13, atol=1e-6)
            estimates_checked += 1
        receipt_path = out / "jobs" / record["reservoir_id"] / record["profile_id"] / "receipt.json"
        if not receipt_path.exists():
            assert not record["simulation_ready"]
            continue
        receipt = w.read(receipt_path)
        assert receipt["profile_identity"] == fleet.digest(record)
        assert receipt["new_optimizer_evaluations"] == 0
        for name, expected in receipt["artifact_sha256"].items():
            assert w.sha(receipt_path.parent / name) == expected
        jobs_checked += 1
        if receipt["status"] != "completed":
            assert receipt.get("reason")
            continue
        simulated = pd.read_csv(receipt_path.parent / "simulation.csv", index_col="date", parse_dates=True, float_precision="round_trip")
        original = fleet.read_frame(common / "observations_si.csv").loc[simulated.index]
        roles = pd.Series(selection["roles"], index=frame.index).loc[simulated.index]
        constants = w.read(pdir / "model_constants.json")
        assert constants == physical  # default campaign retains channel/physics
        s, q, incoming, dt = [simulated[c].to_numpy() for c in ("storage_m3", "release_m3s", "inflow_m3s", "step_seconds")]
        assert np.isfinite(s).all() and np.isfinite(q).all() and (s >= 0).all() and (s <= physical["capacity_m3"]).all() and (q >= 0).all()
        assert s[0] == physical["initial_storage_m3"] and q[0] == physical["initial_release_m3s"]
        residual = np.diff(s) - dt[1:] / 2 * (physical["inflow_correction"] * (incoming[1:] + incoming[:-1]) - q[1:] - q[:-1])
        maximum = float(np.abs(residual).max())
        assert maximum <= w.read(common / "numerical_profile.json")["atol_storage_m3"]
        assert maximum == receipt["maximum_independent_balance_residual_m3"]
        max_balance = max(max_balance, maximum)
        for scores in w.read(receipt_path.parent / "metrics.json"):
            for column, label in (("storage_m3", "storage"), ("release_m3s", "release")):
                fixed_mask = roles.eq(scores["phase"]) & original[column].notna()
                observed, predicted = original.loc[fixed_mask, column].to_numpy(), simulated.loc[fixed_mask, column].to_numpy()
                assert len(observed) == scores[label + "_pairs"]
                expected = independent_nse(observed, predicted)
                actual = scores[label + "_nse"]
                if expected is None:
                    assert actual is None
                else:
                    error = abs(expected - actual)
                    assert error < 1e-10
                    max_error = max(max_error, error)
                nse_checked += 1
            expected = independent_nse(np.sqrt(original.loc[(roles == scores["phase"]) & original.release_m3s.notna(), "release_m3s"].to_numpy()),
                                       np.sqrt(simulated.loc[(roles == scores["phase"]) & original.release_m3s.notna(), "release_m3s"].to_numpy()))
            actual = scores["release_sqrt_nse"]
            assert expected is None and actual is None or expected is not None and abs(expected - actual) < 1e-10
    # Verify the original scientific release rather than rewriting it.
    catalog = w.read(ROOT / "outputs/calibrated_paper_cohort_v1_20261017_final/catalog.json")
    old = {**catalog["application_source_sha256"], catalog["source_table"]: catalog["source_table_sha256"]}
    for entry in catalog["reservoirs"]:
        old.update(entry.get("input_sha256", {}))
    for relative, expected in old.items():
        assert w.sha(ROOT / relative) == expected
    performance = pd.read_csv(report / "performance_comparison.csv")
    held = performance.loc[performance.phase.eq("assessment")]
    observed = held.loc[held.method.eq("observed_monthly")].set_index("reservoir_id")
    flow = held.loc[held.method.eq("flow_only")].set_index("reservoir_id")
    common = observed.index.intersection(flow.index)
    paired = observed.loc[common, ["name", "storage_nse", "release_nse"]].rename(columns={"storage_nse": "observed_monthly_storage_nse", "release_nse": "observed_monthly_release_nse"}).join(
        flow.loc[common, ["storage_nse", "release_nse"]].rename(columns={"storage_nse": "flow_only_storage_nse", "release_nse": "flow_only_release_nse"}))
    paired.to_csv(report / "common_case_recipe_comparison.csv", index_label="reservoir_id", float_format="%.17g")
    summary = {"independently_checked_estimator_profiles": estimates_checked, "verified_job_receipts": jobs_checked,
               "independently_recomputed_raw_nse": nse_checked,
               "maximum_raw_nse_difference": max_error, "maximum_independent_balance_residual_m3": max_balance,
               "frozen_cohort_sources_and_inputs_unchanged": len(old),
               "common_completed_reservoirs": len(common),
               "observed_monthly_better_storage_on_common_cases": int((paired.observed_monthly_storage_nse > paired.flow_only_storage_nse).sum()),
               "observed_monthly_better_release_on_common_cases": int((paired.observed_monthly_release_nse > paired.flow_only_release_nse).sum()),
               "verification_tool_sha256": w.sha(Path(__file__)), "new_reservoir_simulations": 0, "new_optimizer_evaluations": 0}
    w.write(report / "scientific_verification.json", summary)
    w.write(report / "manifest.json", {"artifact_sha256": {p.name: w.sha(p) for p in report.iterdir() if p.is_file() and p.name != "manifest.json"}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_campaign(args.campaign), indent=2))
