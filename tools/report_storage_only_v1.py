"""Assess sealed storage-only experiments from cached results, without model calls."""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from workflows import storage_calibration_v1 as storage
from workflows import reservoir_v2 as native
from workflows.atomic_io_v1 import write

IMPLEMENTATION_SHA = native.sha(__file__)


def report(managed, output):
    if native.sha(__file__) != IMPLEMENTATION_SHA: raise ValueError("Reporting source changed after import; use a fresh process")
    managed = Path(managed).resolve(strict=True); output = Path(output).resolve()
    if not output.is_relative_to(ROOT / "outputs") or output.exists(): raise ValueError("Use a fresh project report folder")
    experiments = {}; summaries = []; all_scores = []; inputs = {str(Path(__file__)): IMPLEMENTATION_SHA}
    for experiment in ("mean_flow_calibration", "naturalized_overlap"):
        campaign, cfg = storage.verify(managed / "stages" / experiment / "campaign")
        budget = cfg["optimization"]["evaluation_budget"]; release = campaign / "releases" / f"budget_{budget}"; storage.checked(release)
        inputs[str(release / "manifest.json")] = native.sha(release / "manifest.json")
        scores = pd.read_csv(release / "baseline_comparison.csv", float_precision="round_trip"); scores["experiment"] = experiment
        all_scores.append(scores); summary = native.read(release / "summary.json")
        selected = scores[(scores.selected == True) & (scores.phase == "assessment")]
        summary.update(experiment=experiment, held_out_profiles=int(selected.calibrated_storage_nse.notna().sum()),
            median_storage_nse=selected.calibrated_storage_nse.median(), median_hidden_release_nse=selected.calibrated_release_nse.median(),
            baseline_comparable=int(selected.improvement_storage_nse.notna().sum()),
            improved_storage=int((selected.improvement_storage_nse > 1e-12).sum()),
            both_nse_positive=int(((selected.calibrated_storage_nse > 0) & (selected.calibrated_release_nse > 0)).sum()))
        summaries.append(summary); experiments[experiment] = (campaign, cfg, release, scores)
    output.mkdir(parents=True)
    pd.DataFrame(summaries).to_csv(output / "experiment_summary.csv", index=False, float_format="%.17g")
    pd.concat(all_scores, ignore_index=True).to_csv(output / "all_seed_phase_scores.csv", index=False, float_format="%.17g")
    primary, cfg, release, scores = experiments["mean_flow_calibration"]
    inventory = pd.read_csv(release / "storage_nse/reservoir_parameterization_by_GRAND_ID.csv", float_precision="round_trip")
    for key in ("mean_source", "mean_discharge_m3s", "donor_count", "target_levels_truncated"):
        if key not in inventory: inventory[key] = None
    for i, row in inventory.iterrows():
        folder = primary / "prepared" / row.reservoir_id
        if (folder / "release_prior/receipt.json").exists():
            prior = native.read(folder / "release_prior/receipt.json")
            for key in ("mean_source", "mean_discharge_m3s", "donor_count", "target_levels_truncated"):
                inventory.loc[i, key] = prior.get(key)
        if row.parameters_available:
            inventory.loc[i, "benchmark_status"] = "calibrated_assessed_exploratory" if row.assessment_status == "completed" else "calibrated_assessment_failed"
        else: inventory.loc[i, "benchmark_status"] = "no_feasible_candidate_at_budget" if row.calibration_ready else row.status
    inventory.to_csv(output / "reservoir_parameterization_all_37_by_GRAND_ID.csv", index=False, float_format="%.17g")
    selected = scores[scores.selected == True].copy()
    previous_path = ROOT / "outputs/calibrated_paper_cohort_v1_20261017_final/cohort_summary.csv"
    old = pd.read_csv(previous_path, float_precision="round_trip"); inputs[str(previous_path)] = native.sha(previous_path)
    flow_path = ROOT / "outputs/managed_flow_only_paper_v1_20261020/stages/flow_calibration/campaign/releases/budget_64/baseline_comparison.csv"
    flow = pd.read_csv(flow_path, float_precision="round_trip"); inputs[str(flow_path)] = native.sha(flow_path)
    flow = flow[(flow.selected == True) & (flow.objective == "sqrt_release_nse")]
    comparison = selected.merge(flow[["reservoir_id", "phase", "calibrated_storage_nse", "calibrated_release_nse"]], on=["reservoir_id", "phase"], how="left", suffixes=("", "_previous_flow_only"), validate="one_to_one")
    comparison = comparison.merge(old[["reservoir_id", "calibration_storage_nse", "calibration_release_nse", "assessment_storage_nse", "assessment_release_nse"]], on="reservoir_id", how="left", validate="many_to_one")
    comparison.to_csv(output / "comparison_with_previous_routes.csv", index=False, float_format="%.17g")
    checks, seasonal, regimes, failures = [], [], [], []
    for experiment, (campaign, protocol, arm_release, _) in experiments.items():
        for choice in native.read(arm_release / "selection.json")["records"]:
            rid = choice["reservoir_id"]; folder = campaign / "prepared" / rid
            obj = storage.objective_for(campaign, rid, "storage_nse")
            hidden = storage.availability.read_frame(folder / "hidden_assessment_observations_si.csv")
            diag = arm_release / "diagnostics" / rid / "storage_nse"
            sim = storage.availability.read_frame(diag / "simulation.csv")
            profile, parameters = native.load_profile(arm_release / "storage_nse/registry.json", identifier=rid)
            prior = pd.read_csv(folder / "release_prior/monthly_release_prior_si.csv", index_col="month", float_precision="round_trip")
            np.testing.assert_array_equal(parameters.monthly_targets.iloc[:, 3:], prior)
            if "release_m3s" in obj.frame: raise ValueError("Hidden release entered objective frame")
            residual = native.independent_balance(sim, parameters.inflow_correction)
            if residual > obj.metadata["atol_storage_m3"]: raise ValueError("Independent cached mass balance failed")
            dates = obj.frame.index[obj.mask]; labels = obj.selection.table().regime.to_numpy()
            groups = [obj.mask] if protocol["scoring"] == "pooled" else obj.groups()
            cost = 1 - np.mean([storage.nse(obj.frame.loc[obj.frame.index[g], "storage_m3"], sim.loc[obj.frame.index[g], "storage_m3"]) for g in groups])
            np.testing.assert_allclose(cost, choice["cost"], rtol=0, atol=1e-10)
            provenance = profile["parameterization"]; prior_receipt = provenance["release_prior"]
            checks.append({"experiment": experiment, "reservoir_id": rid, "maximum_independent_balance_residual_m3": residual,
                "release_prior_fixed": True, "hidden_release_excluded": True, "cached_calibration_cost_reproduced": True})
            assessment_ok = native.read(diag / "assessment.json")["status"] == "completed"
            roles = np.array(obj.selection.roles)
            for phase in ("calibration", "assessment"):
                if phase == "assessment" and not assessment_ok: continue
                for month in range(1, 13):
                    dates = hidden.index[(roles == phase) & (hidden.index.month == month)]
                    if not len(dates) or not dates.isin(sim.index).all(): continue
                    row = {"experiment": experiment, "reservoir_id": rid, "phase": phase, "month": month}
                    for name, column in (("storage", "storage_m3"), ("release", "release_m3s")):
                        error = (sim.loc[dates, column] - hidden.loc[dates, column]).dropna()
                        row[name + "_mean_bias"] = error.mean(); row[name + "_pairs"] = len(error)
                    seasonal.append(row)
                for regime in sorted(set(labels[roles == phase])):
                    dates = hidden.index[(roles == phase) & (labels == regime)]
                    if not dates.isin(sim.index).all(): continue
                    row = {"experiment": experiment, "reservoir_id": rid, "phase": phase, "regime": regime}
                    for name, column in (("storage_nse", "storage_m3"), ("release_nse", "release_m3s")):
                        try: row[name] = storage.nse(hidden.loc[dates, column], sim.loc[dates, column])
                        except ValueError: row[name] = None
                    regimes.append(row)
        for path in (campaign / "jobs").glob("*/storage_nse/seed_*/budget_" + str(protocol["optimization"]["evaluation_budget"]) + "/result.json"):
            counts = Counter(t["reason"] for t in native.read(path)["trials"] if not t["feasible"])
            for reason, count in counts.items(): failures.append({"experiment": experiment, "reservoir_id": path.parents[3].name, "seed": path.parents[1].name, "reason": reason, "count": count})
    pd.DataFrame(checks).to_csv(output / "scientific_checks.csv", index=False)
    pd.DataFrame(seasonal).to_csv(output / "seasonal_bias.csv", index=False, float_format="%.17g")
    pd.DataFrame(regimes).to_csv(output / "regime_scores.csv", index=False, float_format="%.17g")
    pd.DataFrame(failures).to_csv(output / "infeasible_candidate_reasons.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    held = selected[selected.phase == "assessment"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(held.calibrated_release_nse, held.calibrated_storage_nse, color="#2379ad")
    axes[1].scatter(held.improvement_release_nse, held.improvement_storage_nse, color="#2379ad")
    for ax, threshold in zip(axes, (1, .1)):
        ax.axhline(0, color="grey", lw=.8); ax.axvline(0, color="grey", lw=.8); ax.grid(alpha=.2)
        ax.set_xscale("symlog", linthresh=threshold); ax.set_yscale("symlog", linthresh=threshold)
    axes[0].set(xlabel="Hidden-release NSE", ylabel="Held-out storage NSE", title="Storage-only mean-flow prior")
    axes[1].set(xlabel="Release NSE change from fixed-prior baseline", ylabel="Storage NSE change from baseline", title="Held-out changes after storage calibration")
    fig.suptitle(f"Storage-only DDS: {cfg['optimization']['evaluation_budget']} evaluations per seed; convergence unproven")
    fig.text(.5, .01, "Symmetric log axes, linear near zero. Negative values and missing scores remain explicit.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(output / "cohort_comparison.png", dpi=160); fig.savefig(output / "cohort_comparison.pdf"); plt.close(fig)
    # Examples are declared before reviewing their scores.
    examples = []
    for rid in ("ex14_Gardiner", "ex30_Big_horn", "ex5_Garrison"):
        if rid not in set(held.reservoir_id): continue
        folder = primary / "prepared" / rid; diag = release / "diagnostics" / rid / "storage_nse"
        obs = storage.availability.read_frame(folder / "hidden_assessment_observations_si.csv")
        sim = storage.availability.read_frame(diag / "simulation.csv")
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for ax, column, scale, label in zip(axes, ("storage_m3", "release_m3s"), (1e6, 1), ("Storage (million m³)", "Release (m³/s)")):
            ax.plot(obs[column].resample("MS").mean() / scale, color="#333333", label="Observed")
            ax.plot(sim[column].resample("MS").mean() / scale, color="#2379ad", label="Storage-only calibrated", alpha=.85)
            base = release / "baselines" / rid / "simulation.csv"
            if base.exists(): ax.plot(storage.availability.read_frame(base)[column].resample("MS").mean() / scale, color="#999999", ls="--", label="Uncalibrated fixed-prior baseline", alpha=.6)
            ax.set_ylabel(label); ax.grid(alpha=.2)
        axes[0].legend(fontsize=8, ncol=3); fig.suptitle(rid + ": monthly means for display; release hidden during fitting")
        fig.tight_layout(); name = rid + "_simulation.png"; fig.savefig(output / name, dpi=150); plt.close(fig); examples.append(name)
    lines = ["# Storage-only DZTR benchmark", "", "Storage observations determine the monthly storage targets and the scalar DDS objective. Release targets remain fixed; observed release is read only after calibration choices are frozen.", "",
        "The primary experiment implements the user's proposed mean-flow relationship. For each target, Q(P) = mean-flow scale × median of external donor P/mean ratios for P5/P30/P97. Etienne's archived simulated naturalized flows supply donor shapes; the target is excluded. GRAND DIS_AVG_LS is divided by 1,000 to obtain m³/s. Missing verified GRAND means use a separately identified volume-weighted calibration effective-inflow proxy. Source capacity, dead storage and channel capacity are fixed. Release targets exceeding that channel limit are explicitly truncated and recorded.", "",
        "These are exploratory transferred priors from seven Saskatchewan-basin model records. They are not naturalized target-reservoir time series, and the donor spread is not a confidence interval. The direct naturalized-overlap experiment is reported separately. Inflow or physical metadata may depend on original observations; this remains a conditional retrospective benchmark.", "",
        "Etienne's paper explicitly omitted the storage-observed/outflow-unobserved scenario. Its CR1 naturalized-flow approach and an archived Big Horn storage-NSE trial inform this extension; this is not a reproduction of a published storage-only experiment. [Published study](https://doi.org/10.1139/cjce-2021-0507).", "",
        "| Experiment | Ready reservoirs | Searches | Evaluations | Selected profiles | Median held-out storage NSE | Median hidden-release NSE | Storage improved / comparable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries: lines.append(f"| {s['experiment']} | {s['ready_reservoirs']} | {s['searches']} | {s['evaluations']} | {s['selected_profiles']} | {s['median_storage_nse']:.3f} | {s['median_hidden_release_nse']:.3f} | {s['improved_storage']}/{s['baseline_comparable']} |")
    mean_counts = inventory[inventory.calibration_ready == True].mean_source.value_counts().to_dict()
    lines += ["", f"Primary mean-flow sources: {mean_counts}.", "", "![Cohort performance](cohort_comparison.png)", "",
        "## Primary held-out results", "", "| Reservoir | Storage NSE | Hidden-release NSE | Storage NSE change | Assessment |", "|---|---:|---:|---:|---|"]
    for _, r in held.sort_values("reservoir_id").iterrows(): lines.append(f"| {r.reservoir_id} | {r.calibrated_storage_nse:.3f} | {r.calibrated_release_nse:.3f} | {r.improvement_storage_nse:.3f} | {r.assessment_status} |")
    lines += ["", "## Inventory and scientific checks", "", "All 37 paper rows are retained in [the GRAND parameter table](reservoir_parameterization_all_37_by_GRAND_ID.csv). Missing IDs stay blank. No prior or parameter is invented for source-gated cases. Benchmark status distinguishes missing sources, unsuccessful searches and calibrated profiles.", "",
        f"Verified {len(checks)} selected profiles from cached trajectories: fixed release targets, exclusion of release from the objective, reproduced calibration costs and independent mass balance. No reporting simulation or optimizer evaluation was run. Maximum balance residual: {max([r['maximum_independent_balance_residual_m3'] for r in checks], default=float('nan')):.6g} m³.", "",
        "[All per-seed phase scores](all_seed_phase_scores.csv), [seasonal bias](seasonal_bias.csv), [regime scores](regime_scores.csv), [failure reasons](infeasible_candidate_reasons.csv), and [comparison with previous routes](comparison_with_previous_routes.csv). The previous 512-evaluation joint calibration is context; it has different data access and parameter dimensions. Held-out evidence never chooses a seed, objective or prior relationship.", "",
        "## Predeclared examples", ""]
    for name in examples: lines += [f"![{name}]({name})", ""]
    lines += ["Convergence, independent forcing validity, and global transferability remain unproven. The frozen DDS horizon is 512; this report evaluates its first 64-evaluation checkpoint.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    write(output / "manifest.json", {"new_model_evaluations": 0, "input_sha256": inputs,
        "artifact_sha256": {p.name: native.sha(p) for p in output.iterdir() if p.is_file()}})
    return output


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--managed", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); print(report(a.managed, a.output))
