"""Report sealed flow-only benchmark outputs without running the model."""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from workflows import flow_calibration_v1 as flow
from workflows import reservoir_v2 as native
from workflows.atomic_io_v1 import write


def report(campaign, budget, output):
    campaign, cfg = flow.verify(campaign)
    release = campaign / "releases" / f"budget_{budget}"; flow.checked(release)
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / "outputs") or output.exists(): raise ValueError("Use a fresh project report folder")
    output.mkdir(parents=True)
    baseline = pd.read_csv(release / "baseline_comparison.csv", float_precision="round_trip")
    selected = baseline[baseline.selected == True].copy()
    previous_path = ROOT / "outputs/calibrated_paper_cohort_v1_20261017_final/cohort_summary.csv"
    previous = pd.read_csv(previous_path, float_precision="round_trip")
    selected = selected.merge(previous[["reservoir_id", "calibration_release_nse", "calibration_storage_nse", "assessment_release_nse", "assessment_storage_nse"]],
        on="reservoir_id", how="left", validate="many_to_one", suffixes=("", "_previous_joint"))
    selected.to_csv(output / "selected_comparison_with_previous_joint.csv", index=False, float_format="%.17g")
    held = selected[selected.phase == "assessment"]
    summary = []
    for objective in cfg["objectives"]:
        arm = held[held.objective == objective]
        row = {"objective": objective, "selected_profiles": len(arm), "held_out_release_scores": int(arm.calibrated_release_nse.notna().sum()),
               "held_out_storage_scores": int(arm.calibrated_storage_nse.notna().sum())}
        for metric in ("release_nse", "sqrt_release_nse", "storage_nse"):
            row["median_held_out_" + metric] = arm["calibrated_" + metric].median()
            delta = arm["improvement_" + metric].dropna()
            row["baseline_comparable_" + metric] = len(delta)
            row["improved_" + metric] = int((delta > 1e-12).sum())
            row["worsened_" + metric] = int((delta < -1e-12).sum())
        row["both_held_out_nse_positive"] = int(((arm.calibrated_release_nse > 0) & (arm.calibrated_storage_nse > 0)).sum())
        summary.append(row)
    pd.DataFrame(summary).to_csv(output / "cohort_summary.csv", index=False, float_format="%.17g")
    # Differences compare identical metric definitions, masks and cold starts;
    # they do not subtract sqrt-Q NSE from raw-Q NSE.
    left = held[held.objective == "release_nse"]
    right = held[held.objective == "sqrt_release_nse"]
    paired = left.merge(right, on="reservoir_id", suffixes=("_raw_objective", "_sqrt_objective"), validate="one_to_one")
    for metric in ("release_nse", "sqrt_release_nse", "storage_nse"):
        paired["sqrt_minus_raw_" + metric] = paired["calibrated_" + metric + "_sqrt_objective"] - paired["calibrated_" + metric + "_raw_objective"]
    paired.to_csv(output / "paired_objective_comparison.csv", index=False, float_format="%.17g")
    seasonal, regime_rows, failure_rows = [], [], []
    balances = []
    status_table = pd.read_csv(release / "search_status.csv")
    for _, job in status_table.iterrows():
        snapshot = campaign / "jobs" / job.reservoir_id / job.objective / f"seed_{int(job.seed)}" / f"budget_{budget}" / "result.json"
        counts = Counter(t["reason"] for t in native.read(snapshot)["trials"] if not t["feasible"])
        for reason, count in counts.items():
            failure_rows.append({"reservoir_id": job.reservoir_id, "objective": job.objective, "seed": int(job.seed), "reason": reason, "count": count})
    pd.DataFrame(failure_rows).to_csv(output / "infeasible_candidate_reasons.csv", index=False)
    for choice in native.read(release / "selection.json")["records"]:
        identifier, objective = choice["reservoir_id"], choice["objective"]
        prepared = campaign / "prepared" / identifier
        obs = flow.availability.read_frame(prepared / "hidden_assessment_observations_si.csv")
        selection = flow.YearSelection(**native.read(prepared / "selection.json"))
        sim = flow.availability.read_frame(release / "diagnostics" / identifier / objective / "simulation.csv")
        assessment_ok = native.read(release / "diagnostics" / identifier / objective / "assessment.json")["status"] == "completed"
        model = native.read(prepared / "model_config.json")
        residual = native.independent_balance(sim, model["physical_constants"]["inflow_correction"])
        if residual > model["atol_storage_m3"]: raise ValueError("Cached selected trajectory fails independent mass balance")
        balances.append(residual)
        profile, parameters = native.load_profile(release / objective / "registry.json", identifier=identifier)
        cap = parameters.capacity_m3; dead = parameters.dead_storage_fraction * cap
        basis = profile["parameterization"]["storage_basis"]
        targets = np.array([.2, .3, .97]) * (cap if basis == "total_capacity" else cap - dead)
        if basis == "active_capacity": targets += dead
        np.testing.assert_array_equal(parameters.monthly_targets.iloc[:, :3], np.tile(targets, (12, 1)))
        assert parameters.initial_storage_m3 == targets[1]
        training = flow.availability.read_frame(prepared / "observations_si.csv")
        assert training.storage_m3.isna().all()
        dates = obs.index[selection.mask(obs.index, "calibration")]
        labels_cal = selection.table().loc[dates, "regime"]
        groups = [dates] if cfg["scoring"] == "pooled" else [dates[labels_cal == label] for label in sorted(set(labels_cal))]
        cost = 1 - np.mean([flow.nse(obs.loc[d, "release_m3s"], sim.loc[d, "release_m3s"], square_root=objective == "sqrt_release_nse") for d in groups])
        np.testing.assert_allclose(cost, choice["cost"], rtol=0, atol=1e-10)
        roles = np.array(selection.roles); labels = selection.table().regime.to_numpy()
        for phase in ("calibration", "assessment"):
            if phase == "assessment" and not assessment_ok: continue
            for month in range(1, 13):
                dates = obs.index[(roles == phase) & (obs.index.month == month)]
                if not len(dates) or not dates.isin(sim.index).all(): continue
                row = {"reservoir_id": identifier, "objective": objective, "phase": phase, "month": month}
                for variable, column in (("storage", "storage_m3"), ("release", "release_m3s")):
                    values = (sim.loc[dates, column] - obs.loc[dates, column]).dropna()
                    row[variable + "_mean_bias"] = values.mean(); row[variable + "_pairs"] = len(values)
                seasonal.append(row)
            for regime in sorted(set(labels[roles == phase])):
                dates = obs.index[(roles == phase) & (labels == regime)]
                if not dates.isin(sim.index).all(): continue
                row = {"reservoir_id": identifier, "objective": objective, "phase": phase, "regime": regime}
                for metric, column, square in (("release_nse", "release_m3s", False), ("sqrt_release_nse", "release_m3s", True), ("storage_nse", "storage_m3", False)):
                    try: row[metric] = flow.nse(obs.loc[dates, column], sim.loc[dates, column], square_root=square)
                    except ValueError: row[metric] = None
                regime_rows.append(row)
    pd.DataFrame(seasonal).to_csv(output / "seasonal_bias.csv", index=False, float_format="%.17g")
    pd.DataFrame(regime_rows).to_csv(output / "regime_scores.csv", index=False, float_format="%.17g")
    write(output / "scientific_checks.json", {"status": "passed", "selected_profiles_checked": len(balances),
        "fixed_storage_targets_verified": True, "storage_independent_initialization_verified": True,
        "calibration_choices_reproduced_from_cached_release_only": True, "training_storage_hidden": True,
        "maximum_independent_balance_residual_m3": max(balances) if balances else None,
        "new_model_evaluations": 0, "source_certified": False, "convergence_established": False})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"release_nse": "#1b6ca8", "sqrt_release_nse": "#cb6d32"}
    for objective in cfg["objectives"]:
        arm = held[held.objective == objective]
        label = "Raw-Q objective" if objective == "release_nse" else "Square-root-Q objective"
        axes[0].scatter(arm.calibrated_release_nse, arm.calibrated_storage_nse, label=label, color=colors[objective], alpha=.8)
        axes[1].scatter(arm.improvement_release_nse, arm.improvement_storage_nse, label=label, color=colors[objective], alpha=.8)
    for ax in axes:
        ax.axhline(0, color="grey", lw=.8); ax.axvline(0, color="grey", lw=.8); ax.grid(alpha=.2)
    axes[0].set(xlabel="Held-out release NSE", ylabel="Hidden-storage NSE", title="Release performance and storage plausibility")
    axes[1].set(xlabel="Release NSE change from flow-only baseline", ylabel="Storage NSE change from flow-only baseline", title="Calibration changes on the held-out years")
    axes[0].legend(fontsize=8)
    for ax, threshold in zip(axes, (1, .1)):
        ax.set_xscale("symlog", linthresh=threshold); ax.set_yscale("symlog", linthresh=threshold)
    fig.suptitle(f"Flow-only comparison: {budget} evaluations per search; not a convergence result")
    fig.text(.5, .01, "Symmetric log axes: linear within ±1 NSE (left) and ±0.1 change (right); logarithmic beyond.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(output / "cohort_comparison.png", dpi=160); fig.savefig(output / "cohort_comparison.pdf"); plt.close(fig)
    campaign_summary = native.read(release / "summary.json")
    lines = ["# Flow-only paper-reservoir benchmark", "", f"Completed checkpoint: **{budget} evaluations per search**, within a fixed **{cfg['optimization']['horizon']}-evaluation DDS horizon**.", "",
        f"Requested: {campaign_summary['requested_reservoirs']} reservoirs. Ready: {campaign_summary['ready_reservoirs']}. Searches: {campaign_summary['searches']}. Actual optimizer evaluations: {campaign_summary['evaluations']}. Selected profiles: {campaign_summary['selected_profiles']}.", "",
        "Storage observations were hidden from targets, bounds, initial conditions, objective evaluation and parameter selection. Initial storage equals the assumed normal target. Continuous forcing carries state across the original mixed-year masks; at least 65% is calibration. The published sources/forcing may depend on original observations, so this is a conditional benchmark, not an independently ungauged test.", "",
        "Etienne-inspired methods: fixed capacity fractions, monthly observed release targets, single-objective DDS and square-root-Q NSE. Raw-Q NSE is a predeclared comparison arm. Valid zero releases are retained; Hazen remains explicit. Annual Etienne-style release envelopes and linear percentiles are supported options, not used in this first controlled comparison. Capacity, dead storage and channel metadata remain fixed. Dickson, E. B. Campbell and Yellowtail use an explicitly exploratory active-storage fraction basis.", "",
        "| Objective | Profiles | Median held release NSE | Median hidden-storage NSE | Release improved / comparable baselines | Storage improved / comparable baselines | Both NSE positive |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['objective']} | {r['selected_profiles']} | {r['median_held_out_release_nse']:.3f} | {r['median_held_out_storage_nse']:.3f} | {r['improved_release_nse']}/{r['baseline_comparable_release_nse']} | {r['improved_storage_nse']}/{r['baseline_comparable_storage_nse']} | {r['both_held_out_nse_positive']} |")
    lines += ["", "![Cohort performance](cohort_comparison.png)", "", "Each baseline has the same fixed storage targets, initial conditions, forcing and numerical route as its calibrated profile. Missing baseline scores reflect recorded simulation failures. Missing optimized scores or profiles remain explicit in search_status.csv and the 37-row parameter inventories. Held-out scores never select the objective, seed or parameter vector.", "",
        "The previous joint storage/release calibration appears in selected_comparison_with_previous_joint.csv as context. It used observed storage, different initial conditions, 72 fitted targets and a 512-evaluation NSGA-II search. Differences from that result cannot isolate the objective or optimizer effect.", "",
        "## Per-reservoir held-out results", "", "| Reservoir | Objective | Release NSE | Hidden-storage NSE | Baseline release change | Assessment |", "|---|---|---:|---:|---:|---|"]
    for _, r in held.sort_values(["reservoir_id", "objective"]).iterrows():
        lines.append(f"| {r.reservoir_id} | {r.objective} | {r.calibrated_release_nse:.3f} | {r.calibrated_storage_nse:.3f} | {r.improvement_release_nse:.3f} | {r.assessment_status} |")
    phase_link = Path(__import__('os').path.relpath(release / "baseline_comparison.csv", output)).as_posix()
    lines += ["", "## Artifacts", "", f"- [All selected and per-seed phase scores]({phase_link})",
        "- [Comparison with previous joint calibration](selected_comparison_with_previous_joint.csv)", "- [Paired raw/sqrt objective comparison](paired_objective_comparison.csv)", "- [Seasonal bias](seasonal_bias.csv) and [regime scores](regime_scores.csv)", "",
        "Convergence and source certification remain unproven. Storage disagreement is a diagnostic outcome, not a criterion used to discard candidates or choose the best validation seed.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    write(output / "manifest.json", {"new_model_evaluations": 0,
        "input_sha256": {str(release / "manifest.json"): native.sha(release / "manifest.json"), str(previous_path): native.sha(previous_path),
                         str(Path(__file__)): native.sha(Path(__file__))},
        "artifact_sha256": {p.name: native.sha(p) for p in output.iterdir() if p.is_file()}})
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--campaign", required=True); p.add_argument("--budget", type=int, required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); print(report(a.campaign, a.budget, a.output))
