"""Figures and descriptive summaries for a matched full-data scalar experiment."""
from pathlib import Path

import numpy as np
import pandas as pd


def render(build, campaign, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from workflows import reservoir_v2 as native
    from workflows.flow_calibration_v2 import checked

    build, campaign = Path(build), Path(campaign)
    table = pd.read_csv(build / "baseline_comparison.csv")
    status = pd.read_csv(build / "search_status.csv")
    checks = pd.read_csv(build / "scientific_checks.csv")
    progress = pd.read_csv(build / "convergence.csv")
    figures = build / "figures"; figures.mkdir()
    metrics = ["calibrated_storage_nse", "calibrated_release_nse"]
    if table.empty:
        (build / "README.md").write_text("No feasible candidate was found. Inspect search_status.csv and per-job logs.\n", encoding="utf-8")
        return
    summary = table.groupby(["reservoir_id", "objective", "optimizer", "phase"], as_index=False)[metrics].median()
    summary.to_csv(build / "median_performance.csv", index=False, float_format="%.17g")
    paired = []
    for (rid, objective, phase), group in table.groupby(["reservoir_id", "objective", "phase"]):
        dds = group[group.optimizer == "dds"].set_index("seed")
        ga = group[group.optimizer == "ga"].set_index("seed")
        for seed in sorted(set(dds.index) & set(ga.index)):
            row = {"reservoir_id": rid, "objective": objective, "phase": phase, "seed": seed}
            for metric in metrics: row["ga_minus_dds_" + metric] = ga.loc[seed, metric] - dds.loc[seed, metric]
            paired.append(row)
    pd.DataFrame(paired).to_csv(build / "paired_seed_comparison.csv", index=False, float_format="%.17g")
    lines = ["# Full-data GA and DDS comparison", "", "Both algorithms fit the same 72 observed-monthly storage/release targets, starting from the same calibration-only Hazen percentiles and monthly empirical bounds. Physical constants, initial state, forcing, numerical solver and frozen mixed-year mask are shared.", "",
        "Scalar loss: `w_storage * (1 - NSE_storage) + w_release * (1 - NSE_release)`, with weights normalized to sum to one. NSE is dimensionless. Regime scoring is a separate choice: `" + cfg["scoring"] + "`.", "",
        "In release-only fitting the storage weight is zero; storage observations still inform initialization and bounds. This tests objective choice with full data, rather than the storage-hidden flow-only route.", "",
        f"Completed {len(status)} searches and {int(status.evaluations.sum())} optimizer evaluations. Rejections count toward the budget. GA population is configured per algorithm; DDS uses its frozen search horizon.", "",
        "Each seed's vector and the exported seed are selected solely by minimum calibration loss. Assessment responses are read only after selection. The simulation remains continuous across unscored and assessment years; no observed-state resets occur.", "",
        "| Reservoir | Objective | Algorithm | Calibration S NSE | Calibration Q NSE | Assessment S NSE | Assessment Q NSE |", "|---|---|---|---:|---:|---:|---:|"]
    for (rid, objective, optimizer), group in summary.groupby(["reservoir_id", "objective", "optimizer"]):
        phases = group.set_index("phase")
        values = [phases.loc[p, m] for p in ("calibration", "assessment") for m in metrics]
        lines.append(f"| {rid} | {objective} | {optimizer.upper()} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += ["", "The table shows medians across seeds, not validation-selected profiles. Individual seeds, baseline improvements and calibration-selected profiles are in [baseline_comparison.csv](baseline_comparison.csv). Matched-seed differences are in [paired_seed_comparison.csv](paired_seed_comparison.csv).", ""]
    for rid in sorted(table.reservoir_id.unique()):
        group = table[table.reservoir_id == rid]
        objectives = cfg["objectives"]
        fig, axes = plt.subplots(1, len(objectives), figsize=(6 * len(objectives), 4), squeeze=False)
        for ax, objective in zip(axes[0], objectives):
            for algorithm, color in (("dds", "tab:blue"), ("ga", "tab:orange")):
                sub = progress[(progress.reservoir_id == rid) & (progress.objective == objective["id"]) & (progress.optimizer == algorithm)]
                for i, (seed, trace) in enumerate(sub.groupby("seed")):
                    ax.plot(trace.evaluations, trace.best_cost, color=color, alpha=.65, label=algorithm.upper() if i == 0 else None)
            ax.set(title=objective["id"], xlabel="Objective evaluations", ylabel="Best calibration loss")
            ax.grid(alpha=.2); ax.legend()
        fig.tight_layout(); fig.savefig(figures / f"{rid}_convergence.png", dpi=150); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        held = group[group.phase == "assessment"]
        arms = [(o["id"], a["optimizer"]) for o in objectives for a in cfg["optimization"]["algorithms"]]
        for ax, metric, title in zip(axes, metrics, ("Assessment storage NSE", "Assessment release NSE")):
            for i, (objective, algorithm) in enumerate(arms):
                part = held[(held.objective == objective) & (held.optimizer == algorithm)]
                ax.scatter(np.full(len(part), i), part[metric], s=40)
                if len(part): ax.plot([i-.2, i+.2], [part[metric].median()] * 2, color="black", lw=2)
            baseline = held["baseline_" + metric.removeprefix("calibrated_")].dropna()
            if len(baseline): ax.axhline(baseline.iloc[0], color="gray", ls="--", label="Generalized baseline")
            ax.set_xticks(range(len(arms)), [f"{o}\n{a.upper()}" for o, a in arms], rotation=15)
            ax.set_title(title); ax.grid(axis="y", alpha=.2); ax.legend()
        fig.tight_layout(); fig.savefig(figures / f"{rid}_assessment.png", dpi=150); plt.close(fig)
        obs = pd.read_csv(campaign / "handoff" / "prepared" / rid / "observations_si.csv", index_col="date", parse_dates=True, float_precision="round_trip")
        selection = native.read(campaign / "handoff" / "prepared" / rid / "selection.json")
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        for ax, col, scale, ylabel in zip(axes, ("storage_m3", "release_m3s"), (1e9, 1), ("Storage (billion m³)", "Release (m³/s)")):
            ax.plot(obs.index, obs[col] / scale, color="black", lw=.6, alpha=.6, label="Observed")
            for year in sorted({y for y, role in zip(selection["years"], selection["roles"]) if role == "assessment"}):
                dates = obs.index[np.array(selection["years"]) == year]
                ax.axvspan(dates[0], dates[-1], color="gray", alpha=.12)
            for objective, algorithm in arms:
                chosen = group[(group.objective == objective) & (group.optimizer == algorithm) & group.selected & (group.phase == "calibration")]
                if chosen.empty: continue
                seed = int(chosen.iloc[0].seed)
                sim = pd.read_csv(build / "trajectories" / rid / objective / algorithm / f"seed_{seed}" / "simulation.csv", index_col="date", parse_dates=True)
                ax.plot(sim.index, sim[col] / scale, lw=.65, alpha=.8, label=f"{objective} {algorithm.upper()} (seed {seed})")
            ax.set_ylabel(ylabel); ax.grid(alpha=.2)
        axes[0].legend(fontsize=8, ncol=2); axes[0].set_title("Calibration-selected profiles; gray shading marks assessment years")
        fig.tight_layout(); fig.savefig(figures / f"{rid}_trajectories.png", dpi=150); plt.close(fig)
        lines += [f"## {rid}", "", f"![Convergence](figures/{rid}_convergence.png)", "", f"![Assessment](figures/{rid}_assessment.png)", "", f"![Trajectories](figures/{rid}_trajectories.png)", ""]
        handoff = campaign / "handoff" / "prepared" / rid
        shared = {name: native.sha(handoff / name) for name in ("observations_si.csv", "selection.json", "initial_targets_si.csv", "bounds_si.csv", "model_config.json")}
        native.write(build / (rid + "_shared_inputs.json"), shared)
    residuals = checks.maximum_balance_residual_m3.dropna()
    failures = checks[checks.status != "completed"]
    lines += [f"Maximum independently recomputed mass-balance residual across reported trajectories: **{residuals.max():.3g} m³**. Full-period simulation failures: **{len(failures)}**. See [scientific_checks.csv](scientific_checks.csv).", "",
        f"This study uses {len(cfg['profiles'])} reservoir(s), {len(cfg['optimization']['seeds'])} seed(s) and a requested budget of {cfg['optimization']['evaluation_budget']} evaluations per search. This is not a general ranking of algorithms or proof of convergence. Same seed labels do not create identical random proposals across different algorithms. Reservoir forcings and initial-state provenance are inherited from the prepared source; this is a conditional retrospective test, not an independent ungauged evaluation.", "",
        "Weights and optimizer choices must be fixed using application priorities and calibration evidence; assessment scores remain an evaluation. New weights require a separate campaign because checkpoints include the objective identity.", ""]
    (build / "README.md").write_text("\n".join(lines), encoding="utf-8")
