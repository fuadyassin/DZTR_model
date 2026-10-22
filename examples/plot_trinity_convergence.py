"""Plot calibration search progress and descriptive variation in monthly targets."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#6b7280")
MARKERS = ("o", "s", "^")
TARGETS = (
    "storage_critical_m3", "storage_normal_m3", "storage_flood_m3",
    "release_critical_m3s", "release_normal_m3s", "release_flood_m3s",
)


def plot_progress(folder, report, checkpoints, selected):
    seeds = sorted(checkpoints.seed.unique())
    budgets = sorted(checkpoints.budget.unique())
    reference = np.asarray(report["study"]["hypervolume_reference"], dtype=float)
    baseline = selected.loc[selected.role.eq("generalized_baseline") & selected.period.eq("calibration")].iloc[0]
    baseline_hv = max(reference[0] + baseline.release_nse, 0) * max(reference[1] + baseline.storage_nse, 0)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), constrained_layout=True)
    panels = (
        ("hypervolume", "(a) Calibration hypervolume", "Dominated area", baseline_hv),
        ("max_release_nse", "(b) Best release fit", "Maximum release NSE", baseline.release_nse),
        ("max_storage_nse", "(c) Best storage fit", "Maximum storage NSE", baseline.storage_nse),
    )
    for axis, (column, title, ylabel, baseline_value) in zip(axes, panels):
        for index, seed in enumerate(seeds):
            rows = checkpoints.loc[checkpoints.seed.eq(seed)].sort_values("budget")
            axis.plot(rows.budget, rows[column], label=f"Seed {int(seed)}",
                      color=COLORS[index % len(COLORS)], marker=MARKERS[index % len(MARKERS)],
                      markersize=5, linewidth=1.7, zorder=3)
        axis.axhline(baseline_value, color="#6b7280", linestyle="--", linewidth=1,
                     zorder=1, label="Generalized baseline")
        axis.set_title(title, loc="left", pad=10)
        axis.set_xlabel("Search evaluations per seed")
        axis.set_ylabel(ylabel)
        axis.set_xticks(budgets)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.grid(color="#d9dee4", linewidth=0.6, alpha=0.8)
        axis.margins(x=0.06, y=0.12)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    reservoir = report.get("reservoir", {}).get("name", "Trinity")
    fig.suptitle(f"{reservoir}: calibration progress across seeds and nested budgets", fontsize=14, fontweight="semibold")
    fig.supxlabel(
        f"Hypervolume reference in negative-NSE coordinates: [{reference[0]:g}, {reference[1]:g}]. "
        "Higher values are better in all panels.\n"
        f"Checkpoints are prefixes of {len(seeds)} searches; progress does not establish global convergence.",
        fontsize=9,
    )
    output = folder / "trinity_convergence.png"
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)
    return output


def plot_parameter_variation(folder, report, stability, targets):
    budget = int(stability.budget.max())
    rows = stability.loc[stability.budget.eq(budget) & stability.role.eq("balanced")]
    seed_targets = targets.loc[targets.budget.eq(budget) & targets.role.eq("balanced")]
    seeds = sorted(seed_targets.seed.unique())
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.21, top=0.88, hspace=0.35, wspace=0.29)
    for index, (axis, column) in enumerate(zip(axes.flat, TARGETS)):
        table = rows.loc[rows.target_column.eq(column)].sort_values("month")
        if table.month.tolist() != list(range(1, 13)):
            raise ValueError(f"Parameter stability requires twelve months for {column}.")
        scale = 1e6 if column.startswith("storage") else 1.0
        months = table.month.to_numpy()
        array = lambda name: table[name].to_numpy(dtype=float) / scale
        axis.fill_between(months, array("lower_si"), array("upper_si"), color="#dde1e6",
                          alpha=0.65, label="Calibration percentile bounds", zorder=0)
        axis.fill_between(months, array("min_si"), array("max_si"), color="#62a894",
                          alpha=0.3, label="Balanced seed range", zorder=1)
        for seed_index, seed in enumerate(seeds):
            values = seed_targets.loc[seed_targets.seed.eq(seed)].sort_values("month")
            if values.month.tolist() != list(range(1, 13)):
                raise ValueError(f"Representative targets require twelve months for seed {seed}.")
            axis.plot(values.month, values[column] / scale, color=COLORS[seed_index % len(COLORS)],
                      linewidth=1.1, alpha=0.9, label=f"Seed {int(seed)}", zorder=2)
        axis.plot(months, array("median_si"), color="#27313b", linestyle="--", linewidth=1.6,
                  label="Median across seeds", zorder=3)
        level = ("Critical", "Normal", "Flood")[index % 3]
        kind = "storage" if index < 3 else "release"
        axis.set_title(f"{level} {kind}", loc="left", pad=8)
        axis.set_ylabel("Storage target (million m³)" if index < 3 else "Release target (m³/s)")
        axis.set_xticks([1, 3, 5, 7, 9, 11], ["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
        axis.set_xlim(1, 12)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.grid(color="#d9dee4", linewidth=0.6, alpha=0.8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.075),
               ncol=3, frameon=False, fontsize=9)
    reservoir = report.get("reservoir", {}).get("name", "Trinity")
    fig.suptitle(f"{reservoir}: monthly targets from balanced calibration representatives", fontsize=14, fontweight="semibold", y=0.96)
    fig.text(0.5, 0.025,
             f"{len(seeds)} seeds at {budget:,} evaluations each. The range describes seed variation, not a confidence interval.\n"
             "Each seed selects its balanced candidate from its own calibration front; validation does not choose these targets.",
             ha="center", fontsize=9)
    output = folder / "trinity_parameter_stability.png"
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/trinity_convergence")
    args = parser.parse_args()
    folder = args.output_dir
    required = ("summary.json", "checkpoint_metrics.csv", "parameter_stability.csv",
                "representative_targets_si.csv", "selected_metrics.csv")
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        parser.error("Missing study artifacts: " + ", ".join(missing)
                     + ". Run python -m dztr.calibration.study first.")
    report = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    checkpoints = pd.read_csv(folder / "checkpoint_metrics.csv")
    stability = pd.read_csv(folder / "parameter_stability.csv")
    targets = pd.read_csv(folder / "representative_targets_si.csv")
    selected = pd.read_csv(folder / "selected_metrics.csv")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    print(plot_progress(folder, report, checkpoints, selected))
    print(plot_parameter_variation(folder, report, stability, targets))


if __name__ == "__main__":
    main()
