"""Export retrospective calibration stopping diagnostics from saved archives."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd


COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#6b7280")
MARKERS = ("o", "s", "^")


def plot_diagnostics(folder: Path, diagnostics: pd.DataFrame, policy: dict) -> Path:
    seeds = sorted(diagnostics.seed.unique())
    counts = sorted(diagnostics.evaluations.unique())
    minimum = int(policy["minimum_evaluations"])
    window = int(policy["window_evaluations"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.28, wspace=0.3)
    panels = (
        ("normalized_hv_gain", 100.0, policy["normalized_hv_tolerance"] * 100,
         "(a) Fixed-reference hypervolume gain", "Gain (% of theoretical area)"),
        ("front_movement", 1.0, policy["front_tolerance"],
         "(b) Full-front movement", r"Symmetric Hausdorff distance ($L_\infty$)"),
        ("extrema_drift", 1.0, policy["extrema_tolerance"],
         "(c) Objective extrema drift", "Maximum extreme-score change (NSE)"),
    )
    for axis, (column, scale, threshold, title, ylabel) in zip(axes, panels):
        for index, seed in enumerate(seeds):
            rows = diagnostics.loc[diagnostics.seed.eq(seed)].sort_values("evaluations")
            axis.plot(rows.evaluations, rows[column] * scale, label=f"Seed {int(seed)}",
                      color=COLORS[index % len(COLORS)], marker=MARKERS[index % len(MARKERS)],
                      markersize=4.8, linewidth=1.6, zorder=3)
        axis.axhline(threshold, color="#4b5563", linestyle="--", linewidth=1.2,
                     label="Illustrative threshold", zorder=2)
        axis.axvline(minimum, color="#8b5e34", linestyle=":", linewidth=1.2,
                     label=f"Minimum budget ({minimum:,})", zorder=1)
        axis.annotate(f"{threshold:g}", xy=(0.98, threshold),
                      xycoords=axis.get_yaxis_transform(), ha="right", va="bottom",
                      fontsize=8, color="#4b5563",
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
        axis.set_title(title, loc="left", pad=9)
        axis.set_xlabel("Recorded search evaluations per seed")
        axis.set_ylabel(ylabel)
        tick_counts = counts if len(counts) <= 6 else counts[::2] + [counts[-1]]
        axis.set_xticks(sorted(set(tick_counts)))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.set_ylim(bottom=0)
        axis.margins(x=0.035, y=0.12)
        axis.grid(color="#d9dee4", linewidth=0.6, alpha=0.8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.13),
               ncol=min(len(labels), 5), frameon=False, fontsize=9)
    fig.suptitle("Calibration search: retrospective budget diagnostics", fontsize=14,
                 fontweight="semibold", y=0.97)
    reference = policy["hypervolume_reference"]
    fig.text(0.5, 0.90,
             f"Each point compares the feasible archive with its {window:,}-evaluation-earlier prefix; "
             f"fixed negative-NSE reference [{reference[0]:g}, {reference[1]:g}].",
             ha="center", fontsize=9)
    fig.text(0.5, 0.025,
             f"Illustrative policy requires all three signals for {int(policy['patience'])} consecutive eligible checks. "
             "Thresholds are not validated production recommendations.\n"
             "Saved calibration archives only: no new search evaluations and no proof of global convergence.",
             ha="center", fontsize=9)
    output = folder / "budget_diagnostics.png"
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/trinity_budget_review")
    args = parser.parse_args()
    folder = args.output_dir
    required = ("checkpoint_diagnostics.csv", "resolved_policy.json")
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        parser.error("Missing review artifacts: " + ", ".join(missing)
                     + ". Run python -m dztr.calibration.budget_review first.")
    diagnostics = pd.read_csv(folder / "checkpoint_diagnostics.csv")
    columns = {"seed", "evaluations", "normalized_hv_gain", "front_movement", "extrema_drift"}
    if diagnostics.empty or not columns.issubset(diagnostics.columns):
        parser.error("Checkpoint diagnostics must contain seed, evaluations and the three progress signals.")
    policy = json.loads((folder / "resolved_policy.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10})
    print(plot_diagnostics(folder, diagnostics, policy))


if __name__ == "__main__":
    main()
