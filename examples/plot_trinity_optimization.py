"""Plot the calibration archive and frozen representatives' held-out scores."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROLE_LABELS = {
    "generalized_baseline": "Generalized baseline",
    "best_release": "Best release",
    "best_storage": "Best storage",
    "balanced": "Balanced",
}
ROLE_STYLE = {
    "generalized_baseline": ("*", "#292f36", 180),
    "best_release": ("^", "#0072b2", 100),
    "best_storage": ("s", "#d55e00", 85),
    "balanced": ("o", "#009e73", 100),
}


def _finite_scores(frame):
    values = frame[["release_nse", "storage_nse"]].apply(pd.to_numeric, errors="coerce")
    return np.isfinite(values.to_numpy(dtype=float)).all(axis=1)


def _draw_representatives(axis, selected, period):
    """Draw one marker per evaluation ID even when several roles share it."""
    table = selected.loc[selected.period.eq(period)]
    unavailable = []
    for evaluation_id, rows in table.groupby("evaluation_id", sort=True):
        roles = [role for role in ROLE_LABELS if role in set(rows.role)]
        row = rows.iloc[0]
        label = " + ".join(ROLE_LABELS[role] for role in roles)
        if not _finite_scores(rows.iloc[:1])[0]:
            unavailable.append(f"#{evaluation_id}: {label}")
            continue
        # Preserve the baseline star if a selected role refers to the baseline;
        # otherwise use the balanced marker for shared representative roles.
        priority = next(role for role in ("generalized_baseline", "balanced", "best_release", "best_storage") if role in roles)
        marker, color, size = ROLE_STYLE[priority]
        axis.scatter(row.release_nse, row.storage_nse, marker=marker, s=size,
                     facecolor=color, edgecolor="white", linewidth=0.8,
                     zorder=5, label=f"{label} (#{int(evaluation_id)})")
        axis.annotate(f"#{int(evaluation_id)}", (row.release_nse, row.storage_nse),
                      xytext=(5, 7), textcoords="offset points", fontsize=8,
                      color=color, zorder=6)
    if unavailable:
        axis.text(0.03, 0.03, "Scores unavailable: " + "; ".join(unavailable),
                  transform=axis.transAxes, fontsize=8, va="bottom", wrap=True)


def _period_label(report, name):
    period = report["periods"][name]
    return f"{period['start'][:4]}–{period['end'][:4]}"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/trinity_optimization")
    args = parser.parse_args()
    folder = args.output_dir
    required = ("summary.json", "evaluations.csv", "pareto_front.csv", "selected_metrics.csv")
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        parser.error("Missing optimization artifacts: " + ", ".join(missing)
                     + ". Run python -m dztr.calibration.search first.")
    report = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    evaluated = pd.read_csv(folder / "evaluations.csv")
    front = pd.read_csv(folder / "pareto_front.csv")
    selected = pd.read_csv(folder / "selected_metrics.csv")
    feasible = evaluated.loc[evaluated.feasible.eq(True) & _finite_scores(evaluated)]
    if feasible.empty or front.empty:
        parser.error("The saved search has no finite feasible calibration front to plot.")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    axes[0].scatter(feasible.release_nse, feasible.storage_nse, color="#bdc5ce",
                    alpha=0.65, s=19, linewidths=0, zorder=2,
                    label="Feasible search evaluations")
    line = front.loc[_finite_scores(front)].drop_duplicates(["release_nse", "storage_nse"]).sort_values("release_nse")
    axes[0].plot(line.release_nse, line.storage_nse, color="#596a80",
                 marker="o", markersize=3, linewidth=1.2, zorder=3,
                 label="Nondominated calibration archive")
    for axis, period, panel in zip(axes, ("calibration", "validation"), ("a", "b")):
        _draw_representatives(axis, selected, period)
        axis.set_title(f"({panel}) {period.capitalize()} · {_period_label(report, period)}", loc="left", pad=12)
        axis.set_xlabel("Release NSE (higher is better)")
        axis.set_ylabel("Storage NSE (higher is better)")
        axis.grid(color="#d9dee4", linewidth=0.6, alpha=0.8, zorder=0)
        axis.set_axisbelow(True)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.margins(x=0.12, y=0.14)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.legend(frameon=False, fontsize=8, loc="best", handletextpad=0.5)
    reservoir = report.get("reservoir", {}).get("name", "Trinity")
    fig.suptitle(f"{reservoir} reservoir: NSGA-II calibration pilot", fontsize=15, fontweight="semibold")
    seed = report["optimization"]["seed"]
    fig.supxlabel(
        f"Seed {seed} · {report['search_evaluations']} search evaluations. "
        "Representative roles were fixed from calibration scores; validation reports those choices.\n"
        "The archive describes this limited search; it does not establish a global Pareto front.",
        fontsize=9,
    )
    output = folder / "trinity_pareto.png"
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
