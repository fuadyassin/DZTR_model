"""Plot calibration-only population comparisons from a verified complete study.

Lines show population medians and shading shows the descriptive range across
three seeds. No search or daily simulations run during this export.
"""

import argparse
from pathlib import Path
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dztr.calibration.checkpoint import _replace_checkpoint
from dztr.calibration.population_study import verify_population_study


COLORS = {16: "#0072b2", 32: "#d55e00", 64: "#009e73"}
PANELS = (
    ("hypervolume", "Calibration dominated area", "Hypervolume"),
    ("max_release_nse", "Best calibration release NSE", "Release fit"),
    ("max_storage_nse", "Best calibration storage NSE", "Storage fit"),
)


def read_verified_tables(folder):
    """Require verified source jobs and mutually consistent complete tables."""
    folder = Path(folder)
    verified = verify_population_study(folder)
    metrics = pd.read_csv(folder / "checkpoint_metrics.csv", float_precision="round_trip")
    grouped = pd.read_csv(folder / "population_summary.csv", float_precision="round_trip")
    agreement = pd.read_csv(folder / "front_agreement.csv", float_precision="round_trip")
    keys = ["population_size", "seed", "evaluations"]
    required_metrics = {*keys, *(name for name, _, _ in PANELS)}
    if not required_metrics.issubset(metrics.columns) or metrics.duplicated(keys).any():
        raise ValueError("Population checkpoint metrics have missing columns or duplicate run prefixes.")
    populations = tuple(sorted(metrics.population_size.unique()))
    seeds = tuple(sorted(metrics.seed.unique()))
    counts = tuple(sorted(metrics.evaluations.unique()))
    if populations != (16, 32, 64) or seeds != (42, 43, 44) or counts != (512, 1024, 2048):
        raise ValueError("This figure requires populations 16/32/64, seeds 42/43/44 and counts 512/1024/2048.")
    expected = {(population, seed, count) for population in populations
                for seed in seeds for count in counts}
    if set(metrics[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("All nine planned jobs must have every matched checkpoint before plotting.")
    group_keys = ["population_size", "evaluations"]
    required_summary = {*group_keys, "parameter_spread_median", "parameter_spread_maximum",
                        *(f"{name}_{stat}" for name, _, _ in PANELS for stat in ("min", "median", "max"))}
    if not required_summary.issubset(grouped.columns) or grouped.duplicated(group_keys).any():
        raise ValueError("Population summary has missing columns or duplicate population prefixes.")
    expected_groups = {(population, count) for population in populations for count in counts}
    if set(grouped[group_keys].itertuples(index=False, name=None)) != expected_groups:
        raise ValueError("Population summary does not cover the matched evaluation counts.")
    for row in grouped.itertuples(index=False):
        members = metrics.loc[(metrics.population_size == row.population_size)
                              & (metrics.evaluations == row.evaluations)]
        for name, _, _ in PANELS:
            values = members[name].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Population comparison requires finite calibration metrics for every seed.")
            for stat, calculated in (("min", np.min(values)), ("median", np.median(values)), ("max", np.max(values))):
                reported = getattr(row, f"{name}_{stat}")
                if not np.isfinite(reported) or not np.isclose(reported, calculated, rtol=1e-12, atol=1e-12):
                    raise ValueError("Population summary statistics differ from their checkpoint metrics.")
        ranges = np.asarray([row.parameter_spread_median, row.parameter_spread_maximum], dtype=float)
        if not np.isfinite(ranges).all() or not 0 <= ranges[0] <= ranges[1] <= 1 + 1e-12:
            raise ValueError("Normalized target ranges must be finite fractions between zero and one.")
    agreement_keys = ["population_size", "evaluations", "first_seed", "second_seed"]
    required_agreement = {*agreement_keys, "front_distance"}
    if not required_agreement.issubset(agreement.columns) or agreement.duplicated(agreement_keys).any():
        raise ValueError("Full-front agreement has missing columns or duplicate seed pairs.")
    pairs = ((42, 43), (42, 44), (43, 44))
    expected_pairs = {(population, count, first, second) for population in populations
                      for count in counts for first, second in pairs}
    if set(agreement[agreement_keys].itertuples(index=False, name=None)) != expected_pairs:
        raise ValueError("Full-front agreement must include every planned seed pair at each matched count.")
    distances = agreement.front_distance.to_numpy(dtype=float)
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("Full-front distances must be finite and nonnegative.")
    return grouped, agreement, verified


def plot_population_comparison(grouped, agreement, folder):
    """Export medians and descriptive seed ranges with agreement context."""
    folder = Path(folder)
    counts = sorted(grouped.evaluations.unique())
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(14, 7.0))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.38, top=0.77, wspace=0.31)
    for index, (name, ylabel, title) in enumerate(PANELS):
        axis = axes[index]
        for population, color in COLORS.items():
            points = grouped.loc[grouped.population_size.eq(population)].sort_values("evaluations")
            x = points.evaluations.to_numpy(dtype=int)
            axis.fill_between(x, points[f"{name}_min"].to_numpy(dtype=float),
                              points[f"{name}_max"].to_numpy(dtype=float), color=color, alpha=0.13)
            axis.plot(x, points[f"{name}_median"].to_numpy(dtype=float), color=color,
                      linewidth=2.0, marker="o", markersize=4.2, zorder=3)
        axis.set_title(f"({chr(ord('a') + index)}) {title}", loc="left", pad=10)
        axis.set_xlabel("Cumulative search evaluations per seed")
        axis.set_ylabel(ylabel)
        axis.set_xticks(counts, [f"{count:,}" for count in counts])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.grid(color="#d9dee4", linewidth=0.6, alpha=0.8)
        axis.margins(x=0.05, y=0.16)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    handles = [Line2D([0], [0], color=color, linewidth=2, marker="o", markersize=4,
                      label=f"Population {population}") for population, color in COLORS.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.88),
               ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Trinity calibration: population size at equal evaluation budgets",
                 fontsize=14, fontweight="semibold", y=0.97)
    final_count = counts[-1]
    context_rows = []
    for population in COLORS:
        front_distances = agreement.loc[agreement.population_size.eq(population)
                                       & agreement.evaluations.eq(final_count), "front_distance"]
        row = grouped.loc[grouped.population_size.eq(population)
                          & grouped.evaluations.eq(final_count)].iloc[0]
        context_rows.append([
            f"Population {population}",
            f"{front_distances.min():.5f} to {front_distances.max():.5f}",
            f"{row.parameter_spread_median:.2%} / {row.parameter_spread_maximum:.2%}",
        ])
    context_axis = fig.add_axes([0.12, 0.105, 0.76, 0.145])
    context_axis.axis("off")
    table = context_axis.table(cellText=context_rows,
                               colLabels=[f"At {final_count:,} evaluations", "Seed-front distance range (NSE units)",
                                          "Balanced-target range: median / max"],
                               loc="center", cellLoc="center", colWidths=[0.21, 0.42, 0.37])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dee4")
        cell.set_linewidth(0.5)
        if row_index == 0:
            cell.set_facecolor("#eef2f6")
            cell.set_text_props(weight="semibold")
    fig.text(0.5, 0.285,
             "Lines: median; shading: min-to-max across three seeds (descriptive ranges, not confidence intervals).",
             ha="center", fontsize=9)
    fig.text(0.5, 0.035,
             "Calibration only; higher plotted metrics are better. Fixed hypervolume reference: [1, 1] in negative-NSE space.\n"
             "Full-front distances use symmetric Hausdorff distance with the infinity norm. Target ranges divide by fixed calibration bound widths.\n"
             "Fixed budgets disable automatic stopping. One reservoir and three seeds do not establish global convergence or a fleet default.",
             ha="center", fontsize=8.8)
    output = folder / "population_comparison.png"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".population_comparison_", suffix=".png",
                                         dir=folder, delete=False) as stream:
            temporary = Path(stream.name)
        fig.savefig(temporary, dpi=220, facecolor="white")
        verify_population_study(folder)
        _replace_checkpoint(temporary, output)
        temporary = None
    finally:
        plt.close(fig)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    root = args.repo.resolve()
    folder = (root / "outputs/trinity_population_study" if args.output_dir is None
              else root / args.output_dir).resolve()
    if not folder.is_relative_to(root):
        parser.error("Study output must remain within --repo.")
    try:
        grouped, agreement, _ = read_verified_tables(folder)
        output = plot_population_comparison(grouped, agreement, folder)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print("Verified complete population study and source job receipts; no search or simulations performed.")
    print(output)


if __name__ == "__main__":
    main()
