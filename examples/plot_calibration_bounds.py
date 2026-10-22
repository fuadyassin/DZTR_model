"""Plot the monthly calibration bounds and generalized starting candidate."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/trinity_calibration")
    args = parser.parse_args()
    bounds = pd.read_csv(args.output_dir / "bounds_si.csv", index_col="month")
    baseline = pd.read_csv(args.output_dir / "baseline_targets_si.csv", index_col="month")
    report = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    months = np.arange(1, 13)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, constrained_layout=True)
    for col, level in enumerate(("critical", "normal", "flood")):
        for row, stem, suffix, scale, unit in ((0, "storage", "m3", 1e6, "million m³"),
                                               (1, "release", "m3s", 1, "m³/s")):
            axis = axes[row, col]
            column = f"{stem}_{level}_{suffix}"
            axis.fill_between(months, bounds[f"lower_{column}"] / scale, bounds[f"upper_{column}"] / scale,
                              color="#dce9f5", label="Percentile bounds")
            axis.plot(months, baseline[column] / scale, color="#2465a5", linewidth=1.5, label="Generalized starting values")
            axis.set_title(f"{level.capitalize()} {stem}", fontsize=11)
            axis.set_ylabel(unit)
            axis.set_xticks([1, 3, 5, 7, 9, 11], ["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
            axis.set_xlim(1, 12)
            axis.grid(alpha=0.2)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    period = report["bounds"]["estimation_period"]
    fig.suptitle(f"{report['reservoir']['name']} — bounds from calibration observations\n"
                 f"{period['start']} to {period['end']}; {report['bounds']['quantile_method']} percentiles", fontsize=13)
    output = args.output_dir / "trinity_calibration_bounds.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
