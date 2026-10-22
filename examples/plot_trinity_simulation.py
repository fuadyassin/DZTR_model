"""Plot the saved Trinity comparison after running its simulation benchmark."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from dztr.data import read_reservoir_observations


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/trinity_simulation")
    args = parser.parse_args()
    observed = read_reservoir_observations(root / "ResDataInflowStorageRelease/Trinity_131.xlsx").frame
    imported = pd.read_csv(args.output_dir / "daily_imported_matlab.csv", index_col="date", parse_dates=True)
    current = pd.read_csv(args.output_dir / "daily_generalized_linear.csv", index_col="date", parse_dates=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    for ax, column, scale, label in (
        (axes[0], "release_m3s", 1.0, "Release (m³/s)"),
        (axes[1], "storage_m3", 1e6, "Storage (million m³)"),
    ):
        for frame, title, color in ((observed, "Observed", "#343a40"),
                                    (imported, "Imported MATLAB targets", "#d99c28"),
                                    (current, "Current Python targets (linear)", "#2465a5")):
            values = frame.loc["1998-01-01":"2000-12-31", column] / scale
            ax.plot(values.index, values, label=title, color=color, linewidth=1.1)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    axes[0].set_title("Trinity reservoir — imported and current parameterizations", loc="left")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_xlabel("Parameters estimated over 1970–2000; display window 1998–2000")
    fig.savefig(args.output_dir / "trinity_comparison.png", dpi=180)
    plt.close(fig)
    print(args.output_dir / "trinity_comparison.png")


if __name__ == "__main__":
    main()
