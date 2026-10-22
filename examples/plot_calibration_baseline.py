"""Plot monthly means from the Trinity calibration/validation baseline."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/trinity_baseline")
    args = parser.parse_args()
    report = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(args.output_dir / "daily_simulation.csv", index_col="date", parse_dates=True)
    columns = ("release_m3s", "observed_release_m3s", "storage_m3", "observed_storage_m3")
    monthly = daily[list(columns)].resample("MS").mean()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    for axis, column, scale, label in (
        (axes[0], "release_m3s", 1.0, "Monthly mean release (m³/s)"),
        (axes[1], "storage_m3", 1e6, "Monthly mean storage (million m³)"),
    ):
        for key, color in (("spinup", "#e5e7eb"), ("calibration", "#edf4fa"), ("validation", "#fff5df")):
            period = report["periods"][key]
            if period:
                axis.axvspan(pd.Timestamp(period["start"]), pd.Timestamp(period["end"]) + pd.Timedelta(days=1),
                             color=color, zorder=0)
        axis.plot(monthly.index, monthly[f"observed_{column}"] / scale, label="Observed", color="#343a40", linewidth=1.1)
        axis.plot(monthly.index, monthly[column] / scale, label="Generalized DZTR baseline", color="#2465a5", linewidth=1.1)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        axis.set_xlim(daily.index[0], daily.index[-1])
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    for key in ("calibration", "validation"):
        period = report["periods"][key]
        center = pd.Timestamp(period["start"]) + (pd.Timestamp(period["end"]) - pd.Timestamp(period["start"])) / 2
        axes[0].text(center, 0.97, key.capitalize(), transform=axes[0].get_xaxis_transform(), ha="center", va="top")
    axes[0].legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.02),
                   borderaxespad=0, fontsize=9)
    reservoir = report["reservoir"]["name"]
    axes[0].set_title(f"{reservoir} — {report['quantile_method'].capitalize()} targets estimated from calibration observations",
                      loc="left", pad=32)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_xlabel("Monthly means shown; performance metrics use daily values. Validation continues the simulated state.")
    output = args.output_dir / "trinity_baseline_overview.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
