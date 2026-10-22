"""Plot saved shared-geometry results without calling a solver or optimizer."""
import argparse
import calendar
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from workflows import reservoir_v2 as native

COLORS = {"observed": "#252A32", "baseline": "#D58A23", "selected": "#0072B2", "proxy": "#009879", "band": "#A6D5CA"}
NOTE = "16-evaluation integration demonstration · convergence untested"


def frame(path):
    return pd.read_csv(path, index_col="date", parse_dates=True, float_precision="round_trip")


def blocks(index, roles, role):
    chosen = roles.eq(role).to_numpy()
    edges = np.diff(np.r_[False, chosen, False].astype(int))
    return [(index[a], index[b - 1] + pd.Timedelta(days=1))
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def save(fig, folder, name):
    fig.savefig(folder / (name + ".png"), dpi=180, facecolor="white")
    fig.savefig(folder / (name + ".pdf"), facecolor="white")
    plt.close(fig)


def generate(run, output):
    run = Path(run).resolve(strict=True)
    output = Path(output).resolve()
    if not run.is_relative_to(ROOT / "outputs") or not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("Use scientific inputs and fresh figure outputs inside project outputs")
    if output.exists(): raise ValueError("Figure outputs are immutable; use a fresh directory")
    consumed = {}
    def pin(path):
        path = Path(path).resolve(strict=True)
        consumed[str(path)] = native.sha(path)
        return path
    completion = native.read(pin(run / "completion_manifest.json"))
    for name, expected in completion["artifact_sha256"].items():
        if native.sha(pin(run / name)) != expected: raise ValueError("Completed run artifact changed: " + name)
    for name, expected in completion["implementation_source_sha256"].items():
        if native.sha(pin(ROOT / name)) != expected: raise ValueError("Implementation changed: " + name)
    state = native.read(run / "status.json")
    if state["status"] != "completed": raise ValueError("Plot a completed run")
    for row in state["stages"]:
        receipt = native.read(pin(run / "stages" / row["id"] / "receipt.json"))
        for path, expected in receipt["artifact_sha256"].items():
            if native.sha(pin(path)) != expected: raise ValueError("Stage data changed: " + path)
    campaign = run / "stages/calibration/campaign"
    cfg = native.read(pin(campaign / "config.json"))
    if len(cfg["profiles"]) != 1: raise ValueError("This report currently expects one reservoir per managed run")
    identifier = cfg["profiles"][0]["reservoir_id"]
    budget = cfg["optimization"]["evaluation_budget"]
    release = campaign / "releases" / f"budget_{budget}"
    registry = native.verify_registry(pin(release / "registry.json"))
    profile_path = release / registry["profiles_by_reservoir_id"][identifier]
    profile = native.read(pin(profile_path))
    prepared = campaign / "prepared" / identifier
    observed = frame(pin(prepared / "supplied_observations_si.csv"))
    proxy = frame(pin(prepared / "observations_si.csv"))
    baseline = frame(pin(prepared / "baseline/simulation.csv"))
    selected = frame(pin(run / "stages/replay/campaign/simulation.csv"))
    band = frame(pin(prepared / "storage_band_si.csv"))
    selection = native.read(pin(prepared / "selection.json"))
    if list(observed.index.strftime("%Y-%m-%d")) != selection["dates"]:
        raise ValueError("Year roles must match the original calendar")
    roles = pd.Series(selection["roles"], index=observed.index)
    for values in (proxy, baseline, selected, band):
        if not values.index.equals(observed.index): raise ValueError("Scientific timelines differ")
    initial = pd.read_csv(pin(prepared / "initial_targets_si.csv"), index_col="month", float_precision="round_trip")
    calibrated = pd.read_csv(pin(profile_path.parent / "monthly_targets_si.csv"), index_col="month", float_precision="round_trip")
    if initial.shape != (12, 6) or not initial.columns.equals(calibrated.columns): raise ValueError("Expected twelve paired monthly targets")
    performance = pd.read_csv(pin(release / "baseline_comparison.csv"), float_precision="round_trip")
    performance = performance[performance.role.eq("selected")].set_index("phase")
    # Independently reproduce displayed scores on the exact phase support.
    for phase in ("calibration", "assessment"):
        mask = roles.eq(phase)
        for name, column in (("storage", "storage_m3"), ("release", "release_m3s")):
            x = observed.loc[mask, column].to_numpy()
            for label, values in (("baseline", baseline), ("calibrated", selected)):
                y = values.loc[mask, column].to_numpy()
                nse = 1 - np.sum((x - y) ** 2) / np.sum((x - x.mean()) ** 2)
                np.testing.assert_allclose(performance.loc[phase, label + "_" + name + "_nse"], nse, atol=1e-12)
    output.mkdir(parents=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "axes.titleweight": "bold",
                         "savefig.bbox": "tight"})
    name = identifier.split("_", 1)[-1].replace("_", " ")
    title = f"{name} · GRAND {profile['grand_id']}"
    demo_note = NOTE if budget == 16 else f"{budget}-evaluation checkpoint · convergence untested"

    # Monthly target curves, with volume converted only for display.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout="constrained")
    category_colors = ["#B96D19", "#0072B2", "#009879"]
    months = np.arange(1, 13)
    for ax, prefix, scale, ylabel in ((axes[0], "storage", 1e6, "Storage target (million m³)"),
                                      (axes[1], "release", 1., "Release target (m³/s)")):
        for category, color in zip(("critical", "normal", "flood"), category_colors):
            column = next(c for c in initial if c.startswith(prefix + "_" + category + "_"))
            ax.plot(months, initial[column] / scale, "--", color=color, marker="o", markersize=4, markerfacecolor="white", linewidth=1.4)
            ax.plot(months, calibrated[column] / scale, "-", color=color, marker="o", markersize=4, linewidth=1.8)
        ax.set(xticks=months, xticklabels=[calendar.month_abbr[m] for m in months], ylabel=ylabel,
               title="Storage targets" if prefix == "storage" else "Release targets")
        if prefix == "storage":
            ax.axhline(profile["physical_constants"]["capacity_m3"] / scale, color="#777777", linestyle=":", linewidth=1)
            ax.text(.01, .97, "Dotted line: adopted capacity", transform=ax.transAxes, fontsize=8, va="top")
    handles = [Line2D([], [], color=c, label=k.title()) for k, c in zip(("critical", "normal", "flood"), category_colors)]
    handles += [Line2D([], [], color="#444444", linestyle="--", label="Initial estimates"), Line2D([], [], color="#444444", label="Calibrated selection")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle(title + " — initial and calibrated monthly parameters\n" + demo_note, fontsize=13)
    save(fig, output, "01_monthly_parameters")

    monthly = pd.DataFrame({"provided_storage_m3": observed.storage_m3, "proxy_storage_m3": proxy.storage_m3,
        "storage_lower_m3": band.storage_lower_m3, "storage_upper_m3": band.storage_upper_m3,
        "baseline_storage_m3": baseline.storage_m3, "selected_storage_m3": selected.storage_m3,
        "observed_release_m3s": observed.release_m3s, "baseline_release_m3s": baseline.release_m3s,
        "selected_release_m3s": selected.release_m3s}, index=observed.index).resample("MS").mean()
    monthly.to_csv(output / "monthly_plot_data_si.csv", index_label="date", float_format="%.17g")
    performance.to_csv(output / "performance_plot_data.csv", float_format="%.17g")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), layout="constrained")
    ax = axes[0]
    ax.fill_between(monthly.index, monthly.storage_lower_m3 / 1e6, monthly.storage_upper_m3 / 1e6,
                    color=COLORS["band"], alpha=.55, label="Reference residual band")
    ax.plot(monthly.index, monthly.provided_storage_m3 / 1e6, color=COLORS["observed"], linewidth=1., label="Supplied storage")
    ax.plot(monthly.index, monthly.proxy_storage_m3 / 1e6, color=COLORS["proxy"], linewidth=1., label="Geometry-derived proxy")
    ax.set(title="Monthly averages of daily storage / proxies", ylabel="Storage (million m³)")
    ax.legend(fontsize=8, loc="lower left")
    mask = roles.eq("assessment")
    x, y = observed.loc[mask, "storage_m3"].to_numpy(), proxy.loc[mask, "storage_m3"].to_numpy()
    ax = axes[1]; ax.scatter(x / 1e6, y / 1e6, s=7, alpha=.3, color=COLORS["proxy"], rasterized=True)
    lo, hi = min(x.min(), y.min()) / 1e6, max(x.max(), y.max()) / 1e6
    ax.plot([lo, hi], [lo, hi], color="#555555", linestyle="--", linewidth=1.)
    nse = 1 - np.sum((x - y) ** 2) / np.sum((x - x.mean()) ** 2)
    rmse = np.sqrt(np.mean((x - y) ** 2)) / 1e6
    ax.text(.04, .94, f"Held-out conversion NSE: {nse:.3f}\nRMSE: {rmse:.1f} million m³", transform=ax.transAxes, va="top")
    ax.set(title="Held-out conversion agreement", xlabel="Supplied storage (million m³)", ylabel="Geometry proxy (million m³)")
    fig.suptitle(title + " — level-to-storage conversion\nReference band describes residual spread, not a confidence interval", fontsize=13)
    save(fig, output, "02_storage_conversion")

    # Daily detail year is chosen by independent forcing, never fit quality.
    held_inflow = observed.loc[mask, "inflow_m3s"].groupby(observed.index[mask].year).mean()
    detail_year = int(held_inflow.idxmax())
    detail = observed.index.year == detail_year
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.7), layout="constrained")
    for row, (column, label, scale) in enumerate((("storage_m3", "Storage (million m³)", 1e6), ("release_m3s", "Release (m³/s)", 1.))):
        for col, daily in enumerate((False, True)):
            ax = axes[row, col]
            for label_key, values, color in (("Supplied observations", observed, COLORS["observed"]),
                                            ("Initial-target simulation", baseline, COLORS["baseline"]),
                                            ("Calibrated simulation", selected, COLORS["selected"])):
                series = values.loc[detail, column] if daily else values[column].resample("MS").mean()
                ax.plot(series.index, series / scale, label=label_key, color=color,
                        linewidth=.9 if label_key == "Supplied observations" else 1.1, alpha=.9)
            if not daily:
                for start, end in blocks(observed.index, roles, "assessment"):
                    ax.axvspan(start, end, color="#B8A5D5", alpha=.22, linewidth=0)
            ax.set(ylabel=label)
            if row == 0:
                ax.set_title("Full record — monthly averages" if not daily else f"Held-out {detail_year} — daily detail")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Patch(color="#B8A5D5", alpha=.35)); labels.append("Held-out year")
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(title + " — simulation against supplied observations\n" + demo_note, fontsize=13)
    save(fig, output, "03_storage_release_simulation")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), layout="constrained")
    phases = ["calibration", "assessment"]; positions = np.arange(2)
    for ax, variable in zip(axes, ("storage", "release")):
        for offset, prefix, color, label in ((-.18, "baseline", COLORS["baseline"], "Initial-target simulation"),
                                              (.18, "calibrated", COLORS["selected"], "Calibrated selection")):
            values = performance.loc[phases, prefix + "_" + variable + "_nse"].to_numpy()
            bars = ax.bar(positions + offset, values, width=.34, color=color, label=label)
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4, fontsize=10)
        delta = performance.loc[phases, "improvement_" + variable + "_nse"].to_numpy()
        for pos, value in zip(positions, delta):
            ax.text(pos, .97, f"Δ {value:+.3f}", ha="center", va="top", color="#1A7655" if value >= 0 else "#B33A3A", fontsize=10)
        ax.set(xticks=positions, xticklabels=["Calibration", "Held-out assessment"], ylabel="NSE against supplied observations",
               ylim=(0, 1.04), title=variable.title() + " performance")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(title + " — performance before and after calibration\n" + demo_note, fontsize=13)
    save(fig, output, "04_performance_comparison")

    lines = [f"# {title}: parameterization and test-run plots", "",
        f"These figures use saved outputs from the {budget}-evaluation integration demonstration. No solver, optimizer or year selector was called. The original verified run and its completion report remain unchanged.", "",
        "## 1. Monthly parameter targets", "", "Dashed curves are initial estimates; solid curves are the calibration-selected targets. Storage is displayed in million m³ and release in m³/s. These curves are operating targets, not the daily storage/release trajectory.", "",
        "![Monthly parameter targets](01_monthly_parameters.png)", "", "[PDF](01_monthly_parameters.pdf)", "",
        "## 2. Level-to-storage conversion", "", "The reference residual band was fitted using calibration years only. It is an empirical compatibility band, not a probabilistic confidence interval. Volume transformed from daily mean level remains an approximate proxy. Conversion agreement is separate from reservoir-simulation performance.", "",
        "![Storage conversion](02_storage_conversion.png)", "", "[PDF](02_storage_conversion.pdf)", "",
        "## 3. Storage and release simulation", "", f"Left panels show monthly averages over the full continuous record. Shading identifies the frozen held-out years. Right panels show daily data in {detail_year}, chosen because it has the highest mean inflow among the held-out years; it was not chosen by fit quality. Supplied storage timing retains the original archive convention; the geometry proxy's mean-level approximation remains explicit.", "",
        "![Storage and release](03_storage_release_simulation.png)", "", "[PDF](03_storage_release_simulation.pdf)", "",
        "## 4. Performance comparison", "", "Bars use the same supplied observations for both simulations. The optimizer used release NSE and storage-band compatibility; the storage bars below show ordinary NSE against supplied storage. Calibration storage NSE decreases while release NSE increases; both held-out NSE values increase in this small demonstration. These results do not establish convergence or superiority over the existing completed paper calibrations.", "",
        "![Performance comparison](04_performance_comparison.png)", "", "[PDF](04_performance_comparison.pdf)", "",
        "The raw figure inputs are pinned in manifest.json; monthly_plot_data_si.csv and performance_plot_data.csv provide the displayed numerical summaries. New model runs: **0**. New optimizer evaluations: **0**.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    # Hash protection prevents plotting from altering any numerical source/artifact.
    for path, expected in consumed.items():
        if native.sha(path) != expected: raise ValueError("Source changed while plotting: " + path)
    proof = {"kind": "cached_shared_geometry_figures_v1", "source_run": str(run), "reservoir_id": identifier,
        "source_sha256": consumed, "plotting_script_sha256": native.sha(__file__), "detail_year": detail_year,
        "detail_year_rule": "highest mean inflow among frozen held-out years", "displayed_nse_verified": True,
        "new_model_runs": 0, "new_optimizer_evaluations": 0,
        "artifact_sha256": {p.name: native.sha(p) for p in output.iterdir() if p.is_file()}}
    native.write(output / "manifest.json", proof)
    return {"status": "completed", "gallery": str(output / "README.md"), "png_figures": 4,
            "pdf_figures": 4, "new_model_runs": 0, "new_optimizer_evaluations": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); print(json.dumps(generate(args.run, args.output), indent=2))
