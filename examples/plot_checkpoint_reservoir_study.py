"""Export calibration progress from verified, completed adaptive checkpoints.

This analysis reconstructs archive prefixes; it makes no simulation or search
calls. Trinity and Gardiner occupy separate panels because the experiments have
different data, periods and model qualifications.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dztr.calibration.adaptive import (
    ARTIFACT_NAMES, CHECKPOINT_NAME, IDENTITY_NAME, MANIFEST_NAME,
)
from dztr.calibration.checkpoint import load_optimizer_checkpoint
from dztr.calibration.convergence import checkpoint_metrics
from dztr.calibration.stopping import StoppingPolicy


COLORS = {42: "#0072b2", 43: "#d55e00", 44: "#009e73"}
EXPECTED_JOBS = {("Trinity", 42), ("Trinity", 43), ("Trinity", 44), ("Gardiner", 42)}
PANELS = (
    ("hypervolume", "Hypervolume", "Calibration dominated area"),
    ("max_release_nse", "Best release fit", "Maximum calibration release NSE"),
    ("max_storage_nse", "Best storage fit", "Maximum calibration storage NSE"),
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_verified_prefixes(root, folder):
    """Validate every published job artifact before reconstructing its metrics."""
    study = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if study.get("experiment") != "exploratory_checkpoint_reservoir_study":
        raise ValueError("Expected the completed checkpoint-reservoir study summary.")
    jobs = study.get("jobs", [])
    if len(jobs) != len(EXPECTED_JOBS) or {
        (row["job"]["reservoir"], row["job"]["seed"]) for row in jobs
    } != EXPECTED_JOBS or any(row.get("status") != "complete" for row in jobs):
        raise ValueError("All four planned jobs must complete before plotting this study.")
    rows, source_checks = [], []
    for result in sorted(jobs, key=lambda row: (row["job"]["reservoir"], row["job"]["seed"])):
        job = result["job"]
        output = (root / job["folder"]).resolve()
        if not output.is_relative_to(root):
            raise ValueError("An adaptive job folder escapes the repository.")
        manifest_path = output / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = json.loads((output / IDENTITY_NAME).read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            raise ValueError(f"Run/manifest identities differ: {output}")
        hashes = manifest.get("artifact_sha256", {})
        if set(hashes) != set(ARTIFACT_NAMES):
            raise ValueError(f"Incomplete adaptive artifact manifest: {output}")
        for name, digest in hashes.items():
            path = output / name
            if not path.is_file() or _sha(path) != digest:
                raise ValueError(f"Adaptive artifact integrity failed: {path}")
        checkpoint_path = output / CHECKPOINT_NAME
        if _sha(checkpoint_path) != manifest.get("checkpoint_sha256"):
            raise ValueError(f"Checkpoint differs from its published manifest: {output}")
        stage_history = json.loads((output / "stage_summary.json").read_text(encoding="utf-8"))
        if stage_history != result.get("stage_history"):
            raise ValueError(f"Job/study stage histories differ: {output}")
        if stage_history.get("last_manifest_sha256") != _sha(manifest_path):
            raise ValueError(f"Stage history does not identify the final manifest: {output}")
        state = load_optimizer_checkpoint(checkpoint_path, identity)
        if state.evaluations != manifest.get("search_evaluations") or state.evaluations != job["budgets"][-1]:
            raise ValueError(f"Checkpoint does not match the final planned study count: {output}")
        report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if report.get("checkpoint_identity") != identity or report.get("search_evaluations") != state.evaluations:
            raise ValueError(f"Adaptive summary does not match the checkpoint: {output}")
        policy = StoppingPolicy(**report["stopping_policy"])
        stopped = stage_history.get("online_stop_evaluations")
        if stopped is not None and (
            isinstance(stopped, bool) or not isinstance(stopped, int)
            or stopped % policy.check_interval or not 0 < stopped <= state.evaluations
        ):
            raise ValueError(f"Invalid recorded online stop: {output}")
        counts = sorted({state.settings.population_size, state.evaluations,
                         *range(policy.check_interval, state.evaluations + 1, policy.check_interval)})
        for count in counts:
            metrics = checkpoint_metrics(state.evaluated, count, reference=policy.hypervolume_reference)
            selected = metrics.pop("representative_ids") or {}
            reference = metrics.pop("reference")
            rows.append({"reservoir": job["reservoir"], "seed": job["seed"],
                         "phase": "audit_extension" if stopped is not None and count > stopped else "online",
                         "online_stop_evaluations": stopped,
                         "reference_release_cost": reference[0], "reference_storage_cost": reference[1],
                         **metrics, **{f"{role}_evaluation_id": value for role, value in selected.items()}})
        source_checks.append({"reservoir": job["reservoir"], "seed": job["seed"],
                              "manifest_sha256": _sha(manifest_path),
                              "checkpoint_sha256": manifest["checkpoint_sha256"],
                              "artifact_count_verified": len(hashes)})
    return pd.DataFrame(rows), source_checks


def plot_prefixes(table, folder):
    """Draw separate reservoir rows with calibration-only objective indicators."""
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.88, hspace=0.48, wspace=0.3)
    any_stop = table.online_stop_evaluations.notna().any()
    any_audit = table.phase.eq("audit_extension").any()
    for row_index, reservoir in enumerate(("Trinity", "Gardiner")):
        reservoir_rows = table.loc[table.reservoir.eq(reservoir)]
        for panel_index, (column, title, ylabel) in enumerate(PANELS):
            axis = axes[row_index, panel_index]
            for seed_index, seed in enumerate(sorted(reservoir_rows.seed.unique())):
                points = reservoir_rows.loc[reservoir_rows.seed.eq(seed)].sort_values("evaluations")
                stop_values = points.online_stop_evaluations.dropna()
                stopped = None if stop_values.empty else int(stop_values.iloc[0])
                online = points if stopped is None else points.loc[points.evaluations.le(stopped)]
                color = COLORS[int(seed)]
                axis.plot(online.evaluations, online[column], color=color, linewidth=1.7,
                          marker="o", markersize=3.2, zorder=3)
                if stopped is not None:
                    audit = points.loc[points.evaluations.ge(stopped)]
                    if len(audit) > 1:
                        axis.plot(audit.evaluations, audit[column], color=color, linewidth=1.7,
                                  linestyle="--", marker="o", markersize=3.2, zorder=3)
                    stop_row = points.loc[points.evaluations.eq(stopped)]
                    if len(stop_row) != 1:
                        raise ValueError(f"Missing scheduled stop prefix for {reservoir} seed {seed}.")
                    value = float(stop_row[column].iloc[0])
                    axis.plot(stopped, value, marker="D", markersize=6.2, linestyle="none",
                              color=color, markeredgecolor="white", markeredgewidth=0.7, zorder=5)
                    if panel_index == 0:
                        axis.annotate(f"Seed {int(seed)} stop: {stopped:,}", xy=(stopped, value),
                                      xytext=(-8, -18 - seed_index * 16), textcoords="offset points",
                                      ha="right", fontsize=8, color=color,
                                      arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.7})
            letter = chr(ord("a") + row_index * 3 + panel_index)
            axis.set_title(f"({letter}) {reservoir}: {title}", loc="left", pad=9)
            axis.set_xlabel("Cumulative search evaluations per seed")
            axis.set_ylabel(ylabel)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.grid(color="#d9dee4", linewidth=0.6, alpha=0.8)
            axis.margins(x=0.035, y=0.13)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
    handles = [Line2D([0], [0], color=COLORS[seed], linewidth=1.8, label=f"Seed {seed}")
               for seed in (42, 43, 44)]
    handles.append(Line2D([0], [0], color="#374151", linewidth=1.8, label="Online stages"))
    if any_audit:
        handles.append(Line2D([0], [0], color="#374151", linewidth=1.8, linestyle="--", label="Explicit audit extension"))
    if any_stop:
        handles.append(Line2D([0], [0], color="#374151", marker="D", linestyle="none", markersize=6, label="Provisional online stop"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.065),
               ncol=3, frameon=False, fontsize=9)
    title = ("Checkpointed reservoir calibration: progress and stopping audits" if any_stop
             else "Checkpointed reservoir calibration: progress across cumulative budgets")
    fig.suptitle(title,
                 fontsize=14, fontweight="semibold", y=0.975)
    stop_note = "Diamonds mark provisional stops; dashed segments are explicit audit extensions." if any_stop else "No provisional online stops occurred within the displayed budgets."
    reference_pairs = table[["reference_release_cost", "reference_storage_cost"]].drop_duplicates().to_numpy()
    reference_note = "; ".join(f"[{x:g}, {y:g}]" for x, y in reference_pairs)
    fig.text(0.5, 0.023,
             f"Calibration only; higher values are better. Fixed hypervolume reference in negative-NSE space: {reference_note}.\n"
             f"{stop_note}\n"
             "Reservoir panels have separate axes and qualified inputs; these two cases do not validate a fleet budget or establish global convergence.",
             ha="center", fontsize=9)
    output = folder / "convergence.png"
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.repo.resolve()
    folder = (root / "outputs/checkpoint_reservoir_study" if args.output_dir is None else args.output_dir).resolve()
    if not folder.is_relative_to(root):
        parser.error("Study output must remain within --repo.")
    try:
        table, verified = read_verified_prefixes(root, folder)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    table.to_csv(folder / "prefix_metrics.csv", index=False)
    output = plot_prefixes(table, folder)
    print(f"Verified {len(verified)} checkpoints and {sum(row['artifact_count_verified'] for row in verified)} published artifacts.")
    print(folder / "prefix_metrics.csv")
    print(output)


if __name__ == "__main__":
    main()
