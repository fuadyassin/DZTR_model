"""Common reservoir reports from saved vectors, metrics, and trajectories only."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archived_reports(out):
    catalog = ROOT / "outputs/multi_reservoir_archive_20261010"
    proof = read(catalog / "workflow_verification.json")
    if proof["status"] != "passed":
        raise ValueError("Archived catalog must have passed verification.")
    for name, digest in proof["artifact_sha256"].items():
        if sha(catalog / name) != digest:
            raise ValueError("Archived report input changed.")
    inputs = {r["reservoir"]["id"]: r for r in read(catalog / "input_catalog.json")["reservoirs"]}
    groups = {}
    for item in read(catalog / "result_index.json")["results"]:
        file = catalog / item["result_file"]
        groups.setdefault(item["reservoir_id"], []).append((file.parent, read(file)))
    emit(groups, inputs, out, "archived", proof)


def completed_reports(fleet, out):
    groups, inputs = {}, {}
    for receipt in sorted((fleet / "pilots").glob("*/seed_*/fleet_completion.json")):
        folder = receipt.parent
        completion = read(receipt)
        if completion["status"] != "passed":
            continue
        for name, digest in completion["artifact_sha256"].items():
            if sha(folder / name) != digest:
                raise ValueError(f"Completed fleet input changed: {folder / name}")
        original = read(folder / "result.json")
        config = read(folder / "resolved_config.json")
        identifier = original["reservoir_id"]
        inputs[identifier] = {"observations_file": str((ROOT / "configs/reservoirs" / config["observations_file"]).resolve()), "sampling": config["sampling"], "physical_constants": config["model"]}
        result = {"reservoir": config["reservoir"], "seed": original["seed"], "optimizer": original["settings"],
                  "periods": config["periods"], "physical_constants": config["model"], "archive_directory": str(folder),
                  "selected_parameters": {r["role"]: {**r, "historical_validation": {"release_nse": r.get("validation_release_nse"), "storage_nse": r.get("validation_storage_nse")}} for r in original["representatives"]}}
        groups.setdefault(identifier, []).append((folder, result))
    emit(groups, inputs, out, "fleet", {"source": str(fleet), "jobs_with_results": sum(map(len, groups.values()))})


def emit(groups, inputs, out, source_kind, evidence):
    if out.exists() or not out.resolve().is_relative_to(ROOT):
        raise ValueError("Choose a fresh report output inside the repository.")
    out.mkdir(parents=True)
    consumed, summaries = {}, []
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for identifier, items in groups.items():
        items = sorted(items, key=lambda item: item[1]["seed"])
        folder = out / identifier
        folder.mkdir()
        first, result = items[0]
        name, seed = result["reservoir"]["name"], result["seed"]
        source = Path(result["archive_directory"])
        if not source.is_absolute():
            source = ROOT / source
        interval = identifier == "ex35_Bull_Lake"
        if interval:
            trajectory = ROOT / f"outputs/bull_lake_interval_population_20261009/scientific_checks/population_32_seed_{seed}.csv"
            parent = trajectory.parents[1]
            trajectory_proof = read(parent / "scientific_checks.json")
            if sha(trajectory) != trajectory_proof["artifact_sha256"][trajectory.name]:
                raise ValueError("Saved interval trajectory changed.")
        else:
            index = result["selected_parameters"]["balanced"]["evaluation_id"]
            trajectory = source / f"selected_{index}_simulation.csv"
        available = trajectory.exists()
        frame = pd.read_csv(trajectory, index_col="date", parse_dates=True, float_precision="round_trip") if available else None
        saved_residual = None
        if available:
            consumed[str(trajectory.relative_to(ROOT))] = sha(trajectory)
            saved_residual = float(frame.balance_residual_m3.abs().max())
            if interval:
                obs_path = Path(inputs[identifier]["observations_file"])
                if not obs_path.is_absolute():
                    obs_path = ROOT / obs_path
                obs = pd.read_csv(obs_path, index_col="date", parse_dates=True, float_precision="round_trip").reindex(frame.index)
                consumed[str(obs_path.relative_to(ROOT))] = sha(obs_path)
                model_release = frame.interval_mean_total_outflow_m3s
                observed_storage, observed_release = obs.storage_m3, obs.release_m3s
            else:
                model_release = frame.release_m3s
                observed_storage, observed_release = frame.observed_storage_m3, frame.observed_release_m3s
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout="constrained")
            for ax, observed, simulated, ylabel in ((axes[0], observed_storage / 1e6, frame.storage_m3 / 1e6, "Storage (million m³)"),
                                                    (axes[1], observed_release, model_release, "Release (m³/s)")):
                ax.plot(observed.index, observed.values, color=".55", lw=.65, label="Observed")
                ax.plot(simulated.index, simulated.values, color="#0072B2", lw=.65, label="Saved simulation")
                period = result["periods"]["validation"]
                if frame.index.max() >= pd.Timestamp(period["start"]):
                    ax.axvspan(pd.Timestamp(period["start"]), frame.index.max(), color="#E69F00", alpha=.12, label="Held-out period")
                ax.set_ylabel(ylabel); ax.grid(alpha=.2)
            axes[0].legend(loc="upper right", fontsize=8)
            fig.suptitle(f"{name}: archived balanced role, seed {seed} · no new simulation")
            for ext in ("png", "pdf"):
                fig.savefig(folder / f"trajectories.{ext}", dpi=160, bbox_inches="tight")
            plt.close(fig)
        vector_source = source / ("evaluated_vectors.csv" if interval else "vectors.csv")
        vectors = pd.read_csv(vector_source, index_col="evaluation_id", float_precision="round_trip")
        consumed[str(vector_source.relative_to(ROOT))] = sha(vector_source)
        vector = vectors.loc[result["selected_parameters"]["balanced"]["evaluation_id"]].values
        targets = vector.reshape(6, 12).T.copy()
        targets[:, :3] *= result["physical_constants"]["capacity_m3"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
        for j, label in enumerate(("Critical", "Normal", "Flood")):
            axes[0].plot(range(1, 13), targets[:, j] / 1e6, marker="o", label=label)
            axes[1].plot(range(1, 13), targets[:, j + 3], marker="o", label=label)
        for ax in axes:
            ax.set_xlabel("Operating month"); ax.set_xticks(range(1, 13)); ax.grid(alpha=.2); ax.legend(fontsize=8)
        axes[0].set_ylabel("Storage target (million m³)"); axes[1].set_ylabel("Release target (m³/s)")
        fig.suptitle(f"{name}: saved monthly targets · seed {seed}")
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"monthly_targets.{ext}", dpi=160, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.5, 4.5), layout="constrained")
        metric_rows = []
        for package, item in items:
            front_path = package / "pareto_front.csv"
            front = pd.read_csv(front_path, float_precision="round_trip")
            consumed[str(front_path.relative_to(ROOT))] = sha(front_path)
            ax.scatter(front.release_nse, front.storage_nse, s=22, label=f"Seed {item['seed']}")
            balanced = item["selected_parameters"]["balanced"]
            val = balanced.get("historical_validation", {})
            metric_rows.append({"seed": item["seed"], "population_size": item["optimizer"]["population_size"],
                                "evaluation_budget": item["optimizer"]["evaluation_budget"],
                                "calibration_release_nse": balanced["calibration_release_nse"], "calibration_storage_nse": balanced["calibration_storage_nse"],
                                "validation_release_nse": val.get("release_nse"), "validation_storage_nse": val.get("storage_nse"),
                                "validation_source": "saved_reporting" if val else "not_available"})
        ax.set(xlabel="Calibration release NSE", ylabel="Calibration storage NSE", title=f"{name}: complete saved seed fronts")
        ax.legend(); ax.grid(alpha=.2)
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"pareto_fronts.{ext}", dpi=160, bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame(metric_rows).to_csv(folder / "metrics.csv", index=False, float_format="%.17g")
        gaps = []
        if not available:
            gaps.append("Saved balanced trajectory unavailable; no rerun attempted.")
        if interval:
            gaps.append("Saved trajectories cover warmup/calibration only; no held-out trajectory or score for these population32 runs.")
            gaps.append("Computed-net inflow and observed block anchors make this a segmented conditional hindcast.")
        negative_seeds = [r["seed"] for r in metric_rows if r["validation_storage_nse"] is not None and r["validation_storage_nse"] < 0]
        if negative_seeds:
            gaps.append(f"Balanced held-out storage NSE is negative for seeds {negative_seeds}; calibration improvement did not transfer for these runs.")
        sampling = inputs[identifier].get("sampling", {})
        if sampling.get("frequency") == "monthly":
            gaps.append(f"Monthly integration uses the inherited timestep of {sampling.get('timestep_days')} days; plotted dates are observation labels.")
        if identifier == "ex18_High_aswan":
            gaps.append("The inherited 50,000 m3/s channel-capacity setting remains an unverified source assumption.")
        gaps.append("Physical source definitions remain provisional; numerical balance does not establish observed closure. Convergence remains unconfirmed.")
        text = f"# {name}\n\nSeed {seed} is the fixed illustrative trajectory, selected by seed order rather than validation scores. All available seed metrics/fronts are retained.\n\n"
        text += f"Saved maximum numerical balance residual: {saved_residual} m³. No model calls were made to generate this report.\n\n"
        text += "\n".join(f"- {g}" for g in gaps) + "\n\n"
        if available:
            text += "![Saved trajectories](trajectories.png)\n\n"
        text += "![Monthly targets](monthly_targets.png)\n\n![Calibration fronts](pareto_fronts.png)\n\n[All seed scores](metrics.csv). PDFs are provided alongside PNGs.\n"
        (folder / "README.md").write_text(text, encoding="utf-8")
        summaries.append({"reservoir_id": identifier, "name": name, "seeds": [i[1]["seed"] for i in items],
                          "trajectory_seed": seed, "trajectory_available": available, "max_saved_balance_residual_m3": saved_residual, "gaps": gaps})
    manifest = {"status": "generated", "source_kind": source_kind, "new_model_evaluations": 0, "validation_calls": 0,
                "reservoirs": summaries, "consumed_files_sha256": consumed, "source_evidence": evidence,
                "artifact_sha256": {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    (out / "report_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Generated {len(groups)} reservoir reports from saved artifacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-fleet", type=Path)
    args = parser.parse_args()
    if args.completed_fleet:
        completed_reports(args.completed_fleet.resolve(), args.output.resolve())
    else:
        archived_reports(args.output.resolve())
