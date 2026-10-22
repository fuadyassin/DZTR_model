"""Hand prepared observed-monthly profiles to the unchanged native v2 optimizer.

The adapter freezes the supplied calendar, year roles and initial targets. It
does not invoke a year selector, change physical constants, or import checkpoints
from another protocol. Native preparations/releases remain independently usable.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd

from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.optimizer import NSGA2Settings
from dztr.calibration.sampled import _observations
from dztr.evaluation.metrics import objective_metric
from dztr.schemas import ReservoirParameters
from dztr.simulation.sampled import SamplingSettings
from strategies.representative_years import (YearSelection, RepresentativeYearObjective, fit_selected_years)
from workflows import data_availability_v1 as availability
from workflows import reservoir_v2 as native
from workflows.geometry_v1 import digest

ROOT = native.ROOT
SOURCES = ("workflows/prepared_calibration_v1.py", "tools/calibrate_prepared_v1.py")


def sources():
    return {**availability.sources(), **{p: native.sha(ROOT / p) for p in SOURCES}}


def config(path):
    path = Path(path).resolve(strict=True)
    cfg = native.read(path)
    allowed = {"kind", "schema_version", "campaign", "profiles", "scoring", "optimization"}
    if set(cfg) - allowed or cfg.get("kind") != "dztr_prepared_calibration_v1" or cfg.get("schema_version") != 1:
        raise ValueError("Unknown prepared-calibration configuration")
    if cfg.get("scoring", "pooled") not in {"pooled", "equal_regime"}:
        raise ValueError("Choose pooled or equal_regime scoring")
    profiles = cfg.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("Declare prepared reservoir profiles")
    identifiers = []
    for entry in profiles:
        if set(entry) - {"reservoir_id", "method", "profile_id"} or entry.get("method") != "observed_monthly":
            raise ValueError("This adapter supports explicit observed_monthly profiles only")
        identifiers.append(availability.safe_id(entry["reservoir_id"]))
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Use one prepared profile per reservoir in a calibration campaign")
    settings = dict(seeds=[42, 43, 44], population=32, evaluation_budget=512, workers=3)
    if set(cfg.get("optimization", {})) - set(settings):
        raise ValueError("Unknown optimization setting")
    settings.update(cfg.get("optimization", {}))
    if not isinstance(settings["seeds"], list) or not settings["seeds"] or len(set(settings["seeds"])) != len(settings["seeds"]):
        raise ValueError("Declare unique search seeds")
    for seed in settings["seeds"]:
        NSGA2Settings(seed=seed, population_size=settings["population"], evaluation_budget=settings["evaluation_budget"])
    if settings["evaluation_budget"] % settings["population"]:
        raise ValueError("Evaluation budget must contain complete populations")
    if isinstance(settings["workers"], bool) or not isinstance(settings["workers"], int) or settings["workers"] < 1:
        raise ValueError("Workers must be a positive integer")
    cfg.update(scoring=cfg.get("scoring", "pooled"), optimization=settings)
    cfg["campaign"] = str((path.parent / cfg["campaign"]).resolve(strict=True))
    return path, cfg


def prepare(config_path, output):
    path, cfg = config(config_path)
    upstream, profiles = availability.verify(cfg["campaign"])
    pending, inputs = [], {str(path): native.sha(path)}

    def pin(source):
        source = Path(source).resolve(strict=True)
        inputs[str(source)] = native.sha(source)
        return source

    for name in ("protocol.json", "profile_registry.json", "config.json"):
        pin(upstream / name)
    for request in cfg["profiles"]:
        candidates = [p for p in profiles if p["reservoir_id"] == request["reservoir_id"]
                      and p["method"] == request["method"] and ("profile_id" not in request or p["profile_id"] == request["profile_id"])]
        if len(candidates) != 1:
            raise ValueError("Prepared profile is unavailable or ambiguous")
        record = candidates[0]
        if not record["parameters_ready"] or not record["simulation_ready"] or record["reasons"]:
            raise ValueError("Prepared profile has unresolved source/physical/forcing decisions")
        if record["observation_basis"] != "native_archive" or record["identity"]["quantile_method"] != "hazen":
            raise ValueError("Native observed-storage timing and Hazen percentiles are required")
        prepared = upstream / record["prepared_folder"]
        common = prepared.parent
        metadata = native.read(pin(common / "numerical_profile.json"))
        if metadata.get("route") != "standard_native_v1":
            raise ValueError("Signed or interval routes need their own calibration adapter")
        frame = availability.read_frame(pin(common / "observations_si.csv"))
        # The native optimizer expects already prepared SI input; no preprocessing is repeated.
        obs = native.observations(common / "observations_si.csv")
        selection = YearSelection(**native.read(pin(common / "selection.json")))
        selection.mask(frame.index, "calibration")
        if selection.signature != record["identity"]["selection_signature"]:
            raise ValueError("Prepared target table and frozen year selection disagree")
        physical = native.read(pin(prepared / "model_constants.json"))
        if physical != record["identity"]["physical_constants"] or physical != metadata["physical_constants"]:
            raise ValueError("Prepared physical constants disagree; no automatic capacity/channel repair is allowed")
        targets = pd.read_csv(pin(prepared / "monthly_targets_si.csv"), index_col="month", float_precision="round_trip")
        template = ReservoirParameters(targets, **physical)
        _, bounds = fit_selected_years(obs, selection, physical, allow_feasible_initialization=False)
        bounds.validate(parameters_to_vector(template), template)
        # Reproduce the declared source recipe on calibration rows only.
        np.testing.assert_array_equal(targets.to_numpy(), bounds.generalized_targets.to_numpy())
        model = {**metadata, "cost_mode": "pooled_nse" if cfg["scoring"] == "pooled" else "regime_balanced_nse",
                 "quantile_method": "hazen", "source_preprocessing_already_applied": True}
        RepresentativeYearObjective(obs, selection, template, sampling=SamplingSettings(**metadata["sampling"]),
            bounds=bounds, policy=metadata["root_policy"], atol_storage_m3=metadata["atol_storage_m3"], cost_mode=model["cost_mode"])
        baseline = upstream / "jobs" / record["reservoir_id"] / record["profile_id"]
        baseline_files = []
        if (baseline / "receipt.json").exists():
            receipt = native.read(pin(baseline / "receipt.json"))
            if receipt["profile_identity"] != digest(record):
                raise ValueError("Baseline receipt belongs to another prepared profile")
            for name, expected in receipt["artifact_sha256"].items():
                if native.sha(pin(baseline / name)) != expected:
                    raise ValueError("Prepared baseline artifact changed")
            baseline_files = [baseline / "receipt.json"]
            if receipt["status"] == "completed":
                baseline_files += [baseline / "simulation.csv", baseline / "metrics.json"]
        pending.append((record, common, prepared, selection, model, bounds, baseline_files))
    grand_ids = [r[0].get("GRAND_ID") for r in pending if r[0].get("GRAND_ID") is not None]
    if len(grand_ids) != len(set(grand_ids)):
        raise ValueError("Duplicate verified GRAND_ID values would make the exported registry ambiguous")
    out = native.fresh_output(output)
    native_cfg = {"schema_version": 2, "year_selection": "frozen_prepared", "scoring": cfg["scoring"],
                  "optimization": cfg["optimization"], "reservoirs": []}
    handoffs = []
    for record, common, prepared, selection, model, bounds, baseline_files in pending:
        identifier = record["reservoir_id"]
        folder = out / "prepared" / identifier
        folder.mkdir(parents=True)
        for source, name in ((common / "observations_si.csv", "observations_si.csv"),
                             (common / "selection.json", "selection.json"),
                             (prepared / "monthly_targets_si.csv", "initial_targets_si.csv"),
                             (prepared / "profile.json", "source_profile.json"),
                             (common / "preprocessing.json", "preprocessing.json")):
            shutil.copyfile(pin(source), folder / name)
        native.write(folder / "model_config.json", model)
        bounds.empirical.monthly_bounds.to_csv(folder / "bounds_si.csv", index_label="month", float_format="%.17g")
        bounds.generalized_targets.to_csv(folder / "raw_generalized_targets_si.csv", index_label="month", float_format="%.17g")
        selection.table().to_csv(folder / "row_selection.csv", index_label="date")
        native.write(folder / "initialization.json", {"kind": "explicit_calibration_initializer", "basis": "calibration_only",
            "source_profile_id": record["profile_id"], "source_targets_sha256": native.sha(prepared / "monthly_targets_si.csv"),
            "selection_sha256": selection.signature, "capacity_or_target_repairs": False})
        native.write(folder / "selection_metadata.json", {"kind": "frozen_prepared_selection", "selection_recomputed": False,
            "calibration_fraction": selection.roles.count("calibration") / len(selection.roles),
            "selection_sha256": selection.signature, "source_review_required": False})
        if baseline_files:
            cache = folder / "baseline"; cache.mkdir()
            for source in baseline_files:
                shutil.copyfile(source, cache / source.name)
        native_cfg["reservoirs"].append({"reservoir_id": identifier, "grand_id": record.get("GRAND_ID")})
        handoffs.append({"reservoir_id": identifier, "GRAND_ID": record.get("GRAND_ID"), "method": record["method"],
            "source_profile_id": record["profile_id"], "selection_sha256": selection.signature,
            "source_targets_sha256": native.sha(prepared / "monthly_targets_si.csv"),
            "calibration_fraction": selection.roles.count("calibration") / len(selection.roles),
            "calibration_ready": True, "source_certified": False, "original_registry_modified": False})
    native.write(out / "config.json", native_cfg)
    native.write(out / "handoff_config.json", cfg)
    native.write(out / "preparation.json", {"kind": "prepared_profile_calibration_v1", "reservoirs": handoffs,
                 "selection_recomputed": False, "new_evaluations": 0})
    for source, expected in inputs.items():
        if native.sha(source) != expected:
            raise ValueError("Upstream preparation changed during handoff")
    native.write(out / "protocol.json", {"kind": "unified_reservoir_workflow_v2", "adapter": "prepared_profile_calibration_v1",
        "code_sha256": sources(), "input_sha256": inputs, "config_sha256": native.sha(out / "config.json"),
        "adapter_sha256": {name: native.sha(out / name) for name in ("handoff_config.json", "preparation.json")},
        "prepared_sha256": {p.relative_to(out).as_posix(): native.sha(p) for p in (out / "prepared").rglob("*") if p.is_file()}})
    return out


def verify(output):
    out, protocol, cfg = native.check_protocol(output)
    if protocol.get("adapter") != "prepared_profile_calibration_v1":
        raise ValueError("Not a prepared-profile calibration campaign")
    for name in ("handoff_config.json", "preparation.json"):
        # These are additionally pinned in the adapter protocol to prevent dispatch changes.
        if native.sha(out / name) != protocol["adapter_sha256"][name]:
            raise ValueError("Prepared adapter decisions changed")
    return out, cfg


def _metrics(source, simulation, selection, phase):
    dates = source.index[selection.mask(source.index, phase)]
    if not dates.isin(simulation.index).all():
        raise ValueError("Baseline simulation does not cover fixed scored support")
    result = {}
    labels = selection.table().loc[dates, "regime"]
    for variable, column in (("storage", "storage_m3"), ("release", "release_m3s")):
        if not np.isfinite(simulation.loc[dates, column]).all() or simulation.loc[dates, column].lt(0).any():
            raise ValueError("Invalid simulated value on baseline scored support")
        result[variable + "_nse"] = objective_metric(source.loc[dates, column], simulation.loc[dates, column])
        values = []
        for regime in sorted(set(labels)):
            days = dates[labels == regime]
            value = objective_metric(source.loc[days, column], simulation.loc[days, column])
            result[regime + "_" + variable + "_nse"] = value
            values.append(value)
        result["regime_balanced_" + variable + "_nse"] = float(np.mean(values))
    return result


def finish(output, budget=None):
    out, cfg = verify(output)
    budget = cfg["optimization"]["evaluation_budget"] if budget is None else budget
    final = out / "published" / f"budget_{budget}"
    if final.exists():
        native.verify_registry(final / "registry.json")
        return final / "registry.json"
    registry = native.finish(out, budget)
    native.verify_registry(registry)
    build = final.parent / f".building_budget_{budget}_{uuid.uuid4().hex}"
    build.parent.mkdir(exist_ok=True)
    shutil.copytree(registry.parent, build)
    table = pd.read_csv(build / "reservoir_parameterization_by_GRAND_ID.csv", float_precision="round_trip")
    comparisons, baseline_receipts = [], []
    registry_data = native.read(build / "registry.json")
    selected = native.read(build / "selection.json")["records"]
    seed_scores = pd.read_csv(build / "seed_scores.csv", float_precision="round_trip")
    for entry in cfg["reservoirs"]:
        identifier = entry["reservoir_id"]
        folder = out / "prepared" / identifier
        source_profile = native.read(folder / "source_profile.json")
        source = availability.read_frame(folder / "observations_si.csv")
        selection = YearSelection(**native.read(folder / "selection.json"))
        baseline_cache = folder / "baseline"
        receipt_path = baseline_cache / "receipt.json"
        baseline_metrics = {}
        receipt = native.read(receipt_path) if receipt_path.exists() else {"status": "not_run", "reason": "No completed fixed baseline was supplied"}
        if receipt["status"] == "completed":
            simulation = availability.read_frame(baseline_cache / "simulation.csv")
            model = native.read(folder / "model_config.json")
            residual = native.independent_balance(simulation, model["physical_constants"]["inflow_correction"])
            if residual > model["atol_storage_m3"]:
                raise ValueError("Cached baseline fails independent mass balance")
            baseline_metrics = {p: _metrics(source, simulation, selection, p) for p in ("calibration", "assessment")}
        baseline_receipts.append({"reservoir_id": identifier, "status": receipt["status"], "reason": receipt.get("reason"),
                                  "simulation_reused": receipt["status"] == "completed"})
        profile_path = build / registry_data["profiles_by_reservoir_id"][identifier]
        profile = native.read(profile_path)
        provenance = {"method": source_profile["method"], "source_profile_id": source_profile["profile_id"],
                      "selection_sha256": selection.signature, "source_targets_sha256": native.sha(folder / "initial_targets_si.csv"),
                      "selection_recomputed": False, "source_certified": False}
        profile.update(parameterization=provenance, source_kind="prepared_profile_calibration_v1",
                       inflow_preprocessing=native.read(folder / "preprocessing.json"))
        native.write(profile_path, profile)
        mask = table.reservoir_id.eq(identifier)
        for key, value in {**provenance, **profile["physical_constants"], "root_policy": profile["root_policy"],
                           "calibration_fraction": selection.roles.count("calibration") / len(selection.roles)}.items():
            if key not in table:
                table[key] = pd.Series([None] * len(table), dtype=object)
            table.loc[mask, key] = value
        chosen = next(r for r in selected if r["reservoir_id"] == identifier)
        rows = [("selected", chosen["seed"], chosen["evaluation_id"], profile["calibration"], profile["assessment"]["metrics"], profile["assessment"]["feasible"])]
        for row in seed_scores.loc[seed_scores.reservoir_id.eq(identifier)].to_dict("records"):
            rows.append(("seed", row["seed"], row.get("evaluation_id"),
                {k.removeprefix("calibration_"): v for k, v in row.items() if k.startswith("calibration_")},
                {k.removeprefix("assessment_"): v for k, v in row.items() if k.startswith("assessment_")}, row["assessment_feasible"]))
        for role, seed, evaluation, calibration, assessment, feasible in rows:
            for phase, metrics in (("calibration", calibration), ("assessment", assessment)):
                baseline = baseline_metrics.get(phase, {})
                item = {"reservoir_id": identifier, "GRAND_ID": entry.get("grand_id"), "role": role, "seed": seed,
                        "evaluation_id": evaluation, "phase": phase, "baseline_status": receipt["status"],
                        "assessment_feasible": bool(feasible), "selection_uses_assessment": False}
                for metric in ("storage_nse", "release_nse", "regime_balanced_storage_nse", "regime_balanced_release_nse"):
                    item["baseline_" + metric] = baseline.get(metric)
                    item["calibrated_" + metric] = metrics.get(metric)
                    item["improvement_" + metric] = (metrics[metric] - baseline[metric]) if metric in metrics and metric in baseline else None
                comparisons.append(item)
    table["GRAND_ID"] = table["GRAND_ID"].astype("Int64")
    table.to_csv(build / "reservoir_parameterization_by_GRAND_ID.csv", index=False, float_format="%.17g")
    pd.DataFrame(comparisons).to_csv(build / "baseline_comparison.csv", index=False, float_format="%.17g")
    native.write(build / "baseline_reuse.json", {"reservoirs": baseline_receipts, "new_baseline_simulations": 0})
    registry_data["provenance"].update(adapter="prepared_profile_calibration_v1", handoff_protocol_sha256=native.sha(out / "protocol.json"))
    native.write(build / "registry.json", registry_data)
    (build / "README.md").write_text(
        "# Prepared-profile calibration release\n\n"
        "The original prepared targets, physical constants, calendar and year mask were retained. "
        "Empirical search bounds use calibration rows only. Candidate and seed selection uses the pooled nondominated calibration fronts; held-out assessment does not select parameters.\n\n"
        "[Baseline and calibrated comparison](baseline_comparison.csv), [parameter table by GRanD ID](reservoir_parameterization_by_GRAND_ID.csv), "
        "[registry](registry.json), [seed scores](seed_scores.csv), [baseline reuse](baseline_reuse.json).\n\n"
        "Profiles remain exploratory. The evaluation budget is a checkpoint, not evidence of convergence.\n", encoding="utf-8")
    native.write(build / "manifest.json", {"code_sha256": sources(), "artifact_sha256": {
        p.relative_to(build).as_posix(): native.sha(p) for p in build.rglob("*") if p.is_file() and p.name != "manifest.json"}})
    native.verify_registry(build / "registry.json")
    build.replace(final)
    return final / "registry.json"


def run(config_path, output, budget=None):
    path, cfg = config(config_path)
    out = Path(output).resolve()
    if out.exists():
        verify(out)
        if native.read(out / "handoff_config.json") != cfg:
            raise ValueError("Prepared calibration decisions changed")
    else:
        out = prepare(path, out)
    _, native_cfg = verify(out)
    budget = cfg["optimization"]["evaluation_budget"] if budget is None else budget
    for seed in cfg["optimization"]["seeds"]:
        NSGA2Settings(seed=seed, population_size=cfg["optimization"]["population"], evaluation_budget=budget)
    if budget % cfg["optimization"]["population"]:
        raise ValueError("Continuation budget must contain complete populations")
    lock = out / "run.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    jobs = [(e["reservoir_id"], seed) for e in native_cfg["reservoirs"] for seed in cfg["optimization"]["seeds"]]
    def execute(job):
        identifier, seed = job
        folder = out / "jobs" / identifier / f"seed_{seed}"; folder.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, "-B", str(ROOT / "tools/calibrate_prepared_v1.py"), "job", "--output", str(out),
                "--reservoir-id", identifier, "--seed", str(seed), "--evaluation-budget", str(budget)]
        with (folder / "execution.log").open("a", encoding="utf-8") as stream:
            result = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return {"reservoir_id": identifier, "seed": seed, "exit_code": result.returncode}
    try:
        native.write(out / "status.json", {"stage": "calibrating", "budget": budget, "jobs": len(jobs)})
        with ThreadPoolExecutor(max_workers=cfg["optimization"]["workers"]) as pool:
            results = [f.result() for f in as_completed([pool.submit(execute, j) for j in jobs])]
        native.write(out / f"execution_budget_{budget}.json", {"jobs": results})
        if any(r["exit_code"] for r in results):
            raise ValueError("Prepared calibration job failed; inspect persistent execution logs")
        native.write(out / "status.json", {"stage": "assessing_exporting", "budget": budget})
        registry = finish(out, budget)
        native.write(out / "status.json", {"stage": "completed", "budget": budget, "registry": str(registry)})
        return registry
    except Exception as error:
        native.write(out / "status.json", {"stage": "failed", "budget": budget, "reason": str(error)})
        raise
    finally:
        lock.unlink(missing_ok=True)
