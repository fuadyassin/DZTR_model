"""Checkpointed calibration of explicit geometry proxies; old releases stay frozen."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
import pandas as pd

from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.checkpoint import load_optimizer_checkpoint, save_optimizer_checkpoint
from dztr.calibration.optimizer import NSGA2Settings, optimize_nsga2
from dztr.calibration.sampled import _observations
from dztr.schemas import ReservoirParameters
from dztr.simulation.sampled import SamplingSettings
from strategies.geometry_objective_v1 import GeometryObjective
from strategies.representative_years import YearSelection, fit_selected_years
from strategies.strategy_comparison import pooled_choice
from workflows import data_availability_v2 as availability
from workflows import reservoir_v2 as native
from workflows.geometry_v1 import digest
from workflows.atomic_io_v1 import write

ROOT = native.ROOT


def publish_directory(build, final):
    """Publish a sealed fresh release despite brief Windows directory leases."""
    if final.exists(): raise ValueError("Never overwrite an existing release")
    for attempt in range(10):
        try: return build.replace(final)
        except OSError as exc:
            if getattr(exc, "winerror", None) not in (5, 32) or attempt == 9: raise
            time.sleep(.1)


def sources():
    return {**availability.sources(), **{p: native.sha(ROOT / p) for p in (
        "strategies/geometry_objective_v1.py", "workflows/geometry_calibration_v1.py", "tools/calibrate_geometry_v1.py")}}


def config(path):
    path = Path(path).resolve(strict=True); cfg = native.read(path)
    if set(cfg) != {"kind", "schema_version", "campaign", "profiles", "scoring", "storage_objective", "optimization"} or cfg["kind"] != "dztr_geometry_calibration_v1" or cfg["schema_version"] != 1:
        raise ValueError("Invalid geometry-calibration configuration")
    if cfg["scoring"] not in {"pooled", "equal_regime"} or cfg["storage_objective"] not in {"proxy_nse", "reference_band_nse"}:
        raise ValueError("Declare supported scoring and storage-objective policies")
    ids = []
    for request in cfg["profiles"]:
        if set(request) - {"reservoir_id", "method", "profile_id"} or request.get("method") != "level_geometry_monthly":
            raise ValueError("Geometry calibration requires level_geometry_monthly profiles")
        ids.append(availability.safe_id(request["reservoir_id"]))
    if not ids or len(set(ids)) != len(ids): raise ValueError("Declare one profile per reservoir")
    settings = cfg["optimization"]
    if set(settings) != {"seeds", "population", "evaluation_budget", "workers"}:
        raise ValueError("Declare all optimizer settings explicitly")
    if not isinstance(settings["seeds"], list) or not settings["seeds"] or len(set(settings["seeds"])) != len(settings["seeds"]):
        raise ValueError("Seeds must be unique")
    for seed in settings["seeds"]: NSGA2Settings(seed=seed, population_size=settings["population"], evaluation_budget=settings["evaluation_budget"])
    if settings["evaluation_budget"] % settings["population"] or isinstance(settings["workers"], bool) or not isinstance(settings["workers"], int) or settings["workers"] < 1:
        raise ValueError("Use complete populations and positive worker count")
    cfg["campaign"] = str((path.parent / cfg["campaign"]).resolve(strict=True))
    return path, cfg


def prepare(config_path, output):
    path, cfg = config(config_path)
    upstream, profiles = availability.verify(cfg["campaign"])
    inputs = {str(path): native.sha(path)}
    def pin(p):
        p = Path(p).resolve(strict=True); inputs[str(p)] = native.sha(p); return p
    for name in ("protocol.json", "profile_registry.json", "config.json"): pin(upstream / name)
    pending = []
    for request in cfg["profiles"]:
        found = [p for p in profiles if p["reservoir_id"] == request["reservoir_id"] and p["method"] == request["method"]
                 and ("profile_id" not in request or p["profile_id"] == request["profile_id"])]
        if len(found) != 1: raise ValueError("Geometry profile is unavailable or ambiguous")
        record = found[0]
        if not record["parameters_ready"] or not record["simulation_ready"] or record["reasons"]:
            raise ValueError("Unresolved geometry source/physical/forcing decisions")
        prepared = upstream / record["prepared_folder"]; common = prepared.parent
        metadata = native.read(pin(common / "numerical_profile.json"))
        if metadata.get("route") != "standard_native_v1" or record["identity"]["quantile_method"] != "hazen":
            raise ValueError("This geometry adapter supports standard-native Hazen profiles")
        conversion = native.read(pin(common / "level_conversion.json"))
        if "reference_method" not in conversion or "uncertainty" not in conversion:
            raise ValueError("Use the shared explicit reference and uncertainty interface")
        frame = availability.read_frame(pin(common / "level_observations_si.csv"))
        original = availability.read_frame(pin(common / "observations_si.csv"))
        if not original.index.equals(frame.index): raise ValueError("Proxy and supplied calendars disagree")
        selection = YearSelection(**native.read(pin(common / "selection.json")))
        if selection.signature != record["identity"]["selection_signature"]: raise ValueError("Year-selection identity changed")
        bands = availability.read_frame(pin(common / "storage_band_si.csv"))
        targets = pd.read_csv(pin(prepared / "monthly_targets_si.csv"), index_col="month", float_precision="round_trip")
        physical = native.read(pin(prepared / "model_constants.json"))
        if physical != metadata["physical_constants"] or physical != record["identity"]["physical_constants"]:
            raise ValueError("Physical constants changed during handoff")
        _, bounds = fit_selected_years(_observations(frame), selection, physical, allow_feasible_initialization=False)
        np.testing.assert_array_equal(targets.to_numpy(), bounds.generalized_targets.to_numpy())
        template = ReservoirParameters(targets, **physical); bounds.validate(parameters_to_vector(template), template)
        GeometryObjective(_observations(frame), selection, template, sampling=SamplingSettings(**metadata["sampling"]),
            bounds=bounds, policy=metadata["root_policy"], atol_storage_m3=metadata["atol_storage_m3"],
            cost_mode="pooled_nse" if cfg["scoring"] == "pooled" else "regime_balanced_nse",
            storage_band=bands, storage_objective=cfg["storage_objective"], provenance=conversion)
        baseline = upstream / "jobs" / record["reservoir_id"] / record["profile_id"]
        files = []
        if (baseline / "receipt.json").exists():
            receipt = native.read(pin(baseline / "receipt.json"))
            if receipt["profile_identity"] != digest(record): raise ValueError("Baseline profile identity changed")
            files = [baseline / "receipt.json"]
            for name, expected in receipt["artifact_sha256"].items():
                p = pin(baseline / name)
                if native.sha(p) != expected: raise ValueError("Baseline artifact changed")
                files.append(p)
        pending.append((record, common, prepared, metadata, bounds, files))
    gids = [p[0].get("GRAND_ID") for p in pending if p[0].get("GRAND_ID") is not None]
    if len(gids) != len(set(gids)): raise ValueError("Duplicate verified GRAND IDs")
    out = native.fresh_output(output)
    handoffs = []
    for record, common, prepared, metadata, bounds, baseline_files in pending:
        folder = out / "prepared" / record["reservoir_id"]; folder.mkdir(parents=True)
        for source, name in ((common / "level_observations_si.csv", "observations_si.csv"),
                             (common / "observations_si.csv", "supplied_observations_si.csv"),
                             (common / "selection.json", "selection.json"), (common / "storage_band_si.csv", "storage_band_si.csv"),
                             (common / "level_conversion.json", "level_conversion.json"),
                             (common / "preprocessing.json", "preprocessing.json"),
                             (prepared / "monthly_targets_si.csv", "initial_targets_si.csv"),
                             (prepared / "profile.json", "source_profile.json")):
            shutil.copyfile(pin(source), folder / name)
        write(folder / "model_config.json", metadata)
        bounds.empirical.monthly_bounds.to_csv(folder / "bounds_si.csv", float_format="%.17g")
        if baseline_files:
            (folder / "baseline").mkdir()
            for p in baseline_files: shutil.copyfile(p, folder / "baseline" / p.name)
        handoffs.append({"reservoir_id": record["reservoir_id"], "GRAND_ID": record.get("GRAND_ID"),
                         "method": record["method"], "calibration_ready": True, "source_certified": False,
                         "selection_signature": record["identity"]["selection_signature"], "original_profile_id": record["profile_id"]})
    write(out / "config.json", cfg); write(out / "preparation.json", {"kind": "geometry_calibration_handoff_v1", "reservoirs": handoffs, "new_optimizer_evaluations": 0})
    for p, expected in inputs.items():
        if native.sha(p) != expected: raise ValueError("Source changed during handoff")
    write(out / "protocol.json", {"kind": "geometry_calibration_protocol_v1", "source_sha256": sources(), "input_sha256": inputs,
        "prepared_sha256": {p.relative_to(out).as_posix(): native.sha(p) for p in out.rglob("*") if p.is_file()}})
    return out


def verify(output):
    out = Path(output).resolve(strict=True)
    if not out.is_relative_to(ROOT / "outputs") or out == ROOT / "outputs": raise ValueError("Use project outputs")
    proof = native.read(out / "protocol.json")
    if proof["kind"] != "geometry_calibration_protocol_v1": raise ValueError("Wrong calibration protocol")
    for name, expected in proof["source_sha256"].items():
        if native.sha(ROOT / name) != expected: raise ValueError("Geometry calibration code changed: " + name)
    for name, expected in proof["prepared_sha256"].items():
        p = (out / name).resolve(strict=True)
        if not p.is_relative_to(out) or native.sha(p) != expected: raise ValueError("Geometry preparation changed: " + name)
    return out, native.read(out / "config.json")


def objective_for(output, identifier):
    out, cfg = verify(output); folder = out / "prepared" / identifier
    frame = availability.read_frame(folder / "observations_si.csv")
    selection = YearSelection(**native.read(folder / "selection.json")); model = native.read(folder / "model_config.json")
    template, bounds = fit_selected_years(_observations(frame), selection, model["physical_constants"], allow_feasible_initialization=False)
    targets = pd.read_csv(folder / "initial_targets_si.csv", index_col="month", float_precision="round_trip")
    np.testing.assert_array_equal(template.monthly_targets.to_numpy(), targets.to_numpy())
    objective = GeometryObjective(_observations(frame), selection, template,
        sampling=SamplingSettings(**model["sampling"]), bounds=bounds, policy=model["root_policy"],
        atol_storage_m3=model["atol_storage_m3"], cost_mode="pooled_nse" if cfg["scoring"] == "pooled" else "regime_balanced_nse",
        storage_band=availability.read_frame(folder / "storage_band_si.csv"), storage_objective=cfg["storage_objective"],
        provenance=native.read(folder / "level_conversion.json"))
    return objective, template, bounds, selection, frame, model


def calibrate(output, identifier, seed, budget=None):
    out, cfg = verify(output); settings = cfg["optimization"]
    if identifier not in {p["reservoir_id"] for p in cfg["profiles"]} or seed not in settings["seeds"]:
        raise ValueError("Undeclared geometry search")
    budget = settings["evaluation_budget"] if budget is None else budget
    options = NSGA2Settings(seed=seed, population_size=settings["population"], evaluation_budget=budget)
    if budget % settings["population"]: raise ValueError("Use complete populations")
    folder = out / "jobs" / identifier / f"seed_{seed}"; folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "calibration.lock"
    with lock.open("x") as stream: stream.write(str(os.getpid()))
    try:
        objective, template, bounds, _, _, _ = objective_for(out, identifier)
        identity = {"kind": "geometry_calibration_v1", "protocol_sha256": native.sha(out / "protocol.json"),
            "reservoir_id": identifier, "seed": seed, "population": settings["population"], "objective": objective.identity,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
        state = None
        if (folder / "checkpoint.npz").exists():
            if native.read(folder / "identity.json") != identity: raise ValueError("Checkpoint identity changed")
            state = load_optimizer_checkpoint(folder / "checkpoint.npz", identity)
            if budget < len(state.evaluated): raise ValueError("Budget cannot shrink")
        snapshot = folder / f"budget_{budget}"
        if (snapshot / "manifest.json").exists():
            if state is None or len(state.evaluated) < budget: raise ValueError("Incomplete checkpoint")
            for name, expected in native.read(snapshot / "manifest.json")["artifact_sha256"].items():
                if native.sha(snapshot / name) != expected: raise ValueError("Snapshot changed")
            return snapshot
        write(folder / "identity.json", identity)
        def checkpoint(current):
            save_optimizer_checkpoint(folder / "checkpoint.npz", current, identity)
            write(folder / "progress.json", {"evaluations": len(current.evaluated), "generation": current.generation})
        result = optimize_nsga2(objective, parameters_to_vector(template), options, bounds=bounds, resume_state=state, checkpoint_callback=checkpoint)
        if result.evaluations != budget: raise ValueError("Search stopped before its checkpoint budget")
        snapshot.mkdir(exist_ok=True)
        pd.DataFrame([{"evaluation_id": p.evaluation_id, "generation": p.generation, "feasible": p.feasible,
                       "release_cost": p.objectives[0], "storage_cost": p.objectives[1], "reason": p.reason} for p in result.evaluated]).to_csv(snapshot / "trials.csv", index=False, float_format="%.17g")
        pd.DataFrame([p.vector for p in result.evaluated], index=[p.evaluation_id for p in result.evaluated]).to_csv(snapshot / "vectors.csv", index_label="evaluation_id", float_format="%.17g")
        pd.DataFrame([{"evaluation_id": p.evaluation_id, "release_cost": p.objectives[0], "storage_cost": p.objectives[1]} for p in result.front], columns=["evaluation_id", "release_cost", "storage_cost"]).to_csv(snapshot / "front.csv", index=False, float_format="%.17g")
        pd.DataFrame([asdict(g) for g in result.history]).to_csv(snapshot / "history.csv", index=False, float_format="%.17g")
        write(snapshot / "result.json", {"status": "completed", "evaluations": result.evaluations, "assessment_used": False})
        write(snapshot / "manifest.json", {"artifact_sha256": {p.name: native.sha(p) for p in snapshot.iterdir() if p.is_file()}})
        return snapshot
    finally:
        lock.unlink(missing_ok=True)


def finish(output, budget=None):
    out, cfg = verify(output); budget = cfg["optimization"]["evaluation_budget"] if budget is None else budget
    final = out / "releases" / f"budget_{budget}"
    if final.exists(): native.verify_registry(final / "registry.json"); return final / "registry.json"
    selections, seed_selections, snapshots = [], [], {}
    for request in cfg["profiles"]:
        identifier = request["reservoir_id"]; parts = []
        for seed in cfg["optimization"]["seeds"]:
            folder = out / "jobs" / identifier / f"seed_{seed}" / f"budget_{budget}"
            for name, expected in native.read(folder / "manifest.json")["artifact_sha256"].items():
                if native.sha(folder / name) != expected: raise ValueError("Search snapshot changed")
            table = pd.read_csv(folder / "front.csv", float_precision="round_trip"); table["seed"] = seed
            parts.append(table); snapshots[(identifier, seed)] = folder
            if len(table):
                point, _ = pooled_choice([table])
                seed_selections.append({"reservoir_id": identifier, "seed": seed, "evaluation_id": int(point.evaluation_id),
                    "release_cost": float(point.release_cost), "storage_cost": float(point.storage_cost)})
        point, _ = pooled_choice(parts)
        selections.append({"reservoir_id": identifier, "seed": int(point.seed), "evaluation_id": int(point.evaluation_id),
                           "release_cost": float(point.release_cost), "storage_cost": float(point.storage_cost)})
    build = final.parent / (".building_" + uuid.uuid4().hex); build.mkdir(parents=True)
    write(build / "selection.json", {"selection_uses_assessment": False, "rule": "pooled nondominated calibration costs; normalized ideal distance", "records": selections, "seed_records": seed_selections})
    profiles, rows, provenance_rows = [], [], []
    for chosen in selections:
        identifier = chosen["reservoir_id"]; folder = out / "prepared" / identifier
        objective, _, _, selection, proxy, model = objective_for(out, identifier)
        source_profile = native.read(folder / "source_profile.json"); conversion = native.read(folder / "level_conversion.json")
        original = availability.read_frame(folder / "supplied_observations_si.csv")
        vectors = pd.read_csv(snapshots[(identifier, chosen["seed"])] / "vectors.csv", index_col="evaluation_id", float_precision="round_trip")
        evaluation = objective.evaluate(vectors.loc[chosen["evaluation_id"]].to_numpy())
        if not evaluation.feasible: raise ValueError("Selected geometry replay failed")
        np.testing.assert_allclose(evaluation.objectives, [chosen["release_cost"], chosen["storage_cost"]], rtol=0, atol=1e-10)
        # Choices are already frozen; held-out records are consumed only now.
        assessed, simulation = native.assessment(evaluation.parameters, objective.sampling, objective.root_policy,
            objective.atol_storage_m3, original, selection)
        proxy_assessed = objective.evaluate_validation(vectors.loc[chosen["evaluation_id"]].to_numpy())
        diag = build / "diagnostics" / identifier; diag.mkdir(parents=True)
        if simulation is not None: simulation.to_csv(diag / "simulation.csv", index_label="date", float_format="%.17g")
        from workflows.prepared_calibration_v1 import _metrics
        common_cal = _metrics(original, evaluation.simulation.frame, selection, "calibration")
        baseline = folder / "baseline"
        baseline_metrics = {}
        if (baseline / "receipt.json").exists() and native.read(baseline / "receipt.json")["status"] == "completed":
            fixed = availability.read_frame(baseline / "simulation.csv")
            if native.independent_balance(fixed, model["physical_constants"]["inflow_correction"]) > model["atol_storage_m3"]: raise ValueError("Baseline balance failed")
            baseline_metrics = {p: _metrics(original, fixed, selection, p) for p in ("calibration", "assessment")}
        for phase, metrics in (("calibration", common_cal), ("assessment", assessed["metrics"])):
            row = {"reservoir_id": identifier, "GRAND_ID": source_profile.get("GRAND_ID"), "role": "selected",
                   "seed": chosen["seed"], "evaluation_id": chosen["evaluation_id"], "phase": phase,
                   "assessment_reference": "supplied_storage" if original.storage_m3.notna().any() else "storage_unavailable",
                   "selection_uses_assessment": False}
            for metric in ("storage_nse", "release_nse"):
                row["calibrated_" + metric] = metrics.get(metric)
                row["baseline_" + metric] = baseline_metrics.get(phase, {}).get(metric)
                row["improvement_" + metric] = metrics[metric] - baseline_metrics[phase][metric] if metric in metrics and phase in baseline_metrics else None
            rows.append(row)
        provenance = {"method": "level_geometry_monthly", "source_profile_id": source_profile["profile_id"],
            "conversion": conversion, "observation_basis": source_profile["observation_basis"],
            "storage_objective": cfg["storage_objective"], "selection_recomputed": False,
            "source_certified": False, "independent_ungauged_test": False}
        profile = {**chosen, "grand_id": source_profile.get("GRAND_ID"), "year_selection": "frozen_prepared",
            "scoring": cfg["scoring"], "physical_constants": model["physical_constants"], "sampling": model["sampling"],
            "root_policy": model["root_policy"], "atol_storage_m3": model["atol_storage_m3"],
            "initial_date": proxy.index[0].date().isoformat(), "selection_sha256": selection.signature,
            "evaluation_budget": budget, "calibration": evaluation.metrics, "assessment": assessed,
            "proxy_assessment": {"feasible": proxy_assessed.feasible, "metrics": proxy_assessed.metrics, "reason": proxy_assessed.reason},
            "calibration_against_supplied_observations": common_cal, "parameterization": provenance,
            "inflow_preprocessing": native.read(folder / "preprocessing.json"), "source_kind": "geometry_calibration_v1",
            "physical_source_certified": False, "route": "standard_native_v1"}
        profiles.append((profile, evaluation.parameters.monthly_targets))
        reference = conversion["reference"]
        provenance_rows.append({"reservoir_id": identifier, "parameterization_method": "level_geometry_monthly",
            "level_reference_method": conversion["reference_method"], "level_datum": reference["level_datum"],
            "reference_datum": reference["reference_datum"], "reference_zero_level_m": reference["zero_level_m"],
            "reference_uses_storage_observations": reference["uses_storage_observations"],
            "storage_uncertainty_kind": conversion["uncertainty"]["kind"], "storage_objective": cfg["storage_objective"],
            "geometry_signature": conversion["geometry_signature"], "selection_signature": selection.signature,
            "source_certified": False, "calibration_fraction": selection.roles.count("calibration") / len(selection.roles)})
        for seed_choice in [r for r in seed_selections if r["reservoir_id"] == identifier]:
            if seed_choice["seed"] == chosen["seed"] and seed_choice["evaluation_id"] == chosen["evaluation_id"]:
                seed_cal, seed_assessed = common_cal, assessed
            else:
                seed_vectors = pd.read_csv(snapshots[(identifier, seed_choice["seed"])] / "vectors.csv", index_col="evaluation_id", float_precision="round_trip")
                seed_evaluation = objective.evaluate(seed_vectors.loc[seed_choice["evaluation_id"]].to_numpy())
                if not seed_evaluation.feasible: raise ValueError("Frozen seed replay failed")
                np.testing.assert_allclose(seed_evaluation.objectives, [seed_choice["release_cost"], seed_choice["storage_cost"]], rtol=0, atol=1e-10)
                seed_cal = _metrics(original, seed_evaluation.simulation.frame, selection, "calibration")
                seed_assessed, _ = native.assessment(seed_evaluation.parameters, objective.sampling, objective.root_policy,
                    objective.atol_storage_m3, original, selection)
            for phase, metrics in (("calibration", seed_cal), ("assessment", seed_assessed["metrics"])):
                row = {"reservoir_id": identifier, "GRAND_ID": source_profile.get("GRAND_ID"), "role": "seed",
                    "seed": seed_choice["seed"], "evaluation_id": seed_choice["evaluation_id"], "phase": phase,
                    "assessment_reference": "supplied_storage" if original.storage_m3.notna().any() else "storage_unavailable",
                    "selection_uses_assessment": False}
                for metric in ("storage_nse", "release_nse"):
                    row["calibrated_" + metric] = metrics.get(metric)
                    row["baseline_" + metric] = baseline_metrics.get(phase, {}).get(metric)
                    row["improvement_" + metric] = metrics[metric] - baseline_metrics[phase][metric] if metric in metrics and phase in baseline_metrics else None
                rows.append(row)
    pd.DataFrame(rows).to_csv(build / "baseline_comparison.csv", index=False, float_format="%.17g")
    native.export_release(build, profiles, {"protocol_sha256": native.sha(out / "protocol.json"), "budget": budget, "adapter": "geometry_calibration_v1"})
    table = pd.read_csv(build / "reservoir_parameterization_by_GRAND_ID.csv", float_precision="round_trip")
    table = table.merge(pd.DataFrame(provenance_rows), on="reservoir_id", validate="one_to_one")
    table["GRAND_ID"] = table.GRAND_ID.astype("Int64"); table.to_csv(build / "reservoir_parameterization_by_GRAND_ID.csv", index=False, float_format="%.17g")
    (build / "README.md").write_text("# Geometry-proxy calibration release\n\nCalibration-only reference, uncertainty, targets, bounds and candidate/seed selection. Supplied storage is retained as a separate assessment reference. Band scores are compatibility scores, not probabilistic confidence. Profiles remain exploratory; this checkpoint does not establish convergence.\n\n[Comparison](baseline_comparison.csv), [GRAND-indexed parameters](reservoir_parameterization_by_GRAND_ID.csv), [registry](registry.json).\n", encoding="utf-8")
    write(build / "manifest.json", {"code_sha256": sources(), "artifact_sha256": {p.relative_to(build).as_posix(): native.sha(p) for p in build.rglob("*") if p.is_file() and p.name != "manifest.json"}})
    native.verify_registry(build / "registry.json"); publish_directory(build, final)
    return final / "registry.json"


def run(config_path, output, budget=None):
    path, cfg = config(config_path); out = Path(output).resolve()
    if out.exists():
        verify(out)
        if native.read(out / "config.json") != cfg: raise ValueError("Calibration decisions changed")
    else: prepare(path, out)
    budget = cfg["optimization"]["evaluation_budget"] if budget is None else budget
    lock = out / "run.lock"
    with lock.open("x") as stream: stream.write(str(os.getpid()))
    jobs = [(p["reservoir_id"], s) for p in cfg["profiles"] for s in cfg["optimization"]["seeds"]]
    def execute(job):
        identifier, seed = job
        folder = out / "jobs" / identifier / f"seed_{seed}"; folder.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, "-B", str(ROOT / "tools/calibrate_geometry_v1.py"), "job", "--output", str(out),
                "--reservoir-id", identifier, "--seed", str(seed), "--evaluation-budget", str(budget)]
        with (folder / "execution.log").open("a", encoding="utf-8") as stream:
            result = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return {"reservoir_id": identifier, "seed": seed, "exit_code": result.returncode}
    try:
        write(out / "status.json", {"status": "calibrating", "budget": budget})
        with ThreadPoolExecutor(max_workers=cfg["optimization"]["workers"]) as pool: results = list(pool.map(execute, jobs))
        write(out / f"execution_budget_{budget}.json", {"jobs": results})
        if any(r["exit_code"] for r in results): raise ValueError("Geometry job failed; inspect persistent execution logs")
        registry = finish(out, budget)
        write(out / "status.json", {"status": "completed", "budget": budget, "registry": str(registry)})
        return registry
    except Exception as exc:
        write(out / "status.json", {"status": "failed", "budget": budget, "reason": str(exc)}); raise
    finally: lock.unlink(missing_ok=True)
