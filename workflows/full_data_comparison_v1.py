"""Checkpointed, matched GA/DDS experiments on full-data scalar objectives."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd

from dztr.calibration.candidates import parameters_to_vector
from strategies.full_data_scalar_v1 import FullDataObjective, weights
from strategies.representative_years import YearSelection, fit_selected_years
from strategies.single_objective_v1 import options, search
from workflows import prepared_calibration_v1 as prepared
from workflows import reservoir_v2 as native
from workflows import profile_application_v1 as application
from workflows.atomic_io_v1 import write
from workflows.flow_calibration_v2 import scores, checked, simulate
from workflows.geometry_v1 import digest
from workflows.geometry_calibration_v1 import publish_directory

ROOT = native.ROOT


def sources():
    from workflows import flow_calibration_v2 as flow
    return {**prepared.sources(), **flow.sources(), **{p: native.sha(ROOT / p) for p in (
        "strategies/full_data_scalar_v1.py", "workflows/full_data_comparison_v1.py", "workflows/full_data_report_v1.py", "tools/compare_full_data_v1.py")}}


def normalize(cfg):
    if set(cfg) != {"kind", "schema_version", "campaign", "profiles", "objectives", "scoring", "optimization"} or (cfg["kind"], cfg["schema_version"]) != ("dztr_full_data_comparison_v1", 1):
        raise ValueError("Declare the full-data comparison configuration")
    if cfg["scoring"] not in {"pooled", "equal_regime"}: raise ValueError("Unknown regime scoring")
    profiles = cfg["profiles"]
    if not isinstance(profiles, list) or not profiles: raise ValueError("Declare reservoir profiles")
    ids = []
    for entry in profiles:
        if set(entry) != {"reservoir_id"}: raise ValueError("Declare reservoir_id only; full observed-monthly profiles are required")
        ids.append(prepared.availability.safe_id(entry["reservoir_id"]))
    if len(ids) != len(set(ids)): raise ValueError("Duplicate reservoirs")
    objectives = cfg["objectives"]
    if not isinstance(objectives, list) or not objectives: raise ValueError("Declare weighted objectives")
    names = []
    for entry in objectives:
        if set(entry) != {"id", "weights"}: raise ValueError("Declare objective id and weights")
        names.append(prepared.availability.safe_id(entry["id"]))
        entry["weights"] = weights(entry["weights"])
    if len(names) != len(set(names)): raise ValueError("Duplicate objective ids")
    opt = cfg["optimization"]
    if set(opt) != {"seeds", "horizon", "evaluation_budget", "workers", "algorithms"} or not isinstance(opt["algorithms"], list) or not opt["algorithms"]:
        raise ValueError("Declare shared search controls and algorithm settings")
    common = {k: opt[k] for k in ("seeds", "horizon", "evaluation_budget", "workers")}
    algorithms = []
    for entry in opt["algorithms"]:
        if not isinstance(entry, dict) or "optimizer" not in entry or set(entry) - {"optimizer", "ga", "perturbation"}:
            raise ValueError("Declare explicit optimizer-specific controls")
        validated = options({**common, **entry})
        algorithms.append({k: v for k, v in validated.items() if k not in common})
    if len({a["optimizer"] for a in algorithms}) != len(algorithms): raise ValueError("Duplicate algorithms")
    opt["algorithms"] = algorithms
    return cfg


def config(path):
    path = Path(path).resolve(strict=True)
    cfg = normalize(native.read(path))
    cfg["campaign"] = str((path.parent / cfg["campaign"]).resolve(strict=True))
    return path, cfg


def algorithm_options(cfg, optimizer):
    found = [a for a in cfg["optimization"]["algorithms"] if a["optimizer"] == optimizer]
    if len(found) != 1: raise ValueError("Undeclared optimizer")
    return options({**{k: v for k, v in cfg["optimization"].items() if k != "algorithms"}, **found[0]})


def numerical_route(metadata, frame):
    if not np.isfinite(frame.inflow_m3s.to_numpy()).all(): raise ValueError("Prepared inflow must be finite")
    route = metadata.get("route")
    if route is None:
        if metadata["sampling"].get("allow_signed_inflow", False) or frame.inflow_m3s.lt(0).any() or "forcing_policy" in metadata:
            raise ValueError("An absent route cannot be inferred for signed or adapted forcing")
        note = "Explicit standard route inferred from unsigned sampling and finite nonnegative prepared inflow; original metadata omitted route"
    else: note = "Retained explicit source route"
    if route not in {None, "standard_native_v1"}: raise ValueError("Signed or interval forcing requires its own adapter")
    if frame.inflow_m3s.lt(0).any() or metadata["sampling"].get("allow_signed_inflow", False) or "forcing_policy" in metadata:
        raise ValueError("Standard full-data routing requires unsigned forcing without a signed adapter")
    return {**metadata, "route": "standard_native_v1"}, note


def prepare(config_path, output):
    path, cfg = config(config_path)
    upstream, records = prepared.availability.verify(cfg["campaign"])
    out = native.fresh_output(output)
    write(out / "config.json", cfg)
    inputs = {str(path): native.sha(path)}
    def pin(source):
        source = Path(source).resolve(strict=True)
        inputs[str(source)] = native.sha(source)
        return source
    for name in ("protocol.json", "profile_registry.json", "config.json"): pin(upstream / name)
    admissions = []
    for request in cfg["profiles"]:
        rid = request["reservoir_id"]
        found = [r for r in records if r["reservoir_id"] == rid and r["method"] == "observed_monthly"]
        if len(found) != 1: raise ValueError("Observed-monthly source is unavailable or ambiguous")
        record = found[0]
        if not record["parameters_ready"] or not record["simulation_ready"] or record["reasons"]:
            raise ValueError("Source profile has unresolved parameter, forcing or source decisions")
        if record["observation_basis"] != "native_archive" or record["identity"]["quantile_method"] != "hazen":
            raise ValueError("This full-data adapter requires native observations and Hazen percentiles")
        source = upstream / record["prepared_folder"]; common = source.parent
        frame = prepared.availability.read_frame(pin(common / "observations_si.csv"))
        selection = YearSelection(**native.read(pin(common / "selection.json")))
        metadata = native.read(pin(common / "numerical_profile.json"))
        metadata, route_note = numerical_route(metadata, frame)
        physical = native.read(pin(source / "model_constants.json"))
        if physical != metadata["physical_constants"] or physical != record["identity"]["physical_constants"]:
            raise ValueError("Source physical constants disagree")
        if selection.signature != record["identity"]["selection_signature"]: raise ValueError("Source selection identity changed")
        template, bounds = fit_selected_years(native.observations(common / "observations_si.csv"), selection, physical)
        targets = pd.read_csv(pin(source / "monthly_targets_si.csv"), index_col="month", float_precision="round_trip")
        np.testing.assert_array_equal(template.monthly_targets.to_numpy(), targets.to_numpy())
        bounds.validate(parameters_to_vector(template), template)
        metadata["cost_mode"] = "pooled_nse" if cfg["scoring"] == "pooled" else "regime_balanced_nse"
        folder = out / "handoff/prepared" / rid; folder.mkdir(parents=True)
        for original, name in ((common / "observations_si.csv", "observations_si.csv"), (common / "selection.json", "selection.json"),
            (common / "preprocessing.json", "preprocessing.json"), (source / "monthly_targets_si.csv", "initial_targets_si.csv")):
            shutil.copyfile(pin(original), folder / name)
        write(folder / "model_config.json", metadata)
        bounds.empirical.monthly_bounds.to_csv(folder / "bounds_si.csv", index_label="month", float_format="%.17g")
        write(folder / "initialization.json", {"kind": "explicit_calibration_initializer", "basis": "calibration_only",
            "selection_sha256": selection.signature, "source_targets_sha256": native.sha(source / "monthly_targets_si.csv")})
        admissions.append({"reservoir_id": rid, "GRAND_ID": record.get("GRAND_ID"), "source_profile_id": record["profile_id"],
            "route_decision": route_note, "original_registry_modified": False, "calibration_fraction": float(np.mean(selection.mask(frame.index, "calibration")))})
    write(out / "handoff/preparation.json", {"reservoirs": admissions, "selection_recomputed": False, "new_evaluations": 0})
    for entry in cfg["profiles"]:
        for objective in cfg["objectives"]: objective_for(out, entry["reservoir_id"], objective["id"])
    for source, expected in inputs.items():
        if native.sha(source) != expected: raise ValueError("Source changed during handoff")
    write(out / "protocol.json", {"kind": "full_data_comparison_protocol_v1", "source_sha256": sources(),
        "input_sha256": inputs, "prepared_sha256": {
            p.relative_to(out).as_posix(): native.sha(p) for p in out.rglob("*") if p.is_file()}})
    return out


def verify(output):
    out = Path(output).resolve(strict=True)
    if not out.is_relative_to(ROOT / "outputs") or out == ROOT / "outputs": raise ValueError("Use project outputs")
    proof = native.read(out / "protocol.json")
    if proof["kind"] != "full_data_comparison_protocol_v1": raise ValueError("Wrong comparison protocol")
    for p, expected in proof["source_sha256"].items():
        if native.sha(ROOT / p) != expected: raise ValueError("Comparison code changed: " + p)
    for p, expected in proof["prepared_sha256"].items():
        target = (out / p).resolve(strict=True)
        if not target.is_relative_to(out) or native.sha(target) != expected: raise ValueError("Comparison preparation changed: " + p)
    return out, native.read(out / "config.json")


def objective_for(out, identifier, objective):
    cfg = native.read(Path(out) / "config.json")
    if identifier not in {p["reservoir_id"] for p in cfg["profiles"]}: raise ValueError("Undeclared reservoir")
    selected = [o for o in cfg["objectives"] if o["id"] == objective]
    if len(selected) != 1: raise ValueError("Undeclared objective")
    _, template, bounds, selection, frame, metadata = native.objective_for(Path(out) / "handoff", identifier)
    admission = next(r for r in native.read(Path(out) / "handoff/preparation.json")["reservoirs"] if r["reservoir_id"] == identifier)
    metadata = {**metadata, "grand_id": admission.get("GRAND_ID")}
    prep = native.read(Path(out) / "handoff" / "prepared" / identifier / "preprocessing.json")
    return FullDataObjective(frame, selection, template, bounds, metadata, prep,
        objective_weights=selected[0]["weights"], scoring=cfg["scoring"])


def job_folder(out, identifier, objective, optimizer, seed):
    return Path(out) / "jobs" / identifier / objective / optimizer / f"seed_{seed}"


def calibrate(output, identifier, objective, optimizer, seed, budget=None):
    out, cfg = verify(output)
    opt = algorithm_options(cfg, optimizer)
    budget = opt["evaluation_budget"] if budget is None else budget
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= opt["horizon"]: raise ValueError("Invalid checkpoint budget")
    if seed not in opt["seeds"]: raise ValueError("Undeclared seed")
    obj = objective_for(out, identifier, objective)
    folder = job_folder(out, identifier, objective, optimizer, seed); folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "calibration.lock"
    with lock.open("x") as stream: stream.write(str(os.getpid()))
    try:
        snapshot = folder / f"budget_{budget}"
        if (snapshot / "manifest.json").exists(): checked(snapshot); return snapshot
        identity = {"kind": "full_data_scalar_job_v1", "protocol_sha256": native.sha(out / "protocol.json"),
            "reservoir_id": identifier, "objective": objective, "weights": obj.weights, "optimizer": optimizer, "seed": seed,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
        state = None
        if (folder / "checkpoint.json").exists():
            receipt = native.read(folder / "checkpoint.json")
            if receipt["identity"] != identity or digest(receipt["state"]) != receipt["state_sha256"]: raise ValueError("Checkpoint identity/content changed")
            state = receipt["state"]
        def save(current, force=False):
            count = len(current["trials"])
            if force or count % 16 == 0 or count == budget:
                write(folder / "checkpoint.json", {"identity": identity, "state": current, "state_sha256": digest(current)})
                finite = [r["cost"] for r in current["trials"] if r["cost"] is not None]
                write(folder / "progress.json", {"evaluations": count, "budget": budget, "horizon": opt["horizon"],
                    "best_cost": min(finite) if finite else None})
        result = search(obj, parameters_to_vector(obj.template), obj.lower, obj.upper, seed=seed,
            budget=budget, optimization=opt, state=state, checkpoint=save)
        save(result, True)
        snapshot.mkdir(exist_ok=True)
        write(snapshot / "result.json", result)
        pd.DataFrame([{k: v for k, v in t.items() if k != "vector"} for t in result["trials"]]).to_csv(snapshot / "trials.csv", index=False)
        write(snapshot / "manifest.json", {"artifact_sha256": {p.name: native.sha(p) for p in snapshot.iterdir() if p.is_file() and p.name != "manifest.json"}})
        return snapshot
    finally: lock.unlink(missing_ok=True)


def finish(output, budget=None):
    out, cfg = verify(output)
    budget = cfg["optimization"]["evaluation_budget"] if budget is None else budget
    final = out / "releases" / f"budget_{budget}"
    if final.exists(): checked(final); return final
    choices, status = [], []
    for entry in cfg["profiles"]:
        rid = entry["reservoir_id"]
        for objective in cfg["objectives"]:
            for alg in cfg["optimization"]["algorithms"]:
                candidates = []
                for seed in cfg["optimization"]["seeds"]:
                    snapshot = job_folder(out, rid, objective["id"], alg["optimizer"], seed) / f"budget_{budget}"
                    checked(snapshot)
                    state = native.read(snapshot / "result.json"); finite = [t for t in state["trials"] if t["cost"] is not None]
                    row = {"reservoir_id": rid, "objective": objective["id"], "optimizer": alg["optimizer"], "seed": seed,
                        "evaluations": len(state["trials"]), "feasible_evaluations": len(finite), "termination": state.get("terminal_reason")}
                    status.append(row)
                    if finite: candidates.append({**min(finite, key=lambda t: (t["cost"], t["evaluation_id"])), **row})
                if candidates:
                    winner = min(candidates, key=lambda t: (t["cost"], t["seed"], t["evaluation_id"]))
                    choices.extend([{**c, "selected": c["seed"] == winner["seed"]} for c in candidates])
    build = final.parent / (".building_" + uuid.uuid4().hex); build.mkdir(parents=True)
    write(build / "selection.json", {"selection_uses_assessment": False,
        "rule": "minimum calibration scalar cost within objective/optimizer; ties seed then evaluation ID", "records": choices})
    rows, physical, exports, baselines, progress = [], [], {}, {}, []
    for choice in choices:
        rid, objective, optimizer, seed = [choice[k] for k in ("reservoir_id", "objective", "optimizer", "seed")]
        obj = objective_for(out, rid, objective)
        evaluation = obj.evaluate(choice["vector"])
        if not evaluation.feasible or not np.isclose(evaluation.cost, choice["cost"], rtol=0, atol=1e-10): raise ValueError("Calibration replay changed")
        # Selection is frozen above before reading the assessment responses.
        full_obs = prepared.availability.read_frame(out / "handoff" / "prepared" / rid / "observations_si.csv")
        if rid not in baselines:
            try:
                b, residual = application.run_frame(obj.frame, obj.metadata, obj.template, forcing_stage="prepared", preprocessing=obj.preprocessing)
                baselines[rid] = b.frame
                physical.append({"reservoir_id": rid, "case": "baseline", "maximum_balance_residual_m3": residual, "status": "completed"})
                dest = build / "baselines" / rid; dest.mkdir(parents=True)
                b.frame.to_csv(dest / "simulation.csv", index_label="date", float_format="%.17g")
            except (ValueError, RuntimeError) as exc:
                baselines[rid] = None
                physical.append({"reservoir_id": rid, "case": "baseline", "status": "failed", "reason": str(exc)})
        try:
            full, residual = application.run_frame(obj.frame, obj.metadata, evaluation.parameters, forcing_stage="prepared", preprocessing=obj.preprocessing)
            held_status, reason = "completed", None
        except (ValueError, RuntimeError) as exc:
            full, residual, held_status, reason = None, evaluation.balance, "failed", str(exc)
        physical.append({"reservoir_id": rid, "objective": objective, "optimizer": optimizer, "seed": seed,
            "status": held_status, "reason": reason, "maximum_balance_residual_m3": residual,
            "minimum_storage_m3": float(full.frame.storage_m3.min()) if full else None,
            "maximum_storage_m3": float(full.frame.storage_m3.max()) if full else None,
            "capacity_m3": obj.template.capacity_m3, "tolerance_m3": obj.metadata["atol_storage_m3"]})
        for phase in ("calibration", "assessment"):
            trajectory = evaluation.simulation.frame if phase == "calibration" else full.frame if full else None
            metrics = scores(full_obs, trajectory, obj.selection, phase) if trajectory is not None else {}
            base = scores(full_obs, baselines[rid], obj.selection, phase) if baselines[rid] is not None else {}
            row = {k: choice[k] for k in ("reservoir_id", "objective", "optimizer", "seed", "selected", "evaluations", "feasible_evaluations")}
            row.update(GRAND_ID=obj.metadata.get("grand_id"), phase=phase, calibration_cost=choice["cost"],
                storage_weight=obj.weights["storage"], release_weight=obj.weights["release"], assessment_status=held_status, assessment_reason=reason)
            for key, value in metrics.items(): row["calibrated_" + key] = value
            for key, value in base.items(): row["baseline_" + key] = value
            for key in ("storage_nse", "release_nse"):
                a, b = metrics.get(key), base.get(key)
                row["improvement_" + key] = a - b if a is not None and b is not None else None
            rows.append(row)
        dest = build / "trajectories" / rid / objective / optimizer / f"seed_{seed}"; dest.mkdir(parents=True)
        (full.frame if full else evaluation.simulation.frame).to_csv(dest / "simulation.csv", index_label="date", float_format="%.17g")
        if choice["selected"]:
            # Copy only model settings; stale metrics/seed from an upstream profile cannot propagate.
            metadata = {k: obj.metadata[k] for k in ("physical_constants", "sampling", "root_policy", "atol_storage_m3", "initial_date", "route") if k in obj.metadata}
            metadata.update({k: choice[k] for k in ("reservoir_id", "objective", "optimizer", "seed", "evaluation_id")})
            metadata.update(grand_id=obj.metadata.get("grand_id"), year_selection="frozen_mixed_years", scoring=cfg["scoring"],
                objective_weights=obj.weights, evaluation_budget=budget, actual_evaluations=choice["evaluations"],
                selection_sha256=obj.selection.signature, selection_uses_assessment=False, convergence_established=False,
                optimizer_settings=algorithm_options(cfg, optimizer), calibration_cost=choice["cost"],
                inflow_preprocessing=obj.preprocessing, assessment_status=held_status, parameterization_method="observed_monthly",
                adjustable_targets=72, storage_observations_used_for_bounds=True,
                calibration_fraction=float(obj.mask.mean()), source_certified=False)
            exports.setdefault((objective, optimizer), []).append((metadata, evaluation.parameters.monthly_targets))
    for entry in cfg["profiles"]:
        for objective in cfg["objectives"]:
            for alg in cfg["optimization"]["algorithms"]:
                for seed in cfg["optimization"]["seeds"]:
                    trials = native.read(job_folder(out, entry["reservoir_id"], objective["id"], alg["optimizer"], seed) / f"budget_{budget}" / "result.json")["trials"]
                    best = None
                    for t in trials:
                        if t["cost"] is not None: best = min(best, t["cost"]) if best is not None else t["cost"]
                        progress.append({"reservoir_id": entry["reservoir_id"], "objective": objective["id"], "optimizer": alg["optimizer"],
                            "seed": seed, "evaluations": t["evaluation_id"] + 1, "best_cost": best})
    parameter_tables = []
    for (objective, optimizer), profiles in exports.items():
        folder = build / objective / optimizer; folder.mkdir(parents=True)
        native.export_release(folder, profiles, {"method": "observed_monthly", "calibration_backend": "prepared_full_data_v1",
            "objective": objective, "optimizer": optimizer, "campaign": str(out)})
        table = pd.read_csv(folder / "reservoir_parameterization_by_GRAND_ID.csv")
        table["objective"], table["optimizer"] = objective, optimizer
        objective_spec = next(o for o in cfg["objectives"] if o["id"] == objective)
        for variable, value in objective_spec["weights"].items(): table[variable + "_weight"] = value
        table["adjustable_targets"] = 72
        parameter_tables.append(table)
        table.to_csv(folder / "reservoir_parameterization_by_GRAND_ID.csv", index=False, float_format="%.17g")
        write(folder / "manifest.json", {"code_sha256": sources(), "artifact_sha256": {
            p.relative_to(folder).as_posix(): native.sha(p) for p in folder.rglob("*") if p.is_file() and p.name != "manifest.json"}})
    (pd.concat(parameter_tables, ignore_index=True) if parameter_tables else pd.DataFrame(columns=["reservoir_id", "GRAND_ID", "objective", "optimizer"])).to_csv(
        build / "selected_parameterization_by_GRAND_ID.csv", index=False, float_format="%.17g")
    pd.DataFrame(rows, columns=list(rows[0]) if rows else ["reservoir_id", "objective", "optimizer", "phase"]).to_csv(build / "baseline_comparison.csv", index=False, float_format="%.17g")
    pd.DataFrame(physical, columns=list(dict.fromkeys(k for row in physical for k in row)) if physical else ["reservoir_id", "status", "maximum_balance_residual_m3"]).to_csv(build / "scientific_checks.csv", index=False, float_format="%.17g")
    pd.DataFrame(status).to_csv(build / "search_status.csv", index=False)
    pd.DataFrame(progress).to_csv(build / "convergence.csv", index=False, float_format="%.17g")
    write(build / "summary.json", {"searches": len(status), "evaluations": sum(r["evaluations"] for r in status),
        "selected_profiles": sum(c["selected"] for c in choices), "selection_uses_assessment": False,
        "convergence_established": False, "fitted_targets": 72})
    from workflows.full_data_report_v1 import render
    render(build, out, cfg)
    write(build / "manifest.json", {"code_sha256": sources(), "artifact_sha256": {
        p.relative_to(build).as_posix(): native.sha(p) for p in build.rglob("*") if p.is_file() and p != build / "manifest.json"}})
    publish_directory(build, final)
    return final


def run(config_path, output, budget=None):
    _, cfg = config(config_path)
    out = Path(output).resolve()
    if out.exists(): verify(out)
    else: prepare(config_path, out)
    if native.read(out / "config.json") != cfg: raise ValueError("Comparison decisions changed")
    jobs = [(r["reservoir_id"], o["id"], a["optimizer"], s) for r in cfg["profiles"] for o in cfg["objectives"]
        for a in cfg["optimization"]["algorithms"] for s in cfg["optimization"]["seeds"]]
    def launch(job):
        rid, objective, optimizer, seed = job
        folder = job_folder(out, *job); folder.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, "-B", str(ROOT / "tools/compare_full_data_v1.py"), "job", "--output", str(out),
            "--reservoir-id", rid, "--objective", objective, "--optimizer", optimizer, "--seed", str(seed)]
        if budget is not None: args += ["--evaluation-budget", str(budget)]
        with (folder / "execution.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return {"job": job, "returncode": result.returncode}
    with ThreadPoolExecutor(max_workers=cfg["optimization"]["workers"]) as executor: results = list(executor.map(launch, jobs))
    write(out / "execution.json", {"jobs": results})
    if any(j["returncode"] for j in results): raise ValueError("Comparison jobs failed; inspect execution.json and per-job logs")
    return finish(out, budget)
