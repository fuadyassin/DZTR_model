"""Checkpointed CR2-style release-only calibration and hidden-storage assessment."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import platform
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd

from strategies.dds_v1 import search
from strategies.flow_only_v1 import FlowObjective, fit, nse
from strategies.representative_years import YearSelection
from workflows import data_availability_v2 as availability
from workflows import profile_application_v1 as application
from workflows import reservoir_v2 as native
from workflows.atomic_io_v1 import write
from workflows.geometry_v1 import digest
from workflows.geometry_calibration_v1 import publish_directory

ROOT = native.ROOT
OBJECTIVES = {"release_nse", "sqrt_release_nse"}


def sources():
    return {**availability.sources(), **{p: native.sha(ROOT / p) for p in (
        "strategies/dds_v1.py", "strategies/flow_only_v1.py", "workflows/flow_calibration_v1.py",
        "tools/calibrate_flow_v1.py", "workflows/geometry_calibration_v1.py")}}


def config(path):
    path = Path(path).resolve(strict=True); cfg = native.read(path)
    if set(cfg) != {"kind", "schema_version", "campaign", "profiles", "objectives", "scoring", "bounds_policy", "optimization"}:
        raise ValueError("Declare the complete release-only calibration configuration")
    if (cfg["kind"], cfg["schema_version"]) != ("dztr_flow_calibration_v1", 1): raise ValueError("Wrong flow calibration schema")
    if not isinstance(cfg["objectives"], list) or not cfg["objectives"] or len(set(cfg["objectives"])) != len(cfg["objectives"]) or not set(cfg["objectives"]) <= OBJECTIVES:
        raise ValueError("Declare unique raw and/or square-root release objectives")
    if cfg["scoring"] not in {"pooled", "equal_regime"} or cfg["bounds_policy"] not in {"monthly", "etienne_annual_envelope"}:
        raise ValueError("Unknown scoring or release-bound policy")
    ids = []
    for p in cfg["profiles"]:
        if set(p) - {"reservoir_id", "storage_basis", "assumption_evidence"} or "reservoir_id" not in p: raise ValueError("Invalid flow profile request")
        ids.append(availability.safe_id(p["reservoir_id"]))
        if p.get("storage_basis", "total_capacity") not in {"total_capacity", "active_capacity"}: raise ValueError("Unknown storage basis")
        if p.get("storage_basis") == "active_capacity" and not p.get("assumption_evidence"):
            raise ValueError("An active-storage adaptation requires explicit evidence/assumption rationale")
    if not ids or len(ids) != len(set(ids)): raise ValueError("Declare unique reservoir requests")
    o = cfg["optimization"]
    if set(o) != {"seeds", "horizon", "evaluation_budget", "perturbation", "workers"}: raise ValueError("Declare all DDS settings")
    if not isinstance(o["seeds"], list) or not o["seeds"] or len(set(o["seeds"])) != len(o["seeds"]): raise ValueError("Declare unique seeds")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in o["seeds"]): raise ValueError("Invalid seed")
    if any(isinstance(o[k], bool) or not isinstance(o[k], int) for k in ("horizon", "evaluation_budget", "workers")):
        raise ValueError("DDS budgets and workers must be integers")
    if not 1 <= o["evaluation_budget"] <= o["horizon"] or o["horizon"] < 2 or o["workers"] < 1 or not 0 < o["perturbation"] <= 1:
        raise ValueError("Invalid DDS settings")
    cfg["campaign"] = str((path.parent / cfg["campaign"]).resolve(strict=True))
    return path, cfg


def prepare(config_path, output):
    path, cfg = config(config_path); upstream, profiles = availability.verify(cfg["campaign"])
    inputs = {str(path): native.sha(path)}
    def pin(p):
        p = Path(p).resolve(strict=True); inputs[str(p)] = native.sha(p); return p
    for name in ("protocol.json", "profile_registry.json", "config.json"): pin(upstream / name)
    out = native.fresh_output(output); admissions = []
    requests = {p["reservoir_id"]: p for p in cfg["profiles"]}
    flow_profiles = [p for p in profiles if p["method"] == "flow_only"]
    if set(requests) - {p["reservoir_id"] for p in flow_profiles}: raise ValueError("Requested reservoir is absent upstream")
    for record in flow_profiles:
        identifier = record["reservoir_id"]
        admission = {"reservoir_id": identifier, "GRAND_ID": record.get("GRAND_ID"), "calibration_ready": False,
                     "status": "deferred_source_route", "reason": "; ".join(record.get("reasons", [])), "source_certified": False}
        if identifier not in requests:
            admissions.append(admission); continue
        request = requests[identifier]
        # Upstream may have rejected total-capacity fractions against dead
        # storage. Only that physical decision can be revisited explicitly here.
        if not record.get("target_estimation_ready"):
            admission["status"] = "source_gated"; admissions.append(admission); continue
        common = (upstream / record["prepared_folder"]).parent
        frame = availability.read_frame(pin(common / "observations_si.csv"))
        selection = YearSelection(**native.read(pin(common / "selection.json")))
        metadata = native.read(pin(common / "numerical_profile.json"))
        preprocessing = native.read(pin(common / "preprocessing.json"))
        metadata.setdefault("route", "standard_native_v1")
        if metadata["route"] not in {"standard_native_v1", "signed_net_native_v1"}: raise ValueError("Unsupported numerical route")
        folder = out / "prepared" / identifier; folder.mkdir(parents=True)
        basis = request.get("storage_basis", "total_capacity")
        try:
            template, lower, upper = fit(frame, selection, metadata["physical_constants"],
                quantile_method=record["identity"]["quantile_method"], storage_basis=basis, bounds_policy=cfg["bounds_policy"])
            metadata = {**metadata, "physical_constants": {k: getattr(template, k) for k in metadata["physical_constants"]}}
            for objective in cfg["objectives"]:
                FlowObjective(frame, selection, template, lower, upper, metadata, preprocessing, objective=objective, scoring=cfg["scoring"])
        except ValueError as exc:
            admission.update(status="parameter_or_scoring_gated", reason=str(exc)); admissions.append(admission); continue
        # Training inputs are physically separate from the hidden response file.
        training = frame.copy(); training["storage_m3"] = np.nan
        training.to_csv(folder / "observations_si.csv", index_label="date", float_format="%.17g")
        frame.to_csv(folder / "hidden_assessment_observations_si.csv", index_label="date", float_format="%.17g")
        write(folder / "selection.json", selection.as_dict()); write(folder / "model_config.json", metadata)
        write(folder / "preprocessing.json", preprocessing)
        template.monthly_targets.to_csv(folder / "initial_targets_si.csv", float_format="%.17g")
        pd.DataFrame({"lower": lower, "upper": upper}).to_csv(folder / "release_bounds_si.csv", index=False, float_format="%.17g")
        write(folder / "provenance.json", {"method": "flow_only", "source_profile_id": record["profile_id"],
            "GRAND_ID": record.get("GRAND_ID"), "storage_fractions": [.20, .30, .97], "storage_basis": basis,
            "storage_assumption_evidence": request.get("assumption_evidence", "CR2-style fractions of total adopted capacity"),
            "initial_storage_rule": "assumed normal storage target", "initial_release_rule": "first effective inflow clipped to [0, channel]",
            "storage_observations_used_for_fitting": False, "quantile_method": record["identity"]["quantile_method"],
            "valid_zero_release_retained": True, "capacity_channel_dead_storage_fixed": True,
            "forcing_and_initialization_independence_verified": False,
            "experiment_kind": "conditional_storage_hidden_benchmark", "source_certified": False,
            "selection_signature": selection.signature, "calibration_fraction": selection.roles.count("calibration") / len(selection.roles)})
        admission.update(calibration_ready=True, status="ready", reason=None, storage_basis=basis)
        admissions.append(admission)
    write(out / "config.json", cfg); write(out / "preparation.json", {"reservoirs": admissions, "selection_recomputed": False, "new_optimizer_evaluations": 0})
    for p, expected in inputs.items():
        if native.sha(p) != expected: raise ValueError("Source changed during preparation")
    write(out / "protocol.json", {"kind": "flow_calibration_protocol_v1", "source_sha256": sources(), "input_sha256": inputs,
        "prepared_sha256": {p.relative_to(out).as_posix(): native.sha(p) for p in out.rglob("*") if p.is_file()}})
    return out


def verify(output):
    out = Path(output).resolve(strict=True)
    if not out.is_relative_to(ROOT / "outputs") or out == ROOT / "outputs": raise ValueError("Use project outputs")
    proof = native.read(out / "protocol.json")
    if proof["kind"] != "flow_calibration_protocol_v1": raise ValueError("Wrong flow protocol")
    for p, expected in proof["source_sha256"].items():
        if native.sha(ROOT / p) != expected: raise ValueError("Flow calibration code changed: " + p)
    for p, expected in proof["prepared_sha256"].items():
        target = (out / p).resolve(strict=True)
        if not target.is_relative_to(out) or native.sha(target) != expected: raise ValueError("Flow preparation changed: " + p)
    return out, native.read(out / "config.json")


def objective_for(out, identifier, objective):
    folder = Path(out) / "prepared" / identifier; cfg = native.read(Path(out) / "config.json")
    frame = availability.read_frame(folder / "observations_si.csv")
    model = native.read(folder / "model_config.json"); selection = YearSelection(**native.read(folder / "selection.json"))
    provenance = native.read(folder / "provenance.json")
    template, lower, upper = fit(frame, selection, model["physical_constants"], quantile_method=provenance["quantile_method"],
        storage_basis=provenance["storage_basis"], bounds_policy=cfg["bounds_policy"])
    stored = pd.read_csv(folder / "initial_targets_si.csv", index_col="month", float_precision="round_trip")
    bounds = pd.read_csv(folder / "release_bounds_si.csv", float_precision="round_trip")
    np.testing.assert_array_equal(template.monthly_targets.to_numpy(), stored.to_numpy())
    np.testing.assert_array_equal(lower, bounds.lower); np.testing.assert_array_equal(upper, bounds.upper)
    return FlowObjective(frame, selection, template, lower, upper, model, native.read(folder / "preprocessing.json"),
                         objective=objective, scoring=cfg["scoring"])


def checked(path):
    proof = native.read(path / "manifest.json")
    for name, expected in proof["artifact_sha256"].items():
        target = (path / name).resolve(strict=True)
        if not target.is_relative_to(path.resolve()) or native.sha(target) != expected: raise ValueError("Flow snapshot changed")


def calibrate(output, identifier, objective, seed, budget=None):
    out, cfg = verify(output); options = cfg["optimization"]; budget = options["evaluation_budget"] if budget is None else budget
    ready = {p["reservoir_id"] for p in native.read(out / "preparation.json")["reservoirs"] if p["calibration_ready"]}
    if identifier not in ready or objective not in cfg["objectives"] or seed not in options["seeds"]: raise ValueError("Undeclared flow search")
    if not 1 <= budget <= options["horizon"]: raise ValueError("Requested checkpoint exceeds the frozen DDS horizon")
    folder = out / "jobs" / identifier / objective / f"seed_{seed}"; folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "calibration.lock"
    with lock.open("x") as stream: stream.write(str(os.getpid()))
    try:
        snapshot = folder / f"budget_{budget}"
        if snapshot.exists() and (snapshot / "manifest.json").exists(): checked(snapshot); return snapshot
        obj = objective_for(out, identifier, objective)
        identity = {"kind": "flow_dds_v1", "protocol_sha256": native.sha(out / "protocol.json"),
                    "reservoir_id": identifier, "objective": objective, "seed": seed,
                    "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
        state = None
        if (folder / "checkpoint.json").exists():
            receipt = native.read(folder / "checkpoint.json")
            if receipt["identity"] != identity or digest(receipt["state"]) != receipt["state_sha256"]: raise ValueError("DDS checkpoint identity or content changed")
            state = receipt["state"]
            if len(state["trials"]) > budget: raise ValueError("DDS budget cannot shrink")
        def save(current):
            count = len(current["trials"])
            if count % 8 == 0 or count == budget:
                write(folder / "checkpoint.json", {"identity": identity, "state": current, "state_sha256": digest(current)})
                write(folder / "progress.json", {"evaluations": count, "horizon": options["horizon"]})
        result = search(obj, obj.template.monthly_targets.iloc[:, 3:].to_numpy().ravel(), obj.lower, obj.upper,
            seed=seed, horizon=options["horizon"], budget=budget, perturbation=options["perturbation"], state=state, checkpoint=save)
        snapshot.mkdir(exist_ok=True)
        write(snapshot / "result.json", result)
        pd.DataFrame([{k: v for k, v in r.items() if k != "vector"} for r in result["trials"]]).to_csv(snapshot / "trials.csv", index=False, float_format="%.17g")
        write(snapshot / "manifest.json", {"artifact_sha256": {p.name: native.sha(p) for p in snapshot.iterdir() if p.is_file()}})
        return snapshot
    finally: lock.unlink(missing_ok=True)


def scores(source, simulation, selection, phase):
    dates = source.index[selection.mask(source.index, phase)]
    if not dates.isin(simulation.index).all(): raise ValueError("Missing fixed assessment support")
    result = {}
    for name, column, square in (("release_nse", "release_m3s", False), ("sqrt_release_nse", "release_m3s", True), ("storage_nse", "storage_m3", False)):
        try: result[name] = nse(source.loc[dates, column], simulation.loc[dates, column], square_root=square)
        except ValueError: result[name] = None
    for variable, column in (("storage", "storage_m3"), ("release", "release_m3s")):
        o, s = source.loc[dates, column].to_numpy(), simulation.loc[dates, column].to_numpy()
        keep = np.isfinite(o)
        result[variable + "_pairs"] = int(keep.sum())
        if keep.any() and np.isfinite(s[keep]).all():
            result[variable + "_mean_bias"] = float(np.mean(s[keep] - o[keep]))
            result[variable + "_pbias"] = float(100 * np.sum(s[keep] - o[keep]) / np.sum(o[keep])) if np.sum(o[keep]) else None
    return result


def finish(output, budget=None):
    out, cfg = verify(output); budget = cfg["optimization"]["evaluation_budget"] if budget is None else budget
    final = out / "releases" / f"budget_{budget}"
    if final.exists():
        checked(final)
        return final
    ready = [p for p in native.read(out / "preparation.json")["reservoirs"] if p["calibration_ready"]]
    choices, seed_choices, statuses = [], [], []
    for admission in ready:
        identifier = admission["reservoir_id"]
        for objective in cfg["objectives"]:
            candidates = []
            for seed in cfg["optimization"]["seeds"]:
                snapshot = out / "jobs" / identifier / objective / f"seed_{seed}" / f"budget_{budget}"; checked(snapshot)
                trials = native.read(snapshot / "result.json")["trials"]
                finite = [r for r in trials if r["cost"] is not None]
                if finite:
                    point = min(finite, key=lambda r: (r["cost"], r["evaluation_id"]))
                    record = {**point, "reservoir_id": identifier, "seed": seed, "objective": objective}
                    candidates.append(record); seed_choices.append(record)
                statuses.append({**admission, "objective": objective, "seed": seed, "evaluations": len(trials),
                    "feasible_evaluations": len(finite), "search_status": "completed" if finite else "no_feasible_candidate"})
            if candidates: choices.append(min(candidates, key=lambda r: (r["cost"], r["seed"], r["evaluation_id"])))
    build = final.parent / (".building_" + uuid.uuid4().hex); build.mkdir(parents=True)
    # No hidden storage or held-out release has been read by selection.
    write(build / "selection.json", {"selection_uses_assessment": False, "rule": "minimum calibration release cost; ties seed then evaluation ID",
                                    "records": choices, "seed_records": seed_choices})
    rows, diagnostics, exported = [], [], {o: [] for o in cfg["objectives"]}
    baseline_cache = {}
    for choice in seed_choices:
        identifier, objective = choice["reservoir_id"], choice["objective"]
        obj = objective_for(out, identifier, objective)
        evaluation = obj.evaluate(choice["vector"])
        if not evaluation.feasible or not np.isclose(evaluation.cost, choice["cost"], rtol=0, atol=1e-10): raise ValueError("Frozen flow calibration replay changed")
        folder = out / "prepared" / identifier
        hidden = availability.read_frame(folder / "hidden_assessment_observations_si.csv")
        provenance = native.read(folder / "provenance.json")
        if identifier not in baseline_cache:
            try:
                result, residual = application.run_frame(obj.frame, obj.metadata, obj.template, forcing_stage="prepared", preprocessing=obj.preprocessing)
                baseline_cache[identifier] = {"status": "completed", "frame": result.frame, "balance": residual}
            except (ValueError, RuntimeError) as exc: baseline_cache[identifier] = {"status": "failed", "reason": str(exc)}
        baseline = baseline_cache[identifier]
        try:
            full, residual = application.run_frame(obj.frame, obj.metadata, evaluation.parameters, forcing_stage="prepared", preprocessing=obj.preprocessing)
            assessment_status, assessment_reason = "completed", None
        except (ValueError, RuntimeError) as exc:
            full, residual = None, evaluation.balance; assessment_status, assessment_reason = "failed", str(exc)
        selected = any(c["reservoir_id"] == identifier and c["objective"] == objective and c["seed"] == choice["seed"] for c in choices)
        cal_scores = scores(hidden, evaluation.simulation.frame, obj.selection, "calibration")
        held_scores = scores(hidden, full.frame, obj.selection, "assessment") if full is not None else {}
        for phase, metrics in (("calibration", cal_scores), ("assessment", held_scores)):
            baseline_scores = scores(hidden, baseline["frame"], obj.selection, phase) if baseline["status"] == "completed" else {}
            row = {"reservoir_id": identifier, "GRAND_ID": provenance["GRAND_ID"], "objective": objective,
                "seed": choice["seed"], "evaluation_id": choice["evaluation_id"], "selected": selected, "phase": phase,
                "calibration_cost": choice["cost"], "assessment_status": assessment_status, "assessment_reason": assessment_reason,
                "baseline_status": baseline["status"], "baseline_reason": baseline.get("reason"), "storage_basis": provenance["storage_basis"],
                "maximum_independent_balance_residual_m3": residual, "selection_uses_assessment": False}
            for metric in ("release_nse", "sqrt_release_nse", "storage_nse"):
                a, b = metrics.get(metric), baseline_scores.get(metric)
                row["calibrated_" + metric], row["baseline_" + metric] = a, b
                row["improvement_" + metric] = a - b if a is not None and b is not None else None
            for metric in ("storage_pairs", "release_pairs", "storage_mean_bias", "release_mean_bias", "storage_pbias", "release_pbias"):
                row["calibrated_" + metric] = metrics.get(metric); row["baseline_" + metric] = baseline_scores.get(metric)
            rows.append(row)
        if selected:
            diag = build / "diagnostics" / identifier / objective; diag.mkdir(parents=True)
            (full.frame if full is not None else evaluation.simulation.frame).to_csv(diag / "simulation.csv", index_label="date", float_format="%.17g")
            write(diag / "assessment.json", {"status": assessment_status, "reason": assessment_reason, "calibration": cal_scores, "assessment": held_scores})
            profile = {k: v for k, v in choice.items() if k != "vector"}
            profile.update(grand_id=provenance["GRAND_ID"], year_selection="frozen_mixed_years", scoring=cfg["scoring"],
                physical_constants={k: getattr(evaluation.parameters, k) for k in obj.metadata["physical_constants"]},
                sampling=obj.metadata["sampling"], root_policy=obj.metadata["root_policy"], atol_storage_m3=obj.metadata["atol_storage_m3"],
                route=obj.metadata.get("route", "standard_native_v1"), initial_date=obj.frame.index[0].date().isoformat(),
                selection_sha256=obj.selection.signature, evaluation_budget=budget, dds_horizon=cfg["optimization"]["horizon"],
                parameterization=provenance, inflow_preprocessing=obj.preprocessing, assessment_status=assessment_status,
                fitting_years=sorted(set(obj.frame.index[obj.mask].year)), bounds_policy=cfg["bounds_policy"],
                physical_source_certified=False, calibration=cal_scores, assessment=held_scores)
            if "forcing_policy" in obj.metadata: profile["forcing_policy"] = obj.metadata["forcing_policy"]
            exported[objective].append((profile, evaluation.parameters.monthly_targets))
            diagnostics.append({"reservoir_id": identifier, "objective": objective, "seed": choice["seed"],
                "assessment_status": assessment_status, "calibration_fraction": provenance["calibration_fraction"],
                "storage_basis": provenance["storage_basis"], "calibration_cost": choice["cost"]})
    baseline_folder = build / "baselines"; baseline_folder.mkdir()
    for identifier, baseline in baseline_cache.items():
        folder = baseline_folder / identifier; folder.mkdir()
        write(folder / "receipt.json", {k: v for k, v in baseline.items() if k != "frame"})
        if "frame" in baseline: baseline["frame"].to_csv(folder / "simulation.csv", index_label="date", float_format="%.17g")
    pd.DataFrame(rows).to_csv(build / "baseline_comparison.csv", index=False, float_format="%.17g")
    pd.DataFrame(statuses).to_csv(build / "search_status.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(build / "selected_profiles.csv", index=False)
    inventory = native.read(out / "preparation.json")["reservoirs"]
    for objective, profiles in exported.items():
        arm = build / objective; arm.mkdir()
        if profiles:
            native.export_release(arm, profiles, {"method": "flow_only", "objective": objective, "campaign": str(out)})
            table = pd.read_csv(arm / "reservoir_parameterization_by_GRAND_ID.csv")
        else: table = pd.DataFrame(columns=["reservoir_id", "GRAND_ID"])
        # Preserve all paper inventory rows, including gated/no-solution records.
        inv = pd.DataFrame(inventory)
        table = inv.merge(table.drop(columns="GRAND_ID", errors="ignore"), on="reservoir_id", how="left", validate="one_to_one")
        table["parameterization_method"] = "flow_only"; table["objective"] = objective
        table["parameters_available"] = table.reservoir_id.isin([p[0]["reservoir_id"] for p in profiles])
        for profile, _ in profiles:
            keep = table.reservoir_id == profile["reservoir_id"]; provenance = profile["parameterization"]
            for key, value in {"quantile_method": provenance["quantile_method"], "calibration_fraction": provenance["calibration_fraction"],
                "selection_signature": profile["selection_sha256"], "fitting_years": ",".join(map(str, profile["fitting_years"])),
                "assessment_status": profile["assessment_status"], "initial_storage_m3": profile["physical_constants"]["initial_storage_m3"],
                "initial_release_m3s": profile["physical_constants"]["initial_release_m3s"], "storage_basis": provenance["storage_basis"],
                "root_policy": profile["root_policy"], "numerical_route": profile["route"]}.items():
                if key not in table: table[key] = None
                table.loc[keep, key] = value
        table["optimizer"] = "DDS"; table["dds_horizon"] = cfg["optimization"]["horizon"]; table["evaluation_budget"] = budget
        table["bounds_policy"] = cfg["bounds_policy"]; table["scoring"] = cfg["scoring"]
        table["storage_fractions"] = "0.20,0.30,0.97"; table["independent_ungauged_test"] = False
        table["storage_observations_used_for_fitting"] = False; table["source_certified"] = False
        table["convergence_established"] = False
        table.to_csv(arm / "reservoir_parameterization_by_GRAND_ID.csv", index=False, float_format="%.17g")
        if profiles:
            write(arm / "manifest.json", {"code_sha256": sources(), "artifact_sha256": {
                p.relative_to(arm).as_posix(): native.sha(p) for p in arm.rglob("*") if p.is_file() and p.name != "manifest.json"}})
    write(build / "summary.json", {"requested_reservoirs": len(cfg["profiles"]), "ready_reservoirs": len(ready),
        "searches": len(statuses), "evaluations": sum(r["evaluations"] for r in statuses),
        "selected_profiles": len(choices), "convergence_established": False, "selection_uses_assessment": False})
    write(build / "manifest.json", {"code_sha256": sources(), "artifact_sha256": {
        p.relative_to(build).as_posix(): native.sha(p) for p in build.rglob("*") if p.is_file() and p != build / "manifest.json"}})
    publish_directory(build, final)
    return final


def run(config_path, output, budget=None):
    _, cfg = config(config_path); out = Path(output).resolve()
    if out.exists(): verify(out)
    else: prepare(config_path, out)
    if native.read(out / "config.json") != cfg: raise ValueError("Flow campaign decisions changed")
    ready = [p["reservoir_id"] for p in native.read(out / "preparation.json")["reservoirs"] if p["calibration_ready"]]
    def launch(job):
        identifier, objective, seed = job
        folder = out / "jobs" / identifier / objective / f"seed_{seed}"; folder.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, "-B", str(ROOT / "tools/calibrate_flow_v1.py"), "job", "--output", str(out), "--reservoir-id", identifier,
                "--objective", objective, "--seed", str(seed)]
        if budget is not None: args += ["--evaluation-budget", str(budget)]
        with (folder / "execution.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return {"reservoir_id": identifier, "objective": objective, "seed": seed, "returncode": result.returncode}
    jobs = [(r, o, s) for r in ready for o in cfg["objectives"] for s in cfg["optimization"]["seeds"]]
    with ThreadPoolExecutor(max_workers=cfg["optimization"]["workers"]) as executor: results = list(executor.map(launch, jobs))
    write(out / "execution.json", {"jobs": results})
    if any(j["returncode"] for j in results): raise ValueError("Flow jobs failed; inspect execution.json and per-job logs; completed checkpoints are preserved")
    return finish(out, budget)


def simulate(registry, forcing_csv, output, *, identifier=None, grand_id=None, initial=None):
    profile, parameters = native.load_profile(registry, identifier=identifier, grand_id=grand_id)
    frame = availability.read_frame(forcing_csv)
    result, residual = application.run_frame(frame, profile, parameters, forcing_stage="prepared",
        preprocessing=profile["inflow_preprocessing"], initial=initial)
    out = native.fresh_output(output)
    result.frame.to_csv(out / "simulation.csv", index_label="date", float_format="%.17g")
    write(out / "receipt.json", {"status": "completed", "registry": str(Path(registry).resolve()),
        "forcing_sha256": native.sha(forcing_csv), "forcing_stage": "prepared", "observed_storage_used": False,
        "maximum_independent_balance_residual_m3": residual, "new_optimizer_evaluations": 0})
    return out
