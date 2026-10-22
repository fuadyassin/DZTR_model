"""Persistent fixed-profile no-observation runs; original equations stay unchanged."""
from pathlib import Path
import os

import numpy as np
import pandas as pd

from strategies.no_observations_v1 import parameters
from strategies.representative_years import YearSelection
from workflows import data_availability_v2 as availability
from workflows import release_prior_v1 as priors
from workflows import profile_application_v1 as application
from workflows import reservoir_v2 as native
from workflows.atomic_io_v1 import write
from workflows.flow_calibration_v1 import checked, scores
from workflows.geometry_v1 import digest

ROOT = native.ROOT


def sources():
    return {**availability.sources(), **{p: native.sha(ROOT / p) for p in (
        "strategies/no_observations_v1.py", "workflows/no_observations_v1.py",
        "workflows/mean_flow_prior_v1.py", "workflows/release_prior_v1.py",
        "workflows/profile_application_v1.py", "workflows/flow_calibration_v1.py",
        "tools/no_observations_v1.py")}}


def config(path):
    path = Path(path).resolve(strict=True); cfg = native.read(path)
    if set(cfg) != {"kind", "schema_version", "campaign", "profiles", "release_prior"} or (cfg["kind"], cfg["schema_version"]) != ("dztr_no_observations_v1", 1):
        raise ValueError("Declare the complete no-observation configuration")
    if cfg["release_prior"].get("source_kind") != "mean_flow_scaled_donor_prior":
        raise ValueError("This version requires the explicit mean-flow donor prior")
    ids = []
    for row in cfg["profiles"]:
        if set(row) - {"reservoir_id", "storage_basis", "assumption_evidence"} or not {"reservoir_id", "storage_basis"} <= set(row):
            raise ValueError("Declare reservoir ID, storage basis and any adaptation evidence")
        ids.append(availability.safe_id(row["reservoir_id"]))
        if row["storage_basis"] not in {"total_capacity", "active_capacity"}:
            raise ValueError("Unknown storage basis")
        if row["storage_basis"] == "active_capacity" and not row.get("assumption_evidence"):
            raise ValueError("Active-storage fractions require explicit assumption evidence")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Declare unique reservoir requests")
    cfg["campaign"] = str((path.parent / cfg["campaign"]).resolve(strict=True))
    return path, cfg


def prepare(path, output):
    path, cfg = config(path); upstream, records = availability.verify(cfg["campaign"])
    inputs = {str(path): native.sha(path)}
    def pin(value):
        value = Path(value).resolve(strict=True); inputs[str(value)] = native.sha(value); return value
    for name in ("protocol.json", "profile_registry.json", "config.json"):
        pin(upstream / name)
    unique = {}
    for record in records:
        rid = record["reservoir_id"]
        if rid not in unique or record["method"] == "observed_monthly": unique[rid] = record
    requests = {r["reservoir_id"]: r for r in cfg["profiles"]}
    if set(requests) - set(unique): raise ValueError("Unknown reservoir request")
    out = native.fresh_output(output); admissions = []
    for rid, record in unique.items():
        row = {"reservoir_id": rid, "GRAND_ID": record.get("GRAND_ID"), "name": record.get("name", rid),
               "parameters_available": False, "status": "not_requested", "reason": None,
               "source_certified": False, "original_recommended_method": record.get("recommended_method"),
               "capabilities": record.get("capabilities", {}), "original_route": record.get("original_route")}
        if "prepared_folder" not in record:
            row.update(status="deferred_source_route", reason="; ".join(record.get("reasons", [])) or record.get("next_action"))
        elif rid in requests:
            gates = record.get("capabilities", {}).get("source_gates", [])
            if gates:
                row.update(status="source_gated", reason="; ".join(gates))
            else:
                common = (upstream / record["prepared_folder"]).parent
                observed = availability.read_frame(pin(common / "observations_si.csv"))
                # The estimator receives this forcing-only frame. Responses are archived separately.
                forcing = observed[["inflow_m3s"]].copy()
                selection = YearSelection(**native.read(pin(common / "selection.json")))
                if selection.roles.count("calibration") / len(selection.roles) + 1e-12 < .65:
                    raise ValueError("Retain at least 65% calibration support for controlled comparisons")
                original = native.read(pin(common / "numerical_profile.json"))
                model = {k: original[k] for k in ("physical_constants", "sampling", "root_policy", "atol_storage_m3", "initial_date")}
                model["route"] = original.get("route", "standard_native_v1")
                if "forcing_policy" in original: model["forcing_policy"] = original["forcing_policy"]
                if model["route"] not in {"standard_native_v1", "signed_net_native_v1"}: raise ValueError("Unsupported native route")
                preprocessing = native.read(pin(common / "preprocessing.json"))
                folder = out / "prepared" / rid; folder.mkdir(parents=True)
                forcing.to_csv(folder / "forcing_si.csv", index_label="date", float_format="%.17g")
                write(folder / "selection.json", selection.as_dict()); write(folder / "preprocessing.json", preprocessing)
                try:
                    release, prior = priors.prepare(cfg["release_prior"], forcing.index, selection,
                        quantile_method=record["quantile_method"], folder=folder / "release_prior", pin=pin,
                        base=path.parent, frame=forcing, metadata=model, reservoir_id=rid, grand_id=record.get("GRAND_ID"))
                    target = parameters(forcing, model["physical_constants"], release, storage_basis=requests[rid]["storage_basis"])
                except ValueError as exc:
                    observed.to_csv(folder / "hidden_assessment_observations_si.csv", index_label="date", float_format="%.17g")
                    row.update(status="prior_or_parameter_gated", reason=str(exc)); admissions.append(row); continue
                model["physical_constants"] = {k: getattr(target, k) for k in model["physical_constants"]}
                forcing.to_csv(folder / "forcing_si.csv", index_label="date", float_format="%.17g")
                target.monthly_targets.to_csv(folder / "monthly_targets_si.csv", index_label="month", float_format="%.17g")
                provenance = {"method": "ungauged_mean_flow", "storage_basis": requests[rid]["storage_basis"],
                    "assumption_evidence": requests[rid].get("assumption_evidence"), "storage_fractions": [.2, .3, .97],
                    "release_prior": prior, "storage_observations_used_for_fitting": False,
                    "release_observations_used_for_fitting": False, "initial_storage_source": "assumed_normal_target",
                    "initial_release_source": "first_effective_inflow_clipped_to_channel", "optimizer_evaluations": 0,
                    "capacity_channel_dead_storage_fixed": True, "independent_ungauged_test": False,
                    "selection_signature": selection.signature, "quantile_method": record["quantile_method"],
                    "calibration_fraction": selection.roles.count("calibration") / len(selection.roles)}
                write(folder / "provenance.json", provenance); write(folder / "model_config.json", model)
                write(folder / "selection.json", selection.as_dict()); write(folder / "preprocessing.json", preprocessing)
                # Responses are saved only after parameters and initialization have been frozen.
                observed.to_csv(folder / "hidden_assessment_observations_si.csv", index_label="date", float_format="%.17g")
                row.update(parameters_available=True, status="parameters_ready", storage_basis=requests[rid]["storage_basis"],
                    mean_source=prior["mean_source"], mean_discharge_m3s=prior["mean_discharge_m3s"], donor_count=prior["donor_count"],
                    target_levels_truncated=prior["target_levels_truncated"], numerical_route=model["route"],
                    calibration_fraction=provenance["calibration_fraction"])
        admissions.append(row)
    write(out / "config.json", cfg); write(out / "preparation.json", {"reservoirs": admissions, "selection_recomputed": False})
    for p, expected in inputs.items():
        if native.sha(p) != expected: raise ValueError("Input changed during preparation")
    write(out / "protocol.json", {"kind": "no_observations_protocol_v1", "source_sha256": sources(), "input_sha256": inputs,
        "prepared_sha256": {p.relative_to(out).as_posix(): native.sha(p) for p in out.rglob("*") if p.is_file()}})
    return out


def verify(output):
    out = Path(output).resolve(strict=True)
    if not out.is_relative_to(ROOT / "outputs") or out == ROOT / "outputs": raise ValueError("Use project outputs")
    proof = native.read(out / "protocol.json")
    if proof.get("kind") != "no_observations_protocol_v1": raise ValueError("Unknown no-observation protocol")
    for rel, expected in proof["source_sha256"].items():
        if native.sha(ROOT / rel) != expected: raise ValueError("Scientific code changed: " + rel)
    for rel, expected in proof["prepared_sha256"].items():
        target = (out / rel).resolve(strict=True)
        if not target.is_relative_to(out) or native.sha(target) != expected: raise ValueError("Prepared artifact changed: " + rel)
    for p, expected in proof["input_sha256"].items():
        if native.sha(p) != expected: raise ValueError("Original input changed: " + p)
    return out, native.read(out / "config.json")


def fixed_job(output, rid):
    out, _ = verify(output)
    ready = {r["reservoir_id"] for r in native.read(out / "preparation.json")["reservoirs"] if r["parameters_available"]}
    if rid not in ready: raise ValueError("Reservoir has no valid prepared parameters")
    folder = out / "jobs" / rid; folder.mkdir(parents=True, exist_ok=True)
    signature = native.sha(out / "protocol.json")
    if (folder / "manifest.json").exists():
        checked(folder); receipt = native.read(folder / "receipt.json")
        if receipt["protocol_sha256"] != signature: raise ValueError("Job belongs to different preparation")
        return receipt
    lock = folder / "run.lock"
    with lock.open("x") as stream: stream.write(str(os.getpid()))
    try:
        prepared = out / "prepared" / rid
        forcing = availability.read_frame(prepared / "forcing_si.csv")
        assert list(forcing.columns) == ["inflow_m3s"]
        model = native.read(prepared / "model_config.json")
        targets = pd.read_csv(prepared / "monthly_targets_si.csv", index_col="month", float_precision="round_trip")
        from dztr.schemas import ReservoirParameters
        parameter = ReservoirParameters(targets, **model["physical_constants"])
        receipt = {"reservoir_id": rid, "protocol_sha256": signature, "optimizer_evaluations": 0,
                   "responses_used_for_initialization": False, "responses_used_for_parameter_estimation": False}
        try:
            result, residual = application.run_frame(forcing, model, parameter, forcing_stage="prepared",
                preprocessing=native.read(prepared / "preprocessing.json"))
            result.frame.to_csv(folder / "simulation.csv", index_label="date", float_format="%.17g")
            hidden = availability.read_frame(prepared / "hidden_assessment_observations_si.csv")
            plan = YearSelection(**native.read(prepared / "selection.json"))
            receipt.update(status="completed", maximum_independent_balance_residual_m3=residual,
                scores={phase: scores(hidden, result.frame, plan, phase) for phase in ("calibration", "assessment")})
        except (ValueError, RuntimeError) as exc:
            receipt.update(status="simulation_failed", reason=str(exc), scores={})
        write(folder / "receipt.json", receipt)
        write(folder / "manifest.json", {"artifact_sha256": {p.name: native.sha(p) for p in folder.iterdir()
            if p.is_file() and p.name not in {"manifest.json", "run.lock"}}})
        return receipt
    finally:
        lock.unlink(missing_ok=True)


def finish(output):
    out, _ = verify(output); final = out / "release"
    if final.exists(): checked(final); return final
    final.mkdir(); profiles = []; inventory = []; phase_rows = []
    for admission in native.read(out / "preparation.json")["reservoirs"]:
        row = {k: v for k, v in admission.items() if k != "capabilities"}; rid = row["reservoir_id"]
        row.update(parameterization_method="ungauged_mean_flow", optimizer_evaluations=0,
                   storage_observations_used_for_fitting=False, release_observations_used_for_fitting=False)
        if row["parameters_available"]:
            folder = out / "prepared" / rid; model = native.read(folder / "model_config.json")
            checked(out / "jobs" / rid); receipt = native.read(out / "jobs" / rid / "receipt.json")
            row.update(simulation_status=receipt["status"], simulation_reason=receipt.get("reason"))
            provenance = native.read(folder / "provenance.json")
            profile = {**model, "reservoir_id": rid, "grand_id": row["GRAND_ID"], "seed": None, "evaluation_id": None,
                "year_selection": "frozen_mixed_years", "scoring": "none_fixed_profile", "evaluation_budget": 0,
                "inflow_preprocessing": native.read(folder / "preprocessing.json"), "parameterization": provenance,
                "selection_sha256": provenance["selection_signature"], "source_kind": "assumed_no_observations",
                "simulation_status": receipt["status"], "physical_source_certified": False}
            targets = pd.read_csv(folder / "monthly_targets_si.csv", index_col="month", float_precision="round_trip")
            profiles.append((profile, targets))
            for phase in ("calibration", "assessment"):
                phase_rows.append({"reservoir_id": rid, "GRAND_ID": row["GRAND_ID"], "method": "ungauged_mean_flow",
                    "phase": phase, "status": receipt["status"], "reason": receipt.get("reason"), **receipt["scores"].get(phase, {})})
        inventory.append(row)
    if profiles:
        native.export_release(final, profiles, {"method": "ungauged_mean_flow", "campaign": str(out)})
        values = pd.read_csv(final / "reservoir_parameterization_by_GRAND_ID.csv", float_precision="round_trip")
    else:
        values = pd.DataFrame(columns=["reservoir_id", "GRAND_ID"])
        write(final / "registry.json", {"schema_version": 1, "kind": "unified_dztr_profiles", "selection_uses_assessment": False,
            "profiles_by_reservoir_id": {}, "profiles_by_GRAND_ID": {}, "provenance": {"method": "ungauged_mean_flow"}})
    inv = pd.DataFrame(inventory).merge(values.drop(columns="GRAND_ID", errors="ignore"), on="reservoir_id", how="left", validate="one_to_one")
    inv.to_csv(final / "reservoir_parameterization_all_by_GRAND_ID.csv", index=False, float_format="%.17g")
    pd.DataFrame(phase_rows, columns=list(phase_rows[0]) if phase_rows else ["reservoir_id", "phase", "status"]).to_csv(final / "scores.csv", index=False, float_format="%.17g")
    write(final / "summary.json", {"inventory_reservoirs": len(inventory), "parameters_available": len(profiles),
        "completed_simulations": sum(r.get("simulation_status") == "completed" for r in inventory), "optimizer_evaluations": 0,
        "independent_ungauged_test": False, "source_certified": False})
    write(final / "manifest.json", {"code_sha256": sources(), "artifact_sha256": {
        p.relative_to(final).as_posix(): native.sha(p) for p in final.rglob("*") if p.is_file() and p != final / "manifest.json"}})
    return final


def run(config_path, output):
    _, cfg = config(config_path); out = Path(output).resolve()
    if out.exists():
        _, saved = verify(out)
        if saved != cfg: raise ValueError("No-observation decisions changed")
    else: prepare(config_path, out)
    for row in native.read(out / "preparation.json")["reservoirs"]:
        if row["parameters_available"]: fixed_job(out, row["reservoir_id"])
    return finish(out)


def simulate(registry, forcing_csv, output, *, identifier=None, grand_id=None, initial=None, method=None):
    registry = Path(registry).resolve(strict=True)
    if native.read(registry).get("kind") == "availability_route_registry_v1":
        from workflows.availability_comparison_v1 import resolve
        selected = resolve(registry, identifier=identifier, grand_id=grand_id, method=method)
        identifier, grand_id = selected["reservoir_id"], None
        registry = Path(selected["registry"])
    elif method is not None:
        raise ValueError("A method override requires a multi-route registry")
    profile, parameter = native.load_profile(registry, identifier=identifier, grand_id=grand_id)
    forcing = availability.read_frame(forcing_csv)[["inflow_m3s"]]
    result, residual = application.run_frame(forcing, profile, parameter, forcing_stage="prepared",
        preprocessing=profile.get("inflow_preprocessing", {"factor": 1., "offset_m3s": 0.}), initial=initial)
    out = native.fresh_output(output)
    result.frame.to_csv(out / "simulation.csv", index_label="date", float_format="%.17g")
    write(out / "receipt.json", {"status": "completed", "registry": str(registry), "forcing_sha256": native.sha(forcing_csv),
        "maximum_independent_balance_residual_m3": residual, "optimizer_evaluations": 0, "responses_used": False})
    return out
