"""Persistent preparation and fixed-profile benchmarks for data-availability recipes.

The original cohort, equations and optimizer checkpoints are read-only. This
delivery creates targets and fixed-profile simulations, never optimizer jobs.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
import pandas as pd

from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters
from dztr.calibration.sampled import _observations
from dztr.simulation.sampled import SamplingSettings, validate_sampling_index
from dztr.simulation.root_policy import POLICIES
from strategies.representative_years import YearSelection, select_representative_years
from strategies.data_availability_v1 import METHODS, NaturalizedOutflow, build_targets, capabilities, series_signature
from workflows import reservoir_v2 as w
from workflows import profile_application_v1 as app
from workflows.geometry_v1 import (GeometryCurve, LevelReference, calibration_maximum_reference,
                                   convert_levels, digest, read_regeom, read_si_curve, validate_index)
from workflows.observation_operators_v1 import score_native

ROOT = w.ROOT
NEW_SOURCES = ("workflows/geometry_v1.py", "strategies/data_availability_v1.py",
               "workflows/observation_operators_v1.py", "workflows/data_availability_v1.py",
               "tools/data_availability_v1.py")


def sources():
    return {**w.source_hashes(), **{p: w.sha(ROOT / p) for p in NEW_SOURCES},
            "workflows/profile_application_v1.py": w.sha(ROOT / "workflows/profile_application_v1.py")}


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("Unsafe reservoir/profile identifier")
    return value


def resolve(base, value):
    return (Path(base) / value).resolve(strict=True)


def read_frame(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        import csv
        header = next(csv.reader(stream))
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV columns")
    frame = pd.read_csv(path, index_col="date", parse_dates=True, float_precision="round_trip")
    validate_index(frame.index)
    if frame.empty:
        raise ValueError("An input time series cannot be empty")
    for column in frame:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if np.isinf(frame[column].to_numpy()).any():
            raise ValueError("CSV measurements must be finite or missing")
    return frame


def package_geometry(source_folder, catalog_path, destination):
    """Archive only verified cohort-ID curves; no new ID mappings are inferred."""
    catalog = app.read_catalog(catalog_path)
    destination = Path(destination).resolve()
    if destination.exists() or not destination.is_relative_to(ROOT / "data") or destination == ROOT / "data":
        raise ValueError("Use a fresh versioned directory inside project data")
    entries = [e for e in catalog["reservoirs"] if e.get("GRAND_ID") is not None]
    curves = [(e, read_regeom(Path(source_folder) / f"{e['GRAND_ID']}.csv", expected_grand_id=e["GRAND_ID"])) for e in entries]
    destination.mkdir(parents=True)
    records = []
    for entry, curve in curves:
        folder = destination / str(entry["GRAND_ID"])
        folder.mkdir()
        raw = folder / "source_regeom.csv"
        shutil.copyfile(curve.source, raw)
        if w.sha(raw) != curve.source_sha256:
            raise ValueError("Geometry changed while being archived")
        curve.table().to_csv(folder / "curve_si.csv", index=False, float_format="%.17g")
        records.append({"GRAND_ID": entry["GRAND_ID"], "reservoir_id": entry["reservoir_id"],
                        "name": entry["name"], "raw_curve": raw.relative_to(destination).as_posix(),
                        "raw_sha256": w.sha(raw), "si_sha256": w.sha(folder / "curve_si.csv"),
                        "geometry_signature": curve.signature, "diagnostics": curve.diagnostics(),
                        "reference_available": False, "levels_available": False})
    w.write(destination / "index.json", {"kind": "verified_cohort_geometry_curves_v1", "records": records,
        "source_catalog_sha256": w.sha(catalog_path), "dataset_doi": "10.5281/zenodo.1322884",
        "level_references_inferred": False, "new_id_assignments": 0})
    (destination / "README.md").write_text(
        "# Verified cohort geometry curves\n\n"
        f"{len(records)} ReGeom curves are archived for the existing verified paper-cohort GRanD mappings. "
        "Each directory retains the original CSV and an SI representation; index.json pins both. "
        "These are approximate geometric relationships, not surveyed bathymetry. "
        "Curve availability does not establish gauge levels, datum references, storage coordinates or calibration readiness. "
        "No new IDs, level references or capacity adjustments were inferred.\n\n"
        "Dataset: https://doi.org/10.5281/zenodo.1322884\n", encoding="utf-8")
    return destination / "index.json"


def _config(path):
    path = Path(path).resolve(strict=True)
    cfg = w.read(path)
    allowed = {"kind", "schema_version", "catalog", "reservoirs", "geometry_index", "methods",
               "supplemental_inputs", "quantile_method", "channel_policy", "year_selection"}
    if set(cfg) - allowed or cfg.get("kind") != "dztr_data_availability_v1" or cfg.get("schema_version") != 1:
        raise ValueError("Unknown data-availability configuration schema/fields")
    if ("catalog" in cfg) == ("reservoirs" in cfg):
        raise ValueError("Declare either a frozen catalog or standalone reservoirs")
    methods = cfg.get("methods", list(METHODS))
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError("Declare unique supported target recipes")
    cfg["methods"] = methods
    if cfg.get("quantile_method", "hazen") not in {"hazen", "linear"}:
        raise ValueError("Unknown percentile estimator")
    if cfg.get("channel_policy", "source_defined") not in {"source_defined", "fitting_p99"}:
        raise ValueError("Declare source_defined or fitting_p99 channel policy")
    return path, cfg


def _selection(frame, metadata, selection_path, cfg):
    if selection_path is not None:
        selection = YearSelection(**w.read(selection_path))
        selection.mask(frame.index, "calibration")
        return selection
    # Independent declared forcing selects years; responses are not used for fit.
    adjusted = frame.copy()
    adjusted["inflow_m3s"] *= metadata["physical_constants"]["inflow_correction"]
    selection, _, _ = select_representative_years(_observations(adjusted), SamplingSettings(**metadata["sampling"]),
                                                  **cfg.get("year_selection", {}))
    return selection


def _curve(spec, gid, geometry_index, config_parent, pin):
    if "geometry_csv" in spec:
        path = pin(resolve(config_parent, spec["geometry_csv"]))
        if not spec.get("geometry_evidence"):
            raise ValueError("Explicit geometry needs evidence for its reservoir linkage")
        if spec.get("geometry_format") == "regeom":
            expected = spec.get("geometry_grand_id", gid)
            if expected is None:
                raise ValueError("ReGeom requires an independently declared geometry ID")
            return read_regeom(path, expected_grand_id=expected)
        if spec.get("geometry_format") == "surveyed_si":
            return read_si_curve(path, source_kind="surveyed_si", grand_id=gid)
        raise ValueError("Declare regeom or surveyed_si geometry")
    if gid is not None and geometry_index is not None:
        path, index = geometry_index
        found = [e for e in index["records"] if e["GRAND_ID"] == gid]
        if len(found) > 1:
            raise ValueError("Ambiguous geometry index")
        if found:
            record = found[0]
            raw = pin(resolve(path.parent, record["raw_curve"]))
            if w.sha(raw) != record["raw_sha256"]:
                raise ValueError("Packaged geometry checksum changed")
            si = pin(raw.with_name("curve_si.csv"))
            if w.sha(si) != record["si_sha256"]:
                raise ValueError("Packaged SI geometry checksum changed")
            curve = read_regeom(raw, expected_grand_id=gid)
            if curve.signature != record["geometry_signature"]:
                raise ValueError("Packaged scientific geometry identity changed")
            return curve
    return None


def _supplements(frame, selection, spec, curve, parent, pin, folder):
    allowed = {"levels_csv", "level_time_basis", "reference", "allow_calibration_maximum",
               "allow_mean_level_proxy", "geometry_csv", "geometry_format", "geometry_grand_id",
               "geometry_evidence", "naturalized_outflow"}
    if set(spec) - allowed:
        raise ValueError("Unknown supplemental input field")
    converted, conversion, naturalized = None, None, None
    reasons = []
    if "levels_csv" in spec:
        path = pin(resolve(parent, spec["levels_csv"]))
        raw = read_frame(path)
        if list(raw.columns) != ["level_m"]:
            raise ValueError("Level CSV requires date and level_m, in metres")
        shutil.copyfile(path, folder / "source_levels.csv")
        levels = raw.level_m.reindex(frame.index)
        if curve is None:
            reasons.append("needs_usable_geometry")
        else:
            if "reference" in spec:
                reference = LevelReference(**spec["reference"])
                if reference.basis != "source_reference":
                    raise ValueError("Estimated references must be fitted by the calibration-only adapter")
            elif spec.get("allow_calibration_maximum") is True:
                reference = calibration_maximum_reference(levels, curve,
                    pd.Series(selection.mask(frame.index, "calibration"), index=frame.index),
                    fitting_signature=selection.signature, allow_exploratory=True)
            else:
                reference = None
                reasons.append("needs_explicit_level_reference")
            if reference is not None:
                converted, conversion = convert_levels(levels, curve, reference,
                    time_basis=spec["level_time_basis"], allow_mean_level_proxy=spec.get("allow_mean_level_proxy", False))
                converted.to_csv(folder / "level_conversion_si.csv", index_label="date", float_format="%.17g")
                w.write(folder / "level_conversion.json", conversion)
    else:
        reasons.append("no_level_series_supplied")
    if "naturalized_outflow" in spec:
        info = spec["naturalized_outflow"]
        if set(info) != {"csv", "source_kind", "evidence", "independent_of_target_responses"}:
            raise ValueError("Naturalized outflow needs CSV, kind, source evidence and dependency declaration")
        path = pin(resolve(parent, info["csv"]))
        raw = read_frame(path)
        if list(raw.columns) != ["naturalized_outflow_m3s"] or not raw.index.equals(frame.index):
            raise ValueError("Naturalized outflow CSV needs the exact calendar and SI column")
        naturalized = NaturalizedOutflow(raw.iloc[:, 0], info["source_kind"], info["evidence"], info["independent_of_target_responses"])
        shutil.copyfile(path, folder / "naturalized_outflow_si.csv")
    return converted, conversion, naturalized, reasons


def prepare(config_path, output):
    config_path, cfg = _config(config_path)
    upstream = {}

    def pin(path):
        path = Path(path).resolve(strict=True)
        current = w.sha(path)
        if str(path) in upstream and upstream[str(path)] != current:
            raise ValueError("Source changed during preparation")
        upstream[str(path)] = current
        return path

    pin(config_path)
    geometry_index = None
    if cfg.get("geometry_index"):
        path = pin(resolve(config_path.parent, cfg["geometry_index"]))
        index = w.read(path)
        if index.get("kind") != "verified_cohort_geometry_curves_v1":
            raise ValueError("Unknown geometry package")
        geometry_index = path, index
    catalog = None
    if "catalog" in cfg:
        path = pin(resolve(config_path.parent, cfg["catalog"]))
        catalog = app.read_catalog(path)
        entries = catalog["reservoirs"]
    else:
        entries = cfg["reservoirs"]
    identifiers = [safe_id(e["reservoir_id"]) for e in entries]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate reservoir IDs")
    if set(cfg.get("supplemental_inputs", {})) - set(identifiers):
        raise ValueError("Supplemental inputs name an unknown reservoir")
    out = w.fresh_output(output)
    w.write(out / "config.json", cfg)
    inventory, profiles = [], []
    for entry in entries:
        identifier = entry["reservoir_id"]
        gid = entry.get("GRAND_ID")
        if gid is not None and (isinstance(gid, bool) or not isinstance(gid, int) or gid <= 0):
            raise ValueError("GRAND_ID must be a verified positive integer or null")
        folder = out / "prepared" / identifier
        folder.mkdir(parents=True)
        spec = cfg.get("supplemental_inputs", {}).get(identifier, {})
        curve = _curve(spec, gid, geometry_index, config_path.parent, pin)
        if curve is not None:
            curve.table().to_csv(folder / "geometry_si.csv", index=False, float_format="%.17g")
            w.write(folder / "geometry.json", {**asdict(curve), "signature": curve.signature})
        inv = {"reservoir_id": identifier, "GRAND_ID": gid, "name": entry["name"],
               "existing_status": entry.get("status", "new_input"),
               "existing_calibrated_monthly_profile": bool(entry.get("monthly_profile_available", False)),
               "geometry_readable": curve is not None, "level_series_supplied": "levels_csv" in spec,
               "naturalized_outflow_supplied": "naturalized_outflow" in spec,
               "source_certified": False, "original_route": entry.get("route", "standard_native_v1"),
               "storage_unit": "m3", "flow_unit": "m3/s", "level_unit": "m"}
        if catalog is not None and not entry["monthly_profile_available"]:
            raw_source = ROOT / "data/reservoirs" / identifier / "observations_si.csv"
            if raw_source.exists():
                raw_frame = read_frame(pin(raw_source))
                inv.update(_input_inventory(raw_frame))
            inv.update(recommended_method="source_review_required", next_action=entry["next_action"],
                       storage_conversion_ready=False, calibration_fraction=None)
            inventory.append(inv)
            for method in cfg["methods"]:
                profiles.append({**inv, "profile_id": f"{method}_unavailable", "method": method,
                                 "target_estimation_ready": False, "parameters_ready": False,
                                 "simulation_ready": False, "calibration_ready": False,
                                 "reasons": [entry["status"], entry["next_action"]]})
            continue
        if catalog is not None:
            for relative, expected in entry["input_sha256"].items():
                if w.sha(pin(app.reference(relative))) != expected:
                    raise ValueError("Frozen cohort input changed")
            metadata, archived_parameters = w.load_profile(app.reference(entry["registry"]), identifier=identifier)
            frame = read_frame(pin(app.reference(entry["prepared_forcing"])))
            selection_path = pin(app.reference(entry["prepared_forcing"]).with_name("selection.json"))
            preprocessing = entry["inflow_preprocessing"]
        else:
            model_path = pin(resolve(config_path.parent, entry["model_config"]))
            model = w.read(model_path)
            frame = read_frame(pin(resolve(config_path.parent, entry["observations_csv"])))
            if "inflow_m3s" not in frame:
                frame["inflow_m3s"] = np.nan
            for column in ("storage_m3", "release_m3s"):
                if column not in frame:
                    frame[column] = np.nan
            preprocessing = model.get("inflow_preprocessing", {"factor": 1., "offset_m3s": 0.})
            if entry.get("forcing_stage") not in {"canonical", "prepared"}:
                raise ValueError("Standalone forcing needs an explicit canonical/prepared stage")
            if np.isfinite(frame.inflow_m3s).all():
                frame = app.normalize_forcing(frame, stage=entry["forcing_stage"],
                                              factor=preprocessing["factor"], offset=preprocessing["offset_m3s"])
            metadata = {"physical_constants": model["model"], "sampling": model["sampling"],
                        "root_policy": model["root_policy"], "atol_storage_m3": model["atol_storage_m3"],
                        "initial_date": frame.index[0].date().isoformat(), "route": model.get("route", "standard_native_v1")}
            if "forcing_policy" in model:
                metadata["forcing_policy"] = model["forcing_policy"]
            selection_path = pin(resolve(config_path.parent, entry["selection_json"])) if entry.get("selection_json") else None
        selection = _selection(frame, metadata, selection_path, cfg)
        validate_sampling_index(frame.index, SamplingSettings(**metadata["sampling"]).frequency)
        if metadata["root_policy"] not in POLICIES:
            raise ValueError("Unknown native root policy")
        roles = pd.Series(selection.roles, index=frame.index, name="role")
        mask = roles.eq("calibration")
        physical = metadata["physical_constants"]
        inv.update(_input_inventory(frame))
        inv.update(capacity_m3=physical["capacity_m3"], sampling_frequency=metadata["sampling"]["frequency"],
                   root_policy=metadata["root_policy"], input_stage="prepared",
                   scoring_policy_inherited=metadata.get("scoring", metadata.get("cost_mode")),
                   curve_top_relative_capacity_difference=curve.diagnostics(physical["capacity_m3"])["curve_top_relative_capacity_difference"] if curve else None)
        converted, conversion, naturalized, geometry_reasons = _supplements(frame, selection, spec, curve,
                                                                           config_path.parent, pin, folder)
        frame.to_csv(folder / "observations_si.csv", index_label="date", float_format="%.17g")
        w.write(folder / "selection.json", selection.as_dict())
        w.write(folder / "numerical_profile.json", metadata)
        w.write(folder / "preprocessing.json", preprocessing)
        gates = entry.get("source_gates", [])
        if not isinstance(gates, list) or any(not isinstance(g, str) or not g for g in gates):
            raise ValueError("Source gates must be explicit nonempty reason strings")
        caps = capabilities(frame, mask, source_gates=gates)
        if converted is not None:
            derived = frame.copy()
            derived["storage_m3"] = converted.storage_m3
            derived.to_csv(folder / "level_observations_si.csv", index_label="date", float_format="%.17g")
        else:
            derived = frame.copy()
            derived["storage_m3"] = np.nan
        inv.update(recommended_method=caps["recommended_method"], calibration_fraction=float(mask.mean()),
                   storage_calibration_pairs=int(frame.loc[mask, "storage_m3"].notna().sum()),
                   release_calibration_pairs=int(frame.loc[mask, "release_m3s"].notna().sum()),
                   storage_conversion_ready=converted is not None,
                   geometry_reasons=";".join(geometry_reasons), next_action="Prepare fixed-profile baselines; no new optimizer job is prepared")
        if not caps["storage_monthly_available"] and converted is not None:
            gcaps = capabilities(derived, mask, storage_source_kind="level_geometry")
            if gcaps["storage_monthly_available"]:
                inv["recommended_method"] = gcaps["recommended_method"]
        inventory.append(inv)
        for method in cfg["methods"]:
            geometry_method = method == "level_geometry_monthly" or (
                method == "storage_only" and not caps["storage_monthly_available"] and converted is not None)
            working = derived if geometry_method else frame
            storage_kind = "level_geometry" if geometry_method else "provided_storage"
            recipe = build_targets(method, working, mask, capacity_m3=physical["capacity_m3"],
                channel_capacity_m3s=physical["channel_capacity_m3s"] if cfg.get("channel_policy", "source_defined") == "source_defined" else None,
                storage_source_kind=storage_kind, naturalized=naturalized, quantile_method=cfg.get("quantile_method", "hazen"),
                source_gates=gates)
            if geometry_method and converted is None:
                recipe["reasons"].extend(geometry_reasons)
                recipe["target_estimation_ready"] = False
            target_frame = recipe.pop("targets")
            relevant = ["storage_m3"] if method == "storage_only" else ["release_m3s"] if method == "flow_only" else ["storage_m3", "release_m3s"] if method in {"observed_monthly", "level_geometry_monthly"} else []
            fitting = {c: series_signature(working.loc[mask, c]) for c in relevant}
            identity = {"method": method, "reservoir_id": identifier, "selection_signature": selection.signature,
                        "fitting_responses": fitting, "physical_constants": physical,
                        "numerical_sampling": metadata["sampling"], "root_policy": metadata["root_policy"],
                        "quantile_method": recipe["quantile_method"], "channel_policy": cfg.get("channel_policy", "source_defined"),
                        "naturalized_source": recipe["naturalized_source"],
                        "geometry_conversion": conversion if geometry_method else None}
            profile_id = f"{method}_{digest(identity)[:16]}"
            pdir = folder / profile_id
            pdir.mkdir()
            record = {**inv, **recipe, "profile_id": profile_id, "identity": identity,
                      "source_frame": "level_observations_si.csv" if geometry_method else "observations_si.csv",
                      "parameters_ready": False, "simulation_ready": False, "calibration_ready": False,
                      "calibration_readiness_reason": "New-recipe optimizer objectives/bounds/job identity not prepared in this delivery",
                      "experiment_kind": "conditional_controlled_parameterization_comparison",
                      "forcing_and_initialization_independence_verified": False,
                      "observation_basis": "mean_level_volume_proxy" if geometry_method and conversion and conversion["storage_is_mean_level_proxy"] else entry.get("observation_basis", "native_archive"),
                      "geometry_diagnostics": curve.diagnostics(physical["capacity_m3"]) if geometry_method and curve else None,
                      "new_optimizer_evaluations": 0}
            if target_frame is not None:
                target_frame.to_csv(pdir / "monthly_targets_si.csv", float_format="%.17g")
            if recipe["target_estimation_ready"]:
                try:
                    parameters = ReservoirParameters(target_frame, **{**physical, "channel_capacity_m3s": recipe["channel_capacity_m3s"]})
                    record["parameters_ready"] = True
                    record["simulation_ready"] = bool(np.isfinite(working.inflow_m3s).all())
                    if not record["simulation_ready"]:
                        record["reasons"].append("missing_model_inflow")
                    if (working.inflow_m3s.dropna() < 0).any() and metadata.get("route") != "signed_net_native_v1":
                        record["simulation_ready"] = False
                        record["reasons"].append("signed_forcing_requires_explicit_native_policy")
                    if metadata.get("route") == "signed_net_native_v1" and "forcing_policy" not in metadata:
                        record["simulation_ready"] = False
                        record["reasons"].append("missing_signed_net_forcing_policy")
                    w.write(pdir / "model_constants.json", {k: getattr(parameters, k) for k in physical})
                except ValueError as exc:
                    record["reasons"].append("physical_target_validation: " + str(exc))
            record["prepared_folder"] = pdir.relative_to(out).as_posix()
            w.write(pdir / "profile.json", record)
            profiles.append(record)
    for path, expected in upstream.items():
        if w.sha(path) != expected:
            raise ValueError("Source changed during preparation")
    w.write(out / "profile_registry.json", {"kind": "data_availability_profiles_v1", "profiles": profiles})
    pd.DataFrame(inventory).to_csv(out / "reservoir_inventory.csv", index=False)
    if catalog is not None:
        reference = Path(cfg["catalog"])
        source = resolve(config_path.parent, reference).parent / "cohort_summary.csv"
        if source.exists():
            shutil.copyfile(pin(source), out / "archived_cohort_summary.csv")
    prepared = {p.relative_to(out).as_posix(): w.sha(p) for p in out.rglob("*") if p.is_file()}
    w.write(out / "protocol.json", {"kind": "data_availability_campaign_v1", "source_sha256": sources(),
        "prepared_sha256": prepared, "upstream_sha256": upstream, "new_optimizer_evaluations": 0})
    report(out)
    return out


def _input_inventory(frame):
    result = {"observation_start": frame.index[0].date().isoformat(),
              "observation_end": frame.index[-1].date().isoformat(), "input_rows": len(frame)}
    for column in ("storage_m3", "release_m3s", "inflow_m3s"):
        values = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
        result[column + "_finite_rows"] = int(np.isfinite(values.to_numpy()).sum())
        result[column + "_negative_rows"] = int((values < 0).sum())
    return result


def verify(output):
    out = Path(output).resolve(strict=True)
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Campaign must be inside project outputs")
    protocol = w.read(out / "protocol.json")
    if protocol.get("kind") != "data_availability_campaign_v1":
        raise ValueError("Unknown campaign schema")
    for relative, expected in protocol["source_sha256"].items():
        if w.sha(ROOT / relative) != expected:
            raise ValueError("Scientific code changed; prepare a new campaign: " + relative)
    for relative, expected in protocol["prepared_sha256"].items():
        path = (out / relative).resolve(strict=True)
        if not path.is_relative_to(out) or w.sha(path) != expected:
            raise ValueError("Prepared input changed: " + relative)
    return out, w.read(out / "profile_registry.json")["profiles"]


def run_baselines(output, *, reservoir_id=None):
    out, profiles = verify(output)
    attempted, reused = 0, 0
    for record in profiles:
        identifier, pid = record["reservoir_id"], record["profile_id"]
        if reservoir_id is not None and identifier != reservoir_id:
            continue
        if not record["simulation_ready"]:
            continue
        job = out / "jobs" / identifier / pid
        job.mkdir(parents=True, exist_ok=True)
        receipt_path = job / "receipt.json"
        expected_identity = digest(record)
        if receipt_path.exists():
            receipt = w.read(receipt_path)
            if receipt["profile_identity"] != expected_identity:
                raise ValueError("Existing benchmark has a different profile identity")
            for name, expected in receipt.get("artifact_sha256", {}).items():
                if w.sha(job / name) != expected:
                    raise ValueError("Existing benchmark artifact changed")
            reused += 1
            continue
        lock = job / "run.lock"
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        started = time.monotonic()
        receipt = {"reservoir_id": identifier, "profile_id": pid, "method": record["method"],
                   "profile_identity": expected_identity, "new_optimizer_evaluations": 0,
                   "experiment_kind": record["experiment_kind"], "status": "failed", "artifact_sha256": {}}
        try:
            prepared = out / record["prepared_folder"]
            common = prepared.parent
            frame = read_frame(common / record["source_frame"])
            selection = YearSelection(**w.read(common / "selection.json"))
            roles = pd.Series(selection.roles, index=frame.index)
            scored = roles.isin(["calibration", "assessment"]).to_numpy()
            stop = np.flatnonzero(scored)[-1] + 1
            frame, roles = frame.iloc[:stop], roles.iloc[:stop]
            targets = pd.read_csv(prepared / "monthly_targets_si.csv", index_col="month", float_precision="round_trip")
            parameters = ReservoirParameters(targets, **w.read(prepared / "model_constants.json"))
            metadata = w.read(common / "numerical_profile.json")
            preprocessing = w.read(common / "preprocessing.json")
            result, residual = app.run_frame(frame, metadata, parameters, forcing_stage="prepared", preprocessing=preprocessing)
            original = read_frame(common / "observations_si.csv").iloc[:stop]
            # Parameter generation may use geometry-derived storage, but fleet
            # comparison must retain the same independently supplied response.
            has_supplied_storage = bool(original.storage_m3.notna().any())
            scored_frame = original if has_supplied_storage else frame
            basis = "native_archive" if has_supplied_storage else record["observation_basis"]
            metrics = score_native(scored_frame, result.frame, roles, observation_basis=basis)
            for values in metrics:
                values["storage_reference_kind"] = "provided_storage_common_reference" if has_supplied_storage else "geometry_proxy_reference" if record["storage_target_source"] == "level_geometry" else "no_storage_observations"
            result.frame.to_csv(job / "simulation.csv", index_label="date", float_format="%.17g")
            w.write(job / "metrics.json", metrics)
            receipt.update(status="completed", reason=None, maximum_independent_balance_residual_m3=residual,
                           simulated_rows=len(frame), unscored_trailing_context_rows=len(selection.dates) - stop,
                           artifact_sha256={name: w.sha(job / name) for name in ("simulation.csv", "metrics.json")})
        except (ValueError, RuntimeError) as exc:
            receipt["reason"] = str(exc)
        finally:
            receipt["elapsed_seconds"] = time.monotonic() - started
            w.write(receipt_path, receipt)
            lock.unlink()
        attempted += 1
        print(json.dumps({k: receipt[k] for k in ("reservoir_id", "method", "status")}), flush=True)
    report(out)
    return {"new_fixed_profile_attempts": attempted, "verified_jobs_reused": reused, "new_optimizer_evaluations": 0}


def report(output):
    out, profiles = verify(output)
    folder = out / "report"
    folder.mkdir(exist_ok=True)
    inventory, parameters, performance = [], [], []
    reference_path = out / "archived_cohort_summary.csv"
    reference = pd.read_csv(reference_path).set_index("reservoir_id") if reference_path.exists() else None
    for record in profiles:
        row = {k: record.get(k) for k in ("reservoir_id", "GRAND_ID", "name", "profile_id", "method",
            "existing_status", "target_estimation_ready", "parameters_ready", "simulation_ready", "calibration_ready",
            "storage_target_source", "release_target_source", "source_certified", "calibration_fraction", "observation_basis")}
        row["reasons"] = ";".join(record["reasons"])
        receipt_path = out / "jobs" / record["reservoir_id"] / record["profile_id"] / "receipt.json"
        receipt = w.read(receipt_path) if receipt_path.exists() else None
        row["baseline_status"] = receipt["status"] if receipt else "pending" if record["simulation_ready"] else "unavailable"
        row["baseline_reason"] = receipt.get("reason") if receipt else None
        inventory.append(row)
        exported = dict(row)
        exported["parameter_values_available"] = bool(record["parameters_ready"])
        if record["parameters_ready"]:
            targets = pd.read_csv(out / record["prepared_folder"] / "monthly_targets_si.csv", index_col="month", float_precision="round_trip")
            for month in range(1, 13):
                for column in MONTHLY_TARGET_COLUMNS:
                    exported[f"{column}_month_{month:02d}"] = targets.loc[month, column]
        parameters.append(exported)
        if receipt and receipt["status"] == "completed":
            metrics = w.read(receipt_path.parent / "metrics.json")
            for phase in metrics:
                score = {**row, **phase, "maximum_independent_balance_residual_m3": receipt["maximum_independent_balance_residual_m3"]}
                if reference is not None and record["reservoir_id"] in reference.index:
                    old = reference.loc[record["reservoir_id"]]
                    for variable in ("storage", "release"):
                        if variable == "storage" and phase.get("storage_reference_kind") != "provided_storage_common_reference":
                            continue
                        value = old[f"{phase['phase']}_{variable}_nse"]
                        score[f"archived_calibrated_{variable}_nse"] = value
                        current = phase[variable + "_nse"]
                        score[f"baseline_minus_archived_calibrated_{variable}_nse"] = None if current is None else current - value
                performance.append(score)
    frame = pd.DataFrame(inventory)
    frame.to_csv(folder / "profile_inventory.csv", index=False)
    pd.DataFrame(parameters).to_csv(folder / "reservoir_profiles_by_GRAND_ID.csv", index=False, float_format="%.17g")
    pd.DataFrame(performance).to_csv(folder / "performance_comparison.csv", index=False, float_format="%.17g")
    summary = {"reservoirs": int(frame.reservoir_id.nunique()), "recipe_profiles": len(frame),
               "methods": {m: {"profiles": len(g), "valid_parameters": int(g.parameters_ready.sum()),
                                  "completed_baselines": int(g.baseline_status.eq("completed").sum()),
                                  "failed_baselines": int(g.baseline_status.eq("failed").sum()),
                                  "pending_baselines": int(g.baseline_status.eq("pending").sum())}
                           for m, g in frame.groupby("method", sort=False)},
               "performance_rows": len(performance), "new_optimizer_evaluations": 0,
               "new_searches_prepared": 0, "existing_calibrations_modified": False,
               "experiment_kind": "conditional_controlled_parameterization_comparison"}
    assessment_summaries = {}
    if performance:
        values = pd.DataFrame(performance)
        held = values.loc[values.phase.eq("assessment")]
        for method, group in held.groupby("method", sort=False):
            item = {"completed_reservoirs": len(group),
                    "median_release_nse": float(group.release_nse.median()),
                    "median_storage_nse": float(group.storage_nse.median()),
                    "median_release_sqrt_nse": float(group.release_sqrt_nse.median())}
            s, q = "baseline_minus_archived_calibrated_storage_nse", "baseline_minus_archived_calibrated_release_nse"
            if s in group and q in group:
                paired = group[[s, q]].dropna()
                item.update(paired_comparisons_with_archived=len(paired),
                            both_better_than_archived=int(((paired[s] > 0) & (paired[q] > 0)).sum()),
                            both_worse_than_archived=int(((paired[s] < 0) & (paired[q] < 0)).sum()),
                            tradeoff_or_tie=int((~(((paired[s] > 0) & (paired[q] > 0)) | ((paired[s] < 0) & (paired[q] < 0)))).sum()))
            assessment_summaries[method] = item
    summary["held_out_summary_on_completed_subsets"] = assessment_summaries
    w.write(folder / "summary.json", summary)
    lines = ["# Data-availability preparation and fixed-profile baselines", "", "This is a new versioned campaign. Existing calibrations remain unchanged.", "",
             f"Inventory: **{summary['reservoirs']} reservoirs**, **{len(frame)} recipe profiles**. No optimizer evaluation was run.", "",
             "| Method | Valid physical targets | Baselines complete | Baselines failed | Pending |", "|---|---:|---:|---:|---:|"]
    for method, counts in summary["methods"].items():
        lines.append(f"| {method} | {counts['valid_parameters']} | {counts['completed_baselines']} | {counts['failed_baselines']} | {counts['pending_baselines']} |")
    if assessment_summaries:
        lines += ["", "| Method | Completed held-out cases | Median release NSE | Median storage NSE | Median sqrt-release NSE |",
                  "|---|---:|---:|---:|---:|"]
        for method, stats in assessment_summaries.items():
            lines.append(f"| {method} | {stats['completed_reservoirs']} | {stats['median_release_nse']:.3f} | {stats['median_storage_nse']:.3f} | {stats['median_release_sqrt_nse']:.3f} |")
        lines += ["", "These medians use each method's completed subset; they are descriptive and do not rank methods on a common fleet. Use the paired per-reservoir comparisons and retain all failures."]
    lines += ["", "Missing naturalized outflow is never replaced by inflow. Geometry availability does not imply levels or a datum reference are available. Source-gated and conditional interval records retain their status.", "",
              "Baselines use frozen forcing, initial states, sampling and root policies. This is a conditional comparison of target recipes, not a fully ungauged deployment experiment. Original observations remain the common assessment reference; proxy comparisons are labeled.", "",
              "Physical target failures and root/forcing failures remain explicit. No capacity, bound or root policy is changed to make a baseline complete. Invalid simulated values cannot drop scored days.", "",
              "`calibration_ready=false` means new-recipe objective/bounds/checkpoint preparation is still required; target generation and fixed-profile simulation do not constitute a new calibration.", "",
              "See [profile inventory](profile_inventory.csv), [parameter profiles by GRanD ID](reservoir_profiles_by_GRAND_ID.csv), [performance comparison](performance_comparison.csv), [reservoir inventory](../reservoir_inventory.csv), and [protocol](../protocol.json).",
              "", "NSE differences are fixed baseline minus archived calibrated profile, on common native observation support. They are not gains from new optimization. Undefined scores remain blank.", ""]
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")
    w.write(folder / "manifest.json", {"artifact_sha256": {p.name: w.sha(p) for p in folder.iterdir() if p.is_file() and p.name != "manifest.json"}})
    return summary
