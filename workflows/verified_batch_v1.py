"""Import compatible, completed paper calibrations without new model evaluations.

Original experiment identities, numerical routes and source gates are retained.
The batch configuration is an explicit compatibility contract, not a request to
relabel old optimizer checkpoints as new searches.
"""
from datetime import datetime, timezone
from dataclasses import asdict
import copy
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import pandas as pd

from dztr.calibration.bounds import CalibrationBounds
from dztr.calibration.candidates import parameters_from_vector, parameters_to_vector
from dztr.calibration.checkpoint import load_optimizer_checkpoint
from dztr.calibration.periods import DatePeriod
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters
from strategies.representative_years import YearSelection
from strategies.strategy_comparison import pooled_choice
from workflows import profile_application_v1 as app
from workflows import reservoir_v2 as w
from workflows.geometry_v1 import digest

ROOT = w.ROOT
REPORT_FILES = ("cohort_summary.csv", "seed_baseline_comparison.csv", "checkpoint_diagnostics.csv",
                "seed_front_agreement.csv", "seasonal_bias.csv", "zone_diagnostics.csv", "regime_diagnostics.csv")
MODEL_KEYS = ("physical_constants", "sampling", "root_policy", "atol_storage_m3")
INPUT_FILES = ("observations_csv", "selection_json", "model_config", "bounds_csv", "initial_targets_csv")
REQUIREMENTS = {"scoring": "pooled", "year_selection": "mixed", "quantile_method": "hazen",
                "minimum_calibration_fraction": .65, "population": 32, "evaluation_budget": 512, "seeds": [42, 43, 44]}


def sources():
    return {**app.application_sources(), **{p: w.sha(ROOT / p) for p in
        ("workflows/verified_batch_v1.py", "tools/dztr_run_v3.py")}}


def csv(path, dates=False):
    return pd.read_csv(path, index_col="date" if dates else None, parse_dates=dates, float_precision="round_trip")


class Source:
    """One verified read snapshot with shared registry/hash caches."""
    def __init__(self, catalog_path, assessment_path):
        self.hashes, self.registries = {}, {}
        self.catalog_path = Path(catalog_path).resolve(strict=True)
        self.assessment = Path(assessment_path).resolve(strict=True)
        if self.catalog_path.parent != self.assessment:
            raise ValueError("Use the catalog and assessment from the same verified cohort release")
        self.catalog = app.read_catalog(self.catalog_path)
        proof = w.read(self.pin(self.assessment / "manifest.json"))
        for relative, expected in proof["artifact_sha256"].items():
            path = (self.assessment / relative).resolve(strict=True)
            if not path.is_relative_to(self.assessment) or self.pin_hash(path) != expected:
                raise ValueError("Verified cohort report artifact changed")
        self.verification = w.read(self.assessment / "verification.json")
        if self.verification.get("status") != "passed" or self.catalog.get("selection_uses_assessment") is not False:
            raise ValueError("Source cohort has no passing calibration-frozen verification")
        for relative, expected in self.verification["consumed_sha256"].items():
            if self.pin_hash(app.reference(relative)) != expected:
                raise ValueError("Original experiment input/checkpoint changed: " + relative)
        for relative, expected in self.verification["report_source_sha256"].items():
            if self.pin_hash(app.reference(relative)) != expected:
                raise ValueError("Original assessment scientific source changed")
        for relative, expected in self.catalog["application_source_sha256"].items():
            if self.pin_hash(app.reference(relative)) != expected:
                raise ValueError("Original numerical source changed")
        self.table_path = app.reference(self.catalog["source_table"])
        if self.pin_hash(self.table_path) != self.catalog["source_table_sha256"]:
            raise ValueError("Original all-reservoir parameter table changed")
        self.table = csv(self.table_path)
        self.entries = {r["reservoir_id"]: r for r in self.catalog["reservoirs"]}
        if len(self.entries) != len(self.catalog["reservoirs"]):
            raise ValueError("Source catalog contains duplicate reservoir IDs")
        gids = [r["GRAND_ID"] for r in self.entries.values() if r.get("GRAND_ID") is not None]
        if len(gids) != len(set(gids)):
            raise ValueError("Source catalog contains ambiguous GRanD aliases")

    def pin_hash(self, path):
        path = Path(path).resolve(strict=True)
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError("Batch source files must be inside this repository")
        if str(path) not in self.hashes:
            self.hashes[str(path)] = w.sha(path)
        return self.hashes[str(path)]

    def pin(self, path):
        self.pin_hash(path)
        return Path(path).resolve(strict=True)

    def describe(self, identifier):
        entry = self.entries[identifier]
        for relative, expected in entry["input_sha256"].items():
            if self.pin_hash(app.reference(relative)) != expected:
                raise ValueError("Catalog input changed")
        registry_path = app.reference(entry["registry"])
        key = str(registry_path)
        if key not in self.registries:
            self.registries[key] = w.verify_registry(registry_path)
            self.pin(registry_path.parent / "manifest.json")
        registry = self.registries[key]
        parameters_path = (registry_path.parent / registry["profiles_by_reservoir_id"][identifier]).resolve(strict=True)
        if not parameters_path.is_relative_to(registry_path.parent):
            raise ValueError("Profile escapes its original release")
        metadata = w.read(self.pin(parameters_path))
        target_path = (parameters_path.parent / metadata["monthly_targets_file"]).resolve(strict=True)
        if not target_path.is_relative_to(parameters_path.parent) or self.pin_hash(target_path) != metadata["monthly_targets_sha256"]:
            raise ValueError("Selected monthly target file changed")
        targets = pd.read_csv(target_path, index_col="month", float_precision="round_trip")
        parameters = ReservoirParameters(targets, **metadata["physical_constants"])
        if metadata["reservoir_id"] != identifier or metadata.get("selection_uses_assessment") is not False:
            raise ValueError("Profile identity or calibration-only selection is invalid")
        prepared = app.reference(entry["prepared_forcing"]).parent
        legacy = metadata.get("source_kind") == "verified_comparison_archive"
        archive = prepared.parents[2] if legacy else registry_path.parent.parent.parent
        jobs = archive / "jobs" / identifier
        if legacy:
            jobs /= "mixed_pooled"
        model_paths = [prepared / p for p in ("config.json", "model_config.json", "model.json") if (prepared / p).exists()]
        if len(model_paths) != 1:
            raise ValueError("Prepared model configuration is missing or ambiguous")
        inputs = {"observations_csv": prepared / "observations_si.csv", "selection_json": prepared / "selection.json",
                  "model_config": model_paths[0], "bounds_csv": prepared / "bounds_si.csv",
                  "initial_targets_csv": prepared / "initial_targets_si.csv"}
        for path in inputs.values(): self.pin(path)
        return {"entry": entry, "registry_path": registry_path, "parameters_path": parameters_path,
                "target_path": target_path, "metadata": metadata, "parameters": parameters,
                "prepared": prepared, "jobs": jobs, "legacy": legacy, "inputs": inputs}


def expected_model(description):
    metadata = description["metadata"]
    return {**{k: metadata[k] for k in MODEL_KEYS}, "forcing_policy": metadata.get("forcing_policy"),
            "inflow_preprocessing": description["entry"]["inflow_preprocessing"]}


def make_config(catalog_path, assessment_path, identifiers=None):
    source = Source(catalog_path, assessment_path)
    chosen = list(source.entries) if identifiers is None else list(identifiers)
    if not chosen or len(chosen) != len(set(chosen)) or set(chosen) - set(source.entries):
        raise ValueError("Choose unique reservoir IDs from the verified source cohort")
    records = []
    for identifier in chosen:
        entry = source.entries[identifier]
        record = {"reservoir_id": identifier, "GRAND_ID": entry.get("GRAND_ID"), "route": entry["route"],
                  "action": "import_verified" if entry["monthly_profile_available"] else "retain_deferred"}
        if entry["monthly_profile_available"]:
            description = source.describe(identifier)
            record.update(parameterization_method="observed_monthly_calibration", input_stage="prepared",
                inputs={k: {"path": p.relative_to(ROOT).as_posix(), "sha256": source.pin_hash(p)}
                        for k, p in description["inputs"].items()}, expected_model=expected_model(description))
        records.append(record)
    return {"kind": "dztr_verified_batch_v1", "schema_version": 1, "path_base": "repository",
            "catalog": source.catalog_path.relative_to(ROOT).as_posix(),
            "assessment": source.assessment.relative_to(ROOT).as_posix(),
            "scope": "all_paper" if identifiers is None else "explicit", "requirements": copy.deepcopy(REQUIREMENTS),
            "reservoirs": records}


def load_config(path):
    path = Path(path).resolve(strict=True)
    cfg = w.read(path)
    if set(cfg) != {"kind", "schema_version", "path_base", "catalog", "assessment", "scope", "requirements", "reservoirs"}:
        raise ValueError("Unknown batch configuration fields")
    if (cfg["kind"], cfg["schema_version"], cfg["path_base"]) != ("dztr_verified_batch_v1", 1, "repository"):
        raise ValueError("Unknown verified-batch configuration")
    if cfg["scope"] not in {"all_paper", "explicit"} or cfg["requirements"] != REQUIREMENTS:
        raise ValueError("This import protocol retains mixed/pooled/Hazen, 65% minimum, seeds 42/43/44 and 32x512 searches")
    identifiers = [r["reservoir_id"] for r in cfg["reservoirs"]]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("Batch needs unique reservoir entries")
    for r in cfg["reservoirs"]:
        if set(r) - {"reservoir_id", "GRAND_ID", "route", "action", "parameterization_method", "input_stage", "inputs", "expected_model"}:
            raise ValueError("Unknown reservoir import fields")
        if r["action"] not in {"import_verified", "retain_deferred"}:
            raise ValueError("Choose verified import or retain deferred")
    return path, cfg


def compatibility(request, description, source):
    reasons = []
    if request.get("parameterization_method") != "observed_monthly_calibration" or request.get("input_stage") != "prepared":
        reasons.append("method_or_input_stage_differs")
    if request.get("expected_model") != expected_model(description):
        reasons.append("physical_numerical_or_forcing_settings_differ")
    if set(request.get("inputs", {})) != set(INPUT_FILES):
        reasons.append("prepared_input_contract_incomplete")
    else:
        for name, info in request["inputs"].items():
            if set(info) != {"path", "sha256"}:
                raise ValueError("Prepared input needs path and sha256")
            try:
                actual = source.pin_hash(app.reference(info["path"]))
            except (FileNotFoundError, ValueError):
                reasons.append(name + "_unavailable")
                continue
            if actual != info["sha256"] or actual != source.pin_hash(description["inputs"][name]):
                reasons.append(name + "_differs_from_original_experiment")
    return reasons


def _plan(path):
    path, cfg = load_config(path)
    source = Source(app.reference(cfg["catalog"]), app.reference(cfg["assessment"]))
    source.pin(path)
    requested = {r["reservoir_id"] for r in cfg["reservoirs"]}
    if requested - set(source.entries) or (cfg["scope"] == "all_paper" and requested != set(source.entries)):
        raise ValueError("Batch inventory does not match its declared paper scope")
    stages, descriptions = [], {}
    for request in cfg["reservoirs"]:
        identifier = request["reservoir_id"]
        entry = source.entries[identifier]
        if request["GRAND_ID"] != entry.get("GRAND_ID") or request["route"] != entry["route"]:
            raise ValueError("Reservoir ID/GRanD mapping or source route changed")
        expected_action = "import_verified" if entry["monthly_profile_available"] else "retain_deferred"
        if request["action"] != expected_action:
            raise ValueError("Source-gated profiles cannot be promoted or completed profiles silently dropped")
        description = source.describe(identifier) if entry["monthly_profile_available"] else None
        if description:
            descriptions[identifier] = description
        reasons = compatibility(request, description, source) if description else [entry["status"], entry["next_action"]]
        stages.append({"id": identifier, "type": request["action"], "route": entry["route"],
                       "planned_status": "incompatible" if description and reasons else "importable" if description else "deferred",
                       "reasons": reasons})
    body = {"kind": "dztr_verified_batch_plan_v1", "config": cfg, "stages": stages,
            "input_sha256": source.hashes, "code_sha256": sources(),
            "new_optimizer_evaluations": 0, "new_simulations": 0}
    return {**body, "signature": digest(body)}, source, descriptions


def plan(path):
    design, _, _ = _plan(path)
    return design


def verify_source_hashes(design):
    for path, expected in design["input_sha256"].items():
        if w.sha(path) != expected: raise ValueError("Batch source changed during execution: " + path)
    for path, expected in design["code_sha256"].items():
        if w.sha(ROOT / path) != expected: raise ValueError("Batch import source changed: " + path)


def write_status(out, state):
    """Retry transient Windows scanner/reader locks on the atomic status replace."""
    for attempt in range(5):
        try:
            w.write(Path(out) / "status.json", state)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32) or attempt == 4:
                raise
            time.sleep(.1)


def verify_searches(description):
    """Compare original checkpoint arrays, bounds, initializer and frozen choice."""
    metadata, parameters = description["metadata"], description["parameters"]
    selection = YearSelection(**w.read(description["inputs"]["selection_json"]))
    frame = csv(description["inputs"]["observations_csv"], True)
    calibration = selection.mask(frame.index, "calibration")
    if selection.signature != metadata["selection_sha256"] or calibration.mean() < .65:
        raise ValueError("Original profile and fitting-year mask disagree")
    if metadata["scoring"] != "pooled" or metadata["year_selection"] != "mixed" or metadata["evaluation_budget"] != 512:
        raise ValueError("Original split/objective/budget is incompatible")
    model = w.read(description["inputs"]["model_config"])
    for key in MODEL_KEYS:
        if model[key] != metadata[key]: raise ValueError("Original model and selected profile disagree: " + key)
    if model.get("quantile_method", "hazen") != "hazen" or model.get("cost_mode", "pooled_nse") != "pooled_nse":
        raise ValueError("Original percentile estimator/objective differs")
    if model.get("forcing_policy") != metadata.get("forcing_policy"):
        raise ValueError("Original signed-forcing policy differs")
    initial = ReservoirParameters(pd.read_csv(description["inputs"]["initial_targets_csv"], index_col="month", float_precision="round_trip"), **metadata["physical_constants"])
    monthly = pd.read_csv(description["inputs"]["bounds_csv"], index_col="month", float_precision="round_trip")
    dates = frame.index[calibration]
    bounds = CalibrationBounds(monthly, parameters.capacity_m3, parameters.dead_storage_fraction, DatePeriod(dates[0], dates[-1]), "hazen")
    records, fronts, identities = [], [], []
    for seed in REQUIREMENTS["seeds"]:
        job = description["jobs"] / f"seed_{seed}"
        snapshot = job if description["legacy"] else job / "budget_512"
        proof_path = job / "completion_manifest.json" if description["legacy"] else snapshot / "manifest.json"
        for relative, expected in w.read(proof_path)["artifact_sha256"].items():
            target = (snapshot / relative).resolve(strict=True)
            if not target.is_relative_to(snapshot) or w.sha(target) != expected:
                raise ValueError("Original search snapshot changed")
        identity_path = job / ("checkpoint_identity.json" if description["legacy"] else "identity.json")
        identity = w.read(identity_path)
        objective = identity["objective"]
        if objective["selection_sha256"] != selection.signature or objective["cost_mode"] != "pooled_nse" or objective["root_policy"] != metadata["root_policy"] or objective["atol_storage_m3"] != metadata["atol_storage_m3"]:
            raise ValueError("Checkpoint objective/year mask/numerics differ")
        if objective.get("forcing_policy") != metadata.get("forcing_policy"):
            raise ValueError("Checkpoint forcing interpretation differs")
        state = load_optimizer_checkpoint(job / "checkpoint.npz", identity)
        if state.settings.seed != seed or state.settings.population_size != 32 or state.settings.evaluation_budget != 512 or len(state.evaluated) != 512:
            raise ValueError("Incomplete or incompatible original search")
        np.testing.assert_array_equal(state.lower_vector, bounds.lower_vector)
        np.testing.assert_array_equal(state.upper_vector, bounds.upper_vector)
        np.testing.assert_array_equal(state.baseline_vector, parameters_to_vector(initial))
        trials = csv(snapshot / "trials.csv")
        vectors = pd.read_csv(snapshot / "vectors.csv", index_col="evaluation_id", float_precision="round_trip")
        np.testing.assert_array_equal(vectors.to_numpy(), [r.vector for r in state.evaluated])
        np.testing.assert_array_equal(trials[["release_cost", "storage_cost"]], [r.objectives for r in state.evaluated])
        front = csv(snapshot / ("pareto_front.csv" if description["legacy"] else "front.csv"))
        if not front.evaluation_id.isin(trials.evaluation_id).all(): raise ValueError("Unknown front evaluation")
        trial_costs = trials.set_index("evaluation_id").loc[front.evaluation_id, ["release_cost", "storage_cost"]]
        np.testing.assert_array_equal(front[["release_cost", "storage_cost"]], trial_costs)
        front["seed"] = seed; fronts.append(front)
        if seed == metadata["seed"]:
            replayed = parameters_from_vector(vectors.loc[metadata["evaluation_id"]].to_numpy(), parameters)
            np.testing.assert_array_equal(replayed.monthly_targets, parameters.monthly_targets)
        records.append({"seed": seed, "population": 32, "evaluations": 512, "original_identity_file": str(identity_path),
                        "original_optimizer_settings": asdict(state.settings),
                        "original_identity_sha256": w.sha(identity_path), "checkpoint_sha256": w.sha(job / "checkpoint.npz"),
                        "checkpoint_copied_or_reidentified": False})
        identities.append(identity_path)
    choice, _ = pooled_choice(fronts)
    for key in ("seed", "evaluation_id", "release_cost", "storage_cost"):
        if choice[key] != metadata[key]: raise ValueError("Frozen selection does not reproduce calibration-only choice")
    return {"checkpoint_records": records, "selection_sha256": selection.signature,
            "calibration_fraction": float(calibration.mean()), "bounds_and_initial_vector_exact": True,
            "calibration_choice_reproduced": True, "selection_uses_assessment": False}, identities


def copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    expected = w.sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if w.sha(destination) != expected: raise ValueError("Partial/imported artifact differs from its original source")
        return
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    shutil.copyfile(source, temporary)
    if w.sha(temporary) != expected: raise ValueError("Source changed while being copied")
    temporary.replace(destination)


def import_one(out, stage, description, design):
    identifier = stage["id"]
    result = {"reservoir_id": identifier, "route": stage["route"], "reasons": stage["reasons"],
              "new_optimizer_evaluations": 0, "new_simulations": 0, "source_certified": False}
    files = []
    if stage["planned_status"] == "importable":
        evidence, identities = verify_searches(description)
        result.update(evidence, import_status="imported_verified", original_registry=str(description["registry_path"]),
                      original_source_kind=description["metadata"].get("source_kind"),
                      original_selected_seed=description["metadata"]["seed"], original_evaluation_id=description["metadata"]["evaluation_id"])
        profile = out / "release/profiles" / identifier
        copies = [(description["parameters_path"], profile / "parameters.json"),
                  (description["target_path"], profile / description["metadata"]["monthly_targets_file"])]
        prepared = out / "release/prepared" / identifier
        copies += [(p, prepared / {"observations_csv": "observations_si.csv", "selection_json": "selection.json",
                                   "model_config": "model_config.json", "bounds_csv": "bounds_si.csv", "initial_targets_csv": "initial_targets_si.csv"}[k])
                   for k, p in description["inputs"].items()]
        copies += [(p, out / "release/identities" / identifier / f"seed_{seed}.json")
                   for seed, p in zip(REQUIREMENTS["seeds"], identities)]
        for source, target in copies:
            copy_verified(source, target); files.append(target)
    else:
        result["import_status"] = "incompatible" if stage["planned_status"] == "incompatible" else "retained_deferred"
    folder = out / "jobs" / identifier; folder.mkdir(parents=True, exist_ok=True)
    receipt = {"plan_signature": design["signature"], "result": result,
               "artifact_sha256": {str(p.relative_to(out)): w.sha(p) for p in files}}
    w.write(folder / "receipt.json", receipt)
    return result


def comparison_table(summary, seeds):
    rows = []
    for record in summary.to_dict("records"):
        for phase in ("calibration", "assessment"):
            row = {"reservoir_id": record["reservoir_id"], "GRAND_ID": record["GRAND_ID"], "role": "selected",
                   "seed": record["seed"], "evaluation_id": record["evaluation_id"], "period": phase,
                   "baseline_kind": record["baseline_kind"], "baseline_reason": record["baseline_reason"], "selection_uses_assessment": False}
            for variable in ("storage", "release"):
                row[variable + "_nse"] = record[phase + "_" + variable + "_nse"]
                row["baseline_" + variable + "_nse"] = record[phase + "_baseline_" + variable + "_nse"]
                row[variable + "_nse_gain"] = record[phase + "_" + variable + "_nse_gain"]
            rows.append(row)
    by_id = summary.set_index("reservoir_id")
    for record in seeds.to_dict("records"):
        row = dict(record, role="seed", selection_uses_assessment=False, GRAND_ID=by_id.loc[record["reservoir_id"], "GRAND_ID"])
        for variable in ("storage", "release"):
            row["baseline_" + variable + "_nse"] = by_id.loc[record["reservoir_id"], record["period"] + "_baseline_" + variable + "_nse"]
        rows.append(row)
    columns = ["reservoir_id", "GRAND_ID", "role", "seed", "evaluation_id", "period", "baseline_kind", "baseline_reason",
               "assessment_feasible", "selection_uses_assessment"]
    columns += [name for v in ("storage", "release") for name in (v + "_nse", "baseline_" + v + "_nse", v + "_nse_gain")]
    return pd.DataFrame(rows, columns=columns)


def release(out, source, results, design):
    folder = out / "release"; folder.mkdir(exist_ok=True)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        w.verify_registry(folder / "registry.json")
        return
    lookup = {r["reservoir_id"]: r for r in results}
    imported = {i for i, r in lookup.items() if r["import_status"] == "imported_verified"}
    registry = {"schema_version": 1, "kind": "unified_dztr_profiles", "parameter_status": "exploratory",
                "selection_uses_assessment": False, "profiles_by_reservoir_id": {}, "profiles_by_GRAND_ID": {},
                "table": "reservoir_parameterization_by_GRAND_ID.csv",
                "provenance": {"kind": "verified_calibration_import_v1", "plan_signature": design["signature"],
                               "original_experiment_identities_preserved": True, "new_optimizer_evaluations": 0}}
    application = {"kind": "paper_profile_application_v1", "source_table": (folder / registry["table"]).relative_to(ROOT).as_posix(),
                   "application_source_sha256": sources(), "reservoirs": [], "selection_uses_assessment": False, "source_certified": False}
    status_rows = []
    table = source.table.loc[source.table.reservoir_id.isin(lookup)].copy()
    table["original_profile_registry"] = table.profile_registry
    table["batch_import_status"] = table.reservoir_id.map({i: r["import_status"] for i, r in lookup.items()})
    for identifier, result in lookup.items():
        entry = dict(source.entries[identifier])
        row_mask = table.reservoir_id.eq(identifier)
        row = {"reservoir_id": identifier, "name": entry["name"], "GRAND_ID": entry.get("GRAND_ID"), "route": entry["route"],
               "import_status": result["import_status"], "original_status": entry["status"], "next_action": entry["next_action"],
               "reasons": ";".join(result["reasons"]), "parameter_values_available": identifier in imported,
               "simulation_ready": identifier in imported, "source_certified": False, "convergence_established": False}
        if identifier in imported:
            relative = f"profiles/{identifier}/parameters.json"
            registry["profiles_by_reservoir_id"][identifier] = relative
            if entry.get("GRAND_ID") is not None:
                registry["profiles_by_GRAND_ID"][str(entry["GRAND_ID"])] = relative
            entry.update(registry=(folder / "registry.json").relative_to(ROOT).as_posix(),
                         prepared_forcing=(folder / "prepared" / identifier / "observations_si.csv").relative_to(ROOT).as_posix())
            entry["input_sha256"] = {str(p.relative_to(ROOT)): w.sha(p) for p in
                (folder / "prepared" / identifier / "observations_si.csv", folder / "prepared" / identifier / "selection.json")}
            entry["status"] = "imported_verified_exploratory_calibration"
            table.loc[row_mask, "profile_registry"] = entry["registry"]
            row.update(seed=result["original_selected_seed"], evaluation_id=result["original_evaluation_id"],
                       calibration_fraction=result["calibration_fraction"])
        else:
            entry["monthly_profile_available"] = False
            table.loc[row_mask, "parameter_values_available"] = False
            for month in range(1, 13):
                for column in MONTHLY_TARGET_COLUMNS:
                    table.loc[row_mask, f"{column}_month_{month:02d}"] = np.nan
            if result["import_status"] == "incompatible":
                entry["status"] = "batch_contract_incompatible"
                entry["next_action"] = ";".join(result["reasons"])
        application["reservoirs"].append(entry)
        status_rows.append(row)
    table["GRAND_ID"] = table["GRAND_ID"].astype("Int64")
    table.to_csv(folder / registry["table"], index=False, float_format="%.17g")
    application["source_table_sha256"] = w.sha(folder / registry["table"])
    w.write(folder / "catalog.json", application)
    w.write(folder / "registry.json", registry)
    report = folder / "report"; report.mkdir(exist_ok=True)
    status_frame = pd.DataFrame(status_rows)
    status_frame.to_csv(report / "reservoir_status.csv", index=False)
    for name in REPORT_FILES:
        frame = csv(source.assessment / name)
        frame = frame.loc[frame.reservoir_id.isin(imported)]
        frame.to_csv(report / name, index=False, float_format="%.17g")
    selected = csv(report / "cohort_summary.csv")
    seeds = csv(report / "seed_baseline_comparison.csv")
    comparison_table(selected, seeds).to_csv(report / "baseline_comparison_all_seeds_and_selected.csv", index=False, float_format="%.17g")
    outcomes = selected.assessment_baseline_effect.value_counts().to_dict()
    summary = {"inventory_reservoirs": len(lookup), "imported_profiles": len(imported),
        "imported_standard_profiles": sum(r["route"] == "standard_native_v1" and r["import_status"] == "imported_verified" for r in results),
        "imported_signed_profiles": sum(r["route"] == "signed_net_native_v1" and r["import_status"] == "imported_verified" for r in results),
        "retained_deferred": sum(r["import_status"] == "retained_deferred" for r in results),
        "separate_interval_studies": sum(r["import_status"] == "retained_deferred" and r["route"] == "segmented_conditional_interval_v1" for r in results),
        "source_gated": sum(r["import_status"] == "retained_deferred" and r["route"] != "segmented_conditional_interval_v1" for r in results),
        "incompatible": sum(r["import_status"] == "incompatible" for r in results),
        "verified_seed_searches": sum(len(r.get("checkpoint_records", [])) for r in results),
        "cached_evaluations_verified": sum(sum(s["evaluations"] for s in r.get("checkpoint_records", [])) for r in results),
        "parameter_values_preserved": len(imported) * 72, "selected_and_seed_phase_comparisons": len(selected) * 8,
        "assessment_baseline_outcomes": outcomes, "new_optimizer_evaluations": 0, "new_simulations": 0,
        "original_experiment_identities_preserved": True, "source_certified": False, "convergence_established": False}
    w.write(report / "summary.json", summary)
    lines = ["# Verified batch rollout", "", f"**{len(lookup)} reservoirs retained; {len(imported)} completed profiles imported.**", "",
      f"Standard: {summary['imported_standard_profiles']}; signed-inflow: {summary['imported_signed_profiles']}; deferred: {summary['retained_deferred']}; incompatible: {summary['incompatible']}.", "",
      f"Verified {summary['verified_seed_searches']} original seed searches and {summary['cached_evaluations_verified']:,} cached evaluations. **No new optimization or model simulation was run.**", "",
      "Imports preserve original profiles, selected seeds/evaluations, checkpoint identities, bounds, year masks, preprocessing and solver policies. Signed inflows retain their explicit net-boundary interpretation. A verified import does not certify the source definitions or establish convergence.", "",
      "Held-out baseline outcomes: " + ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in outcomes.items()) + ".", "",
      "The baseline may be a generalized estimate or a documented feasible initializer. Unavailable baseline NSE/gains remain blank. Selected and per-seed metrics are copied from the verified assessment, without parameter reselection.", "",
      "[All-reservoir parameter export](reservoir_parameterization_by_GRAND_ID.csv), [status](report/reservoir_status.csv), [selected and all-seed baseline comparisons](report/baseline_comparison_all_seeds_and_selected.csv), [catalog](catalog.json), [registry](registry.json).", "",
      "| Reservoir | Route | Import status | Next action / source reason |", "|---|---|---|---|"]
    for row in status_rows:
        action = row["next_action"] if not row["parameter_values_available"] else "Retain documented assumptions and assess suitability before application."
        lines.append(f"| {row['name']} | {row['route']} | {row['import_status']} | {action} |")
    lines += ["", "Missing GRanD IDs never remove a row. Bull Lake retains its separate conditional interval study. Other deferred cases retain their source decisions; no monthly parameters are invented.", "",
              "[Seasonal diagnostics](report/seasonal_bias.csv), [zone diagnostics](report/zone_diagnostics.csv), [regime diagnostics](report/regime_diagnostics.csv), [checkpoint evidence](report/checkpoint_diagnostics.csv), [seed agreement](report/seed_front_agreement.csv).", ""]
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")
    w.write(manifest_path, {"code_sha256": sources(), "artifact_sha256": {
        p.relative_to(folder).as_posix(): w.sha(p) for p in folder.rglob("*") if p.is_file() and p.name != "manifest.json"}})
    w.verify_registry(folder / "registry.json")
    app.read_catalog(folder / "catalog.json")


def run(path, output, *, resume=False, until=None):
    design, source, descriptions = _plan(path)
    out = Path(output).resolve()
    if not out.is_relative_to(ROOT / "outputs") or out == ROOT / "outputs":
        raise ValueError("Use a batch output inside project outputs")
    if until is not None and until not in [s["id"] for s in design["stages"]]:
        raise ValueError("--until must name a configured reservoir ID")
    if out.exists():
        if not resume or w.read(out / "plan.json") != design:
            raise ValueError("Existing batch needs --resume with unchanged inputs, settings and code")
    else:
        out = w.fresh_output(out); w.write(out / "plan.json", design)
    lock = out / "manager.lock"
    with lock.open("x", encoding="utf-8") as stream: stream.write(str(os.getpid()))
    state = {"status": "running", "signature": design["signature"], "new_optimizer_evaluations": 0,
             "new_simulations": 0, "stages": [{"id": s["id"], "type": s["type"], "status": "pending"} for s in design["stages"]]}
    results = []
    try:
        for stage, row in zip(design["stages"], state["stages"]):
            row["status"] = "running"
            receipt_path = out / "jobs" / stage["id"] / "receipt.json"
            if receipt_path.exists():
                receipt = w.read(receipt_path)
                if receipt["plan_signature"] != design["signature"]: raise ValueError("Import receipt identity changed")
                for relative, expected in receipt["artifact_sha256"].items():
                    if w.sha(out / relative) != expected: raise ValueError("Imported profile artifact changed")
                result = receipt["result"]; row["status"] = "reused"
            else:
                write_status(out, state)
                result = import_one(out, stage, descriptions.get(stage["id"]), design)
                row["status"] = result["import_status"]
            results.append(result); row["result_status"] = result["import_status"]
            print(f"{stage['id']}: {row['status']}", flush=True)
            if stage["id"] == until: break
        verify_source_hashes(design)
        if len(results) == len(design["stages"]):
            release(out, source, results, design)
            state.update(status="completed", release=str(out / "release"), summary=w.read(out / "release/report/summary.json"))
        else: state["status"] = "stopped_at_requested_stage"
    except Exception as error:
        state.update(status="failed", reason=str(error))
        for row in state["stages"]:
            if row["status"] == "running": row["status"] = "failed"
        raise
    finally:
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_status(out, state)
        finally:
            lock.unlink(missing_ok=True)
    return state
