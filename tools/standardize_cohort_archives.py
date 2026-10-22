"""Combine verified interval exports and sampled cohort pilots without evaluation."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("archive_base", Path(__file__).with_name("standardize_archives.py"))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def normalize(name):
    return name.replace("\\", "/")


def verified(root, proof, path, consumed):
    name = path.relative_to(root).as_posix()
    hashes = {normalize(k): v for k, v in proof["artifact_sha256"].items()}
    digest = hashes.get(name)
    if digest is None or base.sha(path) != digest:
        raise ValueError(f"Archived cohort artifact missing or changed: {name}")
    consumed[name] = digest
    return path


def load_pilot(root, folder, entry, proof, protocol, review, consumed):
    """Validate a sampled archive against its checkpoint and recorded protocol."""
    for path in folder.iterdir():
        if path.is_file():
            verified(root, proof, path, consumed)
    result = base.read(folder / "result.json")
    config = base.read(folder / "resolved_config.json")
    identity = base.read(folder / "checkpoint_identity.json")
    selection = base.read(folder / "selection.json")
    identifier, seed = entry["reservoir_id"], entry["seed"]
    settings = result["settings"]
    if (result["reservoir_id"] != identifier or result["seed"] != seed or config["reservoir"]["id"] != identifier
            or result["status"] != "pilot_completed" or result["termination"] != "budget_reached"
            or settings["seed"] != seed or settings["population_size"] != entry["population_size"] or settings["evaluation_budget"] != entry["evaluation_budget"]
            or settings["population_size"] != protocol["pilot"]["population_size"]
            or settings["evaluation_budget"] != protocol["pilot"]["evaluation_budget_per_job"]
            or result["search_evaluations"] != settings["evaluation_budget"] or identity != result["identity"]
            or identity["settings"] != settings or identity["sampling"] != config["sampling"]):
        raise ValueError("Pilot identity or settings do not match the declared historical job.")
    if selection["validation_used"] or not selection["written_before_validation"] or result["selection_uses_validation"]:
        raise ValueError("Pilot selection must have been frozen before validation.")
    if not review["eligible_for_exploratory_calibration"] or not review["pilot_selected"]:
        raise ValueError("Historical cohort review did not select this pilot.")
    for name, digest in identity["input_sha256"].items():
        base.pinned(root, normalize(name), digest)
        consumed[normalize(name)] = digest
    config_ref = next(n for n in identity["input_sha256"] if normalize(n).endswith(f"/{identifier}.json"))
    if base.read(base.inside(root, normalize(config_ref))) != config:
        raise ValueError("Resolved pilot configuration differs from its pinned input.")
    if config["units"] != {"storage": "m3", "flows": "m3/s"} or identity["split"] != config["periods"]:
        raise ValueError("Pilot units or periods do not match the pinned configuration.")
    trials = pd.read_csv(folder / "trials.csv", index_col="evaluation_id", float_precision="round_trip")
    vectors = pd.read_csv(folder / "vectors.csv", index_col="evaluation_id", float_precision="round_trip")
    count = settings["evaluation_budget"]
    if list(trials.index) != list(range(count)) or not trials.index.equals(vectors.index) or list(vectors.columns) != [str(i) for i in range(72)]:
        raise ValueError("Incomplete pilot trial/vector archive.")
    with np.load(folder / "checkpoint.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["vectors"], vectors.values)
        np.testing.assert_array_equal(saved["objectives"], trials[["release_cost", "storage_cost"]].values)
    if int(trials.feasible.sum()) != result["feasible_search_evaluations"]:
        raise ValueError("Pilot feasibility accounting differs from saved trials.")
    rows = {row["role"]: row for row in result["representatives"]}
    if {k: row["evaluation_id"] for k, row in rows.items()} != selection["representatives"]:
        raise ValueError("Selected role IDs differ between historical receipts.")
    roles = {"generalized_baseline": {"evaluation_id": 0,
              "calibration_release_nse": result["baseline_calibration"]["release_nse"],
              "calibration_storage_nse": result["baseline_calibration"]["storage_nse"],
              "validation_feasible": result["baseline_validation"]["feasible"],
              "validation_reason": result["baseline_validation"]["reason"],
              "validation_release_nse": result["baseline_validation"]["metrics"].get("release_nse"),
              "validation_storage_nse": result["baseline_validation"]["metrics"].get("storage_nse")}, **rows}
    targets = {}
    for role, row in roles.items():
        index = row["evaluation_id"]
        if not bool(trials.at[index, "feasible"]):
            raise ValueError("Selected pilot role is infeasible.")
        np.testing.assert_allclose([row["calibration_release_nse"], row["calibration_storage_nse"]],
                                   -trials.loc[index, ["release_cost", "storage_cost"]].values.astype(float), rtol=0, atol=1e-12)
        targets[role] = base.decode_monthly(vectors.loc[index].values, config["model"])
    return result, config, roles, targets


def build(root, worklist_path, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("Choose a fresh output directory inside the repository.")
    worklist_path = Path(worklist_path).resolve()
    if not worklist_path.is_relative_to(root):
        raise ValueError("Worklist must be inside the repository.")
    worklist = base.read(worklist_path)
    if worklist.get("kind") != "multi_reservoir_archive_worklist" or worklist.get("schema_version") != 1 or worklist["execution_mode"] != "archive_export_only":
        raise ValueError("Only a version1 archive-only worklist is supported.")
    consumed = {worklist_path.relative_to(root).as_posix(): base.sha(worklist_path)}
    for name, digest in worklist["pinned_files_sha256"].items():
        base.pinned(root, name, digest)
        consumed[name] = digest
    prior_dir = base.inside(root, worklist["prior_export_directory"])
    prior_proof = base.read(prior_dir / "workflow_verification.json")
    cohort_dir = base.inside(root, worklist["cohort_directory"])
    proof = base.read(cohort_dir / "final_verification.json")
    if prior_proof["status"] != "passed" or proof["status"] != "passed":
        raise ValueError("Both historical verification manifests must have passed.")
    for name, digest in prior_proof["artifact_sha256"].items():
        if base.sha(prior_dir / name) != digest:
            raise ValueError(f"Prior export changed: {name}")
        consumed[(prior_dir / name).relative_to(root).as_posix()] = digest
    for name, digest in prior_proof["new_root_files_sha256"].items():
        base.pinned(root, name, digest)
        consumed[name] = digest
    for name, digest in prior_proof["consumed_files_sha256"].items():
        base.pinned(root, name, digest)
        consumed[name] = digest
    cohort = base.read(verified(root, proof, cohort_dir / "cohort.json", consumed))
    protocol = base.read(verified(root, proof, cohort_dir / "protocol.json", consumed))
    reviews = {r["reservoir_id"]: r for r in cohort["reservoirs"]}
    catalog = base.read(prior_dir / "input_catalog.json")
    inputs = {r["reservoir"]["id"]: r for r in catalog["reservoirs"]}
    pilots, keys = [], set()
    for entry in worklist["pilots"]:
        key = entry["reservoir_id"], entry["seed"]
        if key in keys or key[0] not in inputs or key[0] not in cohort["selected_for_pilot"] or key[1] not in cohort["pilot_seeds"]:
            raise ValueError("Duplicate, excluded or undeclared historical job.")
        keys.add(key)
        folder = base.inside(root, entry["archive_directory"])
        pilots.append((entry, folder, load_pilot(root, folder, entry, proof, protocol, reviews[key[0]], consumed)))
    output.mkdir(parents=True)
    shutil.copytree(prior_dir / "results", output / "results")
    shutil.copyfile(prior_dir / "decision.json", output / "decision.json")
    indexes = base.read(prior_dir / "result_index.json")["results"]
    for entry, source, (original, config, roles, targets) in pilots:
        settings = original["settings"]
        identifier = entry["reservoir_id"]
        job_id = f"{identifier}__seed_{entry['seed']}__pop_{settings['population_size']}__budget_{settings['evaluation_budget']}"
        folder = output / "results" / job_id
        folder.mkdir()
        for name in ("pareto_front.csv", "vectors.csv", "trials.csv", "generation_history.csv", "representatives.csv", "selected_validation.json", "selection.json"):
            shutil.copyfile(source / name, folder / name)
        front = pd.read_csv(source / "pareto_front.csv", float_precision="round_trip")
        vectors = pd.read_csv(source / "vectors.csv", index_col="evaluation_id", float_precision="round_trip")
        vectors.loc[front.evaluation_id].to_csv(folder / "pareto_vectors.csv", float_format="%.17g")
        selected = {}
        for role, monthly in targets.items():
            name = f"{role}_monthly_targets_si.csv"
            monthly.to_csv(folder / name, float_format="%.17g")
            row = roles[role]
            selected[role] = {"evaluation_id": row["evaluation_id"], "monthly_targets_file": name,
                              "calibration_release_nse": row["calibration_release_nse"], "calibration_storage_nse": row["calibration_storage_nse"],
                              "historical_validation": {"feasible": row["validation_feasible"], "reason": row.get("validation_reason"),
                                  "release_nse": row.get("validation_release_nse"), "storage_nse": row.get("validation_storage_nse")}}
        validation = {role: row["historical_validation"] for role, row in selected.items()}
        record = {"schema_version": 1, "kind": "reservoir_calibration_result", "job_id": job_id, "reservoir": config["reservoir"],
                  "seed": entry["seed"], "optimizer": settings, "status": "budget_reached_convergence_unconfirmed", "convergence_established": False,
                  "archive_format": "ready_cohort_sampled_pilot", "archive_directory": entry["archive_directory"],
                  "checkpoint_file": f"{entry['archive_directory']}/checkpoint.npz", "checkpoint_sha256": base.sha(source / "checkpoint.npz"),
                  "checkpoint_identity": original["identity"], "continuation_compatibility": "Historical identity preserved; current runner compatibility not assessed.",
                  "units": config["units"], "sampling": config["sampling"], "periods": config["periods"], "physical_constants": config["model"],
                  "inflow_preprocessing": config["inflow_preprocessing"], "bounds": base.read(source / "calibration_bounds.json"),
                  "solver": protocol["pilot"]["solver"], "overflow_mode": protocol["pilot"]["overflow"], "root_policy": protocol["pilot"]["root_policy"],
                  "objective_order": ["negative_release_nse", "negative_storage_nse"], "objective_direction": "minimize",
                  "search_evaluations": original["search_evaluations"], "feasible_evaluations": original["feasible_search_evaluations"],
                  "rejected_evaluations": original["search_evaluations"] - original["feasible_search_evaluations"], "pareto_count": original["front_size"],
                  "selected_parameters": selected, "selection_uses_validation": False, "validation_scored_by_export": False,
                  "validation_metrics": validation, "validation_provenance": "Historical held-out reporting imported unchanged; no new validation calls or reselection.",
                  "stopping_policy": None, "diagnostic_source": "Historical fixed cap128; no equivalent Bull Lake stopping policy inferred.",
                  "source_review": reviews[identifier], "limitations": worklist["limitations"][identifier], "new_model_evaluations": 0}
        base.write(folder / "result.json", record)
        indexes.append({"job_id": job_id, "reservoir_id": identifier, "seed": entry["seed"], "population_size": settings["population_size"],
                        "evaluations": original["search_evaluations"], "status": record["status"], "pareto_count": record["pareto_count"],
                        "result_file": (folder / "result.json").relative_to(output).as_posix()})
    jobs = []
    for item in indexes:
        result = base.read(output / item["result_file"])
        identifier = item["reservoir_id"]
        balanced = result["selected_parameters"]["balanced"]
        historical = balanced.get("historical_validation", {})
        jobs.append({**item, "name": result["reservoir"]["name"], "frequency": inputs[identifier]["sampling"]["frequency"],
                     "fixed_step_days": inputs[identifier]["sampling"]["fixed_step_days"],
                     "solver": result.get("solver", "interval-volume implicit"), "root_policy": result["root_policy"],
                     "overflow_mode": result.get("overflow_mode", "separate interval spill"),
                     "calibration_start": result["periods"]["calibration"]["start"], "calibration_end": result["periods"]["calibration"]["end"],
                     "balanced_calibration_release_nse": balanced["calibration_release_nse"], "balanced_calibration_storage_nse": balanced["calibration_storage_nse"],
                     "validation_status": "historical_reporting_imported" if historical else "not_scored",
                     "balanced_historical_validation_release_nse": historical.get("release_nse"), "balanced_historical_validation_storage_nse": historical.get("storage_nse"),
                     "source_review_status": "conditional_computed_net_policy" if identifier not in {e['reservoir_id'] for e in worklist['pilots']} else reviews[identifier]["status"],
                     "physical_source_certified": False, "limitations": " | ".join(worklist["limitations"][identifier])})
    for record in catalog["reservoirs"]:
        identifier = record["reservoir"]["id"]
        if identifier in reviews:
            record["cohort_review_snapshot"] = {"review_date": protocol["review_date"], "status": reviews[identifier]["status"],
                                                "eligible_for_exploratory_calibration": reviews[identifier]["eligible_for_exploratory_calibration"],
                                                "physical_source_certified": False, "reasons": reviews[identifier]["reasons"]}
        record["new_job_authorized"] = False
    base.write(output / "input_catalog.json", catalog)
    base.write(output / "result_index.json", {"schema_version": 1, "kind": "reservoir_result_index", "paths_relative_to": "export_directory", "results": indexes})
    pd.DataFrame(indexes).to_csv(output / "result_index.csv", index=False)
    pd.DataFrame(jobs).to_csv(output / "job_comparison.csv", index=False, float_format="%.17g")
    inventory = []
    for record in catalog["reservoirs"]:
        identifier = record["reservoir"]["id"]
        found = [j for j in jobs if j["reservoir_id"] == identifier]
        inventory.append({"reservoir_id": identifier, "name": record["reservoir"]["name"], "frequency": record["sampling"]["frequency"],
                          "archived_jobs_in_this_catalog": len(found), "seeds": ",".join(str(j["seed"]) for j in found),
                          "population_sizes": ",".join(str(v) for v in sorted({j['population_size'] for j in found})),
                          "evaluation_caps": ",".join(str(v) for v in sorted({j['evaluations'] for j in found})),
                          "source_review_status": found[0]["source_review_status"] if found else record.get("cohort_review_snapshot", {}).get("status", "not_assessed"),
                          "review_is_historical": True, "new_job_authorized": False,
                          "limitations": " | ".join(worklist["limitations"].get(identifier, reviews.get(identifier, {}).get("reasons", []))) or "Source assumptions require reservoir-specific review; no physical certification inferred."})
    pd.DataFrame(inventory).to_csv(output / "reservoir_comparison.csv", index=False)
    for name, digest in consumed.items():
        base.pinned(root, name, digest)
    verification = {"status": "passed", "active_reservoirs": len(catalog["reservoirs"]), "exported_jobs": len(indexes),
                    "new_model_evaluations": 0, "validation_calls": 0, "historical_validation_imported_not_recomputed": True,
                    "consumed_files_sha256": consumed, "exporter_sha256": base.sha(Path(__file__)),
                    "artifact_sha256": {p.relative_to(output).as_posix(): base.sha(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    base.write(output / "verification.json", verification)
    return verification


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path, default=root / "configs/reviews/multi_reservoir_archive.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(root, args.worklist, args.output)
    print(json.dumps({k: result[k] for k in ("status", "active_reservoirs", "exported_jobs", "new_model_evaluations", "validation_calls")}, indent=2))
