"""Export pinned reservoir inputs and calibration archives without model calls.

Kept outside src/dztr so reporting changes do not invalidate numerical
checkpoint identities. Paths in the worklist are relative to repository root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import shutil

import numpy as np
import pandas as pd

TARGET_COLUMNS = (
    "storage_critical_m3", "storage_normal_m3", "storage_flood_m3",
    "release_critical_m3s", "release_normal_m3s", "release_flood_m3s",
)
ARCHIVE_NAMES = (
    "evaluations.csv", "evaluated_vectors.csv", "pareto_front.csv", "pareto_vectors.csv",
    "history.csv", "checkpoint_metrics.csv", "stopping_history.csv", "target_sample_support.csv",
    "monthly_targets_si.csv", "monthly_bounds_si.csv", "summary.json", "optimizer_checkpoint.npz",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inside(root, value):
    if (not isinstance(value, str) or not value or Path(value).is_absolute()
            or PureWindowsPath(value).drive or ".." in re.split(r"[\\/]", value)):
        raise ValueError("Worklist paths must stay inside the repository.")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Worklist path resolves outside the repository.")
    return path


def pinned(root, name, digest):
    path = inside(root, name)
    if not isinstance(digest, str) or sha(path) != digest:
        raise ValueError(f"Pinned file changed: {name}")
    return path


def decode_monthly(vector, model):
    """Decode the archived block ordering, never repair targets."""
    x = np.asarray(vector, dtype=float)
    capacity, dead = model["capacity_m3"], model.get("dead_storage_fraction", .1)
    if x.shape != (72,) or not np.isfinite(x).all() or not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("Expected72 finite coordinates and a positive capacity.")
    values = x.reshape(6, 12).T.copy()
    values[:, :3] *= capacity
    sc, sn, sf, qc, qn, qf = values.T
    if not ((sc > capacity * dead).all() and (sc < sn).all() and (sn < sf).all()
            and (sf <= capacity).all() and (qc >= 0).all() and (qc <= qn).all() and (qn <= qf).all()):
        raise ValueError("Archived targets violate physical ordering; no repair performed.")
    return pd.DataFrame(values, index=pd.Index(range(1, 13), name="month"), columns=TARGET_COLUMNS)


def input_catalog(root, worklist, consumed):
    path = pinned(root, worklist["collection_file"], worklist["collection_sha256"])
    consumed[worklist["collection_file"]] = sha(path)
    manifest = read(path)
    if manifest.get("kind") != "reservoir_collection" or manifest.get("schema_version") != 1:
        raise ValueError("Expected an existing version1 reservoir collection.")
    records, configs, identifiers = [], {}, set()
    for entry in manifest["reservoirs"]:
        identifier = entry["reservoir_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", identifier) or identifier.casefold() in identifiers:
            raise ValueError("Unsafe or duplicate reservoir ID.")
        identifiers.add(identifier.casefold())
        for name, digest in entry["files_sha256"].items():
            pinned(root, name, digest)
            consumed[name] = digest
        for name, field in (("config_file", "config_sha256"), ("observations_file", "observations_sha256")):
            if entry["files_sha256"].get(entry[name]) != entry[field]:
                raise ValueError("Required input is missing from the pinned manifest.")
        config = read(inside(root, entry["config_file"]))
        if config.get("kind") != "world_reservoir_sample" or config["reservoir"]["id"] != identifier:
            raise ValueError("Reservoir identity does not match its configuration.")
        if config["units"] != {"storage": "m3", "flows": "m3/s"}:
            raise ValueError("Explicit SI units are required.")
        if config["sampling"]["frequency"] != entry["frequency"] or entry["frequency"] not in ("daily", "monthly"):
            raise ValueError("Sampling frequency must match the pinned configuration.")
        declared_source = (inside(root, entry["config_file"]).parent / config["observations_file"]).resolve()
        if declared_source != inside(root, entry["observations_file"]):
            raise ValueError("Configuration and manifest observation paths differ.")
        records.append({"schema_version": 1, "kind": "reservoir_input_record", "reservoir": config["reservoir"],
                        "config_file": entry["config_file"], "observations_file": entry["observations_file"],
                        "files_sha256": entry["files_sha256"], "units": config["units"],
                        "sampling": config["sampling"], "periods": config["periods"], "physical_constants": config["model"],
                        "rows": entry.get("rows"), "first_date": entry.get("first_date"), "last_date": entry.get("last_date"),
                        "parameterization": {"quantile_method": config.get("quantile_method"), "target_percentiles": config.get("target_percentiles")},
                        "inflow_preprocessing": config.get("inflow_preprocessing", {}),
                        "source_readiness_snapshot": entry.get("readiness", {}),
                        "readiness_interpretation": "Historical packaged snapshot; not a current calibration authorization.",
                        "current_source_review": "not_assessed_by_archive_export", "new_job_authorized": False})
        configs[identifier] = config
    excluded = manifest.get("excluded_reservoirs", [])
    if identifiers & {r["reservoir_id"].casefold() for r in excluded}:
        raise ValueError("Excluded reservoir is present in active membership.")
    if len(records) != manifest["reservoir_count"]:
        raise ValueError("Collection membership count does not match.")
    return records, configs, excluded


def load_archive(root, entry, configs, decision, consumed, input_record):
    identifier, seed = entry["reservoir_id"], entry["seed"]
    if identifier not in configs:
        raise ValueError("Archived job must belong to active membership.")
    folder = inside(root, entry["archive_directory"])
    proof_path = pinned(root, entry["verification_file"], entry["verification_sha256"])
    proof = read(proof_path)
    if proof.get("status") != "passed":
        raise ValueError("Archive verification must have passed.")
    consumed[entry["verification_file"]] = sha(proof_path)
    for name in ARCHIVE_NAMES:
        path = folder / name
        key = path.relative_to(proof_path.parent).as_posix()
        if proof["artifact_sha256"].get(key) != sha(path):
            raise ValueError(f"Archive artifact changed: {key}")
        consumed[path.relative_to(root).as_posix()] = sha(path)
    summary = read(folder / "summary.json")
    if (summary["reservoir"]["id"] != identifier or summary["optimization"]["seed"] != seed
            or summary["optimization"]["population_size"] != decision["population_size"]
            or summary["search_evaluations"] != decision["evaluation_cap"]
            or summary["checkpoint_evaluations"] != decision["evaluation_cap"]):
        raise ValueError("Archived job does not match the declared worklist.")
    if summary["objective_order"] != ["negative_release_nse", "negative_storage_nse"]:
        raise ValueError("Unknown objective ordering.")
    if summary["reporting_validation_evaluations"] != 0:
        raise ValueError("This export requires calibration-only archives.")
    policy_path = pinned(root, entry["policy_file"], entry["policy_sha256"])
    policy = read(policy_path)
    if policy["reservoir_id"] != identifier or policy["policy_id"] != summary["policy_id"]:
        raise ValueError("Reviewed policy does not match archived job.")
    consumed[entry["policy_file"]] = sha(policy_path)
    config = configs[identifier]
    if policy["observations_sha256"] != sha(inside(root, input_record["observations_file"])):
        raise ValueError("Reviewed forcing does not match packaged observations.")
    for name, digest in summary["consumed_files_sha256"].items():
        path = Path(name).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError("Archive consumed input changed or lies outside the repository.")
        consumed[path.relative_to(root).as_posix()] = digest
    archived_input_paths = {Path(name).resolve() for name in summary["consumed_files_sha256"]}
    required_inputs = {inside(root, input_record["config_file"]), inside(root, input_record["observations_file"]), policy_path}
    if not required_inputs.issubset(archived_input_paths):
        raise ValueError("Archive identity must bind the selected configuration, observations and reviewed policy.")
    if summary["periods"]["calibration"] != config["periods"]["calibration"]:
        raise ValueError("Calibration period differs from packaged inputs.")
    trials = pd.read_csv(folder / "evaluations.csv", index_col="evaluation_id", float_precision="round_trip")
    vectors = pd.read_csv(folder / "evaluated_vectors.csv", index_col="evaluation_id", float_precision="round_trip")
    if (list(trials.index) != list(range(decision["evaluation_cap"])) or not trials.index.equals(vectors.index)
            or list(vectors.columns) != [f"x{i:02d}" for i in range(72)]):
        raise ValueError("Archive trial/vector identity is incomplete.")
    targets = {}
    for role, index in summary["selected_candidate_ids"].items():
        if not re.fullmatch(r"[a-z_]+", role) or not bool(trials.at[index, "feasible"]):
            raise ValueError("Selected role must refer to a feasible archived candidate.")
        targets[role] = decode_monthly(vectors.loc[index].values, config["model"])
    return folder, summary, trials, targets, policy


def build(root, worklist_file, output):
    root, worklist_file, output = Path(root).resolve(), Path(worklist_file).resolve(), Path(output).resolve()
    if not worklist_file.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("Worklist and export output must be inside the repository.")
    if output.exists():
        raise FileExistsError("Choose a fresh export directory; existing evidence is preserved.")
    worklist = read(worklist_file)
    if worklist.get("schema_version") != 1 or worklist.get("kind") != "reservoir_archive_worklist" or worklist.get("execution_mode") != "archive_export_only":
        raise ValueError("Only a version1 archive_export_only worklist is supported.")
    decision = worklist["decision"]
    if decision["convergence_established"] or decision["additional_model_evaluations_authorized"]:
        raise ValueError("This exporter cannot establish convergence or authorize evaluations.")
    consumed = {worklist_file.relative_to(root).as_posix(): sha(worklist_file)}
    evidence_path = pinned(root, decision["evidence_file"], decision["evidence_sha256"])
    if read(evidence_path).get("status") != "passed":
        raise ValueError("Decision evidence must have passed verification.")
    consumed[decision["evidence_file"]] = decision["evidence_sha256"]
    records, configs, excluded = input_catalog(root, worklist, consumed)
    input_records = {r["reservoir"]["id"]: r for r in records}
    loaded, keys = [], set()
    for entry in worklist["archives"]:
        key = entry["reservoir_id"], entry["seed"]
        if key in keys or entry["seed"] not in decision["seeds"]:
            raise ValueError("Duplicate job or undeclared seed.")
        keys.add(key)
        if entry["reservoir_id"] not in input_records:
            raise ValueError("Archived job must belong to active membership.")
        loaded.append((entry, load_archive(root, entry, configs, decision, consumed, input_records[entry["reservoir_id"]])))
    output.mkdir(parents=True)
    write(output / "decision.json", decision)
    write(output / "input_catalog.json", {"schema_version": 1, "kind": "reservoir_input_catalog", "paths_relative_to": "repository_root",
                                           "reservoir_count": len(records), "reservoirs": records, "excluded_reservoirs": excluded})
    result_records = []
    for entry, (source, summary, trials, targets, policy) in loaded:
        job_id = f"{entry['reservoir_id']}__seed_{entry['seed']}__pop_{decision['population_size']}__budget_{decision['evaluation_cap']}"
        folder = output / "results" / job_id
        folder.mkdir(parents=True)
        for name in ("pareto_front.csv", "pareto_vectors.csv", "checkpoint_metrics.csv", "stopping_history.csv"):
            shutil.copyfile(source / name, folder / name)
        roles = {}
        for role, monthly in targets.items():
            name = f"{role}_monthly_targets_si.csv"
            monthly.to_csv(folder / name, float_format="%.17g")
            index = summary["selected_candidate_ids"][role]
            roles[role] = {"evaluation_id": index, "monthly_targets_file": name,
                           "calibration_release_nse": float(trials.at[index, "release_nse"]),
                           "calibration_storage_nse": float(trials.at[index, "storage_nse"])}
        result = {"schema_version": 1, "kind": "reservoir_calibration_result", "job_id": job_id,
                  "reservoir": summary["reservoir"], "seed": entry["seed"], "optimizer": summary["optimization"],
                  "status": "budget_reached_convergence_unconfirmed", "convergence_established": False,
                  "archive_directory": entry["archive_directory"], "checkpoint_file": f"{entry['archive_directory']}/optimizer_checkpoint.npz",
                  "checkpoint_sha256": sha(source / "optimizer_checkpoint.npz"),
                  "checkpoint_identity": summary["checkpoint_identity"], "units": {"storage": "m3", "flows": "m3/s"},
                  "physical_constants": configs[entry["reservoir_id"]]["model"], "periods": summary["periods"],
                  "bounds": summary["bounds"], "root_policy": summary["root_policy"],
                  "policy_id": summary["policy_id"], "reviewed_input_policy": policy,
                  "conditional_hindcast": summary["conditional_hindcast"], "objective_order": summary["objective_order"],
                  "objective_direction": summary["objective_direction"], "search_evaluations": summary["search_evaluations"],
                  "feasible_evaluations": summary["feasible_evaluations"], "rejected_evaluations": summary["rejected_evaluations"],
                  "pareto_count": summary["pareto_count"], "stopping_policy": summary["stopping_policy"],
                  "diagnostic_source": "Verified archive; stopping signals do not establish convergence.",
                  "selected_parameters": roles, "validation_scored_by_export": False,
                  "validation_metrics": None, "new_model_evaluations": 0}
        write(folder / "result.json", result)
        result_records.append({"job_id": job_id, "reservoir_id": entry["reservoir_id"], "seed": entry["seed"],
                               "population_size": decision["population_size"], "evaluations": summary["search_evaluations"],
                               "status": result["status"], "pareto_count": result["pareto_count"],
                               "result_file": (folder / "result.json").relative_to(output).as_posix()})
    pd.DataFrame(result_records).to_csv(output / "result_index.csv", index=False)
    write(output / "result_index.json", {"schema_version": 1, "kind": "reservoir_result_index", "paths_relative_to": "export_directory", "results": result_records})
    for name, digest in consumed.items():
        pinned(root, name, digest)
    proof = {"status": "passed", "schema_version": 1, "kind": "reservoir_archive_export_verification",
             "active_reservoirs": len(records), "exported_jobs": len(loaded), "excluded_reservoirs": excluded,
             "new_model_evaluations": 0, "validation_calls": 0, "numerical_source_modified": False,
             "consumed_files_sha256": consumed, "exporter_sha256": sha(Path(__file__)),
             "artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    write(output / "verification.json", proof)
    return proof


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path, default=root / "configs/reviews/archive_workflow.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proof = build(root, args.worklist, args.output)
    print(json.dumps({k: proof[k] for k in ("status", "active_reservoirs", "exported_jobs", "new_model_evaluations", "validation_calls")}, indent=2))


if __name__ == "__main__":
    main()
