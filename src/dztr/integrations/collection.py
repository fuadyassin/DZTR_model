"""Run permanent, hash-pinned reservoir inputs individually or as a collection."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re

import pandas as pd

from dztr.calibration.sampled import _observations
from dztr.evaluation import evaluate_simulation
from .world_reservoirs import _code_hashes, _new_output, _write_json, legacy_replay, run_world_sample, sha256


def _inside(root, relative, label):
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or PureWindowsPath(relative).drive or ".." in re.split(r"[\\/]", relative)):
        raise ValueError(f"{label} must be a relative path inside the repository.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} resolves outside the repository.")
    return path


def load_collection(manifest_file):
    """Validate membership and relative references without accessing original sources."""
    path = Path(manifest_file).expanduser().resolve(strict=True)
    contents = path.read_bytes()
    manifest = json.loads(contents.decode("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("kind") != "reservoir_collection":
        raise ValueError("Expected a version 1 reservoir_collection manifest.")
    reference = manifest.get("repository_root")
    if not isinstance(reference, str) or Path(reference).is_absolute() or PureWindowsPath(reference).drive:
        raise ValueError("repository_root must be relative to the manifest directory.")
    root = (path.parent / reference).resolve(strict=True)
    if root == root.parent or not path.is_relative_to(root):
        raise ValueError("Manifest must be inside its declared repository root.")
    entries = manifest.get("reservoirs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Collection requires at least one active reservoir.")
    ids = []
    for entry in entries:
        identifier = entry.get("reservoir_id") if isinstance(entry, dict) else None
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", identifier):
            raise ValueError("Reservoir IDs must be unique path-safe strings.")
        ids.append(identifier)
        for key in ("config_file", "observations_file"):
            _inside(root, entry.get(key), key)
        if entry.get("frequency") not in ("daily", "monthly"):
            raise ValueError("Each reservoir requires an explicit daily or monthly frequency.")
    if len(set(value.casefold() for value in ids)) != len(ids):
        raise ValueError("Duplicate reservoir IDs in collection.")
    excluded = manifest.get("excluded_reservoirs", [])
    if not isinstance(excluded, list) or any(not isinstance(entry, dict) or not isinstance(entry.get("reservoir_id"), str) for entry in excluded):
        raise ValueError("excluded_reservoirs must contain explicit reservoir IDs.")
    if {identifier.casefold() for identifier in ids} & {entry["reservoir_id"].casefold() for entry in excluded}:
        raise ValueError("An excluded reservoir cannot appear in active membership.")
    return root, manifest, path, hashlib.sha256(contents).hexdigest()


def _selected(manifest, reservoirs, all_reservoirs):
    if bool(reservoirs) == bool(all_reservoirs):
        raise ValueError("Select reservoir IDs/names or all_reservoirs=True, exclusively.")
    if all_reservoirs:
        return manifest["reservoirs"]
    if isinstance(reservoirs, str):
        reservoirs = [reservoirs]
    result = []
    for requested in reservoirs:
        if not isinstance(requested, str):
            raise ValueError("Reservoir selection must use IDs or names.")
        token = requested.strip().casefold()
        matches = [entry for entry in manifest["reservoirs"] if entry["reservoir_id"].casefold() == token]
        if not matches:
            matches = [entry for entry in manifest["reservoirs"] if str(entry.get("name", "")).casefold() == token]
        if len(matches) != 1:
            raise ValueError(f"Reservoir {requested!r} does not uniquely identify an active member.")
        if matches[0] in result:
            raise ValueError("Duplicate reservoir selection.")
        result.append(matches[0])
    return result


def _verified_inputs(root, entry):
    config_path = _inside(root, entry["config_file"], "config_file")
    observations_path = _inside(root, entry["observations_file"], "observations_file")
    hashes = entry.get("files_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Every sample requires a files_sha256 manifest.")
    expected = {}
    for relative, digest in hashes.items():
        path = _inside(root, relative, "files_sha256 path")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Input hashes must be lowercase SHA256 strings.")
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Missing or changed packaged input SHA256: {relative}")
        expected[path] = digest
    for path, field in ((config_path, "config_sha256"), (observations_path, "observations_sha256")):
        if path not in expected or expected[path] != entry.get(field):
            raise ValueError(f"{field} must match the pinned file manifest.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Packaged configuration must be a JSON object.")
    if config.get("schema_version") != 1 or config.get("kind") != "world_reservoir_sample":
        raise ValueError("Expected a version 1 world_reservoir_sample configuration.")
    for field in ("reservoir", "sampling", "model", "periods", "protocol"):
        if not isinstance(config.get(field), dict):
            raise ValueError(f"Configuration {field} must be a JSON object.")
    if config["reservoir"].get("id") != entry["reservoir_id"]:
        raise ValueError("Configuration reservoir ID disagrees with active membership.")
    if config["sampling"].get("frequency") != entry["frequency"]:
        raise ValueError("Configuration frequency disagrees with the collection.")
    if config.get("units") != {"storage": "m3", "flows": "m3/s"}:
        raise ValueError("Packaged observation units must be explicit SI.")
    observation_reference = config.get("observations_file")
    if (not isinstance(observation_reference, str) or Path(observation_reference).is_absolute()
            or PureWindowsPath(observation_reference).drive):
        raise ValueError("Configuration requires a relative observation reference.")
    if (config_path.parent / observation_reference).resolve() != observations_path or config.get("observations_sha256") != expected[observations_path]:
        raise ValueError("Configuration observations disagree with the pinned collection input.")
    for field in ("provenance_file", "matlab_source_file", "legacy_targets_file", "legacy_bounds_file"):
        if field in config:
            relative = config[field]
            if not isinstance(relative, str) or Path(relative).is_absolute() or PureWindowsPath(relative).drive:
                raise ValueError(f"{field} must reference a relative packaged input.")
            target = (config_path.parent / relative).resolve()
            if not target.is_relative_to(root) or target not in expected:
                raise ValueError(f"{field} must be inside the pinned file manifest.")
    frame = pd.read_csv(observations_path, index_col="date", parse_dates=["date"], float_precision="round_trip")
    if list(frame.columns) != ["storage_m3", "inflow_m3s", "release_m3s"]:
        raise ValueError("Packaged observations require exactly storage_m3, inflow_m3s, release_m3s.")
    if (len(frame) != entry.get("rows") or not len(frame)
            or str(frame.index[0].date()) != entry.get("first_date")
            or str(frame.index[-1].date()) != entry.get("last_date")):
        raise ValueError("Packaged row count or date coverage disagrees with the manifest.")
    if any(sha256(path) != digest for path, digest in expected.items()):
        raise ValueError("Packaged inputs changed during loading.")
    return config_path, config, _observations(frame, str(observations_path)), expected


def _run_legacy(config_path, config, observations, output):
    reference = config.get("matlab_source_file")
    if reference is None:
        raise ValueError("Historical replay requires a packaged matlab_source_file.")
    spec = json.loads((config_path.parent / reference).resolve().read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("reservoir_id") != config["reservoir"]["id"]:
        raise ValueError("Packaged historical settings refer to another reservoir.")
    physical, reported = legacy_replay(observations, spec, allow_signed_inflow=True)
    reported.to_csv(output / "legacy_simulation.csv")
    warmup = spec["warmup_rows"]
    observed = _observations(observations.frame.iloc[warmup:], observations.source)
    scored = replace(physical, frame=reported.iloc[warmup:].copy())
    scored.frame["storage_m3"] = scored.frame.legacy_reported_storage_m3
    scored.frame["release_m3s"] = scored.frame.legacy_reported_release_m3s
    report = {"reservoir_id": spec["reservoir_id"], "legacy_status": "completed",
              "metrics": evaluate_simulation(observed, scored), "issues": list(physical.issues),
              "max_abs_balance_residual_m3": float(physical.frame.balance_residual_m3.abs().max()),
              "sampling": spec["sampling"], "warmup_rows": warmup,
              "note": "Unmodified historical baseline logic; shifted reporting is separate from physical states."}
    _write_json(output / "legacy_summary.json", report)
    return report


def run_collection(manifest_file, output_dir, *, reservoirs=None, all_reservoirs=False,
                   mode="baseline", optimization_budget=None, seed=42):
    """Run independent jobs, keeping input failures and scientific blockers explicit."""
    if mode not in ("baseline", "legacy"):
        raise ValueError("mode must be baseline or legacy.")
    if optimization_budget is not None and (mode != "baseline" or isinstance(optimization_budget, bool)
            or not isinstance(optimization_budget, int) or optimization_budget < 2):
        raise ValueError("An optimization budget of at least two is supported only with baseline mode.")
    root, manifest, manifest_path, manifest_hash = load_collection(manifest_file)
    selected = _selected(manifest, reservoirs, all_reservoirs)
    protected = [manifest_path]
    for entry in manifest["reservoirs"]:
        protected.extend(_inside(root, entry[key], key) for key in ("config_file", "observations_file"))
        hashes = entry.get("files_sha256")
        if isinstance(hashes, dict):
            for relative in hashes:
                try:
                    protected.append(_inside(root, relative, "files_sha256 path"))
                except ValueError:
                    # Invalid per-job metadata is rejected by _verified_inputs;
                    # it must not prevent independent valid members running.
                    pass
    output = _new_output(output_dir, protected)
    code = _code_hashes()
    records, verified_hashes = [], {}
    for entry in selected:
        record = {"reservoir_id": entry["reservoir_id"], "name": entry.get("name"),
                  "frequency": entry["frequency"], "mode": mode, "status": "input_error"}
        records.append(record)
        sample_output = output / entry["reservoir_id"]
        try:
            config_path, config, observations, expected = _verified_inputs(root, entry)
            verified_hashes.update(expected)
        except (OSError, ValueError, TypeError, KeyError) as error:
            record["reason"] = str(error)
            continue
        sample_output.mkdir()
        try:
            if mode == "baseline":
                report = run_world_sample(config_path, sample_output / "experiment",
                                          optimization_budget=optimization_budget, seed=seed)
                record["status"] = "completed" if report["baseline_status"] == "completed" else "blocked"
                if optimization_budget is not None and report["optimization_status"] != "pilot_completed":
                    record["status"] = "blocked"
                for key in ("baseline_status", "optimization_status", "parameterization_status", "physical_parameters_status",
                            "baseline_reason", "parameterization_reason", "optimization_reason", "max_abs_balance_residual_m3"):
                    if key in report:
                        record[key] = report[key]
            else:
                report = _run_legacy(config_path, config, observations, sample_output)
                record["status"] = "completed"
                record["legacy_status"] = report["legacy_status"]
        except (OSError, ValueError, TypeError, KeyError) as error:
            report = {"baseline_reason": str(error)}
            record["status"], record["reason"] = "blocked", str(error)
        from dztr.evaluation.readiness import diagnose_sample
        try:
            if config.get("matlab_source_file"):
                source_metadata = json.loads((config_path.parent / config["matlab_source_file"]).resolve().read_text(encoding="utf-8"))
                if not isinstance(source_metadata, dict) or not isinstance(source_metadata.get("model"), dict):
                    raise ValueError("Packaged historical source metadata requires a model object.")
                report = {**report, "source_dead_storage_expression": source_metadata["model"].get("dead_storage_expression")}
            diagnostics = diagnose_sample(observations, config, report if mode == "baseline" else None)
            _write_json(sample_output / "readiness.json", diagnostics)
            issues = diagnostics.get("issues", []) if isinstance(diagnostics, dict) else diagnostics
            record["issue_codes"] = sorted({issue["code"] for issue in issues})
        except (OSError, ValueError, TypeError, KeyError) as error:
            record["diagnostic_reason"] = str(error)
        if any(not path.is_file() or sha256(path) != digest for path, digest in expected.items()):
            record["status"] = "input_error"
            record["reason"] = "Packaged inputs changed during the job; results require a fresh reviewed run."
        _write_json(sample_output / "job.json", record)
    unchanged_manifest = manifest_path.is_file() and sha256(manifest_path) == manifest_hash
    unchanged_inputs = all(path.is_file() and sha256(path) == digest for path, digest in verified_hashes.items())
    unchanged_code = _code_hashes() == code
    integrity = unchanged_manifest and unchanged_inputs and unchanged_code
    summary = {"schema_version": 1, "experiment": "reservoir_collection_run", "mode": mode,
               "status": "completed" if integrity and all(record["status"] == "completed" for record in records)
                         else "completed_with_issues" if integrity else "invalid_integrity",
               "manifest_file": str(manifest_path), "manifest_sha256": manifest_hash,
               "input_files_unchanged": unchanged_inputs, "manifest_file_unchanged": unchanged_manifest,
               "code_files_unchanged": unchanged_code, "code_sha256": code,
               "selected_reservoir_count": len(records), "counts": dict(Counter(record["status"] for record in records)),
               "excluded_reservoirs": manifest.get("excluded_reservoirs", []), "reservoirs": records,
               "notes": ["Each job uses only its packaged inputs; original source paths in provenance are descriptive metadata.",
                         "Baseline parameterization and bounds are calibration-only; validation carries the continuous state.",
                         "Scientific blockers remain explicit, and input failures do not prevent other selected jobs."]}
    _write_json(output / "collection.json", summary)
    pd.DataFrame([{key: value for key, value in record.items() if not isinstance(value, (list, dict))}
                  for record in records]).to_csv(output / "collection.csv", index=False)
    lines = ["# Reservoir collection run", "", f"Mode: {mode}. Selected reservoirs: {len(records)}.", "",
             "| Reservoir | Frequency | Status | Issues |", "| --- | --- | --- | --- |"]
    for record in records:
        issues = ", ".join(record.get("issue_codes", [])) or record.get("reason", "")
        lines.append(f"| {record['reservoir_id']} | {record['frequency']} | {record['status']} | {issues} |")
    (output / "collection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/reservoirs/manifest.json")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--reservoir", action="append", help="Active reservoir ID or exact name; repeat to select several")
    selection.add_argument("--all", action="store_true", help="Run every active member")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--mode", choices=("baseline", "legacy"), default="baseline")
    parser.add_argument("--optimize", type=int, metavar="EVALUATIONS", help="Explicit search budget per selected reservoir")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    summary = run_collection(args.manifest, args.output, reservoirs=args.reservoir, all_reservoirs=args.all,
                             mode=args.mode, optimization_budget=args.optimize, seed=args.seed)
    print(json.dumps({key: summary[key] for key in ("status", "mode", "selected_reservoir_count", "counts",
                                                    "input_files_unchanged", "manifest_file_unchanged", "code_files_unchanged")}, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
