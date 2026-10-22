"""Preflight independent reservoir jobs without simulation or optimization.

The manifest supplies explicit per-job configuration paths and observation
units. Available jobs receive canonical SI copies and resolved configurations;
unavailable inputs and invalid jobs remain visible in an isolated status report.
This is preparation for future fleet execution, not a scheduler or resume system.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from numbers import Real
from pathlib import Path
import platform
import re

import numpy as np
import pandas as pd

from dztr import __version__
from dztr.data import read_reservoir_observations
from dztr.parameterization import estimate_generalized_parameters
from dztr.schemas import ReservoirParameters

from .baseline import json_ready
from .bounds import estimate_calibration_bounds
from .candidates import parameters_to_vector
from .objectives import CalibrationObjective
from .optimizer import NSGA2Settings
from .periods import CalibrationValidationSplit, DatePeriod, select_period
from .stopping import StoppingPolicy


_REQUIRED_MODEL_FIELDS = (
    "capacity_m3", "initial_storage_m3", "initial_release_m3s", "inflow_correction",
)
_MODEL_FIELDS = (
    "capacity_m3", "initial_storage_m3", "initial_release_m3s", "channel_capacity_m3s",
    "dead_storage_fraction", "inflow_correction", "flood_mode",
)
_WINDOWS_RESERVED_IDS = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
}
_STATUS_COLUMNS = (
    "reservoir_id", "status", "reason", "failure_stage", "source_status",
    "configuration_file", "observations_file", "observation_rows", "prepared_directory",
    "observations_si_file", "resolved_configuration_file", "provenance_file",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remember_hash(path: Path, digest: str, hashes: dict[Path, str]) -> None:
    previous = hashes.get(path)
    if previous is not None and previous != digest:
        raise ValueError(f"Input changed during preflight: {path}")
    hashes[path] = digest


def _read_json(path: Path, hashes: dict[Path, str]) -> tuple[dict, str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    _remember_hash(path, digest, hashes)
    value = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON configuration must contain an object: {path}")
    return value, digest


def _relative_input(root: Path, value, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty path relative to repo_root.")
    relative = Path(value)
    if relative.is_absolute() or relative.drive or relative.root:
        raise ValueError(f"{field} must be relative to repo_root.")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"{field} must stay within repo_root; parent-directory escapes are not allowed.")
    return result


def _observation_path(root: Path, value) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("observations_file must be a nonempty local file path.")
    path = Path(value)
    if path.drive and not path.is_absolute():
        raise ValueError("observations_file cannot be an ambiguous drive-relative path.")
    return (root / path).resolve()


def _check_output(output: Path) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Preflight output must be a new or empty directory; existing artifacts are not reused.")


def _validate_jobs(manifest: dict) -> list[dict]:
    version = manifest.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("Fleet manifest schema_version must be the integer 1.")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Fleet manifest jobs must be a nonempty list.")
    seen = set()
    for job in jobs:
        identifier = job.get("reservoir_id") if isinstance(job, dict) else None
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", identifier):
            raise ValueError("Each reservoir_id must be a path-safe name of 1-64 letters, digits, underscores or hyphens.")
        key = identifier.casefold()
        if key in _WINDOWS_RESERVED_IDS:
            raise ValueError(f"reservoir_id {identifier!r} is a reserved filesystem name.")
        if key in seen:
            raise ValueError(f"reservoir_id values must be unique, ignoring case: {identifier!r}")
        seen.add(key)
    return jobs


def _explicit_model(config: dict) -> dict:
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("Each available reservoir requires its own model object.")
    missing = [name for name in _REQUIRED_MODEL_FIELDS if name not in model]
    if missing:
        raise ValueError(f"Each reservoir requires explicit model values for {', '.join(missing)}; no reservoir values are inherited.")
    result = dict(model)
    for name in _MODEL_FIELDS[:-1]:
        if name in result and result[name] is not None:
            value = result[name]
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"model.{name} must be a finite real number.")
    # Only channel capacity has a documented calibration-derived fallback.
    for name in _REQUIRED_MODEL_FIELDS:
        if result[name] is None:
            raise ValueError(f"model.{name} must be an explicit finite real number.")
    return result


def _prepare_job(job: dict, root: Path, output: Path, policy: StoppingPolicy | None,
                 manifest_path: Path, manifest_hash: str, policy_source: dict | None,
                 hashes: dict[Path, str], protected: set[Path]) -> tuple[dict, tuple | None]:
    identifier = job["reservoir_id"]
    row = {name: None for name in _STATUS_COLUMNS}
    row.update({"reservoir_id": identifier, "status": "error", "reason": None,
                "configuration_sha256": None, "source_sha256": None,
                "observation_units": deepcopy(job.get("observation_units")),
                "read_options": deepcopy(job.get("read_options", {}))})
    stage = "configuration"
    try:
        config_path = _relative_input(root, job.get("configuration_file"), "configuration_file")
        protected.add(config_path)
        row["configuration_file"] = str(config_path)
        config, config_hash = _read_json(config_path, hashes)
        row["configuration_sha256"] = config_hash
        source = _observation_path(root, config.get("observations_file"))
        protected.add(source)
        row["observations_file"] = str(source)
        units = job.get("observation_units")
        if not isinstance(units, dict) or set(units) != {"storage", "flow"} or any(
            not isinstance(value, str) or not value.strip() for value in units.values()
        ):
            raise ValueError("observation_units must explicitly specify storage and flow unit strings.")
        read_options = job.get("read_options", {})
        if not isinstance(read_options, dict):
            raise ValueError("read_options must be a JSON object of reader options.")
        stage = "observations"
        source_hash = _sha256(source)
        _remember_hash(source, source_hash, hashes)
        row["source_sha256"] = source_hash
        observed = read_reservoir_observations(source, storage_unit=units["storage"],
            flow_unit=units["flow"], read_options=read_options)
        if _sha256(source) != source_hash:
            raise ValueError(f"Source observations changed during preflight: {source}")
        row.update({"source_status": observed.quality.status,
                    "observation_rows": len(observed.frame), "quality": asdict(observed.quality)})
        if observed.quality.status != "available":
            row.update({"status": f"skipped_{observed.quality.status}",
                        "reason": " ".join(observed.quality.issues) or "No usable observations are available."})
            return row, None

        stage = "periods_and_forcing"
        periods = config["periods"]
        split = CalibrationValidationSplit(DatePeriod(**periods["calibration"]),
            DatePeriod(**periods["validation"]), warmup_steps=periods["warmup_steps"])
        continuous = select_period(observed, split.simulation_period)
        if not np.isfinite(continuous.frame.inflow_m3s.to_numpy()).all():
            raise ValueError("Preflight requires finite inflow throughout spinup, calibration, gaps and validation.")
        calibration = select_period(observed, split.calibration)
        validation = select_period(observed, split.validation)
        CalibrationObjective._require_defined_objectives(calibration)
        CalibrationObjective._require_defined_objectives(validation)

        stage = "parameterization"
        physical = _explicit_model(config)
        supplied_model = dict(physical)
        channel = physical.pop("channel_capacity_m3s", None)
        method = config.get("quantile_method", "hazen")
        target_percentiles = tuple(config.get("target_percentiles", (10, 45, 85)))
        estimates = estimate_generalized_parameters(calibration, target_percentiles,
            channel_capacity_m3s=channel, quantile_method=method)
        template = ReservoirParameters.from_generalized(estimates, **physical)
        bounds = estimate_calibration_bounds(observed, split.calibration, template, quantile_method=method)
        # Catch a custom generalized percentile choice outside the optimization
        # box now; the search requires its first baseline to be inside the box.
        bounds.validate(parameters_to_vector(template), template)

        stage = "objective_configuration"
        CalibrationObjective(observed, split.calibration, template, warmup_steps=split.warmup_steps,
            bounds=bounds, **config.get("simulation", {}))
        settings = NSGA2Settings(**config.get("optimization", {}))
        if policy is not None and settings.population_size != policy.population_size:
            raise ValueError("Optimizer population_size must match the manifest stopping policy population_size.")

        destination = output / identifier
        normalized_file = destination / "observations_si.csv"
        resolved_file = destination / "resolved_config.json"
        provenance_file = destination / "preflight_provenance.json"
        model = {name: getattr(template, name) for name in _MODEL_FIELDS}
        model_sources = {name: "explicit_configuration" if name in supplied_model else "generic_default"
                         for name in _MODEL_FIELDS}
        model_sources["channel_capacity_m3s"] = (
            "explicit_configuration" if channel is not None else "calibration_release_p99")
        resolved = deepcopy(config)
        resolved.update({"observations_file": str(normalized_file), "model": model,
                         "optimization": asdict(settings), "quantile_method": method,
                         "target_percentiles": list(estimates.target_percentiles)})
        resolved["preflight"] = {"reservoir_id": identifier, "source_observations_file": str(source),
            "source_configuration_file": str(config_path), "provenance_file": str(provenance_file),
            "observation_units": {"storage": "m3", "flow": "m3/s"},
            "stopping_policy_role": "illustrative_diagnostic_not_a_validated_production_budget"}
        counts = {"all_observation_rows": len(observed.frame), "simulation_rows": len(continuous.frame),
                  "spinup_rows": split.warmup_steps, "calibration_rows": len(calibration.frame),
                  "validation_rows": len(validation.frame),
                  "unscored_gap_rows": len(continuous.frame) - split.warmup_steps - len(calibration.frame) - len(validation.frame)}
        for label, data in (("calibration", calibration), ("validation", validation)):
            counts[f"{label}_storage_observations"] = int(data.frame.storage_m3.notna().sum())
            counts[f"{label}_release_observations"] = int(data.frame.release_m3s.notna().sum())
        provenance = {
            "schema_version": 1, "reservoir_id": identifier, "status": "ready",
            "source_status": observed.quality.status, "source_file": str(source),
            "configuration_file": str(config_path), "manifest_file": str(manifest_path),
            "source_sha256": source_hash, "configuration_sha256": config_hash,
            "manifest_sha256": manifest_hash, "input_observation_units": deepcopy(units),
            "output_observation_units": {"storage": "m3", "flow": "m3/s"},
            "read_options": deepcopy(read_options), "quality": asdict(observed.quality),
            "counts": counts, "model": model, "model_source_statuses": model_sources,
            "periods": {"simulation": split.simulation_period.as_dict(),
                        "calibration": split.calibration.as_dict(), "validation": split.validation.as_dict()},
            "warmup_steps": split.warmup_steps, "quantile_method": method,
            "target_percentiles": list(estimates.target_percentiles), "bounds": bounds.summary(),
            "optimization": asdict(settings), "stopping_policy": asdict(policy) if policy else None,
            "stopping_policy_source": policy_source,
            "stopping_policy_role": "illustrative_diagnostic_not_a_validated_production_budget",
            "simulation_runs": 0, "optimization_evaluations": 0,
            "notes": [
                "Targets, candidate bounds and inferred channel capacity use calibration observations only.",
                "Validation is checked for reporting coverage and defined NSE; no validation score is computed.",
                "Readiness checks configuration and data, not candidate simulation feasibility or search convergence.",
                "Reservoir capacity, initial state and inflow correction are explicit per-reservoir inputs.",
                "Source observations and configuration are preserved; normalized SI data are new artifacts.",
                "Stopping thresholds and caps are illustrative and have not been validated as fleet budgets.",
            ],
        }
        row.update({"status": "ready", "reason": "Data, parameterization and search configuration passed preflight; no simulation was run.",
            "prepared_directory": str(destination), "observations_si_file": str(normalized_file),
            "resolved_configuration_file": str(resolved_file), "provenance_file": str(provenance_file),
            "counts": counts, "model": model, "model_source_statuses": model_sources})
        return row, (destination, observed.frame.copy(deep=True), resolved, json_ready(provenance))
    # This is the job-isolation boundary: parser backends have several unrelated
    # exception classes for malformed CSV/Excel files. Keep their type/reason
    # visible while allowing the remaining reservoirs to complete preparation.
    except Exception as exc:
        row.update({"status": "error", "reason": f"{type(exc).__name__}: {exc}", "failure_stage": stage})
        return row, None


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_fleet_preflight(manifest_path: str | Path, repo_root: str | Path, output_dir: str | Path) -> dict:
    """Read and validate every job, then write isolated ready-job SI artifacts.

    Manifest/configuration/stopping-policy paths are relative to ``repo_root``;
    the manifest itself may also be an absolute path. Observation paths follow
    the existing search configuration convention: relative to ``repo_root`` or
    an explicit absolute local path. IDs are portable directory names, unique
    ignoring case. Invalid job inputs do not stop other jobs; unsafe manifest
    structure or an unsafe/stale output directory raises before any writes.

    All input reads and hashes complete before the first output is created.
    The output must be absent or empty and cannot contain any configured input,
    even an input currently missing. Artifacts are created exclusively.
    """
    root, output = Path(repo_root).resolve(), Path(output_dir).resolve()
    if not root.is_dir():
        raise ValueError("repo_root must be an existing directory.")
    path = (root / Path(manifest_path)).resolve()
    _check_output(output)
    hashes = {}
    protected = {path}
    manifest, manifest_hash = _read_json(path, hashes)
    jobs = _validate_jobs(manifest)
    policy = None
    policy_source = None
    if "stopping_policy_file" in manifest:
        policy_path = _relative_input(root, manifest["stopping_policy_file"], "stopping_policy_file")
        protected.add(policy_path)
        policy_config, policy_hash = _read_json(policy_path, hashes)
        policy = StoppingPolicy(**policy_config)
        policy_source = {"path": str(policy_path), "sha256": policy_hash, "illustrative": True}

    rows, prepared = [], []
    for job in jobs:
        row, artifacts = _prepare_job(job, root, output, policy, path, manifest_hash, policy_source, hashes, protected)
        rows.append(row)
        if artifacts is not None:
            prepared.append(artifacts)
    if any(source == output or source.is_relative_to(output) for source in protected):
        raise ValueError("The output directory would contain a manifest, configuration or observation input; choose another output directory.")
    # Complete the last reads before creating output. A changing source cannot
    # silently receive a hash that describes a different preflight snapshot.
    for source, digest in hashes.items():
        if _sha256(source) != digest:
            raise ValueError(f"Input changed during preflight: {source}")
    code_hashes = {name: _sha256(Path(__file__).with_name(name)) for name in ("fleet.py", "stopping.py")}
    status_counts = Counter(row["status"] for row in rows)
    report = json_ready({
        "schema_version": 1, "experiment": "reservoir_fleet_preflight",
        "manifest_file": str(path), "manifest_sha256": manifest_hash,
        "repo_root": str(root), "output_directory": str(output), "job_count": len(rows),
        "status_counts": {status: status_counts.get(status, 0) for status in
                          ("ready", "skipped_placeholder", "skipped_unavailable", "error")},
        "stopping_policy": asdict(policy) if policy else None, "stopping_policy_source": policy_source,
        "stopping_policy_role": "illustrative_diagnostic_not_a_validated_production_budget",
        "simulation_runs": 0, "optimization_evaluations": 0, "jobs": rows,
        "code_sha256": code_hashes,
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "dztr": __version__},
        "notes": [
            "This command performs preparation only; no simulation, objective evaluation or optimization is run.",
            "Available reservoirs are checked independently; skipped and invalid inputs remain explicit statuses.",
            "A ready status does not establish candidate simulation feasibility, convergence or parameter identifiability.",
            "Use each ready job's resolved_config.json with existing single-reservoir workflows.",
            "Fleet scheduling, cache/resume and automatic stopping are future work.",
            "The illustrative stopping policy is recorded separately and never substitutes its cap for the configured search budget.",
        ],
    })
    _check_output(output)
    output.mkdir(parents=True, exist_ok=True)
    for destination, frame, resolved, provenance in prepared:
        destination.mkdir()
        with (destination / "observations_si.csv").open("x", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=True, index_label="date")
        _write_json(destination / "resolved_config.json", json_ready(resolved))
        _write_json(destination / "preflight_provenance.json", provenance)
    with (output / "fleet_status.csv").open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame([{name: row.get(name) for name in _STATUS_COLUMNS} for row in rows],
                     columns=_STATUS_COLUMNS).to_csv(stream, index=False)
    _write_json(output / "summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("configs/fleet_manifest.example.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/fleet_preflight"))
    args = parser.parse_args()
    report = run_fleet_preflight(args.manifest, args.repo, args.output)
    print(pd.read_csv(args.output / "fleet_status.csv")[["reservoir_id", "status", "reason"]].to_string(index=False))
    print(f"Prepared {report['status_counts']['ready']} of {report['job_count']} reservoirs; no simulations or optimization evaluations.")
    print(f"Artifacts: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
