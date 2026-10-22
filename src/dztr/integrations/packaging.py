"""Package the declared reservoir samples without refitting or rewriting data."""

from __future__ import annotations

from collections import Counter
import copy
import csv
from datetime import date
import hashlib
import io
import json
import os
from pathlib import Path, PureWindowsPath
import re
import tempfile


_DATA_FILES = ("observations_si.csv", "observation_provenance.json", "matlab_source.json",
               "legacy_targets_si.csv", "legacy_bounds_si.csv")
_REFERENCES = {"provenance_file": "observation_provenance.json",
               "matlab_source_file": "matlab_source.json",
               "legacy_targets_file": "legacy_targets_si.csv", "legacy_bounds_file": "legacy_bounds_si.csv"}
_STAGES = ("baseline_status", "optimization_status", "legacy_status")
_SI = {"storage": "m3", "flows": "m3/s"}


def _digest(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _json(contents: bytes, path: Path) -> dict:
    try:
        value = json.loads(contents.decode("utf-8-sig"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON input: {path.name}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}.")
    return value


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _within(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes the repository: {path}.")
    return resolved


def _relative_path(root: Path, value: str | Path) -> Path:
    text = str(value).replace("\\", "/")
    if not text or Path(text).is_absolute() or PureWindowsPath(text).drive or ".." in Path(text).parts:
        raise ValueError(f"Expected a repository-relative path without traversal: {value}.")
    return _within(root, root / text)


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _id(value: object) -> str:
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value)
            or value.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}):
        raise ValueError(f"Unsafe reservoir_id: {value!r}.")
    return value


def _indexed(rows: object, label: str) -> dict[str, dict]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list.")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} entries must be objects.")
        reservoir_id = _id(row.get("reservoir_id"))
        if reservoir_id.casefold() in {key.casefold() for key in result}:
            raise ValueError(f"Duplicate reservoir_id in {label}: {reservoir_id}.")
        result[reservoir_id] = row
    return result


def _csv_metadata(contents: bytes) -> dict:
    rows = csv.reader(io.StringIO(contents.decode("utf-8-sig"), newline=""))
    if next(rows, None) != ["date", "storage_m3", "inflow_m3s", "release_m3s"]:
        raise ValueError("Normalized observations require date/storage_m3/inflow_m3s/release_m3s columns.")
    count, first, previous = 0, None, None
    for row in rows:
        if not row:
            continue
        if len(row) != 4:
            raise ValueError("Normalized observation rows must contain exactly four columns.")
        current = date.fromisoformat(row[0])
        if previous is not None and current <= previous:
            raise ValueError("Normalized observation dates must be sorted and unique.")
        first = current if first is None else first
        previous, count = current, count + 1
    if not count:
        raise ValueError("Normalized observations must not be empty.")
    return {"rows": count, "first_date": first.isoformat(), "last_date": previous.isoformat()}


class _PackagePlan:
    def __init__(self, root: Path):
        self.root = root
        self.sources: dict[Path, bytes] = {}
        self.destinations: dict[Path, bytes] = {}

    def read(self, path: Path) -> bytes:
        path = _within(self.root, path)
        if path not in self.sources:
            if not path.is_file():
                raise ValueError(f"Required packaging source is missing: {_relative(self.root, path)}.")
            self.sources[path] = path.read_bytes()
        return self.sources[path]

    def add(self, path: Path, contents: bytes) -> None:
        # Reject even in-repository destination aliases: they may point into
        # an archive that this command promises to keep unchanged.
        resolved = _within(self.root, path)
        if resolved != path:
            raise ValueError(f"Destination contains a symbolic path alias: {path}.")
        if path in self.destinations and self.destinations[path] != contents:
            raise ValueError(f"Conflicting planned destination: {path}.")
        self.destinations[path] = contents

    def source_hashes(self) -> dict[str, str]:
        return {_relative(self.root, path): _digest(contents) for path, contents in sorted(self.sources.items())}

    def guard_sources(self) -> None:
        for path, expected in self.sources.items():
            if not path.is_file() or path.read_bytes() != expected:
                raise ValueError(f"Packaging source changed: {_relative(self.root, path)}.")

    def preflight(self) -> None:
        for path, contents in self.destinations.items():
            if path in self.sources:
                raise ValueError(f"Packaging destination overlaps a source input: {path}.")
            if path.exists() and (not path.is_file() or path.read_bytes() != contents):
                raise ValueError(f"Destination differs; existing files will not be overwritten: {_relative(self.root, path)}.")
            for parent in path.parents:
                if parent == self.root:
                    break
                if parent.exists() and not parent.is_dir():
                    raise ValueError(f"A destination parent is not a directory: {parent}.")
        self.guard_sources()

    def install(self, manifest_path: Path) -> None:
        self.preflight()
        created: list[Path] = []
        directories: list[Path] = []
        try:
            # The manifest is the final completeness marker. An interrupted
            # copy cannot advertise a completed collection.
            order = [path for path in self.destinations if path != manifest_path] + [manifest_path]
            for path in order:
                contents = self.destinations[path]
                if path.is_file():
                    if path.read_bytes() != contents:
                        raise ValueError(f"Destination changed during packaging: {path}.")
                    continue
                missing = [parent for parent in path.parents if parent != self.root and not parent.exists()]
                for parent in reversed(missing):
                    parent.mkdir(exist_ok=True)
                    directories.append(parent)
                handle, temporary = tempfile.mkstemp(prefix=".package-", suffix=".tmp", dir=path.parent)
                try:
                    with os.fdopen(handle, "wb") as stream:
                        stream.write(contents)
                        stream.flush()
                        os.fsync(stream.fileno())
                    try:
                        # Atomic publication without replacement, on Windows
                        # and POSIX; the temporary file is on the same volume.
                        os.link(temporary, path)
                        created.append(path)
                    except FileExistsError:
                        if not path.is_file() or path.read_bytes() != contents:
                            raise ValueError(f"Destination appeared with different bytes: {path}.")
                finally:
                    Path(temporary).unlink(missing_ok=True)
            self.guard_sources()
            for path, expected in self.destinations.items():
                if path.read_bytes() != expected:
                    raise ValueError(f"Packaged file changed during installation: {path}.")
        except BaseException:
            # Remove only files created by this call whose bytes remain ours;
            # never remove an existing file or overwrite a concurrent edit.
            for path in reversed(created):
                if path.is_file() and path.read_bytes() == self.destinations[path]:
                    path.unlink()
            for path in reversed(directories):
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
            raise


def package_world_samples(repo_root: str | Path, *, scope_file: str | Path = "configs/world_reservoir_scope.json",
                          manifest_file: str | Path = "data/reservoirs/manifest.json") -> dict:
    """Package only the explicit scope, returning its permanent manifest.

    All source/destination checks finish before any directory or file is
    created. Existing byte-identical files are accepted; differing files are
    never replaced. The command reads normalized archive outputs only, not
    original E-drive inputs, and performs no reservoir simulations or search.
    """
    root = Path(repo_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repo_root must be a directory.")
    plan = _PackagePlan(root)
    scope_path, manifest_path = _relative_path(root, scope_file), _relative_path(root, manifest_file)
    scope = _json(plan.read(scope_path), scope_path)
    if scope.get("schema_version") != 1 or scope.get("kind") != "world_reservoir_scope" or scope.get("paths_relative_to") != "repository_root":
        raise ValueError("Expected a version 1 repository-relative world_reservoir_scope.")
    catalog_path = _relative_path(root, scope["catalog_file"])
    catalog_bytes = plan.read(catalog_path)
    if _digest(catalog_bytes) != scope.get("catalog_sha256"):
        raise ValueError("Scope catalog SHA256 changed; review the scope before packaging.")
    catalog = _json(catalog_bytes, catalog_path)
    if catalog.get("schema_version") != 1 or catalog.get("kind") != "world_reservoir_sample_catalog":
        raise ValueError("Expected a version 1 world_reservoir_sample_catalog.")
    cohort_path = _relative_path(root, catalog["cohort_report"])
    cohort_bytes = plan.read(cohort_path)
    cohort = _json(cohort_bytes, cohort_path)
    if cohort.get("schema_version") != 1 or cohort.get("source_files_unchanged") is not True or cohort.get("code_files_unchanged") is not True:
        raise ValueError("Historical cohort must confirm unchanged source and code files.")
    verification_path = cohort_path.with_name("verification.json")
    if verification_path.is_file():
        verification = _json(plan.read(verification_path), verification_path)
        if verification.get("status") != "passed" or verification.get("cohort_sha256") != _digest(cohort_bytes):
            raise ValueError("Historical verification does not match the cohort SHA256.")
    active = _indexed(scope["active_reservoirs"], "active_reservoirs")
    excluded = _indexed(scope["excluded_reservoirs"], "excluded_reservoirs")
    catalog_rows, cohort_rows = _indexed(catalog["reservoirs"], "catalog"), _indexed(cohort["reservoirs"], "cohort")
    if (set(active) & set(excluded) or set(active) | set(excluded) != set(catalog_rows)
            or set(catalog_rows) != set(cohort_rows)):
        raise ValueError("Scope, exclusions, catalog and cohort reservoir IDs must agree exactly.")
    if (scope.get("source_reservoir_count") != len(catalog_rows)
            or scope.get("active_reservoir_count") != len(active) or not active):
        raise ValueError("Scope reservoir counts disagree with its declared entries.")
    records = []
    for reservoir_id, declared in active.items():
        old_record, historical = catalog_rows[reservoir_id], cohort_rows[reservoir_id]
        if old_record.get("import_status") != "imported" or historical.get("import_status") != "imported":
            raise ValueError(f"Active reservoir is not an imported dataset: {reservoir_id}.")
        config_path = _relative_path(root, declared["sample_config"])
        if config_path != _relative_path(root, old_record["sample_config"]):
            raise ValueError(f"Scope/catalog configuration mismatch: {reservoir_id}.")
        config = _json(plan.read(config_path), config_path)
        if config.get("schema_version") != 1 or config.get("kind") != "world_reservoir_sample" or config.get("reservoir", {}).get("id") != reservoir_id:
            raise ValueError(f"Invalid source sample configuration: {reservoir_id}.")
        if config.get("units") != _SI:
            raise ValueError(f"Source sample units must be SI: {reservoir_id}.")
        for field in ("model", "sampling", "periods"):
            if config.get(field) != old_record.get(field):
                raise ValueError(f"Catalog/source {field} settings differ: {reservoir_id}.")
        frequency = config["sampling"].get("frequency")
        if frequency not in ("daily", "monthly") or any(row.get("frequency") != frequency for row in (old_record, historical)):
            raise ValueError(f"Sampling frequency disagreement: {reservoir_id}.")
        source_observations = _within(root, config_path.parent / config["observations_file"])
        observation_bytes = plan.read(source_observations)
        if _digest(observation_bytes) != config.get("observations_sha256"):
            raise ValueError(f"Source observations SHA256 changed: {reservoir_id}.")
        metadata = _csv_metadata(observation_bytes)
        if any(historical.get(key) != value for key, value in metadata.items()):
            raise ValueError(f"Observation dates/rows disagree with historical cohort: {reservoir_id}.")
        permanent_config = root / "configs" / "reservoirs" / (reservoir_id + ".json")
        permanent_data = root / "data" / "reservoirs" / reservoir_id
        config_out = copy.deepcopy(config)
        config_out["observations_file"] = Path(os.path.relpath(permanent_data / "observations_si.csv", permanent_config.parent)).as_posix()
        record_files = {}
        for filename in _DATA_FILES:
            source = source_observations if filename == "observations_si.csv" else config_path.parent / filename
            contents = plan.read(source)
            destination = permanent_data / filename
            plan.add(destination, contents)
            record_files[_relative(root, destination)] = _digest(contents)
        for key, filename in _REFERENCES.items():
            config_out[key] = Path(os.path.relpath(permanent_data / filename, permanent_config.parent)).as_posix()
        config_bytes = _json_bytes(config_out)
        plan.add(permanent_config, config_bytes)
        record_files[_relative(root, permanent_config)] = _digest(config_bytes)
        readiness = {}
        for stage in _STAGES:
            if not isinstance(historical.get(stage), str) or historical[stage] != old_record.get(stage):
                raise ValueError(f"Historical readiness disagreement for {reservoir_id}: {stage}.")
            readiness[stage] = historical[stage]
        readiness["reasons"] = list(dict.fromkeys(value for key, value in historical.items()
            if (key == "reason" or key.endswith("_reason")) and isinstance(value, str) and value))
        records.append({"reservoir_id": reservoir_id, "name": config["reservoir"]["name"],
                        "config_file": _relative(root, permanent_config), "config_sha256": _digest(config_bytes),
                        "observations_file": _relative(root, permanent_data / "observations_si.csv"),
                        "observations_sha256": _digest(observation_bytes), "frequency": frequency, **metadata,
                        "readiness": readiness, "files_sha256": record_files})
    receipt_path = root / "data" / "reservoirs" / "package_receipt.json"
    manifest = {"schema_version": 1, "kind": "reservoir_collection",
                "repository_root": Path(os.path.relpath(root, manifest_path.parent)).as_posix(),
                "paths_relative_to": "repository_root", "units": _SI,
                "reservoir_count": len(records), "excluded_reservoirs": copy.deepcopy(scope["excluded_reservoirs"]),
                "counts": {"frequency": dict(Counter(record["frequency"] for record in records)),
                           **{stage: dict(Counter(record["readiness"][stage] for record in records)) for stage in _STAGES}},
                "reservoirs": records,
                "provenance": {"scope_file": _relative(root, scope_path), "scope_sha256": _digest(plan.sources[scope_path]),
                               "catalog_file": _relative(root, catalog_path), "catalog_sha256": _digest(catalog_bytes),
                               "cohort_file": _relative(root, cohort_path), "cohort_sha256": _digest(cohort_bytes),
                               "package_receipt_file": _relative(root, receipt_path)}}
    manifest_bytes = _json_bytes(manifest)
    plan.add(manifest_path, manifest_bytes)
    receipt = {"schema_version": 1, "kind": "reservoir_collection_package_receipt",
               "paths_relative_to": "repository_root", "units": _SI,
               "reservoir_count": len(records), "excluded_reservoirs": copy.deepcopy(scope["excluded_reservoirs"]),
               "source_files_sha256": plan.source_hashes(),
               "packaged_files_sha256": {_relative(root, path): _digest(contents) for path, contents in sorted(plan.destinations.items())},
               "manifest_file": _relative(root, manifest_path), "manifest_sha256": _digest(manifest_bytes),
               "source_policy": "Only normalized files declared in the scope; original source drives are not read.",
               "copy_policy": "Byte-exact data/evidence; experiment settings preserved; relative references relocated."}
    plan.add(receipt_path, _json_bytes(receipt))
    plan.install(manifest_path)
    return manifest
