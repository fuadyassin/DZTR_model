"""Atomic, non-pickle storage for complete NSGA-II generation checkpoints.

One compressed NPZ holds numeric arrays and UTF-8 JSON metadata. SHA256 digests
cover each array and canonical metadata; caller-supplied input/code/version
identity must match before a checkpoint can be loaded or replaced. A temporary
file in the destination directory is flushed and atomically replaced, leaving
the preceding checkpoint available if preparation or replacement fails.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from zipfile import BadZipFile

import numpy as np

from .optimizer import (
    EvaluatedCandidate,
    GenerationRecord,
    NSGA2Settings,
    OptimizerState,
    validate_optimizer_state,
)


_FORMAT = "dztr-nsga2-generation-checkpoint"
_SCHEMA_VERSION = 1
_ARRAY_NAMES = ("lower_vector", "upper_vector", "baseline_vector", "vectors", "objectives")
_METADATA_NAME = "metadata_json"
_METADATA_FIELDS = {
    "format", "schema_version", "identity", "settings", "generation", "boundary_type",
    "complete_generation", "rng_state", "evaluated", "history", "population_ids",
    "arrays", "metadata_sha256",
}
_ATOMIC_REPLACE_ATTEMPTS = 5
_ATOMIC_REPLACE_RETRY_SECONDS = 0.1


def _json_bytes(value) -> bytes:
    def check_keys(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("Checkpoint JSON object keys must be strings.")
            for nested in item.values():
                check_keys(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                check_keys(nested)

    check_keys(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Checkpoint metadata and identity must be finite JSON-compatible values.") from exc


def _identity(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Checkpoint identity must be a JSON object of input/code/version provenance.")
    # Canonical JSON also detaches nested caller-owned identity containers.
    return json.loads(_json_bytes(value).decode("utf-8"))


def _unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Checkpoint JSON cannot contain duplicate object keys.")
        value[key] = item
    return value


def _array_descriptor(array: np.ndarray) -> dict:
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def _payload(state: OptimizerState, identity: dict) -> dict[str, np.ndarray]:
    arrays = {
        "lower_vector": state.lower_vector,
        "upper_vector": state.upper_vector,
        "baseline_vector": state.baseline_vector,
        "vectors": np.asarray([point.vector for point in state.evaluated]),
        "objectives": np.asarray([point.objectives for point in state.evaluated]),
    }
    arrays = {name: np.ascontiguousarray(value, dtype="<f8") for name, value in arrays.items()}
    metadata = {
        "format": _FORMAT, "schema_version": _SCHEMA_VERSION, "identity": identity,
        "settings": asdict(state.settings), "generation": state.generation,
        "boundary_type": state.boundary_type, "complete_generation": state.complete_generation,
        "rng_state": state.rng_state, "population_ids": list(state.population_ids),
        "evaluated": [{"evaluation_id": point.evaluation_id, "generation": point.generation,
                       "feasible": point.feasible, "reason": point.reason} for point in state.evaluated],
        "history": [asdict(record) for record in state.history],
        "arrays": {name: _array_descriptor(array) for name, array in arrays.items()},
    }
    metadata["metadata_sha256"] = hashlib.sha256(_json_bytes(metadata)).hexdigest()
    arrays[_METADATA_NAME] = np.frombuffer(_json_bytes(metadata), dtype=np.uint8).copy()
    return arrays


def _replace_checkpoint(source: Path, destination: Path) -> None:
    """Retry transient Windows access/sharing errors without touching old data.

    Scanner or short-lived reader handles can prevent atomic replacement on
    Windows. Retain the already flushed temporary file for at most five attempts;
    other OS failures, including uncoded PermissionError, propagate immediately.
    """
    for attempt in range(_ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32) or attempt == _ATOMIC_REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_ATOMIC_REPLACE_RETRY_SECONDS)


def load_optimizer_checkpoint(path: str | Path, expected_identity: dict) -> OptimizerState:
    """Load and fully validate one complete generation without model calls.

    Identity is compared in canonical JSON form, before state reconstruction.
    Unknown schema/fields, malformed RNG, altered arrays/metadata, incomplete
    generations and inconsistent survivor histories raise ``ValueError``.
    NumPy loading always disables pickle. A missing file remains an ordinary
    ``FileNotFoundError`` so a caller can distinguish it from corrupt content.
    """
    expected = _identity(expected_identity)
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)) or set(archive.files) != {*_ARRAY_NAMES, _METADATA_NAME}:
                raise ValueError("Checkpoint must contain exactly its numeric arrays and JSON metadata.")
            encoded = archive[_METADATA_NAME]
            if encoded.dtype != np.dtype("uint8") or encoded.ndim != 1:
                raise ValueError("Checkpoint metadata must be one-dimensional UTF-8 JSON bytes.")
            metadata = json.loads(encoded.tobytes().decode("utf-8"),
                                  object_pairs_hook=_unique_json_object)
            if not isinstance(metadata, dict) or set(metadata) != _METADATA_FIELDS:
                raise ValueError("Checkpoint metadata fields are missing or unsupported.")
            if metadata["format"] != _FORMAT or isinstance(metadata["schema_version"], bool) or not isinstance(metadata["schema_version"], int) or metadata["schema_version"] != _SCHEMA_VERSION:
                raise ValueError("Unsupported optimizer checkpoint format or schema version.")
            recorded_digest = metadata["metadata_sha256"]
            canonical_metadata = {name: value for name, value in metadata.items() if name != "metadata_sha256"}
            if recorded_digest != hashlib.sha256(_json_bytes(canonical_metadata)).hexdigest():
                raise ValueError("Checkpoint metadata checksum does not match its content.")
            if _json_bytes(_identity(metadata["identity"])) != _json_bytes(expected):
                raise ValueError("Checkpoint identity does not match the requested input/code/version identity.")
            descriptors = metadata["arrays"]
            if not isinstance(descriptors, dict) or set(descriptors) != set(_ARRAY_NAMES):
                raise ValueError("Checkpoint array descriptions are missing or unsupported.")
            arrays = {}
            for name in _ARRAY_NAMES:
                array = archive[name]
                if array.dtype != np.dtype("<f8"):
                    raise ValueError("Checkpoint numeric arrays must be float64 values without pickle.")
                if descriptors[name] != _array_descriptor(array):
                    raise ValueError(f"Checkpoint array checksum, shape or dtype does not match: {name}.")
                arrays[name] = array.copy()
        trials, history = metadata["evaluated"], metadata["history"]
        if not isinstance(trials, list) or not isinstance(history, list):
            raise ValueError("Checkpoint trial/history metadata must be lists.")
        if arrays["vectors"].shape != (len(trials), 72) or arrays["objectives"].shape != (len(trials), 2):
            raise ValueError("Checkpoint candidate matrices must match the trial count and native dimensions.")
        evaluated = []
        for index, trial in enumerate(trials):
            if not isinstance(trial, dict) or set(trial) != {"evaluation_id", "generation", "feasible", "reason"}:
                raise ValueError("Checkpoint trial metadata contains missing or unsupported fields.")
            evaluated.append(EvaluatedCandidate(vector=arrays["vectors"][index],
                objectives=tuple(arrays["objectives"][index]), **trial))
        generation_history = []
        for record in history:
            if not isinstance(record, dict) or set(record) != {"generation", "evaluations", "feasible_count", "front_count"}:
                raise ValueError("Checkpoint generation history contains missing or unsupported fields.")
            generation_history.append(GenerationRecord(**record))
        settings = metadata["settings"]
        if not isinstance(settings, dict) or set(settings) != set(NSGA2Settings.__dataclass_fields__):
            raise ValueError("Checkpoint settings must contain every NSGA-II setting exactly once.")
        state = OptimizerState(NSGA2Settings(**settings), arrays["lower_vector"], arrays["upper_vector"],
            arrays["baseline_vector"], tuple(evaluated), tuple(generation_history),
            metadata["population_ids"], metadata["rng_state"], metadata["generation"],
            metadata["boundary_type"], metadata["complete_generation"])
        return validate_optimizer_state(state)
    except FileNotFoundError:
        raise
    except (BadZipFile, EOFError, OSError, TypeError, ValueError, KeyError, OverflowError) as exc:
        raise ValueError(f"Invalid optimizer checkpoint {source}: {exc}") from exc


def save_optimizer_checkpoint(path: str | Path, state: OptimizerState, identity: dict) -> None:
    """Atomically create/replace an identity-compatible checkpoint in one file.

    Existing destinations must already be valid checkpoints for this identity;
    arbitrary files are protected from replacement. Validation and compression
    occur before atomic replacement. Failed writes/replacements remove only the
    newly created temporary file and preserve the preceding checkpoint.
    """
    canonical_identity = _identity(identity)
    state = validate_optimizer_state(state)
    destination = Path(path)
    if destination.is_symlink():
        raise ValueError("Checkpoint destination must be an actual file, not a symbolic link.")
    if destination.exists():
        if not destination.is_file():
            raise ValueError("Checkpoint destination must be a file path.")
        load_optimizer_checkpoint(destination, canonical_identity)
    arrays = _payload(state, canonical_identity)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp",
                                         dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_checkpoint(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
