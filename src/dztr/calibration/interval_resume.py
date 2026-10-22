"""Verified, calibration-only continuation of a frozen interval seed.

The original input, numerical code, settings and saved costs remain binding.
Native and continued checkpoints retain their original identity. Only an
explicit, parent-manifest-bound migration may change this adapter's hash;
all numerical model modules remain fixed. Stopping never controls search.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pandas as pd

from dztr import __version__

from .baseline import json_ready
from .checkpoint import load_optimizer_checkpoint, save_optimizer_checkpoint
from .convergence import checkpoint_metrics
from .interval_search import (
    ARTIFACT_NAMES, HYPERVOLUME_NORMALIZATION_AREA, HYPERVOLUME_REFERENCE,
    _RecordedObjective, _checkpoints, _module_hashes, _read_object, _sha256,
    _trial_row, _vectors_frame, load_interval_problem,
)
from .optimizer import optimize_nsga2
from .selection import select_pareto_representatives
from .stopping import StoppingPolicy, assess_stopping


DEFAULT_CHECKPOINTS = (16, 32, 64, 96, 128, 160, 192, 224, 256)
APPROVED_ADAPTER = "calibration/interval_resume.py"


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _verified_artifacts(previous: Path):
    """Authenticate receipts using the parent's independently frozen map."""
    verification_path = previous.parent / "verification.json"
    verification = _read_object(verification_path)
    if verification.get("status") != "passed":
        raise ValueError("Previous verification must have status passed.")
    artifact_map = verification.get("artifact_sha256")
    if not isinstance(artifact_map, dict):
        raise ValueError("Previous verification requires an artifact_sha256 map.")
    hashes = {}
    for name in ARTIFACT_NAMES:
        path = previous / name
        key = path.relative_to(verification_path.parent).as_posix()
        expected = artifact_map.get(key)
        if not isinstance(expected, str) or _sha256(path) != expected:
            raise ValueError(f"Previous artifact SHA256 does not match frozen verification: {key}")
        hashes[name] = expected
    return verification_path, _sha256(verification_path), hashes


def _extended_identity(previous_identity, current_hashes, approved_adapter_migration=None):
    old = previous_identity.get("code_sha256")
    if not isinstance(old, dict) or APPROVED_ADAPTER not in current_hashes:
        raise ValueError("Checkpoint requires its historical code hashes and the approved resume adapter.")
    expected = dict(old)
    expected.setdefault(APPROVED_ADAPTER, current_hashes[APPROVED_ADAPTER])
    adapter_changed = APPROVED_ADAPTER in old and old[APPROVED_ADAPTER] != current_hashes[APPROVED_ADAPTER]
    if adapter_changed:
        migration = approved_adapter_migration
        fields = {"old_sha256", "new_sha256", "prior_verification_path", "prior_verification_sha256"}
        if (not isinstance(migration, dict) or set(migration) != fields
                or migration["old_sha256"] != old[APPROVED_ADAPTER]
                or migration["new_sha256"] != current_hashes[APPROVED_ADAPTER]
                or not isinstance(migration["prior_verification_path"], str)
                or not isinstance(migration["prior_verification_sha256"], str)
                or len(migration["prior_verification_sha256"]) != 64):
            raise ValueError("Historical code hash drift requires an explicit matching adapter migration.")
        expected[APPROVED_ADAPTER] = current_hashes[APPROVED_ADAPTER]
    elif approved_adapter_migration is not None:
        raise ValueError("Adapter migration is unnecessary when the adapter hash is unchanged or absent.")
    if current_hashes != expected:
        raise ValueError("Historical code hash drift or an unapproved added module prevents resume.")
    historic = {name: digest for name, digest in old.items() if name != APPROVED_ADAPTER}
    prior_extension = previous_identity.get("identity_extension")
    prior_receipt = previous_identity.get("migration_receipt")
    if (prior_extension is None) != (prior_receipt is None):
        raise ValueError("A continuation identity requires both its bridge and migration receipt.")
    if prior_extension is not None:
        if APPROVED_ADAPTER not in old or not isinstance(prior_extension, dict) or not isinstance(prior_receipt, dict):
            raise ValueError("An original identity cannot contain an unexplained resume bridge.")
        bridge = {name: value for name, value in prior_extension.items() if name != "bridge_sha256"}
        if (prior_extension.get("bridge_sha256") != _digest(bridge)
                or bridge.get("historical_code_sha256") != historic
                or prior_receipt.get("bridge_sha256") != prior_extension["bridge_sha256"]
                or prior_receipt.get("previous_identity_sha256") != bridge.get("previous_identity_sha256")):
            raise ValueError("Previous resume identity bridge does not match the historical code identity.")
        if bridge.get("schema_version") == 1:
            if bridge.get("approved_added_modules") != {APPROVED_ADAPTER: old[APPROVED_ADAPTER]}:
                raise ValueError("Previous adapter addition does not match its saved code identity.")
            origin_codes = dict(historic)
            origin_kind = "pre_adapter"
            additions = dict(bridge["approved_added_modules"])
            migrations = []
        elif bridge.get("schema_version") == 2:
            origin_codes = bridge.get("origin_code_sha256")
            origin_kind = bridge.get("origin_kind")
            additions = bridge.get("approved_added_modules")
            migrations = bridge.get("adapter_code_migrations")
            if (not isinstance(origin_codes, dict) or not isinstance(migrations, list)
                    or {k: v for k, v in origin_codes.items() if k != APPROVED_ADAPTER} != historic
                    or origin_kind not in ("pre_adapter", "native_with_adapter")):
                raise ValueError("Previous native/original code identity is malformed.")
            if origin_kind == "native_with_adapter":
                if APPROVED_ADAPTER not in origin_codes or additions != {}:
                    raise ValueError("Native adapter identity cannot claim an added module.")
                adapter_hash = origin_codes[APPROVED_ADAPTER]
            else:
                if APPROVED_ADAPTER in origin_codes or not isinstance(additions, dict) or set(additions) != {APPROVED_ADAPTER}:
                    raise ValueError("Previous adapter addition is malformed.")
                adapter_hash = additions[APPROVED_ADAPTER]
            for item in migrations:
                if (not isinstance(item, dict) or set(item) != {"old_sha256", "new_sha256", "prior_verification_path", "prior_verification_sha256"}
                        or item["old_sha256"] != adapter_hash
                        or not all(isinstance(item[k], str) for k in item)):
                    raise ValueError("Previous adapter migration chain is malformed.")
                adapter_hash = item["new_sha256"]
            if adapter_hash != old[APPROVED_ADAPTER] or prior_receipt.get("adapter_code_migrations") != migrations:
                raise ValueError("Previous adapter migration chain does not match saved code.")
        else:
            raise ValueError("Unsupported resume identity bridge schema.")
        origin_digest = prior_extension["origin_identity_sha256"]
    else:
        # A fresh run legitimately hashes the installed adapter without ever
        # having been continued. Its original identity remains the origin.
        origin_codes = dict(old)
        origin_kind = "native_with_adapter" if APPROVED_ADAPTER in old else "pre_adapter"
        additions = {} if APPROVED_ADAPTER in old else {APPROVED_ADAPTER: current_hashes[APPROVED_ADAPTER]}
        migrations = []
        origin_digest = _digest(previous_identity)
    migrations = deepcopy(migrations)
    if adapter_changed:
        migrations.append(deepcopy(approved_adapter_migration))
    extension = {
        "schema_version": 2,
        "origin_kind": origin_kind,
        "origin_code_sha256": origin_codes,
        "approved_added_modules": additions,
        "adapter_code_migrations": migrations,
        "historical_code_sha256": historic,
        "previous_identity_sha256": _digest(previous_identity),
        "origin_identity_sha256": origin_digest,
    }
    extension["bridge_sha256"] = _digest(extension)
    identity = deepcopy(previous_identity)
    identity["code_sha256"] = dict(current_hashes)
    identity["identity_extension"] = extension
    identity["migration_receipt"] = {
        "historical_module_count": len(historic), "current_module_count": len(current_hashes),
        "historical_hashes_unchanged": not adapter_changed,
        "historical_numerical_hashes_unchanged": True,
        "origin_module_count": len(origin_codes),
        "only_approved_adapter_added": bool(additions) and not adapter_changed,
        "approved_added_modules": extension["approved_added_modules"],
        "origin_kind": origin_kind,
        "adapter_code_migrations": migrations,
        "adapter_changed_this_extension": adapter_changed,
        "previous_identity_sha256": extension["previous_identity_sha256"],
        "bridge_sha256": extension["bridge_sha256"],
    }
    return identity


def _check_receipt_prefix(previous, report, state):
    count = len(state.evaluated)
    if (report.get("search_evaluations") != count or report.get("checkpoint_evaluations") != count
            or report.get("optimization") != asdict(state.settings)
            or count != state.settings.evaluation_budget
            or report.get("final_population_ids") != list(state.population_ids)):
        raise ValueError("Previous summary must describe the complete saved generation and survivor order.")
    if report.get("checkpoint_saved") is not True:
        raise ValueError("Previous seed has no complete saved checkpoint.")
    trials = pd.read_csv(previous / "evaluations.csv", float_precision="round_trip")
    vectors = pd.read_csv(previous / "evaluated_vectors.csv", index_col="evaluation_id",
                          float_precision="round_trip")
    expected_vectors = _vectors_frame(state.evaluated)
    if (vectors.columns.tolist() != expected_vectors.columns.tolist()
            or vectors.index.tolist() != expected_vectors.index.tolist()
            or not np.array_equal(vectors.to_numpy(), expected_vectors.to_numpy())):
        raise ValueError("Previous evaluated vectors do not exactly match the checkpoint.")
    columns = list(_trial_row(state.evaluated[0]))
    if len(trials) != count or trials.columns.tolist() != columns:
        raise ValueError("Previous trial CSV shape does not match the checkpoint.")
    for actual, point in zip(trials.to_dict("records"), state.evaluated):
        expected = _trial_row(point)
        for name, value in expected.items():
            observed = actual[name]
            if value is None:
                matches = pd.isna(observed)
            elif name == "feasible":
                matches = isinstance(observed, (bool, np.bool_)) and observed == value
            else:
                matches = observed == value
            if not matches:
                raise ValueError(f"Previous trial costs/metadata do not match checkpoint: {name}.")
    history = pd.read_csv(previous / "history.csv", float_precision="round_trip")
    expected_history = pd.DataFrame([asdict(record) for record in state.history])
    if (history.columns.tolist() != expected_history.columns.tolist()
            or not np.array_equal(history.to_numpy(), expected_history.to_numpy())):
        raise ValueError("Previous generation history does not exactly match the checkpoint.")


def _write_preserved_prefix(source, destination, suffix):
    original = source.read_bytes()
    if not original.endswith(b"\n"):
        raise ValueError("Previous CSV prefix must end at a complete row boundary.")
    destination.write_bytes(original + suffix.to_csv(index=False, header=False).encode("utf-8"))
    if destination.read_bytes()[:len(original)] != original:
        raise RuntimeError("Inherited CSV bytes were not preserved.")


def resume_interval_seed(
    previous_seed_dir: str | Path, policy_path: str | Path, output_dir: str | Path, *,
    total_evaluation_budget=256, checkpoints=DEFAULT_CHECKPOINTS,
    stopping_policy: StoppingPolicy | None = None, progress_callback=None,
    approved_adapter_migration: dict | None = None,
) -> dict:
    """Extend a verified complete seed without reevaluating inherited trials.

    ``previous_seed_dir.parent/verification.json`` must independently freeze all
    the prior seed artifacts in ``artifact_sha256``. The checkpoint's complete
    numerical identity is checked before any model evaluation. An original
    checkpoint may gain exactly this adapter. Native checkpoints with the
    adapter already installed need no bridge. Adapter-only changes require
    ``approved_adapter_migration`` with exact old/new hashes and the verified
    parent manifest path/hash; other code drift is refused. Fresh output is
    required. Baseline and rejected trials count in the cumulative budget.
    """
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh output directory; existing results are preserved.")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")
    previous = Path(previous_seed_dir).resolve()
    verification_path, verification_sha, artifact_hashes = _verified_artifacts(previous)
    prior = _read_object(previous / "summary.json")
    old_identity = prior.get("checkpoint_identity")
    if not isinstance(old_identity, dict):
        raise ValueError("Previous summary requires its complete checkpoint identity.")
    for name in ("policy_id", "consumed_files_sha256", "code_sha256", "software_versions",
                 "root_policy", "initial_storage_offset_capacity_fraction", "periods", "bounds"):
        if prior.get(name) != old_identity.get(name):
            raise ValueError(f"Previous summary and checkpoint identity disagree: {name}.")
    code_hashes = _module_hashes()
    if approved_adapter_migration is not None:
        if (not isinstance(approved_adapter_migration, dict)
                or approved_adapter_migration.get("prior_verification_path") != str(verification_path.resolve())
                or approved_adapter_migration.get("prior_verification_sha256") != verification_sha):
            raise ValueError("Adapter migration must name the verified parent manifest and its exact SHA256.")
    checkpoint_identity = _extended_identity(old_identity, code_hashes, approved_adapter_migration)
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "dztr": __version__}
    if versions != old_identity.get("software_versions"):
        raise ValueError("Software version identity drift prevents exact continuation.")
    consumed = old_identity.get("consumed_files_sha256")
    policy_path = Path(policy_path).resolve()
    if not isinstance(consumed, dict) or str(policy_path) not in consumed:
        raise ValueError("Requested policy path is outside the inherited input identity.")
    for name, expected in consumed.items():
        if _sha256(Path(name)) != expected:
            raise ValueError(f"Consumed input SHA256 drift prevents resume: {name}")
    state = load_optimizer_checkpoint(previous / "optimizer_checkpoint.npz", old_identity)
    _check_receipt_prefix(previous, prior, state)
    inherited_count = len(state.evaluated)
    settings = replace(state.settings, evaluation_budget=total_evaluation_budget)
    if settings.evaluation_budget <= state.settings.evaluation_budget:
        raise ValueError("Resume must strictly extend the inherited evaluation budget.")
    if settings.evaluation_budget % settings.population_size:
        raise ValueError("Resume total budget must end at a complete population boundary.")
    checkpoints = _checkpoints(checkpoints, settings.evaluation_budget)
    if old_identity.get("nonbudget_settings") != {
            name: value for name, value in asdict(state.settings).items() if name != "evaluation_budget"}:
        raise ValueError("Checkpoint nonbudget settings disagree with its saved identity.")
    inherited_stopping = prior.get("stopping_policy")
    if inherited_stopping is None:
        raise ValueError("A resumed seed requires an inherited stopping diagnostic policy.")
    policy_values = dict(inherited_stopping)
    policy_values["hypervolume_reference"] = tuple(policy_values["hypervolume_reference"])
    policy_values["max_evaluations"] = settings.evaluation_budget
    expected_stopping = StoppingPolicy(**policy_values)
    if stopping_policy is None:
        stopping_policy = expected_stopping
    if not isinstance(stopping_policy, StoppingPolicy):
        raise TypeError("stopping_policy must be StoppingPolicy or None.")
    if stopping_policy != expected_stopping:
        raise ValueError("Resume stopping diagnostics may extend only max_evaluations; tolerances remain frozen.")
    if (stopping_policy.population_size != settings.population_size
            or stopping_policy.hypervolume_reference != HYPERVOLUME_REFERENCE):
        raise ValueError("Inherited stopping population/reference does not match the interval pilot.")
    problem, baseline, _, context = load_interval_problem(policy_path)
    for name, actual in {
        "policy_id": context["policy"]["policy_id"],
        "consumed_files_sha256": context["consumed_files_sha256"],
        "periods": context["periods"], "bounds": context["bounds"].summary(),
        "root_policy": "reject", "initial_storage_offset_capacity_fraction": 0.0,
    }.items():
        if actual != old_identity.get(name):
            raise ValueError(f"Loaded problem identity drift prevents continuation: {name}.")
    recorded = _RecordedObjective(problem, settings.population_size)
    recorded.records.extend(state.evaluated)
    complete_states = [state]
    new_history = []

    def progress(record):
        new_history.append(record)
        if progress_callback is not None:
            progress_callback(record)

    started = time.perf_counter()
    result = optimize_nsga2(recorded, baseline, settings, resume_state=state,
                           progress_callback=progress,
                           checkpoint_callback=lambda saved: complete_states.__setitem__(slice(None), [saved]))
    elapsed = time.perf_counter() - started
    evaluated, front, final_population = result.evaluated, result.front, result.final_population
    new_calls = len(recorded.records) - inherited_count
    if len(evaluated) != inherited_count + new_calls:
        raise RuntimeError("Resume budget accounting disagrees with recorded objective calls.")
    for original, continued in zip(state.evaluated, evaluated[:inherited_count]):
        if (original.evaluation_id != continued.evaluation_id or original.generation != continued.generation
                or original.feasible != continued.feasible or original.reason != continued.reason
                or original.objectives != continued.objectives
                or not np.array_equal(original.vector, continued.vector)):
            raise RuntimeError("Resume changed an inherited trial vector, cost or receipt.")
    if result.history[:len(state.history)] != state.history:
        raise RuntimeError("Resume changed inherited generation history.")
    for name, expected in consumed.items():
        if _sha256(Path(name)) != expected:
            raise RuntimeError(f"Consumed input changed during resume: {name}")
    if _module_hashes() != code_hashes:
        raise RuntimeError("Model module hashes changed during resume.")
    if (_sha256(verification_path) != verification_sha
            or any(_sha256(previous / name) != digest for name, digest in artifact_hashes.items())):
        raise RuntimeError("Frozen previous verification/artifacts changed during resume.")
    completed_checkpoints = tuple(value for value in checkpoints if value <= len(evaluated))
    diagnostics = []
    for checkpoint in completed_checkpoints:
        row = checkpoint_metrics(evaluated, checkpoint, reference=HYPERVOLUME_REFERENCE)
        row["normalized_hypervolume"] = row["hypervolume"] / HYPERVOLUME_NORMALIZATION_AREA
        diagnostics.append(row)
    assessment = assess_stopping(evaluated, stopping_policy)
    rejected = [point for point in evaluated if not point.feasible]
    representatives = select_pareto_representatives(front) if front else {}
    report = deepcopy(prior)
    report.update({
        "experiment": "reviewed_interval_nsga2_seed_resume", "optimization": asdict(settings),
        "search_evaluations": len(evaluated), "inherited_count": inherited_count,
        "new_calls": new_calls, "total_evaluations": len(evaluated),
        "inherited_evaluations": inherited_count, "new_search_evaluations": new_calls,
        "baseline_reevaluated": False, "inherited_rejected_evaluations": sum(not point.feasible for point in state.evaluated),
        "new_rejected_evaluations": sum(not point.feasible for point in evaluated[inherited_count:]),
        "feasible_evaluations": len(evaluated) - len(rejected), "rejected_evaluations": len(rejected),
        "rejection_reasons": dict(Counter(point.reason for point in rejected)), "pareto_count": len(front),
        "selected_candidate_ids": {"generalized_baseline": 0, **representatives},
        "final_population_ids": [point.evaluation_id for point in final_population],
        "termination": result.termination, "search_elapsed_seconds": elapsed,
        "reporting_calibration_evaluations": 0, "reporting_validation_evaluations": 0,
        "requested_checkpoints": list(checkpoints), "completed_checkpoints": list(completed_checkpoints),
        "stopping_policy": asdict(stopping_policy), "stopping_assessment": asdict(assessment),
        "stopping_controls_search": False, "optimizer_convergence_established": False,
        "code_sha256": code_hashes, "checkpoint_identity": checkpoint_identity,
        "checkpoint_evaluations": len(complete_states[0].evaluated), "checkpoint_saved": True,
        "identity_bridge_sha256": checkpoint_identity["identity_extension"]["bridge_sha256"],
        "migration_receipt": checkpoint_identity["migration_receipt"],
        "resume_provenance": {
            "previous_seed_dir": str(previous), "verification_path": str(verification_path),
            "verification_sha256": verification_sha, "inherited_artifact_sha256": artifact_hashes,
            "previous_checkpoint_identity_sha256": _digest(old_identity),
            "inherited_vectors_and_costs_exact": True, "inherited_history_exact": True,
            "inherited_csv_prefix_bytes_preserved": True,
            "survivor_order_and_rng_restored": True,
        },
    })
    report = json_ready(report)
    output.mkdir(parents=True, exist_ok=False)
    _write_preserved_prefix(previous / "evaluations.csv", output / "evaluations.csv",
                            pd.DataFrame([_trial_row(point) for point in evaluated[inherited_count:]]))
    _write_preserved_prefix(previous / "evaluated_vectors.csv", output / "evaluated_vectors.csv",
                            _vectors_frame(evaluated[inherited_count:]).reset_index())
    _write_preserved_prefix(previous / "history.csv", output / "history.csv",
                            pd.DataFrame([asdict(record) for record in new_history]))
    pd.DataFrame([_trial_row(point) for point in front], columns=list(_trial_row(evaluated[0]))).to_csv(
        output / "pareto_front.csv", index=False)
    _vectors_frame(front).to_csv(output / "pareto_vectors.csv")
    pd.DataFrame(diagnostics).to_csv(output / "checkpoint_metrics.csv", index=False)
    pd.DataFrame(list(assessment.history), columns=None if assessment.history else ["evaluations", "status"]).to_csv(
        output / "stopping_history.csv", index=False)
    for name in ("target_sample_support.csv", "monthly_targets_si.csv", "monthly_bounds_si.csv"):
        shutil.copyfile(previous / name, output / name)
    save_optimizer_checkpoint(output / "optimizer_checkpoint.npz", complete_states[0], checkpoint_identity)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
