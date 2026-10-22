"""Checkpointed calibration with provisional online stopping and audit extensions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd

from dztr import __version__

from .baseline import json_ready
from .checkpoint import _replace_checkpoint, load_optimizer_checkpoint, save_optimizer_checkpoint
from .optimizer import NSGA2Settings
from .search import ARTIFACT_NAMES as SEARCH_ARTIFACT_NAMES, run_configured_optimization
from .stopping import StoppingPolicy, assess_stopping


CHECKPOINT_NAME = "optimizer_checkpoint.npz"
IDENTITY_NAME = "run_identity.json"
MANIFEST_NAME = "adaptive_manifest.json"
ARTIFACT_NAMES = (*SEARCH_ARTIFACT_NAMES, "stopping_history.csv")
_SEARCH_MODULES = (
    "__init__.py", "schemas.py", "data/__init__.py", "data/observations.py", "data/readers.py",
    "parameterization/__init__.py", "parameterization/constants.py", "parameterization/quantiles.py",
    "parameterization/targets.py", "model/__init__.py", "model/balance.py", "model/release.py",
    "simulation/__init__.py", "simulation/runner.py", "evaluation/__init__.py", "evaluation/metrics.py",
    "calibration/__init__.py", "calibration/baseline.py", "calibration/bounds.py", "calibration/candidates.py",
    "calibration/objectives.py", "calibration/optimizer.py", "calibration/periods.py", "calibration/selection.py",
    "calibration/search.py", "calibration/checkpoint.py", "calibration/convergence.py",
    "calibration/stopping.py", "calibration/adaptive.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_hashes() -> dict:
    root = Path(__file__).resolve().parents[1]
    return {name: _sha(root / name) for name in _SEARCH_MODULES}


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _write_json(path, value):
    payload = (json.dumps(json_ready(value), indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_checkpoint(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_adaptive_optimization(config_path: str | Path, repo_root: str | Path, output_dir: str | Path, *,
                              evaluation_budget: int | None = None, seed: int | None = None,
                              resume: bool = False, honor_stopping: bool = True, progress_callback=None) -> dict:
    """Start or resume complete generations, then freeze and report representatives.

    The requested budget is a generation-aligned stage endpoint, bounded by the
    policy's hard cap. Only the endpoint may change between resumes. An explicit
    audit extension can bypass stagnation while retaining the same hard cap and
    stopping history. Validation is reported after calibration-only selection.

    Uncheckpointed work after an interruption is repeated. Saved generation
    boundaries resume without reevaluating their trial vectors. A seed plateau
    is provisional and still requires independent extension and cohort audits.
    """
    config_path, root, output = Path(config_path).resolve(), Path(repo_root).resolve(), Path(output_dir).resolve()
    if not isinstance(resume, bool) or not isinstance(honor_stopping, bool):
        raise ValueError("resume and honor_stopping must be booleans.")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    options = dict(config.get("optimization", {}))
    if evaluation_budget is not None:
        options["evaluation_budget"] = _integer(evaluation_budget, "evaluation_budget")
    if seed is not None:
        options["seed"] = seed
    settings = NSGA2Settings(**options)
    adaptive = config.get("adaptive", {})
    if not isinstance(adaptive, dict) or set(adaptive) - {"checkpoint_interval", "stopping_policy_file"}:
        raise ValueError("adaptive accepts checkpoint_interval and stopping_policy_file only.")
    policy_path = (root / adaptive.get("stopping_policy_file", "configs/stopping_policy.example.json")).resolve()
    policy_bytes = policy_path.read_bytes()
    policy = StoppingPolicy(**json.loads(policy_bytes.decode("utf-8")))
    interval = _integer(adaptive.get("checkpoint_interval", policy.check_interval), "checkpoint_interval")
    if settings.population_size != policy.population_size:
        raise ValueError("Optimizer and stopping policy population_size must match.")
    if interval % settings.population_size or settings.evaluation_budget % settings.population_size:
        raise ValueError("Checkpoint intervals and resumable evaluation budgets must end at complete generations.")
    if settings.evaluation_budget > policy.max_evaluations:
        raise ValueError("The requested evaluation budget exceeds the stopping policy hard cap.")
    source = (root / config["observations_file"]).resolve()
    if any(path == output or path.is_relative_to(output) for path in (source, config_path, policy_path)):
        raise ValueError("Adaptive output would overwrite or contain source observations or configuration; choose another folder.")
    resolved = json.loads(json.dumps(config))
    resolved.pop("study", None)
    resolved["optimization"] = asdict(settings)
    resolved["optimization"].pop("evaluation_budget")
    resolved["adaptive"] = {"checkpoint_interval": interval, "stopping_policy": asdict(policy)}
    resolved.pop("preflight", None)  # Source hashes, not generated reporting paths, establish search identity.
    code = _code_hashes()
    source_digest = _sha(source)
    identity = json_ready({"schema_version": 1, "resolved_search_configuration": resolved,
        "observations_sha256": source_digest, "search_code_sha256": code,
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "dztr": __version__}})
    input_hashes = {"observations": source_digest,
                    "configuration": hashlib.sha256(config_bytes).hexdigest(),
                    "policy": hashlib.sha256(policy_bytes).hexdigest()}
    checkpoint_path = output / CHECKPOINT_NAME
    identity_path = output / IDENTITY_NAME
    manifest_path = output / MANIFEST_NAME
    previous_state = None
    reporting_recovered = False
    if output.exists() and not output.is_dir():
        raise ValueError("Adaptive output must be a directory.")
    if resume:
        if not identity_path.is_file():
            raise ValueError("Resume requires an owned adaptive run directory and run_identity.json.")
        saved_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if saved_identity != identity:
            raise ValueError("Resume differs in input data, settings, policy, code or software versions; choose a new run folder.")
        if not checkpoint_path.is_file():
            raise ValueError("No complete-generation checkpoint exists to resume.")
        previous_state = load_optimizer_checkpoint(checkpoint_path, identity)
        if settings.evaluation_budget < previous_state.settings.evaluation_budget:
            raise ValueError("A resumed stage budget cannot decrease from the saved target budget.")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("identity") != identity or set(manifest.get("artifact_sha256", {})) != set(ARTIFACT_NAMES):
                raise ValueError("Adaptive reporting manifest does not match the run identity or complete artifact set.")
            published_count = manifest["search_evaluations"]
            if previous_state.evaluations < published_count:
                raise ValueError("The checkpoint is older than the published search; refusing rollback and repeated evaluations.")
            if previous_state.evaluations == published_count and _sha(checkpoint_path) != manifest["checkpoint_sha256"]:
                raise ValueError("The checkpoint differs from the checkpoint published with these search artifacts.")
            if any(not (output / name).is_file() or _sha(output / name) != digest
                   for name, digest in manifest["artifact_sha256"].items()):
                if previous_state.evaluations == published_count:
                    raise ValueError("Previously published adaptive artifacts failed their integrity check.")
                # A trusted newer checkpoint survives an interrupted reporting
                # export. Rebuild only owned derived artifacts from that state.
                reporting_recovered = True
    elif output.exists() and any(output.iterdir()):
        raise ValueError("A new adaptive run requires a new or empty output folder; use resume for an owned run.")
    previous_count = 0 if previous_state is None else previous_state.evaluations
    latest_state = previous_state
    saved = []

    def unchanged():
        if input_hashes != {"observations": _sha(source), "configuration": _sha(config_path), "policy": _sha(policy_path)}:
            raise RuntimeError("Adaptive inputs changed during the stage; reporting and continuation require review.")
        if code != _code_hashes():
            raise RuntimeError("Search code changed during the stage; reporting and continuation require review.")

    def checkpoint(state):
        nonlocal latest_state
        latest_state = state
        count = state.evaluations
        if (state.generation == 0 or count % interval == 0 or count % policy.check_interval == 0
                or count == settings.evaluation_budget):
            unchanged()
            save_optimizer_checkpoint(checkpoint_path, state, identity)
            saved.append(count)

    def should_stop(state):
        nonlocal latest_state
        latest_state = state
        if not honor_stopping or state.evaluations % policy.check_interval:
            return False
        return assess_stopping(state.evaluated, policy).status == "stopping_candidate"

    unchanged()
    output.mkdir(parents=True, exist_ok=True)
    if not resume:
        with identity_path.open("x", encoding="utf-8") as stream:
            json.dump(identity, stream, indent=2, allow_nan=False)
            stream.write("\n")
    report = run_configured_optimization(config_path, root, output,
        optimization_overrides={"seed": settings.seed, "evaluation_budget": settings.evaluation_budget},
        progress_callback=progress_callback, resume_state=previous_state,
        checkpoint_callback=checkpoint, stopping_callback=should_stop)
    unchanged()
    if latest_state is None or latest_state.evaluations != report["search_evaluations"]:
        if report["termination"] != "duplicate_exhausted":
            raise RuntimeError("The final result lacks a matching complete-generation checkpoint.")
    if latest_state is not None and latest_state.evaluations == report["search_evaluations"]:
        # A resumed immediate stop or same-budget replay produces no new callback.
        # Refresh its target settings only when a completed generation exists.
        final_saved = load_optimizer_checkpoint(checkpoint_path, identity)
        if (final_saved.evaluations != latest_state.evaluations
                or final_saved.settings != latest_state.settings):
            unchanged()
            save_optimizer_checkpoint(checkpoint_path, latest_state, identity)
            saved.append(latest_state.evaluations)
    state_for_diagnostics = latest_state
    assessment = None if state_for_diagnostics is None else assess_stopping(state_for_diagnostics.evaluated, policy)
    stop_history = [] if assessment is None else [dict(row) for row in assessment.history]
    for row in stop_history:
        row["reasons"] = ";".join(row["reasons"])
    history_columns = ("evaluations", "window_start_evaluations", "eligible", "stable_window",
        "consecutive_stable_windows", "reasons", "hypervolume", "hypervolume_gain", "normalized_hv_gain",
        "hypervolume_normalization_area", "front_movement", "extrema_drift", "newly_feasible_count",
        "feasible_count", "rejected_count", "front_size", "front_points_outside_reference",
        "window_start_front_points_outside_reference", "candidate_invalidated")
    pd.DataFrame(stop_history, columns=history_columns).to_csv(output / "stopping_history.csv", index=False)
    termination_reason = ("stopping_candidate" if report["termination"] == "stopping_candidate"
        else "duplicate_exhausted" if report["termination"] == "duplicate_exhausted"
        else "hard_cap_reached" if report["search_evaluations"] >= policy.max_evaluations else "stage_budget_reached")
    report.update({
        "experiment": "checkpointed_adaptive_calibration", "resumed": resume,
        "reporting_recovered_from_newer_checkpoint": reporting_recovered,
        "new_search_evaluations": report["search_evaluations"] - previous_count,
        "reused_search_evaluations": previous_count, "requested_stage_budget": settings.evaluation_budget,
        "honor_stopping": honor_stopping, "audit_extension": not honor_stopping,
        "termination_reason": termination_reason, "policy_status": "illustrative_unvalidated",
        "stopping_policy": asdict(policy), "stopping_status": None if assessment is None else assessment.status,
        "earliest_stopping_candidate_evaluations": None if assessment is None else assessment.suggested_stop_evaluations,
        "candidate_invalidated": False if assessment is None else assessment.candidate_invalidated,
        "checkpoint_file": CHECKPOINT_NAME, "checkpoint_evaluations": None if latest_state is None else latest_state.evaluations,
        "checkpoint_counts_written_this_invocation": saved,
        "checkpoint_identity": identity, "policy_sha256": input_hashes["policy"],
        "global_convergence_established": False,
    })
    report["notes"].extend([
        "Stopping evaluates the calibration archive on complete generations; validation never controls stopping.",
        "A stopping candidate is provisional; fit quality, independent seeds and cohort audits remain separate.",
        "An audit extension bypasses stagnation but retains the configured policy hard cap.",
        "Previous checkpointed evaluations are reused; uncheckpointed work after interruption must be repeated.",
        "Complete CSV archives from older workflows cannot restore the population or random-generator state.",
    ])
    report = json_ready(report)
    _write_json(output / "summary.json", report)
    manifest = {"identity": identity, "search_evaluations": report["search_evaluations"],
                "checkpoint_sha256": _sha(checkpoint_path) if checkpoint_path.is_file() else None,
                "artifact_sha256": {name: _sha(output / name) for name in ARTIFACT_NAMES}}
    _write_json(manifest_path, manifest)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/trinity_adaptive.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/trinity_adaptive/seed_42"))
    parser.add_argument("--evaluation-budget", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-extension", action="store_true")
    args = parser.parse_args(argv)

    def progress(record):
        if record.generation == 0 or record.evaluations % 128 == 0:
            print(f"Seed {args.seed if args.seed is not None else 'configured'}: {record.evaluations} evaluations, "
                  f"{record.feasible_count} feasible, {record.front_count} nondominated", flush=True)

    report = run_adaptive_optimization(args.repo / args.config, args.repo, args.output,
        evaluation_budget=args.evaluation_budget, seed=args.seed, resume=args.resume,
        honor_stopping=not args.audit_extension, progress_callback=progress)
    print(f"{report['termination_reason']}: {report['search_evaluations']} total; "
          f"{report['new_search_evaluations']} new; {report['reused_search_evaluations']} reused evaluations")
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
