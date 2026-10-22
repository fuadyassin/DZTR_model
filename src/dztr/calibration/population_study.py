"""Compare population settings at equal calibration evaluation budgets.

This orchestration layer leaves the checkpointed search implementation intact.
Completed sources are verified and read without model calls. New jobs run to a
fixed cumulative budget; stopping is assessed retrospectively, never used to
give one population a smaller comparison budget. Validation reports exported
by the adaptive runner do not enter population comparisons.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import platform
import traceback

import numpy as np
import pandas as pd

from dztr import __version__
from dztr.data import read_reservoir_observations
from dztr.parameterization import estimate_generalized_parameters
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters

from .adaptive import (ARTIFACT_NAMES as RUN_ARTIFACT_NAMES, CHECKPOINT_NAME, IDENTITY_NAME,
                       MANIFEST_NAME, _code_hashes, _write_json, run_adaptive_optimization)
from .baseline import json_ready
from .bounds import estimate_calibration_bounds
from .candidates import parameters_to_vector
from .checkpoint import load_optimizer_checkpoint
from .convergence import checkpoint_metrics, prefix_pareto_front
from .optimizer import NSGA2Settings
from .periods import DatePeriod, select_period
from .stopping import StoppingPolicy, assess_stopping


IDENTITY_FILE = "population_identity.json"
RECEIPT_DIRECTORY = "job_receipts"
ARTIFACT_NAMES = ("checkpoint_metrics.csv", "population_summary.csv", "front_agreement.csv",
                  "parameter_stability.csv", "checkpoint_fronts.csv", "stopping_diagnostics.csv",
                  "job_status.csv", "summary.json", "resolved_config.json")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _integers(values, name, minimum):
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ValueError(f"{name} requires at least two distinct integers.")
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < minimum for value in values):
        raise ValueError(f"{name} must contain integers at least {minimum}.")
    result = tuple(int(value) for value in values)
    if len(set(result)) != len(result) or result != tuple(sorted(result)):
        raise ValueError(f"{name} must contain distinct increasing integers.")
    return result


@dataclass(frozen=True)
class PopulationStudySettings:
    population_sizes: tuple[int, ...] = (16, 32, 64)
    seeds: tuple[int, ...] = (42, 43, 44)
    evaluation_checkpoints: tuple[int, ...] = (512, 1024, 2048)
    workers: int = 3
    agreement_tolerance: float = 0.01

    def __post_init__(self):
        for name, minimum in (("population_sizes", 2), ("seeds", 0), ("evaluation_checkpoints", 1)):
            object.__setattr__(self, name, _integers(getattr(self, name), name, minimum))
        if isinstance(self.workers, bool) or not isinstance(self.workers, Integral) or self.workers < 1:
            raise ValueError("workers must be a positive integer.")
        object.__setattr__(self, "workers", int(self.workers))
        if (isinstance(self.agreement_tolerance, bool) or not isinstance(self.agreement_tolerance, Real)
                or not np.isfinite(self.agreement_tolerance) or self.agreement_tolerance < 0):
            raise ValueError("agreement_tolerance must be finite and nonnegative.")
        object.__setattr__(self, "agreement_tolerance", float(self.agreement_tolerance))
        if self.evaluation_checkpoints[0] < max(self.population_sizes):
            raise ValueError("The first evaluation checkpoint must include every initial population.")
        if any(count % size for count in self.evaluation_checkpoints for size in self.population_sizes):
            raise ValueError("Evaluation checkpoints must be aligned with every population size.")


def _job_key(population, seed):
    return f"population_{population}_seed_{seed}"


def _expected_identity(config, policy, population, seed, observation_digest, budget):
    # Mirror the adaptive runner's identity normalization without changing its
    # code bytes or replaying cached searches just to obtain an identity.
    resolved = deepcopy(config)
    resolved.pop("study", None)
    resolved.pop("preflight", None)
    settings = NSGA2Settings(**{**config.get("optimization", {}), "population_size": population,
                               "seed": seed, "evaluation_budget": budget})
    resolved["optimization"] = asdict(settings)
    resolved["optimization"].pop("evaluation_budget")
    resolved["adaptive"] = {
        "checkpoint_interval": config.get("adaptive", {}).get("checkpoint_interval", policy.check_interval),
        "stopping_policy": asdict(policy),
    }
    return json_ready({"schema_version": 1, "resolved_search_configuration": resolved,
        "observations_sha256": observation_digest, "search_code_sha256": _code_hashes(),
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "dztr": __version__}})


def _verify_run(folder, expected_identity=None, budget=None):
    folder = Path(folder)
    identity = json.loads((folder / IDENTITY_NAME).read_text(encoding="utf-8"))
    if expected_identity is not None and identity != expected_identity:
        raise ValueError(f"Population run identity differs in data, settings, policy, code or versions: {folder}")
    manifest_path = folder / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity") != identity or set(manifest.get("artifact_sha256", {})) != set(RUN_ARTIFACT_NAMES):
        raise ValueError(f"Population run has an incomplete or incompatible artifact manifest: {folder}")
    for name, digest in manifest["artifact_sha256"].items():
        if not (folder / name).is_file() or _sha(folder / name) != digest:
            raise ValueError(f"Population source artifact failed its integrity check: {folder / name}")
    if _sha(folder / CHECKPOINT_NAME) != manifest.get("checkpoint_sha256"):
        raise ValueError(f"Population checkpoint differs from its published manifest: {folder}")
    state = load_optimizer_checkpoint(folder / CHECKPOINT_NAME, identity)
    if state.evaluations != manifest.get("search_evaluations") or (budget is not None and state.evaluations != budget):
        raise ValueError(f"Population run does not contain the exact complete comparison budget: {folder}")
    report = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if report.get("checkpoint_identity") != identity or report.get("search_evaluations") != state.evaluations:
        raise ValueError(f"Population report does not match its checkpoint: {folder}")
    proof = {"directory": str(folder.resolve()), "manifest_sha256": _sha(manifest_path),
             "checkpoint_sha256": manifest["checkpoint_sha256"], "identity": identity}
    return state, report, proof


def _verify_proof(proof):
    state, report, actual = _verify_run(proof["directory"], proof["identity"])
    if actual != proof:
        raise ValueError(f"A population source changed after analysis: {proof['directory']}")
    return state, report


def _calibration_vectors(config, root):
    """Recompute the common search box and starting vector without simulation."""
    observed = read_reservoir_observations(root / config["observations_file"])
    period = DatePeriod(**config["periods"]["calibration"])
    physical = dict(config["model"])
    channel = physical.pop("channel_capacity_m3s", None)
    method = config.get("quantile_method", "hazen")
    estimates = estimate_generalized_parameters(select_period(observed, period),
        tuple(config.get("target_percentiles", (10, 45, 85))),
        channel_capacity_m3s=channel, quantile_method=method)
    template = ReservoirParameters.from_generalized(estimates, **physical)
    bounds = estimate_calibration_bounds(observed, period, template, quantile_method=method)
    return {"lower_vector": bounds.lower_vector, "upper_vector": bounds.upper_vector,
            "baseline_vector": parameters_to_vector(template)}


def _verify_vectors(state, expected):
    if any(not np.array_equal(getattr(state, name), vector) for name, vector in expected.items()):
        raise ValueError("Population checkpoint bounds or baseline differ from current calibration-derived vectors.")


def verify_population_study(output_dir):
    """Read historical study results only after report and source verification."""
    output = Path(output_dir).resolve()
    manifest = json.loads((output / "study_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or set(manifest.get("artifact_sha256", {})) != set(ARTIFACT_NAMES):
        raise ValueError("Population study manifest is incomplete or unsupported.")
    if _sha(output / IDENTITY_FILE) != manifest.get("identity_sha256"):
        raise ValueError("Population study identity failed its integrity check.")
    for name, digest in manifest["artifact_sha256"].items():
        if not (output / name).is_file() or _sha(output / name) != digest:
            raise ValueError(f"Population study artifact failed its integrity check: {name}")
    identity = json.loads((output / IDENTITY_FILE).read_text(encoding="utf-8"))
    settings = PopulationStudySettings(**identity["settings"])
    expected_keys = {(size, seed) for size in settings.population_sizes for seed in settings.seeds}
    planned = {(row["population_size"], row["seed"]): row for row in identity["jobs"]}
    if set(planned) != expected_keys or len(identity["jobs"]) != len(expected_keys):
        raise ValueError("Population identity does not describe the complete planned job matrix.")
    receipt_names = {f"{_job_key(*key)}.json" for key in expected_keys}
    if set(manifest.get("receipt_sha256", {})) != receipt_names:
        raise ValueError("Population receipts do not match the complete planned job matrix.")
    receipts = []
    for name, digest in manifest["receipt_sha256"].items():
        receipt_path = output / RECEIPT_DIRECTORY / name
        if Path(name).name != name or not receipt_path.is_file() or _sha(receipt_path) != digest:
            raise ValueError("Population job receipt failed its integrity check.")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        key = (receipt["population_size"], receipt["seed"])
        if key not in planned or name != f"{_job_key(*key)}.json" or receipt.get("status") != "complete":
            raise ValueError("Population receipt identifies the wrong planned job or incomplete status.")
        state, _ = _verify_proof(receipt["source_proof"])
        expected_directory = planned[key]["external_directory"] or str(output / "runs" / f"population_{key[0]}" / f"seed_{key[1]}")
        if (receipt["source_proof"]["identity"] != planned[key]["identity"]
                or (state.settings.population_size, state.settings.seed) != key
                or state.evaluations != settings.evaluation_checkpoints[-1]
                or receipt.get("search_evaluations") != state.evaluations
                or Path(receipt["directory"]).resolve() != Path(expected_directory).resolve()
                or Path(receipt["source_proof"]["directory"]).resolve() != Path(expected_directory).resolve()):
            raise ValueError("Population receipt source differs from its planned identity, directory or budget.")
        receipts.append(receipt)
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    receipts.sort(key=lambda row: (row["population_size"], row["seed"]))
    if (len(expected_keys) != report.get("job_count") or report.get("failed_job_count") != 0
            or report.get("jobs") != receipts
            or report.get("evaluations_per_job") != settings.evaluation_checkpoints[-1]
            or {name: value for name, value in report["settings"].items() if name != "workers"} != identity["settings"]):
        raise ValueError("Population study analysis requires all planned jobs to complete.")
    return report


def _execute_job(job, root, resume):
    """Run one owned job; the caller records failures independently."""
    folder = Path(job["directory"])
    existing = folder.exists() and any(folder.iterdir())
    if existing and not resume:
        raise ValueError(f"Existing population job requires explicit resume: {folder}")

    def progress(record):
        if record.generation == 0 or record.evaluations % 128 == 0:
            print(f"Population {job['population_size']}, seed {job['seed']}: "
                  f"{record.evaluations}/{job['budget']} trials; {record.feasible_count} feasible", flush=True)

    report = run_adaptive_optimization(job["config_path"], root, folder,
        evaluation_budget=job["budget"], seed=job["seed"], resume=existing,
        honor_stopping=False, progress_callback=progress)
    state, _, proof = _verify_run(folder, job["expected_identity"], job["budget"])
    return {"population_size": job["population_size"], "seed": job["seed"], "status": "complete",
        "source_kind": "study_run", "directory": str(folder), "search_evaluations": state.evaluations,
        "new_search_evaluations": report["new_search_evaluations"],
        "reused_search_evaluations": report["reused_search_evaluations"],
        "search_elapsed_seconds_this_invocation": report["search_elapsed_seconds"], "source_proof": proof}


def _distance(first, second):
    if not len(first) or not len(second):
        return None
    distances = np.max(np.abs(first[:, None, :] - second[None, :, :]), axis=2)
    return float(max(np.min(distances, axis=0).max(), np.min(distances, axis=1).max()))


def _analyze(receipts, settings, output, base_config):
    metrics, fronts, diagnostics, agreement, stability = [], [], [], [], []
    states = {}
    canonical = None
    for receipt in receipts:
        state, report = _verify_proof(receipt["source_proof"])
        key = (receipt["population_size"], receipt["seed"])
        if (state.settings.population_size, state.settings.seed) != key or state.evaluations != settings.evaluation_checkpoints[-1]:
            raise ValueError("A population receipt identifies the wrong search or budget.")
        if canonical is None:
            canonical = state
        elif any(not np.array_equal(getattr(state, name), getattr(canonical, name))
                 for name in ("lower_vector", "upper_vector", "baseline_vector")) or state.evaluated[0].objectives != canonical.evaluated[0].objectives:
            raise ValueError("Population searches do not share identical bounds, baseline vector and baseline costs.")
        states[key] = state
        policy = StoppingPolicy(**report["stopping_policy"])
        assessment = assess_stopping(state.evaluated, policy)
        for row in assessment.history:
            diagnostics.append({"population_size": key[0], "seed": key[1], **row,
                                "reasons": ";".join(row["reasons"])})
        for count in settings.evaluation_checkpoints:
            row = checkpoint_metrics(state.evaluated, count, reference=policy.hypervolume_reference)
            selected = row.pop("representative_ids") or {}
            row.pop("reference")
            metrics.append({"population_size": key[0], "seed": key[1], **row,
                "complete_generations_after_initialization": count // key[0] - 1,
                **{f"{role}_evaluation_id": identifier for role, identifier in selected.items()}})
            for point in prefix_pareto_front(state.evaluated, count):
                fronts.append({"population_size": key[0], "seed": key[1], "evaluations": count,
                    "evaluation_id": point.evaluation_id, "release_nse": -point.objectives[0],
                    "storage_nse": -point.objectives[1]})
    metric_frame = pd.DataFrame(metrics)
    if len(states) != len(settings.population_sizes) * len(settings.seeds):
        raise ValueError("Population analysis requires the complete population/seed matrix.")
    summaries = []
    capacity = base_config["model"]["capacity_m3"]
    for population in settings.population_sizes:
        for count in settings.evaluation_checkpoints:
            subset = metric_frame.loc[metric_frame.population_size.eq(population) & metric_frame.evaluations.eq(count)]
            row = {"population_size": population, "evaluations": count, "seed_count": len(subset)}
            for name in ("hypervolume", "max_release_nse", "max_storage_nse", "balanced_release_nse", "balanced_storage_nse"):
                values = subset[name].to_numpy(dtype=float)
                row.update({f"{name}_min": float(values.min()), f"{name}_median": float(np.median(values)),
                            f"{name}_max": float(values.max())})
            vectors, objectives = [], {}
            for seed in settings.seeds:
                state = states[population, seed]
                balanced = int(subset.loc[subset.seed.eq(seed), "balanced_evaluation_id"].iloc[0])
                vectors.append(state.evaluated[balanced].vector)
                objectives[seed] = np.unique(np.asarray([point.objectives for point in prefix_pareto_front(state.evaluated, count)]), axis=0)
            values = np.asarray(vectors)
            width = canonical.upper_vector - canonical.lower_vector
            span = np.divide(np.ptp(values, axis=0), width, out=np.zeros(72), where=width > 0)
            row.update(parameter_spread_median=float(np.median(span)), parameter_spread_maximum=float(span.max()))
            for index in range(72):
                scale = capacity if index < 36 else 1.0
                stability.append({"population_size": population, "evaluations": count,
                    "target": MONTHLY_TARGET_COLUMNS[index // 12], "month": index % 12 + 1,
                    "unit": "m3" if index < 36 else "m3/s", "lower": canonical.lower_vector[index] * scale,
                    "upper": canonical.upper_vector[index] * scale, "minimum": values[:, index].min() * scale,
                    "median": np.median(values[:, index]) * scale, "maximum": values[:, index].max() * scale,
                    "normalized_range": span[index]})
            distances = []
            for offset, seed in enumerate(settings.seeds):
                for other in settings.seeds[offset + 1:]:
                    distance = _distance(objectives[seed], objectives[other])
                    within = distance is not None and distance <= settings.agreement_tolerance
                    agreement.append({"population_size": population, "evaluations": count,
                        "first_seed": seed, "second_seed": other, "front_distance": distance,
                        "within_tolerance": within})
                    distances.append(distance)
            row["maximum_seed_front_distance"] = None if any(value is None for value in distances) else max(distances)
            row["all_seed_fronts_within_tolerance"] = all(value is not None and value <= settings.agreement_tolerance for value in distances)
            summaries.append(row)
    for name, frame in (("checkpoint_metrics.csv", metric_frame), ("population_summary.csv", pd.DataFrame(summaries)),
                        ("front_agreement.csv", pd.DataFrame(agreement)), ("parameter_stability.csv", pd.DataFrame(stability)),
                        ("checkpoint_fronts.csv", pd.DataFrame(fronts)), ("stopping_diagnostics.csv", pd.DataFrame(diagnostics))):
        frame.to_csv(output / name, index=False)
    return summaries


def run_population_study(study_config_path, repo_root, output_dir, *, workers=None, resume=False, analyze_only=False):
    """Run/resume an equal-budget matrix, or analyze complete verified sources.

    Reuse references are immutable read-only inputs. Owned completed jobs are
    reused without representative simulations; partial jobs require resume.
    Failures are recorded as futures finish, while independent jobs continue.
    Population recommendations remain provisional and exclude validation costs.
    """
    if not isinstance(resume, bool) or not isinstance(analyze_only, bool):
        raise ValueError("resume and analyze_only must be booleans.")
    if resume and analyze_only:
        raise ValueError("Choose resume or analyze_only, not both.")
    root = Path(repo_root).resolve()
    output, config_path = (root / output_dir).resolve(), (root / study_config_path).resolve()
    payload = config_path.read_bytes()
    config = json.loads(payload.decode("utf-8"))
    fields = {"schema_version", "base_config_file", "population_sizes", "seeds", "evaluation_checkpoints",
              "workers", "agreement_tolerance", "reuse_runs"}
    if (not isinstance(config, dict) or set(config) - fields
            or type(config.get("schema_version")) is not int or config.get("schema_version") != 1):
        raise ValueError("Population study requires schema_version 1 and supported fields only.")
    settings = PopulationStudySettings(**{name: config[name] for name in PopulationStudySettings.__dataclass_fields__ if name in config})
    if workers is not None:
        settings = PopulationStudySettings(**{**asdict(settings), "workers": workers})
    base_path = (root / config["base_config_file"]).resolve()
    base_bytes = base_path.read_bytes()
    base = json.loads(base_bytes.decode("utf-8"))
    policy_path = (root / base.get("adaptive", {}).get("stopping_policy_file", "configs/stopping_policy.example.json")).resolve()
    policy_bytes = policy_path.read_bytes()
    original_policy = StoppingPolicy(**json.loads(policy_bytes.decode("utf-8")))
    source = (root / base["observations_file"]).resolve()
    source_digest = _sha(source)
    policies = {population: StoppingPolicy(**{**asdict(original_policy), "population_size": population})
                for population in settings.population_sizes}
    interval = base.get("adaptive", {}).get("checkpoint_interval", original_policy.check_interval)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1 or any(interval % size for size in settings.population_sizes):
        raise ValueError("Checkpoint interval must be aligned with every population size.")
    budget = settings.evaluation_checkpoints[-1]
    if budget > original_policy.max_evaluations:
        raise ValueError("Population budget exceeds the policy hard cap.")
    reuse = {}
    reuse_entries = config.get("reuse_runs", [])
    if not isinstance(reuse_entries, list):
        raise ValueError("reuse_runs must be a list of planned job references.")
    for entry in reuse_entries:
        if not isinstance(entry, dict) or set(entry) != {"population_size", "seed", "directory"}:
            raise ValueError("reuse_runs requires population_size, seed and directory for every entry.")
        if (any(isinstance(entry[name], bool) or not isinstance(entry[name], int) for name in ("population_size", "seed"))
                or not isinstance(entry["directory"], str) or not entry["directory"].strip()):
            raise ValueError("Reuse population/seed values must be integers and directory must be a nonempty path string.")
        key = (entry["population_size"], entry["seed"])
        if key not in {(size, seed) for size in settings.population_sizes for seed in settings.seeds} or key in reuse:
            raise ValueError("Reused population/seed entries must be unique planned jobs.")
        reuse[key] = (root / entry["directory"]).resolve()
    protected = (config_path, base_path, policy_path, source, *reuse.values())
    if not output.is_relative_to(root) or any(path == output or path.is_relative_to(output) for path in protected) or any(output.is_relative_to(path) for path in reuse.values()):
        raise ValueError("Population output must stay within the repository and protect all source inputs/runs.")
    if output.exists() and not output.is_dir():
        raise ValueError("Population output must be a directory.")
    identities = {(size, seed): _expected_identity(base, policies[size], size, seed, source_digest, budget)
                  for size in settings.population_sizes for seed in settings.seeds}
    study_identity = {"schema_version": 1, "settings": {name: value for name, value in asdict(settings).items() if name != "workers"},
                      "jobs": [{"population_size": key[0], "seed": key[1], "identity": value,
                                "external_directory": str(reuse[key]) if key in reuse else None} for key, value in identities.items()]}
    study_identity = json_ready(study_identity)
    marker = output / IDENTITY_FILE
    existing = output.exists() and any(output.iterdir())
    if existing:
        if not resume and not analyze_only:
            raise ValueError("Existing population study requires explicit resume or analyze_only.")
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != study_identity:
            raise ValueError("Population study identity differs; use a new output directory.")
    elif analyze_only or resume:
        raise ValueError("Resume or analyze_only requires an existing owned population study.")
    # Reject every incompatible external cache before any search or output write.
    external = {key: _verify_run(folder, identities[key], budget) for key, folder in reuse.items()}
    vectors = _calibration_vectors(base, root)
    completed = {}
    for key, (state, _, _) in external.items():
        _verify_vectors(state, vectors)
    for key, identity in identities.items():
        if key in external:
            continue
        receipt_file = output / RECEIPT_DIRECTORY / f"{_job_key(*key)}.json"
        prior = json.loads(receipt_file.read_text(encoding="utf-8")) if receipt_file.is_file() else None
        if prior is not None and prior.get("status") == "complete":
            folder = output / "runs" / f"population_{key[0]}" / f"seed_{key[1]}"
            state, report, proof = _verify_run(folder, identity, budget)
            _verify_vectors(state, vectors)
            if ((prior.get("population_size"), prior.get("seed")) != key
                    or prior.get("source_proof") != proof):
                raise ValueError("Completed owned population run changed since its receipt was saved.")
            completed[key] = (state, report, proof, prior)
        elif analyze_only:
            raise ValueError("analyze_only requires every planned population job to be complete.")
    input_hashes = {config_path: hashlib.sha256(payload).hexdigest(), base_path: hashlib.sha256(base_bytes).hexdigest(),
                    policy_path: hashlib.sha256(policy_bytes).hexdigest(), source: source_digest}
    analysis_digest = _sha(Path(__file__))

    def unchanged():
        if any(_sha(path) != digest for path, digest in input_hashes.items()) or _sha(Path(__file__)) != analysis_digest or _code_hashes() != next(iter(identities.values()))["search_code_sha256"]:
            raise RuntimeError("Population inputs or code changed during the study; results require review.")

    unchanged()
    output.mkdir(parents=True, exist_ok=True)
    if not existing:
        _write_json(marker, study_identity)
    generated = output / "generated_inputs"
    receipts_path = output / RECEIPT_DIRECTORY
    generated.mkdir(exist_ok=True)
    receipts_path.mkdir(exist_ok=True)
    variants = {}
    for size, policy in policies.items():
        policy_file = generated / f"policy_{size}.json"
        variant = deepcopy(base)
        variant["optimization"] = asdict(NSGA2Settings(**{**base.get("optimization", {}), "population_size": size, "evaluation_budget": budget}))
        variant["adaptive"] = {"checkpoint_interval": interval, "stopping_policy_file": str(policy_file)}
        variant_file = generated / f"population_{size}.json"
        for path, value in ((policy_file, asdict(policy)), (variant_file, variant)):
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8")) != json_ready(value):
                    raise ValueError("Generated population inputs changed; refusing continuation.")
            else:
                if analyze_only:
                    raise ValueError("Completed population study lacks its generated inputs.")
                _write_json(path, value)
        variants[size] = variant_file
        input_hashes[policy_file] = _sha(policy_file)
        input_hashes[variant_file] = _sha(variant_file)
    outcomes, pending = [], []

    def record(receipt):
        outcomes.append(receipt)
        unchanged()
        _write_json(receipts_path / f"{_job_key(receipt['population_size'], receipt['seed'])}.json", receipt)
        pd.DataFrame([{key: value for key, value in row.items() if key != "source_proof"} for row in outcomes]).to_csv(output / "job_status.csv", index=False)
        print(f"Population {receipt['population_size']}, seed {receipt['seed']}: {receipt['status']}", flush=True)

    for key, identity in identities.items():
        folder = reuse.get(key, output / "runs" / f"population_{key[0]}" / f"seed_{key[1]}")
        if key in external:
            state, report, proof = external[key]
            record({"population_size": key[0], "seed": key[1], "status": "complete", "source_kind": "external_cache",
                "directory": str(folder), "search_evaluations": state.evaluations, "new_search_evaluations": 0,
                "reused_search_evaluations": state.evaluations, "search_elapsed_seconds_this_invocation": 0.0, "source_proof": proof})
            continue
        if key in completed:
            state, report, proof, prior = completed[key]
            record({**prior, "new_search_evaluations": 0, "reused_search_evaluations": state.evaluations,
                    "search_elapsed_seconds_this_invocation": 0.0})
            continue
        pending.append({"population_size": key[0], "seed": key[1], "budget": budget, "directory": str(folder),
                        "config_path": str(variants[key[0]]), "expected_identity": identity})

    def failed(job, error):
        return {"population_size": job["population_size"], "seed": job["seed"], "status": "error",
                "source_kind": "study_run", "directory": job["directory"], "error": str(error),
                "traceback": traceback.format_exc(), "new_search_evaluations": None, "reused_search_evaluations": None}

    if settings.workers == 1:
        for job in pending:
            try:
                receipt = _execute_job(job, str(root), resume)
            except Exception as error:
                receipt = failed(job, error)
            record(receipt)
    elif pending:
        with ProcessPoolExecutor(max_workers=min(settings.workers, len(pending))) as pool:
            futures = {pool.submit(_execute_job, job, str(root), resume): job for job in pending}
            for future in as_completed(futures):
                try:
                    receipt = future.result()
                except Exception as error:
                    receipt = failed(futures[future], error)
                record(receipt)
    outcomes.sort(key=lambda row: (row["population_size"], row["seed"]))
    pd.DataFrame([{key: value for key, value in row.items() if key != "source_proof"} for row in outcomes]).to_csv(output / "job_status.csv", index=False)
    failures = [row for row in outcomes if row["status"] != "complete"]
    report = {"experiment": "equal_budget_population_study", "job_count": len(identities), "failed_job_count": len(failures),
        "settings": asdict(settings), "jobs": outcomes, "evaluations_per_job": budget,
        "compared_search_evaluations": sum(row.get("search_evaluations", 0) for row in outcomes),
        "new_search_evaluations_this_invocation": sum(row["new_search_evaluations"] or 0 for row in outcomes),
        "reused_search_evaluations_this_invocation": sum(row["reused_search_evaluations"] or 0 for row in outcomes),
        "calibration_only_analysis": True, "stopping_controls_budget": False,
        "global_convergence_established": False, "validated_population_recommendation": None,
        "qualification": "Operational population settings change initialization and generation counts. Seed labels are reproducibility controls, not common random draws. Seed ranges are descriptive. Validation never determines population comparisons or budgets."}
    _write_json(output / "resolved_config.json", {**config, "workers": settings.workers})
    if failures:
        _write_json(output / "summary.json", report)
        raise RuntimeError(f"{len(failures)} population job(s) failed; independent jobs finished and receipts preserve recovery.")
    report["population_results"] = _analyze(outcomes, settings, output, base)
    unchanged()
    _write_json(output / "summary.json", report)
    for row in outcomes:
        _verify_proof(row["source_proof"])
    manifest = {"schema_version": 1, "identity_sha256": _sha(marker), "analysis_code_sha256": analysis_digest,
        "artifact_sha256": {name: _sha(output / name) for name in ARTIFACT_NAMES},
        "receipt_sha256": {path.name: _sha(path) for path in sorted(receipts_path.glob("*.json"))}}
    _write_json(output / "study_manifest.json", manifest)
    return json_ready(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/trinity_population_study.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/trinity_population_study"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    report = run_population_study(args.config, args.repo, args.output, workers=args.workers,
                                  resume=args.resume, analyze_only=args.analyze_only)
    print(f"Compared {report['compared_search_evaluations']} trials; "
          f"{report['new_search_evaluations_this_invocation']} new this invocation, "
          f"{report['reused_search_evaluations_this_invocation']} reused.")
    print(f"Artifacts: {(args.repo / args.output).resolve()}")


if __name__ == "__main__":
    main()
