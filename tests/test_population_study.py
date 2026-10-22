"""Equal-cost population experiments, read-only archive reuse and job recovery."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

import dztr.calibration.population_study as population_module
import dztr.calibration.search as search_module
from dztr.calibration.adaptive import CHECKPOINT_NAME, IDENTITY_NAME, run_adaptive_optimization
from dztr.calibration.checkpoint import load_optimizer_checkpoint, save_optimizer_checkpoint
from dztr.calibration.objectives import CalibrationObjective
from dztr.calibration.population_study import run_population_study


ANALYSIS_ARTIFACTS = (
    "checkpoint_metrics.csv", "population_summary.csv", "front_agreement.csv",
    "parameter_stability.csv", "checkpoint_fronts.csv", "stopping_diagnostics.csv",
    "job_status.csv", "summary.json", "resolved_config.json", "study_manifest.json",
)


def write_inputs(folder, *, populations=(2, 4), seeds=(23, 24), checkpoints=(16, 24, 32),
                 changed_validation=False):
    folder.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("1999-12-02", "2001-01-10", name="date")
    frame = pd.DataFrame({
        "storage_m3": 50e6 + (dates.day.to_numpy() - 16) * 1e6,
        "inflow_m3s": np.full(len(dates), 2.2),
        "release_m3s": 2.0 + dates.day.to_numpy() * 0.03,
    }, index=dates)
    if changed_validation:
        frame.loc["2001-01-04":, "storage_m3"] *= 3
        frame.loc["2001-01-04":, "release_m3s"] += 20
    source = folder / "observations.csv"
    frame.to_csv(source)
    policy = {"population_size": 4, "check_interval": 8, "window_evaluations": 8,
              "minimum_evaluations": 16, "patience": 2, "max_evaluations": 64}
    policy_path = folder / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    base = {"reservoir": {"name": "Portable population experiment"},
            "observations_file": source.name,
            "periods": {"calibration": {"start": "2000-01-01", "end": "2000-12-31"},
                        "validation": {"start": "2001-01-04", "end": "2001-01-10"},
                        "warmup_steps": 30},
            "model": {"capacity_m3": 80e6, "initial_storage_m3": 55e6,
                      "initial_release_m3s": 2.5},
            "quantile_method": "hazen", "target_percentiles": [10, 45, 85],
            "simulation": {"solver": "implicit", "overflow_mode": "conservative"},
            "optimization": {"seed": 23, "population_size": 4, "evaluation_budget": 32},
            "adaptive": {"checkpoint_interval": 8, "stopping_policy_file": policy_path.name}}
    base_path = folder / "base.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    study = {"schema_version": 1, "base_config_file": base_path.name,
             "population_sizes": list(populations), "seeds": list(seeds),
             "evaluation_checkpoints": list(checkpoints), "workers": 1,
             "agreement_tolerance": 0.01, "reuse_runs": []}
    study_path = folder / "study.json"
    study_path.write_text(json.dumps(study), encoding="utf-8")
    return source, base_path, policy_path, study_path, study


def snapshots(folder):
    return {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def forbidden(*args, **kwargs):
    raise AssertionError("This operation must not run a search or score a model.")


def forbid_scoring(monkeypatch):
    monkeypatch.setattr(CalibrationObjective, "evaluate", forbidden)
    monkeypatch.setattr(CalibrationObjective, "evaluate_validation", forbidden)
    monkeypatch.setattr(population_module, "run_adaptive_optimization", forbidden)


def add_reuse(study_path, study, directory, *, population=4, seed=23):
    study["reuse_runs"] = [{"population_size": population, "seed": seed,
                            "directory": directory}]
    study_path.write_text(json.dumps(study), encoding="utf-8")


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("population_sizes", []),
    ("population_sizes", [4, 4]), ("population_sizes", [True]),
    ("seeds", []), ("seeds", [23, 23]), ("seeds", [-1]),
    ("evaluation_checkpoints", [16, 20, 32]),
    ("evaluation_checkpoints", [24, 16, 32]),
    ("evaluation_checkpoints", [16, 16, 32]),
    ("workers", 0), ("workers", True),
    ("agreement_tolerance", -0.01), ("agreement_tolerance", True),
])
def test_invalid_study_is_rejected_before_creating_outputs(tmp_path, monkeypatch, field, value):
    _, _, _, study_path, study = write_inputs(tmp_path, populations=(4, 8))
    study[field] = value
    study_path.write_text(json.dumps(study), encoding="utf-8")
    before = snapshots(tmp_path)
    forbid_scoring(monkeypatch)
    with pytest.raises((ValueError, TypeError)):
        run_population_study(study_path, tmp_path, tmp_path / "outputs")
    assert snapshots(tmp_path) == before


def test_equal_budget_prefixes_count_rejections_and_preserve_all_input_files(tmp_path, monkeypatch):
    source, base, policy, study_path, _ = write_inputs(
        tmp_path, populations=(2, 4, 8), seeds=(23, 24))
    input_bytes = {path: path.read_bytes() for path in (source, base, policy, study_path)}
    original_optimizer = search_module.optimize_nsga2
    trial_counts = []

    def one_rejected_trial_per_job(problem, *args, **kwargs):
        actual_evaluate = problem.evaluate
        counts = []

        def evaluate(vector):
            counts.append(len(counts))
            evaluated = actual_evaluate(vector)
            if len(counts) == 2:
                return replace(evaluated, feasible=False, objectives=(float("inf"), float("inf")),
                               metrics={}, reason="Controlled rejected trial in population study test")
            return evaluated

        problem.evaluate = evaluate
        try:
            result = original_optimizer(problem, *args, **kwargs)
        finally:
            problem.evaluate = actual_evaluate
        trial_counts.append(len(counts))
        return result

    monkeypatch.setattr(search_module, "optimize_nsga2", one_rejected_trial_per_job)
    output = tmp_path / "outputs"
    report = run_population_study(study_path, tmp_path, output)
    assert trial_counts == [32] * 6
    assert all((output / name).is_file() for name in ANALYSIS_ARTIFACTS)
    metrics = read_csv(output / "checkpoint_metrics.csv")
    assert set(zip(metrics.population_size, metrics.seed, metrics.evaluations)) == {
        (population, seed, budget) for population in (2, 4, 8)
        for seed in (23, 24) for budget in (16, 24, 32)}
    assert len(metrics) == 18
    assert (metrics.feasible_count + metrics.rejected_count == metrics.evaluations).all()
    assert metrics.rejected_count.ge(1).all()
    assert (metrics.complete_generations_after_initialization
            == metrics.evaluations // metrics.population_size - 1).all()
    for _, group in metrics.groupby(["population_size", "seed"]):
        assert np.diff(group.sort_values("evaluations").hypervolume).min() >= -1e-12
    summary = read_csv(output / "population_summary.csv")
    assert set(zip(summary.population_size, summary.evaluations)) == {
        (population, budget) for population in (2, 4, 8) for budget in (16, 24, 32)}
    stability = read_csv(output / "parameter_stability.csv")
    assert len(stability) == 3 * 3 * 72
    for (population, count), group in stability.groupby(["population_size", "evaluations"], sort=False):
        vectors, state = [], None
        for seed in (23, 24):
            folder = output / "runs" / f"population_{population}" / f"seed_{seed}"
            identity = json.loads((folder / IDENTITY_NAME).read_text(encoding="utf-8"))
            state = load_optimizer_checkpoint(folder / CHECKPOINT_NAME, identity)
            row = metrics.loc[(metrics.population_size == population) & (metrics.seed == seed)
                              & (metrics.evaluations == count)].iloc[0]
            vectors.append(state.evaluated[int(row.balanced_evaluation_id)].vector)
        values = np.asarray(vectors)
        scale = np.r_[np.full(36, 80e6), np.ones(36)]
        np.testing.assert_allclose(group.minimum, values.min(axis=0) * scale, rtol=1e-12)
        np.testing.assert_allclose(group.maximum, values.max(axis=0) * scale, rtol=1e-12)
        width = state.upper_vector - state.lower_vector
        expected = np.divide(np.ptp(values, axis=0), width, out=np.zeros(72), where=width > 0)
        np.testing.assert_allclose(group.normalized_range, expected, atol=1e-12)
        assert group.unit.iloc[:36].eq("m3").all()
        assert group.unit.iloc[36:].eq("m3/s").all()
    fronts = read_csv(output / "checkpoint_fronts.csv")
    agreement = read_csv(output / "front_agreement.csv")
    assert len(agreement) == 3 * 3
    for row in agreement.itertuples(index=False):
        subset = fronts.loc[(fronts.population_size == row.population_size)
                            & (fronts.evaluations == row.evaluations)]
        first = subset.loc[subset.seed == row.first_seed, ["release_nse", "storage_nse"]].to_numpy()
        second = subset.loc[subset.seed == row.second_seed, ["release_nse", "storage_nse"]].to_numpy()
        directed = lambda left, right: max(min(np.max(np.abs(a - b)) for b in right) for a in left)
        expected = max(directed(first, second), directed(second, first))
        assert row.front_distance == pytest.approx(expected, abs=1e-12)
        assert bool(row.within_tolerance) == (expected <= 0.01)
    for name in ("checkpoint_metrics.csv", "population_summary.csv", "front_agreement.csv",
                 "parameter_stability.csv", "checkpoint_fronts.csv", "stopping_diagnostics.csv"):
        assert not any("validation" in column.lower() for column in read_csv(output / name).columns)
    assert report["global_convergence_established"] is False
    assert report["calibration_only_analysis"] is True
    assert report["stopping_controls_budget"] is False
    assert report["job_count"] == 6 and report["failed_job_count"] == 0
    assert report["compared_search_evaluations"] == report["new_search_evaluations_this_invocation"] == 192
    assert report["reused_search_evaluations_this_invocation"] == 0
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == report
    manifest = json.loads((output / "study_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["receipt_sha256"]) == 6
    for name, digest in manifest["artifact_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["receipt_sha256"].items():
        assert hashlib.sha256((output / "job_receipts" / name).read_bytes()).hexdigest() == digest
    for path, before in input_bytes.items():
        assert path.read_bytes() == before
    for path in output.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: pytest.fail(f"Nonfinite JSON {token}"))


def test_validation_observations_cannot_change_population_comparisons(tmp_path):
    original, changed = tmp_path / "original", tmp_path / "changed"
    *_, original_study, _ = write_inputs(original, populations=(4, 8), checkpoints=(16, 24))
    *_, changed_study, _ = write_inputs(changed, populations=(4, 8), checkpoints=(16, 24),
                                        changed_validation=True)
    run_population_study(original_study, original, original / "outputs")
    run_population_study(changed_study, changed, changed / "outputs")
    for name in ("checkpoint_metrics.csv", "population_summary.csv", "front_agreement.csv",
                 "parameter_stability.csv", "checkpoint_fronts.csv", "stopping_diagnostics.csv"):
        pd.testing.assert_frame_equal(read_csv(original / "outputs" / name),
                                      read_csv(changed / "outputs" / name))


@pytest.fixture(scope="module")
def archive_template(tmp_path_factory):
    folder = tmp_path_factory.mktemp("population_archive_template")
    source, base, policy, study_path, study = write_inputs(folder)
    configuration = json.loads(base.read_text(encoding="utf-8"))
    original_policy = json.loads(policy.read_text(encoding="utf-8"))
    study["reuse_runs"] = []
    for population in study["population_sizes"]:
        policy_path = folder / f"archive_policy_{population}.json"
        policy_path.write_text(json.dumps({**original_policy, "population_size": population}), encoding="utf-8")
        variant = json.loads(json.dumps(configuration))
        variant["optimization"]["population_size"] = population
        variant["adaptive"]["stopping_policy_file"] = policy_path.name
        variant_path = folder / f"archive_input_{population}.json"
        variant_path.write_text(json.dumps(variant), encoding="utf-8")
        for seed in study["seeds"]:
            archive = folder / f"archive_population_{population}_seed_{seed}"
            run_adaptive_optimization(variant_path, folder, archive, evaluation_budget=32, seed=seed,
                                      honor_stopping=False)
            study["reuse_runs"].append({"population_size": population, "seed": seed, "directory": archive.name})
    study_path.write_text(json.dumps(study), encoding="utf-8")
    return folder


@pytest.fixture
def reusable_archive(tmp_path, archive_template):
    # All search identities use relative source/configuration paths. Copy the
    # verified fixture to isolate tampering cases without repeating simulations.
    shutil.copytree(archive_template, tmp_path, dirs_exist_ok=True)
    source, base, policy, study_path = [tmp_path / name for name in
                                      ("observations.csv", "base.json", "policy.json", "study.json")]
    study = json.loads(study_path.read_text(encoding="utf-8"))
    return source, base, policy, study_path, study, tmp_path / "archive_population_4_seed_23"


def test_verified_complete_archive_reuse_and_analysis_are_read_only(reusable_archive, tmp_path, monkeypatch):
    source, base, policy, study_path, _, archive = reusable_archive
    before_archive = snapshots(archive)
    input_bytes = {path: path.read_bytes() for path in (source, base, policy, study_path)}
    forbid_scoring(monkeypatch)
    output = tmp_path / "outputs"
    first = run_population_study(study_path, tmp_path, output)
    first_metrics = read_csv(output / "checkpoint_metrics.csv")
    second = run_population_study(study_path, tmp_path, output, analyze_only=True)
    assert first["global_convergence_established"] is second["global_convergence_established"] is False
    assert first["new_search_evaluations_this_invocation"] == second["new_search_evaluations_this_invocation"] == 0
    assert first["reused_search_evaluations_this_invocation"] == second["reused_search_evaluations_this_invocation"] == 128
    pd.testing.assert_frame_equal(first_metrics, read_csv(output / "checkpoint_metrics.csv"))
    assert snapshots(archive) == before_archive
    for path, contents in input_bytes.items():
        assert path.read_bytes() == contents


@pytest.mark.parametrize("tamper", ["artifact", "checkpoint", "observations", "operators", "policy"])
def test_invalid_reuse_is_rejected_before_any_model_call(reusable_archive, tmp_path, monkeypatch, tamper):
    source, base, policy, study_path, _, archive = reusable_archive
    if tamper == "artifact":
        path = archive / "evaluated_vectors.csv"
        path.write_bytes(path.read_bytes() + b"\nchanged cached vectors\n")
    elif tamper == "checkpoint":
        path = archive / CHECKPOINT_NAME
        path.write_bytes(path.read_bytes()[:80])
    elif tamper == "observations":
        source.write_bytes(source.read_bytes() + b"\n")
    elif tamper == "operators":
        value = json.loads(base.read_text(encoding="utf-8"))
        value["optimization"]["mutation_eta"] = 21
        base.write_text(json.dumps(value), encoding="utf-8")
    else:
        value = json.loads(policy.read_text(encoding="utf-8"))
        value["front_tolerance"] = 0.02
        policy.write_text(json.dumps(value), encoding="utf-8")
    before = snapshots(archive)
    forbid_scoring(monkeypatch)
    with pytest.raises((ValueError, OSError)):
        run_population_study(study_path, tmp_path, tmp_path / "outputs")
    assert snapshots(archive) == before


def test_explicit_resume_and_analysis_only_reject_ambiguous_or_missing_search(tmp_path, monkeypatch):
    _, _, _, study_path, _ = write_inputs(tmp_path)
    output = tmp_path / "outputs"
    forbid_scoring(monkeypatch)
    for options in ({"resume": True, "analyze_only": True}, {"analyze_only": True}, {"resume": True}):
        with pytest.raises(ValueError):
            run_population_study(study_path, tmp_path, output, **options)
    assert not output.exists()


def test_explicit_reuse_cannot_extend_or_modify_an_incomplete_external_archive(tmp_path, monkeypatch):
    _, base, _, study_path, study = write_inputs(tmp_path)
    archive = tmp_path / "archive"
    run_adaptive_optimization(base, tmp_path, archive, evaluation_budget=16, honor_stopping=False)
    add_reuse(study_path, study, archive.name)
    before = snapshots(archive)
    forbid_scoring(monkeypatch)
    with pytest.raises(ValueError):
        run_population_study(study_path, tmp_path, tmp_path / "outputs")
    assert snapshots(archive) == before


@pytest.mark.parametrize("options", [{"resume": 1}, {"analyze_only": 1}, {"workers": True}, {"workers": 0}])
def test_api_flags_are_typed_and_workers_override_is_validated_before_search(tmp_path, monkeypatch, options):
    _, _, _, study_path, _ = write_inputs(tmp_path)
    before = snapshots(tmp_path)
    forbid_scoring(monkeypatch)
    with pytest.raises((ValueError, TypeError)):
        run_population_study(study_path, tmp_path, tmp_path / "outputs", **options)
    assert snapshots(tmp_path) == before


def test_failed_job_is_recorded_after_other_population_jobs_finish(tmp_path, monkeypatch):
    _, _, _, study_path, _ = write_inputs(tmp_path, checkpoints=(8, 16))
    output = tmp_path / "outputs"
    actual_runner = population_module.run_adaptive_optimization
    visited = []

    def failing_one_job(config_path, root, folder, **kwargs):
        population = json.loads(Path(config_path).read_text(encoding="utf-8"))["optimization"]["population_size"]
        key = (population, kwargs["seed"])
        visited.append(key)
        if key == (2, 23):
            raise RuntimeError("Controlled one-job input failure")
        assert kwargs["honor_stopping"] is False
        return actual_runner(config_path, root, folder, **kwargs)

    monkeypatch.setattr(population_module, "run_adaptive_optimization", failing_one_job)
    with pytest.raises(RuntimeError, match="1 population job"):
        run_population_study(study_path, tmp_path, output)
    assert visited == [(2, 23), (2, 24), (4, 23), (4, 24)]
    statuses = read_csv(output / "job_status.csv")
    assert statuses.status.tolist() == ["error", "complete", "complete", "complete"]
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["failed_job_count"] == 1 and report["job_count"] == 4
    assert report["compared_search_evaluations"] == 48
    assert not (output / "study_manifest.json").exists()
    for population, seed in visited[1:]:
        folder = output / "runs" / f"population_{population}" / f"seed_{seed}"
        identity = json.loads((folder / IDENTITY_NAME).read_text(encoding="utf-8"))
        assert load_optimizer_checkpoint(folder / CHECKPOINT_NAME, identity).evaluations == 16


def test_interrupted_job_resumes_saved_trials_and_reuses_completed_jobs(tmp_path, monkeypatch):
    _, _, _, study_path, _ = write_inputs(tmp_path)
    output = tmp_path / "outputs"
    actual_runner = population_module.run_adaptive_optimization

    def interrupted_one_job(config_path, root, folder, **kwargs):
        population = json.loads(Path(config_path).read_text(encoding="utf-8"))["optimization"]["population_size"]
        if (population, kwargs["seed"]) == (2, 23):
            actual_progress = kwargs["progress_callback"]

            def progress(record):
                actual_progress(record)
                if record.evaluations == 8:
                    raise RuntimeError("Interrupted immediately after trusted checkpoint eight")

            kwargs["progress_callback"] = progress
        return actual_runner(config_path, root, folder, **kwargs)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(population_module, "run_adaptive_optimization", interrupted_one_job)
        with pytest.raises(RuntimeError, match="1 population job"):
            run_population_study(study_path, tmp_path, output)
    folder = output / "runs" / "population_2" / "seed_23"
    identity = json.loads((folder / IDENTITY_NAME).read_text(encoding="utf-8"))
    assert load_optimizer_checkpoint(folder / CHECKPOINT_NAME, identity).evaluations == 8
    with monkeypatch.context() as guarded:
        forbid_scoring(guarded)
        with pytest.raises(ValueError, match="every planned population job"):
            run_population_study(study_path, tmp_path, output, analyze_only=True)

    calls = []

    def counted_resume(config_path, root, destination, **kwargs):
        calls.append((kwargs["seed"], kwargs["resume"], kwargs["honor_stopping"]))
        return actual_runner(config_path, root, destination, **kwargs)

    monkeypatch.setattr(population_module, "run_adaptive_optimization", counted_resume)
    report = run_population_study(study_path, tmp_path, output, resume=True)
    assert calls == [(23, True, False)]
    assert report["failed_job_count"] == 0
    assert report["compared_search_evaluations"] == 128
    assert report["new_search_evaluations_this_invocation"] == 24
    assert report["reused_search_evaluations_this_invocation"] == 104
    resumed = load_optimizer_checkpoint(folder / CHECKPOINT_NAME, identity)
    full = tmp_path / "uninterrupted"
    actual_runner(output / "generated_inputs" / "population_2.json", tmp_path, full,
                  evaluation_budget=32, seed=23, honor_stopping=False)
    entire = load_optimizer_checkpoint(full / CHECKPOINT_NAME, identity)
    assert resumed.rng_state == entire.rng_state
    assert resumed.population_ids == entire.population_ids
    assert resumed.history == entire.history
    np.testing.assert_array_equal([point.vector for point in resumed.evaluated],
                                  [point.vector for point in entire.evaluated])
    np.testing.assert_array_equal([point.objectives for point in resumed.evaluated],
                                  [point.objectives for point in entire.evaluated])
    before = snapshots(output / "runs")
    with monkeypatch.context() as guarded:
        forbid_scoring(guarded)
        replay = run_population_study(study_path, tmp_path, output, resume=True)
    assert replay["new_search_evaluations_this_invocation"] == 0
    assert replay["reused_search_evaluations_this_invocation"] == 128
    assert snapshots(output / "runs") == before


def test_api_resolves_relative_config_and_output_under_repo_root(reusable_archive, tmp_path, monkeypatch):
    forbid_scoring(monkeypatch)
    report = run_population_study("study.json", tmp_path, "outputs")
    assert report["job_count"] == 4
    assert report["new_search_evaluations_this_invocation"] == 0
    assert (tmp_path / "outputs" / "study_manifest.json").is_file()


def test_receipt_cannot_reidentify_a_completed_owned_job_before_analysis(tmp_path, monkeypatch):
    _, _, _, study_path, _ = write_inputs(tmp_path, checkpoints=(8, 16))
    output = tmp_path / "outputs"
    run_population_study(study_path, tmp_path, output)
    receipt_path = output / "job_receipts" / "population_2_seed_23.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["population_size"] = 8
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    before = snapshots(output)
    forbid_scoring(monkeypatch)
    with pytest.raises(ValueError):
        run_population_study(study_path, tmp_path, output, analyze_only=True)
    assert snapshots(output) == before


@pytest.mark.parametrize("changed", ["bounds", "baseline"])
def test_current_bounds_and_baseline_are_checked_even_when_archive_hashes_match(
        reusable_archive, tmp_path, monkeypatch, changed):
    _, _, _, study_path, study, archive = reusable_archive
    identity = json.loads((archive / IDENTITY_NAME).read_text(encoding="utf-8"))
    state = load_optimizer_checkpoint(archive / CHECKPOINT_NAME, identity)
    baseline = state.baseline_vector.copy()
    baseline[0] += 0.001
    if changed == "bounds":
        lower, upper = state.lower_vector.copy(), state.upper_vector.copy()
        lower[0] += 0.001
        upper[0] += 0.001
        evaluated = []
        for point in state.evaluated:
            vector = point.vector.copy()
            vector[0] += 0.001
            evaluated.append(replace(point, vector=vector))
        altered = replace(state, lower_vector=lower, upper_vector=upper, baseline_vector=baseline,
                          evaluated=tuple(evaluated))
    else:
        altered = replace(state, baseline_vector=baseline,
                          evaluated=(replace(state.evaluated[0], vector=baseline), *state.evaluated[1:]))
    save_optimizer_checkpoint(archive / CHECKPOINT_NAME, altered, identity)
    manifest_path = archive / "adaptive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_sha256"] = hashlib.sha256((archive / CHECKPOINT_NAME).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    # This remains a structurally valid checkpoint and internally valid archive;
    # its calibration-derived bounds/baseline no longer match current inputs.
    assert load_optimizer_checkpoint(archive / CHECKPOINT_NAME, identity).evaluations == 32
    add_reuse(study_path, study, archive.name)
    before = snapshots(archive)
    forbid_scoring(monkeypatch)
    with pytest.raises(ValueError):
        run_population_study(study_path, tmp_path, tmp_path / "outputs")
    assert not (tmp_path / "outputs").exists()
    assert snapshots(archive) == before

