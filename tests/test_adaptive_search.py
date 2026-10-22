"""Portable checkpointed searches, online stop timing, and resume identity guards."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

import dztr.calibration.adaptive as adaptive_module
import dztr.calibration.search as search_module
from dztr.calibration.adaptive import (
    ARTIFACT_NAMES, CHECKPOINT_NAME, IDENTITY_NAME, MANIFEST_NAME, run_adaptive_optimization,
)
from dztr.calibration.checkpoint import load_optimizer_checkpoint
from dztr.calibration.objectives import CalibrationObjective


def write_inputs(folder, *, check_interval=4, minimum=8, patience=2, maximum=24):
    folder.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("1999-12-02", "2001-01-10", name="date")
    source = folder / "observations.csv"
    pd.DataFrame({"storage_m3": 50e6 + (dates.day.to_numpy() - 16) * 1e6,
                  "inflow_m3s": np.full(len(dates), 2.2),
                  "release_m3s": 2.0 + dates.day.to_numpy() * 0.03}, index=dates).to_csv(source)
    policy = {"population_size": 4, "check_interval": check_interval,
              "window_evaluations": check_interval, "minimum_evaluations": minimum,
              "patience": patience, "max_evaluations": maximum}
    policy_path = folder / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    config = {"reservoir": {"name": "Portable adaptive test"}, "observations_file": source.name,
              "periods": {"calibration": {"start": "2000-01-01", "end": "2000-12-31"},
                          "validation": {"start": "2001-01-04", "end": "2001-01-10"},
                          "warmup_steps": 30},
              "model": {"capacity_m3": 80e6, "initial_storage_m3": 55e6, "initial_release_m3s": 2.5},
              "quantile_method": "hazen", "target_percentiles": [10, 45, 85],
              "simulation": {"solver": "implicit", "overflow_mode": "conservative"},
              "optimization": {"seed": 23, "population_size": 4, "evaluation_budget": 8},
              "adaptive": {"checkpoint_interval": 8, "stopping_policy_file": policy_path.name}}
    configuration = folder / "config.json"
    configuration.write_text(json.dumps(config), encoding="utf-8")
    return source, configuration, policy_path


def snapshot(folder):
    return {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def forbid_scoring(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid resume must fail before any model evaluation.")

    monkeypatch.setattr(CalibrationObjective, "evaluate", forbidden)
    monkeypatch.setattr(CalibrationObjective, "evaluate_validation", forbidden)


def test_staged_resume_matches_full_search_and_accounts_only_new_trials(tmp_path):
    source, configuration, policy_path = write_inputs(tmp_path)
    inputs = {path: path.read_bytes() for path in (source, configuration, policy_path)}
    staged, full = tmp_path / "staged", tmp_path / "full"
    first = run_adaptive_optimization(configuration, tmp_path, staged, evaluation_budget=8, honor_stopping=False)
    second = run_adaptive_optimization(configuration, tmp_path, staged, evaluation_budget=16,
                                       resume=True, honor_stopping=False)
    entire = run_adaptive_optimization(configuration, tmp_path, full, evaluation_budget=16, honor_stopping=False)
    assert first["new_search_evaluations"] == first["search_evaluations"] == 8
    assert first["reused_search_evaluations"] == 0
    assert second["new_search_evaluations"] == second["reused_search_evaluations"] == 8
    assert second["search_evaluations"] == entire["search_evaluations"] == 16
    assert second["checkpoint_evaluations"] == second["requested_stage_budget"] == 16
    assert second["audit_extension"] and not second["honor_stopping"]
    assert second["termination_reason"] == "stage_budget_reached"
    assert second["global_convergence_established"] is False
    assert second["checkpoint_counts_written_this_invocation"] == [12, 16]
    for name in ("evaluations.csv", "evaluated_vectors.csv", "pareto_front.csv", "pareto_vectors.csv", "history.csv"):
        pd.testing.assert_frame_equal(read_csv(staged / name), read_csv(full / name))
    assert second["selected_candidate_ids"] == entire["selected_candidate_ids"]
    assert second["selected_scores"] == entire["selected_scores"]
    identity = json.loads((staged / IDENTITY_NAME).read_text(encoding="utf-8"))
    staged_state = load_optimizer_checkpoint(staged / CHECKPOINT_NAME, identity)
    full_state = load_optimizer_checkpoint(full / CHECKPOINT_NAME, identity)
    assert staged_state.rng_state == full_state.rng_state
    assert staged_state.population_ids == full_state.population_ids
    assert staged_state.history == full_state.history
    assert {path.name for path in staged.iterdir()} == {*ARTIFACT_NAMES, CHECKPOINT_NAME, IDENTITY_NAME, MANIFEST_NAME}
    for path, original in inputs.items():
        assert path.read_bytes() == original
    for path in staged.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: pytest.fail(f"Nonstandard JSON {value}"))


def test_interruption_after_complete_generation_resumes_without_published_artifacts(tmp_path):
    _, configuration, _ = write_inputs(tmp_path)
    interrupted, full = tmp_path / "interrupted", tmp_path / "full"

    def interrupted_progress(record):
        if record.evaluations == 8:
            raise RuntimeError("User progress callback interrupted the completed generation")

    with pytest.raises(RuntimeError, match="progress callback interrupted"):
        run_adaptive_optimization(configuration, tmp_path, interrupted,
                                  honor_stopping=False, progress_callback=interrupted_progress)
    identity = json.loads((interrupted / IDENTITY_NAME).read_text(encoding="utf-8"))
    assert load_optimizer_checkpoint(interrupted / CHECKPOINT_NAME, identity).evaluations == 8
    assert not (interrupted / MANIFEST_NAME).exists()
    actual = run_adaptive_optimization(configuration, tmp_path, interrupted, evaluation_budget=16,
                                      resume=True, honor_stopping=False)
    run_adaptive_optimization(configuration, tmp_path, full, evaluation_budget=16, honor_stopping=False)
    assert actual["new_search_evaluations"] == actual["reused_search_evaluations"] == 8
    for name in ("evaluations.csv", "evaluated_vectors.csv", "history.csv"):
        pd.testing.assert_frame_equal(read_csv(interrupted / name), read_csv(full / name))


def test_newer_verified_checkpoint_recovers_interrupted_reporting_without_new_trials(tmp_path):
    source, configuration, policy_path = write_inputs(tmp_path)
    before_inputs = {path: path.read_bytes() for path in (source, configuration, policy_path)}
    staged, full = tmp_path / "staged", tmp_path / "full"
    run_adaptive_optimization(configuration, tmp_path, staged, honor_stopping=False)

    def interrupted_progress(record):
        if record.evaluations == 16:
            raise RuntimeError("Interrupted after checkpoint before reporting")

    with pytest.raises(RuntimeError, match="before reporting"):
        run_adaptive_optimization(configuration, tmp_path, staged, evaluation_budget=16,
                                  resume=True, honor_stopping=False, progress_callback=interrupted_progress)
    old_manifest = json.loads((staged / MANIFEST_NAME).read_text(encoding="utf-8"))
    identity = json.loads((staged / IDENTITY_NAME).read_text(encoding="utf-8"))
    assert old_manifest["search_evaluations"] == 8
    assert load_optimizer_checkpoint(staged / CHECKPOINT_NAME, identity).evaluations == 16
    # A partial reporting write is recoverable only because the checkpoint is newer than publication.
    (staged / "summary.json").write_text("Interrupted derived report\n", encoding="utf-8")
    actual = run_adaptive_optimization(configuration, tmp_path, staged, evaluation_budget=16,
                                      resume=True, honor_stopping=False)
    entire = run_adaptive_optimization(configuration, tmp_path, full, evaluation_budget=16, honor_stopping=False)
    assert actual["reporting_recovered_from_newer_checkpoint"] is True
    assert actual["new_search_evaluations"] == 0
    assert actual["reused_search_evaluations"] == actual["search_evaluations"] == 16
    assert actual["selected_candidate_ids"] == entire["selected_candidate_ids"]
    assert actual["selected_scores"] == entire["selected_scores"]
    for name in ("evaluations.csv", "evaluated_vectors.csv", "history.csv", "pareto_front.csv", "pareto_vectors.csv"):
        pd.testing.assert_frame_equal(read_csv(staged / name), read_csv(full / name))
    assert json.loads((staged / MANIFEST_NAME).read_text(encoding="utf-8"))["search_evaluations"] == 16
    for path, original in before_inputs.items():
        assert path.read_bytes() == original


def test_online_stopping_checks_only_scheduled_calibration_prefixes_and_saves_before_halting(tmp_path, monkeypatch):
    _, configuration, _ = write_inputs(tmp_path, check_interval=8, minimum=12, patience=1)
    original_assess = adaptive_module.assess_stopping
    assessments, search_counts = [], []

    def controlled_stopping(records, policy):
        # Isolate runner stop timing; geometric criteria are tested independently in test_stopping.
        assessments.append(len(records))
        assessed = original_assess(records, policy)
        if len(records) == 16:
            return replace(assessed, status="stopping_candidate", reason="Controlled scheduled test stop",
                           suggested_stop_evaluations=16)
        return assessed

    original_search = search_module.optimize_nsga2

    def counted_search(objective, *args, **kwargs):
        actual_evaluate = objective.evaluate

        def evaluate(vector):
            search_counts.append(len(search_counts) + 1)
            return actual_evaluate(vector)

        objective.evaluate = evaluate
        result = original_search(objective, *args, **kwargs)
        objective.evaluate = actual_evaluate
        return result

    monkeypatch.setattr(adaptive_module, "assess_stopping", controlled_stopping)
    monkeypatch.setattr(search_module, "optimize_nsga2", counted_search)
    output = tmp_path / "run"
    report = run_adaptive_optimization(configuration, tmp_path, output, evaluation_budget=16)
    assert assessments == [8, 16, 16]  # Two online checks, then final offline diagnostics.
    assert len(search_counts) == report["search_evaluations"] == 16
    assert report["new_search_evaluations"] == 16
    assert report["requested_stage_budget"] == 16
    assert report["termination_reason"] == "stopping_candidate"
    assert report["checkpoint_evaluations"] == 16
    assert report["checkpoint_counts_written_this_invocation"] == [4, 8, 16]
    identity = json.loads((output / IDENTITY_NAME).read_text(encoding="utf-8"))
    assert load_optimizer_checkpoint(output / CHECKPOINT_NAME, identity).evaluations == 16
    # Replaying the saved stop must perform no new candidate calls, while explicit audit may continue.
    search_counts.clear()
    replay = run_adaptive_optimization(configuration, tmp_path, output, evaluation_budget=24, resume=True)
    assert replay["new_search_evaluations"] == 0
    assert search_counts == []
    assert load_optimizer_checkpoint(output / CHECKPOINT_NAME, identity).settings.evaluation_budget == 24
    before = snapshot(output)
    with monkeypatch.context() as guarded:
        forbid_scoring(guarded)
        with pytest.raises(ValueError, match="cannot decrease"):
            run_adaptive_optimization(configuration, tmp_path, output, evaluation_budget=16, resume=True)
    assert snapshot(output) == before
    audit = run_adaptive_optimization(configuration, tmp_path, output, evaluation_budget=24,
                                      resume=True, honor_stopping=False)
    assert audit["new_search_evaluations"] == 8
    assert audit["termination_reason"] == "hard_cap_reached"
    assert audit["audit_extension"]


def test_failed_atomic_reporting_manifest_preserves_old_manifest_and_recovers_new_checkpoint(tmp_path, monkeypatch):
    _, configuration, _ = write_inputs(tmp_path)
    output = tmp_path / "run"
    run_adaptive_optimization(configuration, tmp_path, output, honor_stopping=False)
    old_manifest = (output / MANIFEST_NAME).read_bytes()
    original_replace = adaptive_module._replace_checkpoint

    def failed_manifest_replace(source, destination):
        if destination.name == MANIFEST_NAME:
            raise OSError("Simulated interrupted reporting manifest replacement")
        return original_replace(source, destination)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(adaptive_module, "_replace_checkpoint", failed_manifest_replace)
        with pytest.raises(OSError, match="reporting manifest replacement"):
            run_adaptive_optimization(configuration, tmp_path, output, evaluation_budget=16,
                                      resume=True, honor_stopping=False)
    assert (output / MANIFEST_NAME).read_bytes() == old_manifest
    identity = json.loads((output / IDENTITY_NAME).read_text(encoding="utf-8"))
    assert load_optimizer_checkpoint(output / CHECKPOINT_NAME, identity).evaluations == 16
    assert not any(path.name.endswith(".tmp") for path in output.iterdir())
    recovered = run_adaptive_optimization(configuration, tmp_path, output, evaluation_budget=16,
                                          resume=True, honor_stopping=False)
    assert recovered["new_search_evaluations"] == 0
    assert recovered["reused_search_evaluations"] == 16
    assert recovered["reporting_recovered_from_newer_checkpoint"]


@pytest.mark.parametrize("change", ["observations", "operators", "seed", "policy", "code", "versions", "reporting_artifact", "checkpoint"])
def test_resume_identity_and_integrity_changes_fail_before_model_calls(tmp_path, monkeypatch, change):
    source, configuration, policy_path = write_inputs(tmp_path)
    output = tmp_path / "run"
    run_adaptive_optimization(configuration, tmp_path, output, honor_stopping=False)
    kwargs = {}
    if change == "observations":
        source.write_bytes(source.read_bytes() + b"\n")
    elif change == "operators":
        config = json.loads(configuration.read_text(encoding="utf-8"))
        config["optimization"]["mutation_eta"] = 21
        configuration.write_text(json.dumps(config), encoding="utf-8")
    elif change == "seed":
        kwargs["seed"] = 24
    elif change == "policy":
        value = json.loads(policy_path.read_text(encoding="utf-8"))
        value["front_tolerance"] = 0.01
        policy_path.write_text(json.dumps(value), encoding="utf-8")
    elif change == "code":
        original = adaptive_module._code_hashes
        monkeypatch.setattr(adaptive_module, "_code_hashes", lambda: {**original(), "simulation/runner.py": "0" * 64})
    elif change == "versions":
        monkeypatch.setattr(adaptive_module.platform, "python_version", lambda: "Different Python version")
    elif change == "reporting_artifact":
        artifact = output / "evaluations.csv"
        artifact.write_bytes(artifact.read_bytes() + b"\ncorrupted\n")
    else:
        path = output / CHECKPOINT_NAME
        path.write_bytes(path.read_bytes()[:100])
    before = snapshot(output)
    forbid_scoring(monkeypatch)
    with pytest.raises(ValueError):
        run_adaptive_optimization(configuration, tmp_path, output, evaluation_budget=16,
                                  resume=True, honor_stopping=False, **kwargs)
    assert snapshot(output) == before


def test_older_valid_checkpoint_cannot_roll_back_published_evaluations(tmp_path, monkeypatch):
    _, configuration, _ = write_inputs(tmp_path)
    output = tmp_path / "run"
    run_adaptive_optimization(configuration, tmp_path, output, honor_stopping=False)
    old_checkpoint = (output / CHECKPOINT_NAME).read_bytes()
    run_adaptive_optimization(configuration, tmp_path, output, evaluation_budget=16,
                              resume=True, honor_stopping=False)
    (output / CHECKPOINT_NAME).write_bytes(old_checkpoint)
    before = snapshot(output)
    forbid_scoring(monkeypatch)
    with pytest.raises(ValueError):
        run_adaptive_optimization(configuration, tmp_path, output, evaluation_budget=24,
                                  resume=True, honor_stopping=False)
    assert snapshot(output) == before


@pytest.mark.parametrize("options", [{"evaluation_budget": 10}, {"evaluation_budget": 28}, {"resume": 1}, {"honor_stopping": 1}])
def test_invalid_stage_configuration_writes_nothing(tmp_path, options):
    _, configuration, _ = write_inputs(tmp_path)
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        run_adaptive_optimization(configuration, tmp_path, output, **options)
    assert not output.exists()


def test_adaptive_outputs_cannot_contain_source_or_configuration_files(tmp_path):
    _, configuration, _ = write_inputs(tmp_path)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError):
        run_adaptive_optimization(configuration, tmp_path, tmp_path)
    assert snapshot(tmp_path) == before
