"""Verified archive-only budget review and fixed-coordinate seed agreement."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import dztr.calibration.budget_review as review_module
import dztr.calibration.optimizer as optimizer_module
import dztr.calibration.study as study_module
from dztr.calibration.budget_review import ARTIFACT_NAMES, run_budget_review
from dztr.calibration.objectives import CalibrationObjective
from dztr.calibration.search import ARTIFACT_NAMES as SEARCH_ARTIFACT_NAMES
from dztr.calibration.stopping import StoppingPolicy
from dztr.calibration.study import run_calibration_study


def policy(**changes):
    settings = dict(population_size=4, check_interval=4, window_evaluations=4,
                    minimum_evaluations=8, patience=2, max_evaluations=20,
                    normalized_hv_tolerance=0.001, front_tolerance=0.01,
                    extrema_tolerance=0.01)
    settings.update(changes)
    return StoppingPolicy(**settings)


def snapshots(folder):
    return {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_inputs(folder, *, source_name="observations.csv"):
    folder.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("1999-12-02", "2001-01-10", name="date")
    frame = pd.DataFrame({"storage_m3": 50e6 + (dates.day.to_numpy() - 16) * 1e6,
                          "inflow_m3s": np.full(len(dates), 2.2),
                          "release_m3s": 2.0 + dates.day.to_numpy() * 0.03}, index=dates)
    source = folder / source_name
    frame.to_csv(source, index_label="date")
    config = {"reservoir": {"name": "Portable archive review"}, "observations_file": source_name,
              "periods": {"calibration": {"start": "2000-01-01", "end": "2000-12-31"},
                          "validation": {"start": "2001-01-04", "end": "2001-01-10"},
                          "warmup_steps": 30},
              "model": {"capacity_m3": 80e6, "initial_storage_m3": 55e6, "initial_release_m3s": 2.5},
              "quantile_method": "hazen", "target_percentiles": [10, 45, 85],
              "simulation": {"solver": "implicit", "overflow_mode": "conservative"},
              "optimization": {"seed": 23, "population_size": 4, "evaluation_budget": 8},
              "study": {"seeds": [23, 24], "evaluation_checkpoints": [4, 8],
                        "hypervolume_reference": [1, 1], "workers": 1}}
    configuration = folder / "config.json"
    configuration.write_text(json.dumps(config), encoding="utf-8")
    return source, configuration


def write_archive(study, seed, costs):
    """Build honest counted records with full hashes, independently of optimization."""
    folder = study / "runs" / f"seed_{seed}"
    folder.mkdir(parents=True)
    for name in SEARCH_ARTIFACT_NAMES:
        (folder / name).write_text("{}\n" if name.endswith(".json") else "synthetic\n", encoding="utf-8")
    pd.DataFrame({"evaluation_id": range(len(costs)),
                  "generation": np.arange(len(costs)) // 4, "feasible": True,
                  "reason": None, "negative_release_nse": [pair[0] for pair in costs],
                  "negative_storage_nse": [pair[1] for pair in costs]}).to_csv(
                      folder / "evaluations.csv", index=False)
    pd.DataFrame(np.full((len(costs), 72), 0.5),
                 index=pd.Index(range(len(costs)), name="evaluation_id"),
                 columns=[f"x{i:02d}" for i in range(72)]).to_csv(folder / "evaluated_vectors.csv")
    manifest = {"search_evaluations": len(costs),
                "identity": {"resolved_search_configuration": {
                    "optimization": {"seed": seed, "population_size": 4, "evaluation_budget": len(costs)},
                    "model": {"capacity_m3": 1000}, "observations_file": "historical.csv"},
                    "observations_sha256": "0" * 64, "search_code_sha256": {},
                    "software_versions": {"fixture": "synthetic archive"}},
                "artifact_sha256": {name: digest(folder / name) for name in SEARCH_ARTIFACT_NAMES}}
    (folder / "seed_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def forbid_search_and_scoring(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Archive review must not search or score daily simulations.")

    monkeypatch.setattr(study_module, "run_configured_optimization", forbidden)
    monkeypatch.setattr(optimizer_module, "optimize_nsga2", forbidden)
    monkeypatch.setattr(CalibrationObjective, "evaluate", forbidden)
    monkeypatch.setattr(CalibrationObjective, "evaluate_validation", forbidden)


def test_portable_review_reuses_verified_search_without_simulation_or_validation(tmp_path, monkeypatch):
    source, configuration = write_inputs(tmp_path)
    study = tmp_path / "study"
    run_calibration_study(configuration, tmp_path, study)
    before = snapshots(tmp_path)
    forbid_search_and_scoring(monkeypatch)
    output = tmp_path / "review"
    report = run_budget_review(study, output, policy=policy())
    assert {path.name for path in output.iterdir()} == set(ARTIFACT_NAMES)
    assert report["new_search_evaluations"] == report["reporting_simulations"] == 0
    assert report["validation_used"] is False
    assert report["common_tested_evaluations_per_seed"] == 8
    assert report["independent_seed_count"] == 2
    assert report["production_budget_recommendation"] is None
    assert report["global_convergence_established"] is False
    assert report["cross_seed_front_agreement"] is False
    assert report["suggested_next_experiment_budget_per_seed"] == 16
    statuses = pd.read_csv(output / "seed_status.csv")
    assert statuses.evaluations.tolist() == [8, 8]
    assert (statuses.feasible_count + statuses.rejected_count).eq(8).all()
    assert set(statuses.seed) == {23, 24}
    for seed in (23, 24):
        assert report["seed_manifest_sha256"][str(seed)] == digest(study / "runs" / f"seed_{seed}" / "seed_manifest.json")
    for relative, contents in before.items():
        assert (tmp_path / relative).read_bytes() == contents

    def reject_nonstandard_json(token):
        raise AssertionError(f"Nonstandard JSON: {token}")

    for path in output.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonstandard_json)
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == report
    # Reporting validation results and current input values are outside this archive-only calculation.
    validation_file = study / "selected_metrics.csv"
    validation_file.write_text("Changed validation-only report; do not inspect this file.\n", encoding="utf-8")
    frame = pd.read_csv(source)
    frame.loc[frame.date >= "2001-01-04", "release_m3s"] += 100
    frame.to_csv(source, index=False)
    changed_source = source.read_bytes()
    assert run_budget_review(study, output, policy=policy()) == report
    assert source.read_bytes() == changed_source
    assert configuration.read_bytes() == before["config.json"]


def test_equal_hypervolumes_do_not_establish_cross_seed_front_agreement(tmp_path, monkeypatch):
    study = tmp_path / "study"
    write_archive(study, 1, [(0.0, 0.5)] * 12)
    write_archive(study, 2, [(0.5, 0.0)] * 12)
    write_archive(study, 3, [(0.0, 0.5)] * 12)
    before = snapshots(study)
    forbid_search_and_scoring(monkeypatch)
    output = tmp_path / "review"
    report = run_budget_review(study, output, policy=policy())
    assert all(row["status"] == "stopping_candidate" for row in report["seed_status"])
    assert {row["hypervolume"] for row in report["seed_status"]} == {0.5}
    distances = pd.read_csv(output / "cross_seed_fronts.csv")
    assert distances.loc[distances.evaluations == 12, "front_distance"].max() == pytest.approx(0.5)
    assert report["cross_seed_front_agreement"] is False
    assert report["budget_decision"] == "continue_search"
    assert report["suggested_next_experiment_budget_per_seed"] == 20
    assert snapshots(study) == before


def test_three_agreeing_stagnant_fronts_are_only_a_candidate_for_independent_audit(tmp_path):
    study = tmp_path / "study"
    for seed in (1, 2, 3):
        write_archive(study, seed, [(0.1, 0.2)] * 12)
    report = run_budget_review(study, tmp_path / "review", policy=policy())
    assert report["cross_seed_front_agreement"] is True
    assert report["budget_decision"] == "stopping_candidate_requires_independent_audit"
    assert report["suggested_next_experiment_budget_per_seed"] is None
    assert report["production_budget_recommendation"] is None
    assert report["global_convergence_established"] is False
    assert all(row["earliest_stopping_candidate_evaluations"] == 12 for row in report["seed_status"])


def test_many_small_tolerated_windows_still_fail_cumulative_extension_audit(tmp_path):
    study = tmp_path / "study"
    costs = [(0.1, 0.2)] * 12 + [(0.095, 0.2)] * 4 + [(0.09, 0.2)] * 4
    for seed in (1, 2, 3):
        write_archive(study, seed, costs)
    report = run_budget_review(study, tmp_path / "review", policy=policy(
        max_evaluations=24, normalized_hv_tolerance=0.002,
        front_tolerance=0.006, extrema_tolerance=0.006,
    ))
    assert report["cross_seed_front_agreement"] is True
    assert all(row["status"] == "stopping_candidate" for row in report["seed_status"])
    assert all(row["extension_front_movement"] == pytest.approx(0.01) for row in report["seed_status"])
    assert all(row["extension_exceeds_policy_tolerances"] for row in report["seed_status"])
    assert report["budget_decision"] == "continue_search"
    assert report["suggested_next_experiment_budget_per_seed"] == 24


@pytest.mark.parametrize("tamper", ["artifact", "missing_artifact_hash", "model_identity", "population"])
def test_bad_archive_or_population_configuration_is_rejected_before_output(tmp_path, tamper):
    study = tmp_path / "study"
    first = write_archive(study, 1, [(0.1, 0.2)] * 12)
    second = write_archive(study, 2, [(0.1, 0.2)] * 12)
    if tamper == "artifact":
        (first / "evaluated_vectors.csv").write_bytes((first / "evaluated_vectors.csv").read_bytes() + b"\ncorruption\n")
    elif tamper in {"missing_artifact_hash", "model_identity"}:
        path = second / "seed_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if tamper == "missing_artifact_hash":
            del manifest["artifact_sha256"]["evaluations.csv"]
        else:
            manifest["identity"]["resolved_search_configuration"]["model"]["capacity_m3"] += 1
        path.write_text(json.dumps(manifest), encoding="utf-8")
    settings = policy(population_size=2) if tamper == "population" else policy()
    before = snapshots(study)
    output = tmp_path / "review"
    with pytest.raises(ValueError):
        run_budget_review(study, output, policy=settings)
    assert not output.exists()
    assert snapshots(study) == before


def test_an_archive_change_during_review_aborts_before_publishing(tmp_path, monkeypatch):
    study = tmp_path / "study"
    folder = write_archive(study, 1, [(0.1, 0.2)] * 12)
    original_pairs = review_module._pairs
    changed = []

    def pairs_and_change_artifact(records, count):
        if not changed:
            (folder / "selected_metrics.csv").write_text("Archive changed during review\n", encoding="utf-8")
            changed.append(True)
        return original_pairs(records, count)

    monkeypatch.setattr(review_module, "_pairs", pairs_and_change_artifact)
    output = tmp_path / "review"
    with pytest.raises(RuntimeError, match="artifacts changed"):
        run_budget_review(study, output, policy=policy())
    assert changed == [True]
    assert not output.exists()


@pytest.mark.parametrize("inside", ["", "runs", "runs/seed_1"])
def test_review_output_cannot_modify_the_original_study_tree(tmp_path, inside):
    study = tmp_path / "study"
    write_archive(study, 1, [(0.1, 0.2)] * 12)
    before = snapshots(study)
    with pytest.raises(ValueError, match="outside"):
        run_budget_review(study, study / inside, policy=policy())
    assert snapshots(study) == before


def test_external_output_collision_cannot_overwrite_source_observations(tmp_path):
    source, configuration = write_inputs(tmp_path, source_name="seed_status.csv")
    study = tmp_path / "study"
    run_calibration_study(configuration, tmp_path, study)
    before = snapshots(tmp_path)
    with pytest.raises(ValueError):
        run_budget_review(study, tmp_path, policy=policy())
    assert snapshots(tmp_path) == before
    assert source.read_bytes() == before["seed_status.csv"]


@pytest.mark.parametrize("tolerance", [True, -0.1, np.inf, np.nan])
def test_invalid_cross_seed_tolerance_is_rejected_before_writing(tmp_path, tolerance):
    study = tmp_path / "study"
    write_archive(study, 1, [(0.1, 0.2)] * 12)
    output = tmp_path / "review"
    with pytest.raises(ValueError):
        run_budget_review(study, output, policy=policy(), cross_seed_tolerance=tolerance)
    assert not output.exists()
