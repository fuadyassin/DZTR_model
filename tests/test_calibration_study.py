"""Portable nested-budget study, calibration selection and verified-cache checks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dztr.calibration.study as study_module
from dztr.calibration.study import STUDY_ARTIFACT_NAMES, StudySettings, run_calibration_study
from dztr.data import normalize_observations


MODEL = {"capacity_m3": 80e6, "initial_storage_m3": 55e6, "initial_release_m3s": 2.5}


def source_frame(*, changed_validation=False):
    dates = pd.date_range("1999-12-02", "2001-01-10", name="date")
    frame = normalize_observations(pd.DataFrame({
        "storage_m3": 50e6 + (dates.day.to_numpy() - 16) * 1e6,
        "inflow_m3s": np.full(len(dates), 2.2),
        "release_m3s": 2.0 + dates.day.to_numpy() * 0.03,
    }, index=dates)).frame
    if changed_validation:
        frame.loc["2001-01-04":, "storage_m3"] *= 3
        frame.loc["2001-01-04":, "release_m3s"] += 20
    return frame


def write_inputs(folder, *, changed_validation=False, source_name="observations.csv", config_name="config.json"):
    source = folder / source_name
    source.parent.mkdir(parents=True, exist_ok=True)
    source_frame(changed_validation=changed_validation).to_csv(source, index_label="date")
    config = {
        "reservoir": {"name": "Synthetic nested-budget study"}, "observations_file": source_name,
        "periods": {"calibration": {"start": "2000-01-01", "end": "2000-12-31"},
                    "validation": {"start": "2001-01-04", "end": "2001-01-10"}, "warmup_steps": 30},
        "model": MODEL, "quantile_method": "hazen", "target_percentiles": [10, 45, 85],
        "simulation": {"solver": "implicit", "overflow_mode": "conservative"},
        "optimization": {"seed": 23, "population_size": 4, "evaluation_budget": 8},
        "study": {"seeds": [23, 24], "evaluation_checkpoints": [4, 8],
                  "hypervolume_reference": [1, 1], "workers": 1},
    }
    configuration = folder / config_name
    configuration.parent.mkdir(parents=True, exist_ok=True)
    configuration.write_text(json.dumps(config), encoding="utf-8")
    return source, configuration, config


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def snapshots(folder):
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*") if path.is_file()}


def test_study_exports_honest_checkpoint_counts_pooled_front_and_si_spreads(tmp_path):
    source, configuration, config = write_inputs(tmp_path)
    input_bytes = {path: path.read_bytes() for path in (source, configuration)}
    output = tmp_path / "study"
    report = run_calibration_study(configuration, tmp_path, output)
    assert {path.name for path in output.iterdir()} == {*STUDY_ARTIFACT_NAMES, "runs"}
    assert report["independent_seed_count"] == 2
    assert report["total_search_evaluations"] == 16
    assert report["feasible_evaluations"] + report["rejected_evaluations"] == 16
    assert report["convergence_established"] is False
    code_root = Path(study_module.__file__).resolve().parents[1]
    for name in ("calibration/study.py", "calibration/convergence.py"):
        assert report["analysis_code_sha256"][name] == hashlib.sha256((code_root / name).read_bytes()).hexdigest()
    assert report["study"]["hypervolume_reference"] == [1.0, 1.0]
    assert report["source_sha256"] == {"observations": hashlib.sha256(input_bytes[source]).hexdigest(),
                                       "configuration": hashlib.sha256(input_bytes[configuration]).hexdigest()}
    checkpoints = read_csv(output / "checkpoint_metrics.csv")
    assert checkpoints.shape[0] == 4
    assert set(zip(checkpoints.seed, checkpoints.budget)) == {(23, 4), (23, 8), (24, 4), (24, 8)}
    assert (checkpoints.feasible_count + checkpoints.rejected_count == checkpoints.budget).all()
    for _, group in checkpoints.groupby("seed"):
        assert np.diff(group.sort_values("budget").hypervolume).min() >= -1e-12
    budget_summary = read_csv(output / "budget_summary.csv")
    assert budget_summary.budget.tolist() == [4, 8]
    assert (budget_summary.hypervolume_min <= budget_summary.hypervolume_median).all()
    assert (budget_summary.hypervolume_median <= budget_summary.hypervolume_max).all()
    evaluated = []
    for seed in (23, 24):
        folder = output / "runs" / f"seed_{seed}"
        trials = read_csv(folder / "evaluations.csv")
        assert len(trials) == 8
        assert trials.evaluation_id.tolist() == list(range(8))
        manifest = json.loads((folder / "seed_manifest.json").read_text(encoding="utf-8"))
        assert manifest["search_evaluations"] == 8
        for name, digest in manifest["artifact_sha256"].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
        for row in trials.loc[trials.feasible].itertuples(index=False):
            evaluated.append(((seed, row.evaluation_id), np.array([row.negative_release_nse, row.negative_storage_nse])))
    expected_origins = {
        origin for origin, costs in evaluated
        if not any(np.all(other <= costs) and np.any(other < costs)
                   for other_origin, other in evaluated if other_origin != origin)
    }
    pooled = read_csv(output / "pooled_front.csv")
    assert set(zip(pooled.seed, pooled.evaluation_id)) == expected_origins
    assert len(pooled) == report["pooled_front_count"]
    vectors = read_csv(output / "pooled_vectors.csv")
    assert vectors.shape == (len(pooled), 75)
    assert vectors.pooled_id.tolist() == pooled.pooled_id.tolist()
    representatives = read_csv(output / "representative_vectors.csv")
    assert representatives.shape == (2 * 2 * 3, 76)
    targets = read_csv(output / "representative_targets_si.csv")
    assert len(targets) == 2 * 2 * 3 * 12
    assert targets.storage_critical_m3.gt(MODEL["capacity_m3"] * 0.1).all()
    assert targets.storage_flood_m3.le(MODEL["capacity_m3"]).all()
    stability = read_csv(output / "parameter_stability.csv")
    assert len(stability) == 2 * 3 * 72
    assert stability.loc[stability.vector_index < 36, "unit"].eq("m3").all()
    assert stability.loc[stability.vector_index >= 36, "unit"].eq("m3/s").all()
    native_columns = [f"x{i:02d}" for i in range(72)]
    scales = np.r_[np.full(36, MODEL["capacity_m3"]), np.ones(36)]
    for (budget, role), subset in stability.groupby(["budget", "role"]):
        subset = subset.sort_values("vector_index")
        native = representatives.loc[(representatives.budget == budget) & (representatives.role == role), native_columns].to_numpy()
        si = native * scales
        np.testing.assert_allclose(subset.min_si, si.min(axis=0), rtol=1e-12)
        np.testing.assert_allclose(subset.max_si, si.max(axis=0), rtol=1e-12)
        widths = subset.upper_si.to_numpy() - subset.lower_si.to_numpy()
        expected_spread = np.divide(si.max(axis=0) - si.min(axis=0), widths, out=np.zeros(72), where=widths > 0)
        np.testing.assert_allclose(subset.normalized_spread, expected_spread, atol=1e-12)
    unique_selected = {item["pooled_id"] for item in report["selected_candidates"].values()}
    assert report["reporting_calibration_evaluations"] == report["reporting_validation_evaluations"] == len(unique_selected)
    for role in ("best_release", "best_storage", "balanced"):
        assert report["selected_candidates"][role]["pooled_id"] in set(pooled.pooled_id)
    daily = read_csv(output / "selected_daily_simulations.csv")
    assert daily.loc[daily.period == "validation"].groupby("pooled_id").size().eq(7).all()

    def reject_nonstandard_json(token):
        raise AssertionError(f"Nonstandard JSON: {token}")

    for path in output.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonstandard_json)
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == report
    assert json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))["study"] == report["study"]
    assert config["study"]["evaluation_checkpoints"] == report["study"]["evaluation_checkpoints"]
    for path, contents in input_bytes.items():
        assert path.read_bytes() == contents


def test_validation_measurements_cannot_move_checkpoints_fronts_or_pooled_choices(tmp_path):
    original_root, changed_root = tmp_path / "original", tmp_path / "changed"
    original_root.mkdir()
    changed_root.mkdir()
    original_source, original_config, _ = write_inputs(original_root)
    changed_source, changed_config, _ = write_inputs(changed_root, changed_validation=True)
    before = {path: path.read_bytes() for path in (original_source, original_config, changed_source, changed_config)}
    first_output, second_output = original_root / "study", changed_root / "study"
    first = run_calibration_study(original_config, original_root, first_output)
    second = run_calibration_study(changed_config, changed_root, second_output)
    for name in ("checkpoint_metrics.csv", "budget_summary.csv", "checkpoint_fronts.csv", "pooled_front.csv",
                 "pooled_vectors.csv", "representative_vectors.csv", "parameter_stability.csv"):
        pd.testing.assert_frame_equal(read_csv(first_output / name), read_csv(second_output / name))
    assert first["selected_candidates"] == second["selected_candidates"]
    for role in first["selected_scores"]:
        assert first["selected_scores"][role]["calibration"] == second["selected_scores"][role]["calibration"]
        assert first["selected_scores"][role]["validation"]["metrics"]["release_rmse"] != second["selected_scores"][role]["validation"]["metrics"]["release_rmse"]
    for seed in (23, 24):
        for name in ("evaluations.csv", "evaluated_vectors.csv"):
            pd.testing.assert_frame_equal(read_csv(first_output / "runs" / f"seed_{seed}" / name),
                                          read_csv(second_output / "runs" / f"seed_{seed}" / name))
    for path, contents in before.items():
        assert path.read_bytes() == contents


def test_analyze_only_reuses_verified_search_without_running_optimizer(tmp_path, monkeypatch):
    _, configuration, _ = write_inputs(tmp_path)
    output = tmp_path / "study"
    first = run_calibration_study(configuration, tmp_path, output)
    seed_artifacts = snapshots(output / "runs")

    def forbidden_search(*args, **kwargs):
        raise AssertionError("Analyze-only must reuse verified candidate artifacts.")

    monkeypatch.setattr(study_module, "run_configured_optimization", forbidden_search)
    second = run_calibration_study(configuration, tmp_path, output, analyze_only=True)
    assert all(item["reused"] for item in second["seed_runs"].values())
    assert second["total_search_evaluations"] == first["total_search_evaluations"] == 16
    assert second["selected_candidates"] == first["selected_candidates"]
    assert second["selected_scores"] == first["selected_scores"]
    assert snapshots(output / "runs") == seed_artifacts


def test_analysis_code_change_aborts_before_overwriting_completed_reporting(tmp_path, monkeypatch):
    _, configuration, _ = write_inputs(tmp_path)
    output = tmp_path / "study"
    run_calibration_study(configuration, tmp_path, output)
    completed_artifacts = snapshots(output)
    original_hashes = study_module._analysis_hashes
    calls = []

    def changing_hashes():
        hashes = original_hashes()
        calls.append(1)
        if len(calls) == 2:
            hashes["calibration/study.py"] = "0" * 64
        return hashes

    monkeypatch.setattr(study_module, "_analysis_hashes", changing_hashes)
    with pytest.raises(RuntimeError, match="Analysis code changed during the study"):
        run_calibration_study(configuration, tmp_path, output, analyze_only=True)
    assert len(calls) == 2
    assert snapshots(output) == completed_artifacts


@pytest.mark.parametrize("tamper", ["artifact", "search_settings"])
def test_analyze_only_rejects_tampered_cache_or_changed_settings_before_any_search(tmp_path, monkeypatch, tamper):
    _, configuration, config = write_inputs(tmp_path)
    output = tmp_path / "study"
    run_calibration_study(configuration, tmp_path, output)
    if tamper == "artifact":
        artifact = output / "runs" / "seed_23" / "evaluated_vectors.csv"
        artifact.write_bytes(artifact.read_bytes() + b"\ncorrupted cache\n")
    else:
        config["optimization"]["mutation_eta"] = 21
        configuration.write_text(json.dumps(config), encoding="utf-8")
    before = snapshots(tmp_path)

    def forbidden_search(*args, **kwargs):
        raise AssertionError("An invalid cache must not silently start a replacement search.")

    monkeypatch.setattr(study_module, "run_configured_optimization", forbidden_search)
    with pytest.raises(ValueError, match="integrity|differs"):
        run_calibration_study(configuration, tmp_path, output, analyze_only=True)
    assert snapshots(tmp_path) == before


@pytest.mark.parametrize("collision", ["root_source", "root_config", "seed_source", "seed_config"])
def test_study_and_seed_artifact_destinations_cannot_overwrite_inputs(tmp_path, collision):
    source_name = {"root_source": "study/checkpoint_metrics.csv", "seed_source": "study/runs/seed_23/evaluated_vectors.csv"}.get(collision, "observations.csv")
    config_name = {"root_config": "study/summary.json", "seed_config": "study/runs/seed_23/resolved_config.json"}.get(collision, "config.json")
    _, configuration, _ = write_inputs(tmp_path, source_name=source_name, config_name=config_name)
    before = snapshots(tmp_path)
    with pytest.raises(ValueError, match="overwrite source observations or configuration"):
        run_calibration_study(configuration, tmp_path, tmp_path / "study")
    assert snapshots(tmp_path) == before


def test_intermediate_checkpoint_must_end_at_full_generation_before_search(tmp_path, monkeypatch):
    _, configuration, config = write_inputs(tmp_path)
    config["study"]["evaluation_checkpoints"] = [5, 8]
    configuration.write_text(json.dumps(config), encoding="utf-8")

    def forbidden_search(*args, **kwargs):
        raise AssertionError("Invalid checkpoint alignment must fail before search.")

    monkeypatch.setattr(study_module, "run_configured_optimization", forbidden_search)
    with pytest.raises(ValueError, match="population-sized generations"):
        run_calibration_study(configuration, tmp_path, tmp_path / "study")
    assert not (tmp_path / "study").exists()


@pytest.mark.parametrize("settings", [
    {"seeds": [23]}, {"seeds": [23, 23]}, {"seeds": [23, True]}, {"seeds": [-1, 23]},
    {"evaluation_checkpoints": [8]}, {"evaluation_checkpoints": [8, 4]}, {"evaluation_checkpoints": [4, 4]},
    {"hypervolume_reference": [1]}, {"hypervolume_reference": [np.inf, 1]},
    {"workers": 0}, {"workers": True},
])
def test_invalid_static_study_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        StudySettings(**settings)


def test_budget_and_fleet_analysis_changes_preserve_search_identity_but_optimizer_changes_do_not(tmp_path, monkeypatch):
    _, _, config = write_inputs(tmp_path)
    original_sha = study_module._sha
    before = study_module._search_identity(config, tmp_path, 23, 8)
    analysis_names = {"stopping.py", "budget_review.py", "fleet.py", "study.py", "convergence.py"}

    def changed_analysis_hash(path):
        return "0" * 64 if path.name in analysis_names else original_sha(path)

    monkeypatch.setattr(study_module, "_sha", changed_analysis_hash)
    assert study_module._search_identity(config, tmp_path, 23, 8) == before

    def changed_optimizer_hash(path):
        return "0" * 64 if path.name == "optimizer.py" else original_sha(path)

    monkeypatch.setattr(study_module, "_sha", changed_optimizer_hash)
    assert study_module._search_identity(config, tmp_path, 23, 8) != before
