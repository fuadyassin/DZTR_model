"""Configured optimizer artifacts, reproducibility and validation isolation."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from dztr.calibration.baseline import run_baseline
from dztr.calibration.periods import CalibrationValidationSplit, DatePeriod
from dztr.calibration.search import ARTIFACT_NAMES, run_configured_optimization
from dztr.data import normalize_observations


MODEL = {"capacity_m3": 80e6, "initial_storage_m3": 55e6, "initial_release_m3s": 2.5}


def synthetic_observations():
    dates = pd.date_range("1999-12-02", "2001-01-10", name="date")
    return normalize_observations(pd.DataFrame({
        "storage_m3": 50e6 + (dates.day.to_numpy() - 16) * 1e6,
        "inflow_m3s": np.full(len(dates), 2.2),
        "release_m3s": 2.0 + dates.day.to_numpy() * 0.03,
    }, index=dates))


def split():
    return CalibrationValidationSplit(DatePeriod("2000-01-01", "2000-12-31"),
                                       DatePeriod("2001-01-04", "2001-01-10"), warmup_steps=30)


def write_inputs(folder, *, changed_validation=False, source_name="observations.csv", config_name="config.json"):
    source = folder / source_name
    frame = synthetic_observations().frame.copy()
    if changed_validation:
        frame.loc[split().validation.start:, "storage_m3"] *= 3
        frame.loc[split().validation.start:, "release_m3s"] += 20
    frame.to_csv(source, index_label="date")
    config = {
        "reservoir": {"name": "Synthetic optimizer"}, "observations_file": source.name,
        "periods": {"calibration": split().calibration.as_dict(), "validation": split().validation.as_dict(), "warmup_steps": 30},
        "model": MODEL, "quantile_method": "hazen", "target_percentiles": [10, 45, 85],
        "simulation": {"solver": "implicit", "overflow_mode": "conservative"},
        "optimization": {"seed": 23, "population_size": 4, "evaluation_budget": 6},
    }
    configuration = folder / config_name
    configuration.write_text(json.dumps(config), encoding="utf-8")
    return source, configuration, config


def reject_nonstandard_json(token):
    raise AssertionError(f"Nonstandard JSON token: {token}")


def test_configured_search_saves_counted_native_vectors_and_replay_metadata(tmp_path):
    source, configuration, original_config = write_inputs(tmp_path)
    source_bytes, configuration_bytes = source.read_bytes(), configuration.read_bytes()
    callbacks = []
    output = tmp_path / "optimization"
    report = run_configured_optimization(configuration, tmp_path, output, progress_callback=callbacks.append)
    assert {path.name for path in output.iterdir()} == set(ARTIFACT_NAMES)
    assert report["objective_order"] == ["negative_release_nse", "negative_storage_nse"]
    assert report["objective_direction"] == "minimize"
    assert report["search_evaluations"] == 6
    assert report["feasible_evaluations"] + report["rejected_evaluations"] == 6
    assert [record.evaluations for record in callbacks] == [4, 6]
    assert report["optimization"]["seed"] == 23
    assert report["optimization"]["population_size"] == 4
    assert report["optimization"]["evaluation_budget"] == 6
    selected = report["selected_candidate_ids"]
    assert set(selected) == {"generalized_baseline", "best_release", "best_storage", "balanced"}
    distinct_ids = set(selected.values())
    assert report["reporting_calibration_evaluations"] == report["reporting_validation_evaluations"] == len(distinct_ids)
    assert report["unique_representative_count"] == len({selected[role] for role in ("best_release", "best_storage", "balanced")})
    assert set(report["software_versions"]) == {"python", "numpy", "pandas", "dztr"}
    assert report["random_generator"] == "numpy.random.Generator / PCG64"
    assert report["code_sha256"]
    assert all(len(digest) == 64 for digest in report["code_sha256"].values())
    assert report["source_sha256"] == {
        "observations": hashlib.sha256(source_bytes).hexdigest(),
        "configuration": hashlib.sha256(configuration_bytes).hexdigest(),
    }
    evaluations = pd.read_csv(output / "evaluations.csv")
    vectors = pd.read_csv(output / "evaluated_vectors.csv")
    assert len(evaluations) == len(vectors) == 6
    assert vectors.shape == (6, 73)
    assert vectors.evaluation_id.tolist() == evaluations.evaluation_id.tolist()
    baseline = run_baseline(synthetic_observations(), split(), **MODEL)
    expected_seed = baseline.estimates.monthly_targets.to_numpy().copy()
    expected_seed[:, :3] /= MODEL["capacity_m3"]
    np.testing.assert_allclose(vectors.iloc[0, 1:].to_numpy(dtype=float), expected_seed.T.reshape(72), atol=1e-12)
    for label in ("calibration", "validation"):
        score = report["selected_scores"]["generalized_baseline"][label]
        np.testing.assert_allclose(score["objectives"], [-baseline.metrics[label]["release_nse"], -baseline.metrics[label]["storage_nse"]], atol=1e-12)
        assert score["paired_counts"] == baseline.paired_counts[label]
    selected_targets = pd.read_csv(output / "selected_targets_si.csv")
    assert len(selected_targets) == len(distinct_ids) * 12
    assert (selected_targets.storage_critical_m3 > MODEL["capacity_m3"] * 0.1).all()
    assert (selected_targets.storage_critical_m3 < selected_targets.storage_normal_m3).all()
    assert (selected_targets.storage_normal_m3 < selected_targets.storage_flood_m3).all()
    assert (selected_targets.storage_flood_m3 <= MODEL["capacity_m3"]).all()
    assert (selected_targets.release_critical_m3s <= selected_targets.release_normal_m3s).all()
    assert (selected_targets.release_normal_m3s <= selected_targets.release_flood_m3s).all()
    daily = pd.read_csv(output / "selected_daily_simulations.csv")
    assert set(daily.evaluation_id) == distinct_ids
    assert daily.groupby("evaluation_id").size().eq(split().simulation_period.day_count).all()
    assert daily.loc[daily.period == "validation"].groupby("evaluation_id").size().eq(7).all()
    for name, expected in (("summary.json", report), ("resolved_config.json", report["configuration"])):
        content = json.loads((output / name).read_text(encoding="utf-8"), parse_constant=reject_nonstandard_json)
        assert content == expected
    assert report["configuration"]["optimization"] != original_config["optimization"]  # All defaults were resolved and saved.
    assert source.read_bytes() == source_bytes
    assert configuration.read_bytes() == configuration_bytes


def test_validation_measurements_cannot_change_search_vectors_costs_or_selected_ids(tmp_path):
    original_folder, changed_folder = tmp_path / "original", tmp_path / "changed"
    original_folder.mkdir()
    changed_folder.mkdir()
    original_source, original_config, _ = write_inputs(original_folder)
    changed_source, changed_config, _ = write_inputs(changed_folder, changed_validation=True)
    before = {path: path.read_bytes() for path in (original_source, original_config, changed_source, changed_config)}
    first_output, second_output = original_folder / "output", changed_folder / "output"
    first = run_configured_optimization(original_config, original_folder, first_output)
    second = run_configured_optimization(changed_config, changed_folder, second_output)
    for name in ("evaluated_vectors.csv", "evaluations.csv", "pareto_vectors.csv", "pareto_front.csv", "bounds_si.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(first_output / name), pd.read_csv(second_output / name))
    assert first["selected_candidate_ids"] == second["selected_candidate_ids"]
    for role in first["selected_scores"]:
        assert first["selected_scores"][role]["calibration"] == second["selected_scores"][role]["calibration"]
        assert first["selected_scores"][role]["validation"]["metrics"]["release_rmse"] != second["selected_scores"][role]["validation"]["metrics"]["release_rmse"]
    for path, contents in before.items():
        assert path.read_bytes() == contents


@pytest.mark.parametrize("collision", ["observations", "configuration"])
def test_optimizer_outputs_cannot_overwrite_inputs(tmp_path, collision):
    source, configuration, _ = write_inputs(
        tmp_path,
        source_name="evaluated_vectors.csv" if collision == "observations" else "observations.csv",
        config_name="summary.json" if collision == "configuration" else "config.json",
    )
    before = {path: path.read_bytes() for path in (source, configuration)}
    names_before = {path.name for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="overwrite source observations or configuration"):
        run_configured_optimization(configuration, tmp_path, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == names_before
    for path, contents in before.items():
        assert path.read_bytes() == contents
