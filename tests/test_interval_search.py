"""Reviewed-policy guards and counted, held-out-free interval search receipts."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dztr.calibration.interval_search as adapter
from dztr.calibration.checkpoint import load_optimizer_checkpoint
from dztr.calibration.objectives import CandidateEvaluation
from dztr.calibration.optimizer import NSGA2Settings
from dztr.calibration.stopping import StoppingPolicy


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def reviewed_policy(tmp_path):
    dates = pd.date_range("1999-12-31", "2001-01-30", name="date")
    step = np.arange(len(dates), dtype=float)
    observed = pd.DataFrame({"storage_m3": 40e6 + (step % 31) * 1e6,
                            "release_m3s": 1. + (step % 31) * .1,
                            "inflow_m3s": 3. + (step % 4) * .3}, index=dates)
    observed.loc["2000-01-05", "inflow_m3s"] = -.1
    source = tmp_path / "observations.csv"
    observed.to_csv(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    config = {"reservoir": {"id": "synthetic", "name": "Synthetic"},
        "observations_file": "observations.csv", "observations_sha256": source_hash,
        "units": {"storage": "m3", "flows": "m3/s"},
        "sampling": {"frequency": "daily", "fixed_step_days": 1., "month_clock": "observed"},
        "periods": {"simulation_start": "1999-12-31",
                    "calibration": {"start": "2000-01-01", "end": "2000-12-31"},
                    "validation": {"start": "2001-01-01", "end": "2001-01-30"}},
        "model": {"capacity_m3": 100e6, "initial_storage_m3": 40e6, "initial_release_m3s": 1.,
                  "channel_capacity_m3s": 8., "dead_storage_fraction": .1,
                  "inflow_correction": 1., "flood_mode": "storage"},
        "inflow_preprocessing": {"factor": 1., "offset_m3s": 0.},
        "target_percentiles": [10, 45, 85], "quantile_method": "hazen", "atol_storage_m3": 1e-4}
    policy = {"schema_version": 1, "kind": "reservoir_interval_review_policy", "policy_id": "synthetic_v1",
        "reservoir_config_file": "config.json", "observations_file": "observations.csv",
        "observations_sha256": source_hash,
        "interval_inputs": {"flow_semantics": "computed_net_inflow", "flow_alignment": "interval_mean_end",
                            "nominal_step_seconds": 86400, "inflow_correction": 1.,
                            "inflow_offset_m3s": 0., "additional_loss_model": None},
        "quarantine": {"interval_end_dates": ["2000-01-10"]},
        "future_simulation": {"flood_mode": "storage", "carry_state_across_protocol_period_boundaries": True,
                              "additional_restart_spinup_days": 0}}
    write_json(tmp_path / "config.json", config)
    write_json(tmp_path / "policy.json", policy)
    return tmp_path / "policy.json"


def test_loader_fixes_strict_nominal_settings_and_independent_calibration_masks(reviewed_policy):
    problem, baseline, validation, context = adapter.load_interval_problem(reviewed_policy)
    assert problem.settings.root_policy == "reject"
    assert problem.settings.initial_storage_offset_capacity_fraction == 0.
    assert validation.start == pd.Timestamp("2001-01-01")
    assert problem.bounds.contains(baseline)
    support = context["fit"].support
    assert support.loc[0, "storage_sample_count"] == 364
    assert support.loc[0, "release_sample_count"] == 363
    assert support.loc[1, "storage_sample_count"] == 29
    assert support.loc[1, "release_sample_count"] == 28
    assert context["bounds"].period == problem.calibration_period
    frame = context["preparation"].frame
    assert frame.loc["2000-01-05", "inflow_m3s"] == -.1
    assert not frame.loc["2000-01-10", "storage_target_eligible"]
    assert frame.loc["2000-01-11", "segment_initialization"]
    assert frame.loc["2000-01-11", "storage_target_eligible"]
    assert not frame.loc["2000-01-11", "release_score_eligible"]
    assert all(Path(path).exists() for path in context["consumed_files_sha256"])


def test_validation_observations_never_change_calibration_targets_or_bounds(reviewed_policy):
    first_problem, first_vector, _, first = adapter.load_interval_problem(reviewed_policy)
    source = first["source_path"]
    frame = pd.read_csv(source, parse_dates=["date"], index_col="date")
    frame.loc["2001-01-01":, "storage_m3"] = 900e6
    frame.loc["2001-01-01":, "release_m3s"] = 800.
    frame.to_csv(source)
    changed_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    policy = json.loads(reviewed_policy.read_text())
    config = json.loads(first["config_path"].read_text())
    policy["observations_sha256"] = config["observations_sha256"] = changed_hash
    write_json(reviewed_policy, policy)
    write_json(first["config_path"], config)
    second_problem, second_vector, _, second = adapter.load_interval_problem(reviewed_policy)
    np.testing.assert_array_equal(first_vector, second_vector)
    pd.testing.assert_frame_equal(first["fit"].estimates.monthly_targets, second["fit"].estimates.monthly_targets)
    pd.testing.assert_frame_equal(first["fit"].support, second["fit"].support)
    pd.testing.assert_frame_equal(first["bounds"].monthly_bounds, second["bounds"].monthly_bounds)
    assert first_problem.calibration_period == second_problem.calibration_period


@pytest.mark.parametrize("location,key,value,match", [
    ("interval_inputs", "flow_alignment", "endpoint", "end-labelled"),
    ("interval_inputs", "nominal_step_seconds", 3600, "86400"),
    ("interval_inputs", "nominal_step_seconds", True, "86400"),
    ("interval_inputs", "inflow_offset_m3s", .1, "offset"),
    ("interval_inputs", "additional_loss_model", "fitted_evaporation", "loss"),
    ("interval_inputs", "inflow_correction", 1.1, "Computed net"),
    ("future_simulation", "carry_state_across_protocol_period_boundaries", False, "carry"),
    ("future_simulation", "additional_restart_spinup_days", 3, "spinup"),
])
def test_unsupported_policy_interpretation_rejected_before_search(reviewed_policy, location, key, value, match):
    policy = json.loads(reviewed_policy.read_text())
    policy[location][key] = value
    write_json(reviewed_policy, policy)
    with pytest.raises(ValueError, match=match):
        adapter.load_interval_problem(reviewed_policy)


@pytest.mark.parametrize("location,key,value,match", [
    ("sampling", "frequency", "monthly", "daily"),
    ("sampling", "fixed_step_days", 2., "daily"),
    ("sampling", "month_clock", "legacy_synthetic", "calendar"),
    ("inflow_preprocessing", "factor", 2., "preprocessing"),
    ("inflow_preprocessing", "offset_m3s", 1., "preprocessing"),
    ("inflow_preprocessing", "negative_mode", "clip", "preprocessing"),
    ("model", "flood_mode", "inflow", "Computed net"),
    ("model", "inflow_correction", 1.01, "Computed net"),
    ("model", "evaporation_factor", .1, "extra loss"),
])
def test_unsupported_config_processing_rejected(reviewed_policy, location, key, value, match):
    path = reviewed_policy.parent / "config.json"
    config = json.loads(path.read_text())
    config[location][key] = value
    write_json(path, config)
    with pytest.raises(ValueError, match=match):
        adapter.load_interval_problem(reviewed_policy)


def test_consumed_source_hash_must_match_reviewed_bytes(reviewed_policy):
    path = reviewed_policy.parent / "observations.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256"):
        adapter.load_interval_problem(reviewed_policy)


def synthetic_evaluator(monkeypatch, *, reject_every=None, all_rejected=False, change=None):
    calls = []

    def evaluate(self, vector):
        calls.append(np.asarray(vector).copy())
        if change is not None and len(calls) == 2:
            change()
        lower, upper = self.bounds.lower_vector, self.bounds.upper_vector
        unit = np.divide(vector - lower, upper - lower, out=np.zeros(72), where=upper != lower)
        feasible = not all_rejected and not (reject_every and len(calls) % reject_every == 0)
        # Finite, spectacular costs on rejected candidates must still be excluded.
        costs = (-unit[0], -(1 - unit[0])) if feasible else (-999., -999.)
        return CandidateEvaluation(feasible, costs, {}, {"release": 363, "storage": 363},
            None if feasible else "Synthetic strict-root rejection at 2000-04-03", None, None, self.calibration_period)

    def held_out(*args, **kwargs):
        raise AssertionError("Seed search must not score or replay held-out validation.")

    monkeypatch.setattr(adapter.IntervalCalibrationObjective, "evaluate", evaluate)
    monkeypatch.setattr(adapter.IntervalCalibrationObjective, "evaluate_validation", held_out)
    return calls


def run_small(policy, output, **options):
    return adapter.run_interval_seed(policy, output,
        settings=NSGA2Settings(seed=42, population_size=4, evaluation_budget=16),
        checkpoints=(4, 8, 16), stopping_policy=StoppingPolicy(population_size=4, check_interval=4,
            window_evaluations=4, minimum_evaluations=8, patience=2, max_evaluations=16), **options)


def test_seed_budget_includes_baseline_and_rejected_attempts_and_has_no_validation(reviewed_policy, monkeypatch, tmp_path):
    calls = synthetic_evaluator(monkeypatch, reject_every=3)
    progress = []
    output = tmp_path / "seed"
    report = run_small(reviewed_policy, output, progress_callback=progress.append)
    assert len(calls) == report["search_evaluations"] == 16
    assert report["baseline_included_in_budget"] is True
    assert report["baseline_evaluation_id"] == 0
    assert report["reporting_validation_evaluations"] == report["reporting_calibration_evaluations"] == 0
    assert report["rejected_evaluations"] == 5
    assert report["feasible_evaluations"] == 11
    assert report["rejection_reasons"] == {"Synthetic strict-root rejection at 2000-04-03": 5}
    assert report["root_policy"] == "reject"
    assert report["optimizer_convergence_established"] is False
    assert report["stopping_controls_search"] is False
    assert [record.evaluations for record in progress] == [4, 8, 12, 16]
    assert set(adapter.ARTIFACT_NAMES).issubset(path.name for path in output.iterdir())
    trials = pd.read_csv(output / "evaluations.csv")
    vectors = pd.read_csv(output / "evaluated_vectors.csv", index_col="evaluation_id")
    assert trials.evaluation_id.tolist() == list(range(16))
    np.testing.assert_allclose(vectors.to_numpy(), np.stack(calls), rtol=0, atol=1e-14)
    assert trials.loc[~trials.feasible, "release_nse"].isna().all()
    front = pd.read_csv(output / "pareto_front.csv")
    assert front.feasible.all()
    assert front.negative_release_nse.ge(-1).all()
    saved = json.loads((output / "summary.json").read_text())
    assert saved == report
    checkpoints = pd.read_csv(output / "checkpoint_metrics.csv")
    assert checkpoints.evaluations.tolist() == [4, 8, 16]
    np.testing.assert_allclose(checkpoints.normalized_hypervolume, checkpoints.hypervolume / 4.)
    state = load_optimizer_checkpoint(output / "optimizer_checkpoint.npz", report["checkpoint_identity"])
    assert len(state.evaluated) == report["checkpoint_evaluations"] == 16
    np.testing.assert_allclose(state.baseline_vector, calls[0])


def test_all_rejected_candidates_still_have_complete_receipts(reviewed_policy, monkeypatch, tmp_path):
    calls = synthetic_evaluator(monkeypatch, all_rejected=True)
    output = tmp_path / "failed_seed"
    report = run_small(reviewed_policy, output)
    assert len(calls) == report["search_evaluations"] == report["rejected_evaluations"] == 16
    assert report["termination"] == "no_feasible_candidate"
    assert report["pareto_count"] == report["feasible_evaluations"] == 0
    assert report["selected_candidate_ids"] == {"generalized_baseline": 0}
    assert len(pd.read_csv(output / "evaluations.csv")) == 16
    assert pd.read_csv(output / "pareto_front.csv").empty
    assert pd.read_csv(output / "pareto_vectors.csv").empty
    assert report["optimizer_convergence_established"] is False


def test_existing_output_is_refused_before_loading_or_evaluation(reviewed_policy, monkeypatch, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "important.txt"
    marker.write_text("preserve me", encoding="utf-8")
    monkeypatch.setattr(adapter, "load_interval_problem", lambda *_: pytest.fail("Existing output must fail before loading."))
    with pytest.raises(FileExistsError, match="fresh"):
        run_small(reviewed_policy, output)
    assert marker.read_text() == "preserve me"
    assert list(output.iterdir()) == [marker]


def test_mid_search_source_change_prevents_a_misleading_receipt(reviewed_policy, monkeypatch, tmp_path):
    source = reviewed_policy.parent / "observations.csv"
    calls = synthetic_evaluator(monkeypatch, change=lambda: source.write_bytes(source.read_bytes() + b"\n"))
    output = tmp_path / "changed"
    with pytest.raises(RuntimeError, match="Consumed input changed"):
        run_small(reviewed_policy, output)
    assert len(calls) == 16
    assert not output.exists()


def test_mid_search_module_change_prevents_a_misleading_receipt(reviewed_policy, monkeypatch, tmp_path):
    synthetic_evaluator(monkeypatch)
    hashes = iter([{"solver.py": "original"}, {"solver.py": "changed"}])
    monkeypatch.setattr(adapter, "_module_hashes", lambda: next(hashes))
    output = tmp_path / "changed_code"
    with pytest.raises(RuntimeError, match="module hashes changed"):
        run_small(reviewed_policy, output)
    assert not output.exists()


@pytest.mark.parametrize("checkpoints", [(0,), (17,), (True,), (4, 4), (4.5,), "16"])
def test_invalid_checkpoint_counts_do_not_start_search(reviewed_policy, monkeypatch, tmp_path, checkpoints):
    monkeypatch.setattr(adapter, "load_interval_problem", lambda *_: pytest.fail("Invalid checkpoints must fail before loading."))
    with pytest.raises(ValueError, match="checkpoint|Checkpoint"):
        adapter.run_interval_seed(reviewed_policy, tmp_path / "invalid",
            settings=NSGA2Settings(population_size=4, evaluation_budget=16), checkpoints=checkpoints)


def test_stopping_suggestion_does_not_truncate_fixed_search_budget(reviewed_policy, monkeypatch, tmp_path):
    calls = synthetic_evaluator(monkeypatch)
    policy = StoppingPolicy(population_size=4, check_interval=4, window_evaluations=4,
        minimum_evaluations=8, patience=1, max_evaluations=16,
        normalized_hv_tolerance=100., front_tolerance=100., extrema_tolerance=100.)
    report = adapter.run_interval_seed(reviewed_policy, tmp_path / "diagnostic",
        settings=NSGA2Settings(population_size=4, evaluation_budget=16),
        checkpoints=(4, 8, 16), stopping_policy=policy)
    assert report["stopping_assessment"]["suggested_stop_evaluations"] == 8
    assert report["search_evaluations"] == len(calls) == 16
    assert report["termination"] == "budget_reached"
    assert report["stopping_controls_search"] is False
