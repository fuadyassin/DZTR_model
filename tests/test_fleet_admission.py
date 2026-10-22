"""Admission preserves physical/source gates without requiring one baseline."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("fleet_admission_test", Path(__file__).parents[1] / "tools/fleet_admission.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def box():
    return np.tile([20, 40, 60, 1, 2, 3], (12, 1)), np.tile([35, 55, 95, 2, 3, 4], (12, 1))


def test_ordered_witness_respects_original_box_and_physics():
    lower, upper = box()
    original = lower.copy()
    result = module.ordered_candidate(lower, upper, 100, .1, 2, close_critical_gap=True)
    point = np.array(result["monthly_targets_si"])
    assert result["feasible"] and not result["simulation_verified"]
    assert (point >= lower).all() and (point <= upper).all()
    assert (np.diff(point[:, :3], axis=1) > 0).all()
    assert (point[:, 0] > 10).all() and (point[:, 2] <= 100).all()
    assert (np.diff(point[:, 3:], axis=1) >= 0).all()
    assert (point[:, 0] - 10 >= 2 * point[:, 3]).all()
    np.testing.assert_array_equal(lower, original)


def test_empty_ordered_box_is_not_clipped_or_widened():
    lower, upper = box()
    lower[4, 2] = 101
    upper[4, 2] = 110
    result = module.ordered_candidate(lower, upper, 100, .1, 2)
    assert not result["feasible"] and result["infeasible_months"] == [5]
    assert result["monthly_targets_si"] is None


def test_critical_gap_constraint_is_separate_from_storage_order():
    lower, upper = box()
    assert module.ordered_candidate(lower, upper, 100, .1, 30)["feasible"]
    assert not module.ordered_candidate(lower, upper, 100, .1, 30, close_critical_gap=True)["feasible"]


def evidence():
    return {"calibration_status": "completed", "calibration_quality": {"numerically_valid": True, "physical_storage_valid": True}}


def audit():
    return {"ordered_box": {"feasible": True}, "critical_gap_box": {"feasible": True}}


def test_multiple_root_calibration_evidence_admits_failed_strict_baseline():
    review = {"reservoir_id": "example", "strict_calibration_baseline_feasible": False}
    proof = {**evidence(), "validation_status": "blocked", "metrics": {"validation": {"storage_nse": -1000}}}
    assert module.classify(review, {}, proof, audit())[0] == "ready_explicit_root_policy"


def test_source_flags_are_not_erased_by_successful_solver():
    review = {"reservoir_id": "example", "strict_calibration_baseline_feasible": True, "calibration_flags": {"above_capacity": 3}}
    assert module.classify(review, {}, evidence(), audit())[0] == "source_assumptions_review"


def test_failed_baseline_with_feasible_box_enters_initialization_queue():
    assert module.classify({"reservoir_id": "example"}, {}, {}, audit())[0] == "initialization_preflight_required"


@pytest.mark.parametrize("identifier,status", [("ex12_Trinity", "source_profile_required"), ("ex26_Ghost", "source_profile_required"), ("ex35_Bull_Lake", "cached_interval_results")])
def test_named_profiles_and_cached_results_remain_in_program(identifier, status):
    assert module.classify({"reservoir_id": identifier}, {}, evidence(), audit())[0] == status


def test_bounds_ignore_extreme_held_out_values():
    dates = pd.date_range("2000-01-01", "2003-12-01", freq="MS")
    frame = pd.DataFrame({"storage_m3": 25.0 + dates.month.to_numpy(), "release_m3s": 1 + dates.month.to_numpy() / 10}, index=dates)
    config = {"periods": {"calibration": {"start": "2000-01-01", "end": "2001-12-01"}}, "model": {"capacity_m3": 100, "dead_storage_fraction": .1}, "sampling": {"frequency": "monthly", "fixed_step_days": 30.5}, "quantile_method": "hazen"}
    before = module.bounds_audit(frame, config)
    frame.loc["2002-01-01":] = 1e20
    assert module.bounds_audit(frame, config) == before


def test_external_forcing_cannot_be_promoted_by_root_evidence():
    assert module.classify({"reservoir_id": "example"}, {"simulation_supported": False}, evidence(), audit())[0] == "forcing_adapter_required"
