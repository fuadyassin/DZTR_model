"""Portable diagnoses expose evidence without changing records or policies."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from dztr.evaluation.readiness import diagnose_sample, inspect_implicit_branches, trace_implicit_failure
from dztr.model.balance import _implicit_solution
from dztr.model.release import _monthly_values
from dztr.schemas import DataQuality, MONTHLY_TARGET_COLUMNS, ReservoirObservations, ReservoirParameters
from dztr.simulation.sampled import SamplingSettings


def _sample():
    index = pd.date_range("2018-01-01", periods=60, freq="MS", name="date")
    year = index.year - 2018
    frame = pd.DataFrame({"storage_m3": 2e8 + year * 1e8 + index.month * 1e5,
                          "inflow_m3s": 2 + np.sin(np.arange(60)),
                          "release_m3s": 1 + year * .4 + index.month * .01}, index=index)
    observations = ReservoirObservations(frame, DataQuality("available", 60, {name: 0 for name in frame}, 0), "fixture")
    config = {"reservoir": {"id": "portable"}, "sampling": {"frequency": "monthly", "fixed_step_days": 30.5, "month_clock": "observed", "legacy_start": None, "allow_signed_inflow": False},
              "periods": {"simulation_start": "2018-01-01", "calibration": {"start": "2019-01-01", "end": "2020-12-01"}, "validation": {"start": "2021-01-01", "end": "2022-12-01"}, "warmup_rows": 12},
              "model": {"capacity_m3": 1e9, "initial_storage_m3": 4e8, "initial_release_m3s": 1., "channel_capacity_m3s": 20., "dead_storage_fraction": .1, "inflow_correction": 1., "flood_mode": "storage"},
              "inflow_preprocessing": {"factor": 1., "offset_m3s": 0.}, "simulation_supported": True,
              "target_percentiles": [10, 45, 85], "quantile_method": "hazen", "atol_storage_m3": 1e-4}
    return observations, config


def _codes(result):
    return {issue["code"] for issue in result["issues"]}


def _parameters(*, inflow=False, large_release=False):
    row = [2e8, 5e8, 8e8, 100., 200., 300.] if large_release else [2e8, 5e8, 8e8, 1., 2., 3.]
    targets = pd.DataFrame([row] * 12, columns=MONTHLY_TARGET_COLUMNS, index=pd.Index(range(1, 13), name="month"))
    return ReservoirParameters(targets, 1e9, 4e8, 1., 500. if large_release else 20., .1, 1., "inflow" if inflow else "storage")


def test_ready_diagnosis_is_json_ready_and_does_not_modify_input():
    observations, config = _sample()
    before = observations.frame.copy(deep=True)
    old_config = deepcopy(config)
    result = diagnose_sample(observations, config)
    assert result["summary"]["status"] == "ready_for_baseline"
    assert result["summary"]["calibration_bounds_status"] == "feasible_box"
    assert result["summary"]["convergence_assessed"] is False
    assert result["summary"]["optimizer_executed"] is False
    json.dumps(result, allow_nan=False)
    pd.testing.assert_frame_equal(observations.frame, before, check_exact=True)
    assert config == old_config
    assert all(set(issue) == {"code", "severity", "phase", "message", "evidence", "next_action"} for issue in result["issues"])


def test_targets_and_bounds_exclude_warmup_validation_and_raw_signed_inflow():
    observations, config = _sample()
    expected = diagnose_sample(observations, config)
    altered = observations.frame.copy()
    outside = (altered.index.year == 2018) | (altered.index.year >= 2021)
    altered.loc[outside, "storage_m3"] = 9e8
    altered.loc[outside, "release_m3s"] = 9000.
    altered["inflow_m3s"] = -1.
    actual = diagnose_sample(replace(observations, frame=altered), config)
    assert actual["summary"]["calibration_targets"] == expected["summary"]["calibration_targets"]
    assert actual["summary"]["calibration_bounds"] == expected["summary"]["calibration_bounds"]
    assert "transformed_inflow_gate" in _codes(actual)
    assert "target_estimation_failed" not in _codes(actual)


def test_transformed_forcing_and_external_forcing_are_separate_gates():
    observations, config = _sample()
    config["inflow_preprocessing"] = {"factor": 1., "offset_m3s": -10.}
    result = diagnose_sample(observations, config)
    issue = next(issue for issue in result["issues"] if issue["code"] == "transformed_inflow_gate")
    assert issue["evidence"]["transformed"]["negative_count"] == 60
    config["simulation_supported"] = False
    config["simulation_unsupported_reason"] = "Requires source MESH forcing adapter."
    config["inflow_preprocessing"] = None
    result = diagnose_sample(observations, config)
    assert "external_forcing_required" in _codes(result)
    assert "invalid_inflow_preprocessing" in _codes(result)
    assert result["summary"]["status"] == "blocked"


def test_signed_policy_is_explicit_and_raw_missing_zero_measurements_are_retained():
    observations, config = _sample()
    changed = observations.frame.copy()
    changed["inflow_m3s"] = -1.
    changed.iloc[0, changed.columns.get_loc("storage_m3")] = np.nan
    changed["release_m3s"] = 0.
    config["sampling"]["allow_signed_inflow"] = True
    result = diagnose_sample(replace(observations, frame=changed), config)
    assert {"negative_observations", "missing_observations", "all_zero_observations", "signed_forcing_policy"}.issubset(_codes(result))
    assert "transformed_inflow_gate" not in _codes(result)


@pytest.mark.parametrize("change", ["missing_month", "warmup", "boundary"])
def test_native_sampling_and_split_errors_are_explained(change):
    observations, config = _sample()
    if change == "missing_month":
        observations = replace(observations, frame=observations.frame.drop(observations.frame.index[20]))
    elif change == "warmup":
        config["periods"]["warmup_rows"] = 365
    else:
        config["periods"]["calibration"]["start"] = "2019-01-02"
    result = diagnose_sample(observations, config)
    assert "invalid_sampling_or_periods" in _codes(result)
    assert result["summary"]["status"] == "blocked"


@pytest.mark.parametrize("kind,expected", [("dead", "dead_storage_not_below_critical"), ("equal", "critical_not_below_normal"), ("capacity", "flood_target_above_capacity"), ("initial", "invalid_initial_storage"), ("negative_release", "negative_release_targets"), ("reverse", "release_targets_not_ordered")])
def test_target_violations_report_months_and_values(kind, expected):
    observations, config = _sample()
    frame = observations.frame.copy()
    if kind == "dead":
        config["model"]["dead_storage_fraction"] = .5
    elif kind == "equal":
        frame["storage_m3"] = 3e8
    elif kind == "capacity":
        config["model"]["capacity_m3"] = 3.5e8
        config["model"]["initial_storage_m3"] = 3e8
    elif kind == "initial":
        config["model"]["initial_storage_m3"] = 2e9
    elif kind == "negative_release":
        frame["release_m3s"] = -1.
    else:
        config["target_percentiles"] = [85, 45, 10]
    result = diagnose_sample(replace(observations, frame=frame), config)
    assert expected in _codes(result)
    if kind != "initial":
        matching = [issue for issue in result["issues"] if issue["code"] == expected]
        assert {issue["evidence"]["month"] for issue in matching} == set(range(1, 13))
    if kind == "equal":
        assert "normal_not_below_flood" in _codes(result)


def test_inherited_full_record_dead_storage_note_is_explicit():
    observations, config = _sample()
    result = diagnose_sample(observations, config, {"source_dead_storage_expression": "0.5*min(Lc)"})
    issue = next(issue for issue in result["issues"] if issue["code"] == "inherited_dead_storage")
    assert issue["evidence"]["full_record_dependency"] is True
    assert issue["severity"] == "info"


def test_infeasible_empirical_bounds_are_separate_from_valid_baseline_targets():
    observations, config = _sample()
    config["model"]["dead_storage_fraction"] = .35
    config["target_percentiles"] = [50, 60, 95]
    result = diagnose_sample(observations, config)
    assert "physical_parameters_invalid" not in _codes(result)
    assert "calibration_bounds_invalid" in _codes(result)
    assert result["summary"]["calibration_bounds_status"] == "blocked"
    issue = next(issue for issue in result["issues"] if issue["code"] == "calibration_bounds_invalid")
    assert issue["evidence"]["period"] == config["periods"]["calibration"]


def test_branch_diagnostics_distinguish_multiple_roots_from_no_root():
    parameters = _parameters(inflow=True)
    multiple = inspect_implicit_branches(802e6, 100., 86400., 1, parameters)
    assert multiple["distinct_root_count"] == 2
    assert multiple["flood_boundary"]["release_jump_m3s"] < 0
    assert multiple["policy_selected"] is False
    with pytest.raises(ValueError, match="Multiple"):
        _implicit_solution(802e6, 100., 86400., _monthly_values(1, parameters), 1e-4)
    parameters = _parameters(large_release=True)
    no_root = inspect_implicit_branches(280e6, 100., 30.5 * 86400., 1, parameters)
    assert no_root["distinct_root_count"] == 0
    with pytest.raises(ValueError, match="No feasible"):
        _implicit_solution(280e6, 100., 30.5 * 86400., _monthly_values(1, parameters), 1e-4)
    json.dumps([multiple, no_root], allow_nan=False)


def test_reported_runtime_root_failure_is_not_labeled_target_capacity_violation(monkeypatch):
    observations, config = _sample()
    import dztr.evaluation.readiness as module
    calls = []
    def trace(*args, **kwargs):
        calls.append(kwargs)
        return {"distinct_root_count": 2, "date": "2019-02-01T00:00:00", "policy_selected": False}
    monkeypatch.setattr(module, "trace_implicit_failure", trace)
    report = {"baseline_reason": "Simulation failed at 2019-02-01: Multiple implicit storage/release roots (2) exist.", "calibration_baseline_reason": "Multiple implicit storage/release roots (2) exist."}
    result = diagnose_sample(observations, config, report)
    assert "implicit_multiple_roots" in _codes(result)
    assert "flood_target_above_capacity" not in _codes(result)
    assert len(calls) == 1
    assert calls[0]["stop_date"] == "2019-02-01"


def test_negative_predictor_trace_preserves_prior_state_and_does_not_choose_root():
    observations, _ = _sample()
    observations = replace(observations, frame=observations.frame.iloc[:3].copy())
    observations.frame["inflow_m3s"] = 0.
    parameters = replace(_parameters(), initial_storage_m3=1., initial_release_m3s=100.)
    detail = trace_implicit_failure(observations, parameters, SamplingSettings("monthly", 30.5))
    assert detail["failure_code"] == "implicit_negative_storage"
    assert detail["predictor_m3"] < 0
    assert detail["previous_state"]["storage_m3"] == 1.
    assert detail["policy_selected"] is False
