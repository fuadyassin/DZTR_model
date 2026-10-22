"""Negative-NSE candidate costs, fitting isolation and MATLAB state parity."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dztr.calibration.objectives as objective_module
from dztr.calibration.bounds import CalibrationBounds
from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.objectives import CalibrationObjective
from dztr.calibration.periods import DatePeriod
from dztr.data import normalize_observations
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters
from dztr.simulation import simulate_reservoir


CALIBRATION = DatePeriod("2000-01-01", "2000-01-05")
VALIDATION = DatePeriod("2000-01-06", "2000-01-10")


def template():
    return ReservoirParameters(
        pd.DataFrame(np.tile([20e6, 50e6, 80e6, 1, 2, 3], (12, 1)),
                     index=pd.Index(range(1, 13), name="month"), columns=MONTHLY_TARGET_COLUMNS),
        capacity_m3=100e6, initial_storage_m3=60e6, initial_release_m3s=2.5,
        channel_capacity_m3s=4,
    )


def observations():
    dates = pd.date_range("1999-12-30", "2000-01-10", name="date")
    sequence = np.arange(len(dates), dtype=float)
    return normalize_observations(pd.DataFrame({
        "storage_m3": 35e6 + sequence * 1e6,
        "inflow_m3s": 2.5 + (sequence % 4) * 0.3,
        "release_m3s": 1 + sequence * 0.1,
    }, index=dates), source="synthetic_objective_reference")


def objective(source=None, **options):
    return CalibrationObjective(
        observations() if source is None else source,
        CALIBRATION, template(), warmup_steps=2, **options,
    )


def negative_nse(observed, simulated):
    observed, simulated = np.asarray(observed), np.asarray(simulated)
    valid = np.isfinite(observed) & np.isfinite(simulated)
    x, y = observed[valid], simulated[valid]
    return np.sum((x - y) ** 2) / np.sum((x - x.mean()) ** 2) - 1


def box():
    parameters = template()
    lower, upper = parameters.monthly_targets * 0.9, parameters.monthly_targets * 1.1
    lower.columns = [f"lower_{column}" for column in MONTHLY_TARGET_COLUMNS]
    upper.columns = [f"upper_{column}" for column in MONTHLY_TARGET_COLUMNS]
    return CalibrationBounds(pd.concat([lower, upper], axis=1), parameters.capacity_m3,
                             parameters.dead_storage_fraction, CALIBRATION)


def test_perfect_candidate_has_negative_one_cost_for_both_nse_objectives():
    source = observations()
    simulated = simulate_reservoir(source, template())
    perfect = source.frame.copy()
    perfect[["storage_m3", "release_m3s"]] = simulated.frame[["storage_m3", "release_m3s"]]
    problem = objective(normalize_observations(perfect))
    evaluated = problem.evaluate(parameters_to_vector(template()))
    assert evaluated.feasible
    assert evaluated.reason is None
    np.testing.assert_allclose(evaluated.objectives, [-1, -1], atol=1e-12)
    np.testing.assert_allclose(problem(parameters_to_vector(template())), [-1, -1], atol=1e-12)


def test_objective_signs_order_and_scored_pairs_match_independent_formula():
    source = observations()
    source.frame.loc[pd.Timestamp("2000-01-02"), "release_m3s"] = np.nan
    source.frame.loc[pd.Timestamp("2000-01-04"), "storage_m3"] = np.nan
    evaluated = objective(source).evaluate(parameters_to_vector(template()))
    scored = source.frame.loc[CALIBRATION.start:CALIBRATION.end]
    simulated = evaluated.simulation.frame.loc[CALIBRATION.start:CALIBRATION.end]
    expected = [negative_nse(scored.release_m3s, simulated.release_m3s),
                negative_nse(scored.storage_m3, simulated.storage_m3)]
    assert evaluated.feasible
    assert evaluated.period == CALIBRATION
    assert evaluated.paired_counts == {"release": 4, "storage": 4}
    np.testing.assert_allclose(evaluated.objectives, expected)
    assert evaluated.metrics["release_nse"] == pytest.approx(-expected[0])
    assert evaluated.metrics["storage_nse"] == pytest.approx(-expected[1])
    assert evaluated.simulation.frame.index[-1] == CALIBRATION.end


@pytest.mark.parametrize("changed_period", ["spinup", "validation"])
def test_spinup_or_validation_storage_release_cannot_change_calibration_costs(changed_period):
    source = observations()
    changed = source.frame.copy()
    mask = changed.index < CALIBRATION.start if changed_period == "spinup" else changed.index > CALIBRATION.end
    changed.loc[mask, "storage_m3"] = 1e12
    changed.loc[mask, "release_m3s"] = 1e6
    vector = parameters_to_vector(template())
    original = objective(source).evaluate(vector)
    alternative = objective(normalize_observations(changed)).evaluate(vector)
    np.testing.assert_array_equal(alternative.objectives, original.objectives)
    pd.testing.assert_frame_equal(alternative.simulation.frame, original.simulation.frame)


def test_future_missing_inflow_is_ignored_for_calibration_but_rejected_for_validation():
    source = observations()
    source.frame.loc[VALIDATION.start:, "inflow_m3s"] = np.nan
    problem = objective(source)
    vector = parameters_to_vector(template())
    assert problem.evaluate(vector).feasible
    with pytest.raises(ValueError, match="inflow"):
        problem.evaluate_validation(vector, VALIDATION)


def test_held_out_evaluation_keeps_state_and_scores_every_validation_day():
    source = observations()
    problem = objective(source)
    vector = parameters_to_vector(template())
    calibrated, held_out = problem.evaluate(vector), problem.evaluate_validation(vector, VALIDATION)
    assert held_out.feasible
    assert held_out.period == VALIDATION
    assert held_out.paired_counts == {"release": 5, "storage": 5}
    pd.testing.assert_frame_equal(held_out.simulation.frame.loc[:CALIBRATION.end], calibrated.simulation.frame)
    simulated = held_out.simulation.frame.loc[VALIDATION.start:VALIDATION.end]
    scored = source.frame.loc[VALIDATION.start:VALIDATION.end]
    np.testing.assert_allclose(held_out.objectives, [negative_nse(scored.release_m3s, simulated.release_m3s),
                                                  negative_nse(scored.storage_m3, simulated.storage_m3)])
    reset = simulate_reservoir(normalize_observations(scored), template())
    assert reset.frame.storage_m3.iloc[0] == template().initial_storage_m3
    assert simulated.storage_m3.iloc[0] != pytest.approx(reset.frame.storage_m3.iloc[0], abs=1)


@pytest.mark.parametrize("kind", ["short", "matrix", "nonfinite", "tied_storage", "negative_release", "reversed_release"])
def test_invalid_candidate_is_penalized_without_mutating_inputs(kind):
    source, parameters = observations(), template()
    problem = CalibrationObjective(source, CALIBRATION, parameters, warmup_steps=2)
    candidate = parameters_to_vector(parameters)
    if kind == "short":
        candidate = candidate[:-1]
    elif kind == "matrix":
        candidate = candidate.reshape(6, 12)
    elif kind == "nonfinite":
        candidate[0] = np.nan
    elif kind == "tied_storage":
        candidate[12] = candidate[0]
    elif kind == "negative_release":
        candidate[36] = -1
    else:
        candidate[48] = 0.5
    before_vector = candidate.copy()
    before_parameters, before_source = parameters.monthly_targets.copy(), source.frame.copy()
    evaluated = problem.evaluate(candidate)
    assert not evaluated.feasible
    assert evaluated.reason
    np.testing.assert_array_equal(evaluated.objectives, [np.inf, np.inf])
    np.testing.assert_array_equal(problem(candidate), [np.inf, np.inf])
    np.testing.assert_array_equal(candidate, before_vector)
    pd.testing.assert_frame_equal(parameters.monthly_targets, before_parameters)
    pd.testing.assert_frame_equal(source.frame, before_source)


def test_physical_candidate_outside_calibration_box_is_penalized_only_with_bounds():
    candidate = parameters_to_vector(template())
    candidate[0] = 0.24  # Valid critical storage, above this box's 0.22 upper limit.
    assert objective().evaluate(candidate).feasible
    rejected = objective(bounds=box()).evaluate(candidate)
    assert not rejected.feasible
    assert "bounds" in rejected.reason
    np.testing.assert_array_equal(rejected.objectives, [np.inf, np.inf])


def test_mutating_caller_bounds_after_construction_does_not_change_problem():
    bounds = box()
    problem = objective(bounds=bounds)
    candidate = parameters_to_vector(template())
    before = problem(candidate)
    assert np.isfinite(before).all()
    bounds.monthly_bounds.loc[1, "upper_storage_critical_m3"] = 19e6
    assert not bounds.contains(candidate)
    np.testing.assert_array_equal(problem(candidate), before)


def test_declared_physical_solver_failure_becomes_infeasible_candidate(monkeypatch):
    def failing_simulation(*args, **kwargs):
        raise ValueError("Simulation failed: negative storage in supplied forcing.")

    monkeypatch.setattr(objective_module, "simulate_reservoir", failing_simulation)
    evaluated = objective().evaluate(parameters_to_vector(template()))
    assert not evaluated.feasible
    assert "Simulation failed" in evaluated.reason
    np.testing.assert_array_equal(evaluated.objectives, [np.inf, np.inf])


def test_undefined_auxiliary_kge_does_not_invalidate_finite_nse(monkeypatch):
    original = simulate_reservoir

    def constant_simulation(source, parameters, **kwargs):
        result = original(source, parameters, **kwargs)
        frame = result.frame.copy()
        frame["storage_m3"] = parameters.initial_storage_m3
        frame["release_m3s"] = parameters.initial_release_m3s
        frame["balance_residual_m3"] = 0.0
        return replace(result, frame=frame)

    monkeypatch.setattr(objective_module, "simulate_reservoir", constant_simulation)
    evaluated = objective().evaluate(parameters_to_vector(template()))
    assert evaluated.feasible
    assert np.isfinite(evaluated.objectives).all()
    assert np.isnan(evaluated.metrics["storage_kge"])
    assert np.isnan(evaluated.metrics["release_kge"])


def compatibility_parameters(initial_storage, initial_release):
    return ReservoirParameters(
        pd.DataFrame(np.tile([200, 500, 800, 0.001, 0.01, 0.02], (12, 1)),
                     index=pd.Index(range(1, 13), name="month"), columns=MONTHLY_TARGET_COLUMNS),
        capacity_m3=1000, initial_storage_m3=initial_storage,
        initial_release_m3s=initial_release, channel_capacity_m3s=0.03,
    )


def test_legacy_fixed_point_oscillation_is_rejected_even_without_spill_or_negative_storage():
    parameters = compatibility_parameters(350, 0.0055)
    source = normalize_observations(pd.DataFrame({
        "storage_m3": [350, 400], "release_m3s": [0.0055, 0.006],
        "inflow_m3s": [0.0055, 0.0055],
    }, index=pd.date_range("2000-01-01", periods=2, name="date")))
    problem = CalibrationObjective(source, DatePeriod("2000-01-01", "2000-01-02"), parameters,
                                   warmup_steps=0, solver="matlab_fixed_point", overflow_mode="matlab")
    evaluated = problem.evaluate(parameters_to_vector(parameters))
    assert evaluated.simulation is not None
    frame = evaluated.simulation.frame
    assert frame.storage_m3.gt(0).all()
    assert frame.overflow_volume_m3.eq(0).all()
    assert frame.balance_residual_m3.abs().max() > 500
    assert not evaluated.feasible
    assert "water-balance residual" in evaluated.reason
    np.testing.assert_array_equal(evaluated.objectives, [np.inf, np.inf])


def test_implicit_solver_with_legacy_spill_accounting_remains_an_accepted_candidate():
    parameters = compatibility_parameters(950, 0.01)
    source = normalize_observations(pd.DataFrame({
        "storage_m3": [940, 990], "release_m3s": [0.01, 0.04],
        "inflow_m3s": [0.04, 0.04],
    }, index=pd.date_range("2000-01-01", periods=2, name="date")))
    problem = CalibrationObjective(source, DatePeriod("2000-01-01", "2000-01-02"), parameters,
                                   warmup_steps=0, solver="implicit", overflow_mode="matlab")
    evaluated = problem.evaluate(parameters_to_vector(parameters))
    assert evaluated.feasible
    assert evaluated.reason is None
    assert np.isfinite(evaluated.objectives).all()
    frame = evaluated.simulation.frame
    assert frame.overflow_volume_m3.iloc[1] > 0
    assert frame.balance_residual_m3.abs().max() > 500
    np.testing.assert_allclose(frame.balance_residual_m3 + frame.overflow_volume_m3 / 2, 0, atol=1e-9)


@pytest.mark.parametrize("column", ["storage_m3", "release_m3s"])
@pytest.mark.parametrize("kind", ["constant", "all_missing", "one_value"])
def test_undefined_observed_calibration_nse_is_a_configuration_error(column, kind):
    source = observations()
    source.frame.loc[CALIBRATION.start:CALIBRATION.end, column] = np.nan if kind != "constant" else 4
    if kind == "one_value":
        source.frame.loc[CALIBRATION.start, column] = 5
    with pytest.raises(ValueError, match="at least two nonmissing, nonconstant"):
        objective(source)


@pytest.mark.parametrize("options", [
    {"solver": "unknown"}, {"overflow_mode": "unknown"},
    {"atol_storage_m3": 0}, {"atol_storage_m3": -1},
    {"atol_storage_m3": np.inf}, {"atol_storage_m3": np.nan},
])
def test_bad_solver_configuration_is_rejected_before_candidate_evaluation(options):
    with pytest.raises(ValueError):
        objective(**options)


def test_validation_overlap_and_incomplete_coverage_are_configuration_errors():
    problem, vector = objective(), parameters_to_vector(template())
    for period in (DatePeriod("2000-01-04", "2000-01-06"), DatePeriod("2000-01-11", "2000-01-12")):
        with pytest.raises(ValueError):
            problem.evaluate_validation(vector, period)
    incomplete = observations().frame.drop(index=pd.Timestamp("2000-01-08"))
    problem = objective(normalize_observations(incomplete))
    with pytest.raises(ValueError, match="complete consecutive daily coverage"):
        problem.evaluate_validation(vector, VALIDATION)


def test_observation_or_template_edits_after_construction_do_not_change_problem():
    source, parameters = observations(), template()
    problem = CalibrationObjective(source, CALIBRATION, parameters, warmup_steps=2)
    vector = parameters_to_vector(parameters)
    before = problem(vector)
    source.frame.loc[:, "inflow_m3s"] *= 100
    source.frame.loc[:, "storage_m3"] *= 100
    parameters.monthly_targets.loc[:, "release_normal_m3s"] = 1.5
    np.testing.assert_array_equal(problem(vector), before)


@pytest.mark.parametrize("member", [0, 1, 2])
def test_portable_matlab_1971_candidate_costs_match_independent_saved_state_nse(member):
    path = Path(__file__).parent / "reference_cases/trinity_matlab_reference.npz"
    with np.load(path, allow_pickle=False) as fixture:
        parameters = ReservoirParameters(
            pd.DataFrame(fixture["coefficients"][4:].reshape(19, 12)[1:7].T,
                         index=range(1, 13), columns=MONTHLY_TARGET_COLUMNS),
            float(fixture["capacity_m3"]), float(fixture["initial_storage_m3"]),
            float(fixture["initial_release_m3s"]), float(fixture["channel_capacity_m3s"]),
            dead_storage_fraction=float(fixture["dead_storage_fraction"]),
            inflow_correction=float(fixture["inflow_correction"]),
        )
        frame = pd.DataFrame({"inflow_m3s": fixture["inflow_m3s"],
                              "storage_m3": fixture["observed_storage_m3"],
                              "release_m3s": fixture["observed_release_m3s"]},
                             index=pd.DatetimeIndex(fixture["dates"], name="date"))
        problem = CalibrationObjective(normalize_observations(frame), DatePeriod("1971-01-01", "1971-12-31"),
                                       parameters, warmup_steps=365)
        evaluated = problem.evaluate(fixture["parameter_vectors"][member])
        expected = [negative_nse(fixture["observed_release_m3s"][365:], fixture["release_m3s"][365:, member]),
                    negative_nse(fixture["observed_storage_m3"][365:], fixture["storage_m3"][365:, member])]
        assert evaluated.feasible
        assert evaluated.paired_counts == {"release": 365, "storage": 365}
        np.testing.assert_allclose(evaluated.objectives, expected, rtol=0, atol=5e-10)
        np.testing.assert_allclose(evaluated.simulation.frame.storage_m3, fixture["storage_m3"][:, member], rtol=0, atol=1e-4)
        np.testing.assert_allclose(evaluated.simulation.frame.release_m3s, fixture["release_m3s"][:, member], rtol=0, atol=1e-9)
