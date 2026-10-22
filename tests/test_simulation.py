"""Reservoir integration, conservation and portable MATLAB reference cases."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dztr.data import normalize_observations
from dztr.parameterization import estimate_generalized_parameters, estimate_percentiles
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters
from dztr.simulation import simulate_reservoir


def reservoir(**overrides):
    targets = pd.DataFrame(
        np.tile([40, 60, 80, 1e-4, 2e-4, 3e-4], (12, 1)),
        index=pd.Index(range(1, 13), name="month"), columns=MONTHLY_TARGET_COLUMNS,
    )
    values = dict(monthly_targets=targets, capacity_m3=100,
                  initial_storage_m3=50, initial_release_m3s=1.5e-4,
                  channel_capacity_m3s=1e-3)
    values.update(overrides)
    return ReservoirParameters(**values)


def observations(inflow, start="2001-01-30"):
    index = pd.date_range(start, periods=len(inflow), freq="D", name="date")
    return normalize_observations(pd.DataFrame({
        "storage_m3": np.ones(len(inflow)), "inflow_m3s": inflow,
        "release_m3s": np.zeros(len(inflow)),
    }, index=index))


def test_daily_alignment_initial_state_and_current_month_targets():
    params = reservoir()
    params.monthly_targets.loc[2, list(MONTHLY_TARGET_COLUMNS[3:])] *= 2
    obs = observations([2e-4] * 5)
    output = simulate_reservoir(obs, params)
    assert output.frame.index.equals(obs.frame.index)
    assert output.frame.storage_m3.iloc[0] == 50
    assert output.frame.release_m3s.iloc[0] == 1.5e-4
    assert output.frame.release_m3s.loc["2001-02-01"] > output.frame.release_m3s.loc["2001-01-31"]
    assert np.max(np.abs(output.frame.balance_residual_m3)) < 1e-8
    assert output.issues == ()


def test_mass_conserving_spill_and_explicit_legacy_overflow_difference():
    obs = observations([0.003, 0.003])
    params = reservoir(initial_storage_m3=95)
    conserved = simulate_reservoir(obs, params)
    legacy = simulate_reservoir(obs, params, overflow_mode="matlab")
    assert conserved.frame.storage_m3.iloc[1] == params.capacity_m3
    assert conserved.frame.overflow_volume_m3.iloc[1] > 0
    assert conserved.frame.release_m3s.iloc[1] > params.channel_capacity_m3s
    assert conserved.frame.balance_residual_m3.iloc[1] == pytest.approx(0, abs=1e-9)
    excess = legacy.frame.overflow_volume_m3.iloc[1]
    assert legacy.frame.balance_residual_m3.iloc[1] == pytest.approx(-excess / 2)
    assert any("overflow" in issue for issue in legacy.issues)


def test_legacy_iteration_can_report_negative_storage_without_silent_clipping():
    params = reservoir(initial_storage_m3=1, initial_release_m3s=0.1)
    result = simulate_reservoir(observations([0, 0]), params,
                                solver="matlab_fixed_point", overflow_mode="matlab")
    assert result.frame.storage_m3.iloc[1] < 0
    assert any("negative storage" in issue for issue in result.issues)
    with pytest.raises(ValueError, match="Simulation failed"):
        simulate_reservoir(observations([0, 0]), params)


@pytest.mark.parametrize("inflows", [[0.001, np.nan], [0.001, -0.001]])
def test_missing_or_negative_inflow_is_rejected(inflows):
    with pytest.raises(ValueError):
        simulate_reservoir(observations(inflows), reservoir())


def test_daily_simulation_requires_contiguous_dates():
    obs = observations([0.001] * 3)
    broken = obs.frame.copy()
    broken.index = pd.DatetimeIndex(["2001-01-01", "2001-01-03", "2001-01-04"])
    with pytest.raises(ValueError, match="consecutive"):
        simulate_reservoir(normalize_observations(broken), reservoir())


@pytest.mark.parametrize("unit", ["s", "ms", "us"])
def test_daily_calendar_is_independent_of_datetime_resolution(unit):
    obs = observations([2e-4] * 5)
    expected = simulate_reservoir(obs, reservoir()).frame
    frame = obs.frame.copy()
    # Pandas 1.x promotes these to ns; Pandas 2.x preserves the supplied unit.
    frame.index = pd.DatetimeIndex(frame.index.to_numpy().astype(f"datetime64[{unit}]"), name="date")
    normalized = normalize_observations(frame)
    actual = simulate_reservoir(normalized, reservoir()).frame
    assert actual.index.equals(normalized.frame.index)
    np.testing.assert_array_equal(actual.storage_m3, expected.storage_m3)
    np.testing.assert_array_equal(actual.release_m3s, expected.release_m3s)


def test_duplicate_monthly_target_columns_are_rejected_at_configuration():
    frame = reservoir().monthly_targets
    duplicated = pd.concat([frame, frame[["release_critical_m3s"]]], axis=1)
    with pytest.raises(ValueError, match="column names must be unique"):
        reservoir(monthly_targets=duplicated)


@pytest.mark.parametrize("override", [
    {"capacity_m3": 0}, {"initial_storage_m3": 101},
    {"dead_storage_fraction": 1}, {"inflow_correction": -1},
    {"flood_mode": "unknown"}, {"channel_capacity_m3s": np.nan},
])
def test_invalid_reservoir_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        reservoir(**override)


def test_equal_storage_targets_rejected_before_release_denominator_is_used():
    frame = reservoir().monthly_targets.copy()
    frame.loc[1, "storage_normal_m3"] = frame.loc[1, "storage_critical_m3"]
    with pytest.raises(ValueError, match="Storage targets"):
        reservoir(monthly_targets=frame)


def test_explicit_hazen_compatibility_preserves_existing_linear_default():
    raw = observations([1, 2, 3, 4])
    raw.frame["storage_m3"] = [1, 2, 3, 4]
    linear = estimate_percentiles(raw, (25,)).frame
    hazen = estimate_percentiles(raw, (25,), quantile_method="hazen")
    assert linear.loc[(linear.variable == "storage_m3") & (linear.month == 0), "value"].iloc[0] == 1.75
    assert hazen.frame.loc[(hazen.frame.variable == "storage_m3") & (hazen.frame.month == 0), "value"].iloc[0] == 1.5
    assert hazen.quantile_method == "hazen"
    with pytest.raises(ValueError, match="quantile_method"):
        estimate_percentiles(raw, (25,), quantile_method="midpoint")


@pytest.mark.parametrize("member", [0, 1, 2])
def test_portable_matlab_ensemble_reference_matches_implicit_solver(member):
    from dztr.integrations.matlab import parameters_from_matlab_vector

    path = Path(__file__).parent / "reference_cases/trinity_matlab_reference.npz"
    with np.load(path, allow_pickle=False) as fixture:
        template = ReservoirParameters(
            pd.DataFrame(fixture["coefficients"][4:].reshape(19, 12)[1:7].T,
                         index=range(1, 13), columns=MONTHLY_TARGET_COLUMNS),
            float(fixture["capacity_m3"]), float(fixture["initial_storage_m3"]),
            float(fixture["initial_release_m3s"]), float(fixture["channel_capacity_m3s"]),
            inflow_correction=float(fixture["inflow_correction"]),
        )
        parameters = parameters_from_matlab_vector(fixture["parameter_vectors"][member], template)
        frame = pd.DataFrame({"inflow_m3s": fixture["inflow_m3s"],
                              "storage_m3": fixture["observed_storage_m3"],
                              "release_m3s": fixture["observed_release_m3s"]},
                             index=pd.DatetimeIndex(fixture["dates"], name="date"))
        result = simulate_reservoir(normalize_observations(frame), parameters)
        np.testing.assert_allclose(result.frame.storage_m3, fixture["storage_m3"][:, member], rtol=0, atol=1e-4)
        np.testing.assert_allclose(result.frame.release_m3s, fixture["release_m3s"][:, member], rtol=0, atol=1e-9)
        assert np.max(np.abs(result.frame.balance_residual_m3)) < 1e-4


def test_literal_1000_iteration_solver_matches_short_matlab_reference():
    from dztr.integrations.matlab import parameters_from_matlab_vector

    path = Path(__file__).parent / "reference_cases/trinity_matlab_reference.npz"
    with np.load(path, allow_pickle=False) as fixture:
        template = ReservoirParameters(
            pd.DataFrame(fixture["coefficients"][4:].reshape(19, 12)[1:7].T,
                         index=range(1, 13), columns=MONTHLY_TARGET_COLUMNS),
            float(fixture["capacity_m3"]), float(fixture["initial_storage_m3"]),
            float(fixture["initial_release_m3s"]), float(fixture["channel_capacity_m3s"]),
            inflow_correction=float(fixture["inflow_correction"]),
        )
        parameters = parameters_from_matlab_vector(fixture["parameter_vectors"][0], template)
        frame = pd.DataFrame({"inflow_m3s": fixture["inflow_m3s"][:75],
                              "storage_m3": fixture["observed_storage_m3"][:75],
                              "release_m3s": fixture["observed_release_m3s"][:75]},
                             index=pd.DatetimeIndex(fixture["dates"][:75], name="date"))
        result = simulate_reservoir(normalize_observations(frame), parameters,
                                    solver="matlab_fixed_point", overflow_mode="matlab")
        np.testing.assert_allclose(result.frame.storage_m3, fixture["storage_m3"][:75, 0], rtol=0, atol=1e-4)
        np.testing.assert_allclose(result.frame.release_m3s, fixture["release_m3s"][:75, 0], rtol=0, atol=1e-9)
