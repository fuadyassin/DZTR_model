"""Hand-calculated release laws and implicit storage-balance cases."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dztr.model import dztr_release, dztr_zone, fixed_point_step, solve_implicit_step
from dztr.schemas import ReservoirParameters


@pytest.fixture
def parameters():
    monthly = pd.DataFrame(
        {
            "storage_critical_m3": np.full(12, 200.0),
            "storage_normal_m3": np.full(12, 500.0),
            "storage_flood_m3": np.full(12, 800.0),
            "release_critical_m3s": np.full(12, 1.0),
            "release_normal_m3s": np.full(12, 4.0),
            "release_flood_m3s": np.full(12, 7.0),
        },
        index=pd.Index(range(1, 13), name="month"),
    )
    return ReservoirParameters(
        monthly_targets=monthly,
        capacity_m3=1000.0,
        initial_storage_m3=350.0,
        initial_release_m3s=2.5,
        channel_capacity_m3s=10.0,
    )


@pytest.mark.parametrize(
    "storage, expected_zone, expected_release",
    [
        (-5.0, 0, 0.0), (100.0, 0, 0.0), (105.0, 1, 0.5),
        (150.0, 1, 1.0), (200.0, 1, 1.0), (350.0, 2, 2.5),
        (500.0, 2, 4.0), (650.0, 3, 5.5), (800.0, 3, 7.0),
        (801.0, 4, 7.0), (880.0, 4, 8.0), (900.0, 4, 10.0),
        (10000.0, 4, 10.0),
    ],
)
def test_release_all_zones_and_exact_boundaries(parameters, storage, expected_zone, expected_release):
    assert dztr_zone(storage, 1, parameters) == expected_zone
    assert dztr_release(storage, 20, 1, parameters, 10) == pytest.approx(expected_release)


def test_inflow_option_uses_raw_inflow_and_only_changes_flood_zone(parameters):
    modified = replace(parameters, flood_mode="inflow", inflow_correction=0.5)
    assert dztr_release(650, 20, 1, modified, 10) == pytest.approx(12.0)
    assert dztr_release(650, 2, 1, modified, 10) == pytest.approx(5.5)
    assert dztr_release(350, 20, 1, modified, 10) == pytest.approx(2.5)
    assert dztr_release(800, 20, 1, modified, 10) == pytest.approx(20.0)
    assert dztr_release(801, 20, 1, modified, 10) == pytest.approx(7.0)


def test_channel_capacity_applies_only_above_flood_target(parameters):
    modified = replace(parameters, channel_capacity_m3s=3.0)
    assert dztr_release(800, 20, 1, modified, 10) == 7.0
    assert dztr_release(801, 20, 1, modified, 10) == 3.0


def test_current_month_targets_are_selected(parameters):
    monthly = parameters.monthly_targets.copy()
    monthly.loc[2, ["release_critical_m3s", "release_normal_m3s", "release_flood_m3s"]] = [2, 5, 8]
    modified = replace(parameters, monthly_targets=monthly)
    assert dztr_release(350, 0, 1, modified, 10) == 2.5
    assert dztr_release(350, 0, 2, modified, 10) == 3.5


@pytest.mark.parametrize("storage", [0, 50, 105, 150, 200, 350, 500, 650, 800, 801, 880, 900])
def test_implicit_solver_reproduces_a_steady_state_in_every_segment(parameters, storage):
    release = dztr_release(storage, 0, 1, parameters, 10)
    simulated_storage, simulated_release = solve_implicit_step(storage, release, release, release, 1, parameters, 10)
    assert simulated_storage == pytest.approx(storage)
    assert simulated_release == pytest.approx(release)


def test_implicit_solver_matches_hand_calculation_and_conserves_water(parameters):
    storage, release = solve_implicit_step(350, 2.5, 1, 5, 1, parameters, 10)
    assert storage == pytest.approx(372.5 / 1.05)
    assert release == pytest.approx(0.01 * storage - 1.0)
    assert storage - 350 == pytest.approx(5 * (1 + 5 - 2.5 - release))


def test_inflow_correction_is_applied_to_both_balance_inputs(parameters):
    modified = replace(parameters, inflow_correction=0.5)
    storage, release = solve_implicit_step(350, 2.5, 1, 5, 1, modified, 10)
    assert storage - 350 == pytest.approx(5 * (0.5 * (1 + 5) - 2.5 - release))


def test_raw_solver_returns_storage_above_capacity_for_simulation_to_handle(parameters):
    storage, release = solve_implicit_step(990, 10, 20, 20, 1, parameters, 10)
    assert storage == pytest.approx(1090.0)
    assert release == 10.0


def test_exact_solver_rejects_negative_storage(parameters):
    with pytest.raises(ValueError, match="negative storage"):
        solve_implicit_step(50, 100, 0, 0, 1, parameters, 10)


def test_exact_solver_reports_a_gap_at_discontinuous_critical_boundary(parameters):
    with pytest.raises(ValueError, match="No feasible implicit"):
        solve_implicit_step(200, 0.1, 0.35, 0.35, 1, parameters, 1000)


def test_exact_solver_reports_multiple_roots_from_inflow_discontinuity(parameters):
    modified = replace(parameters, flood_mode="inflow")
    with pytest.raises(ValueError, match="Multiple implicit"):
        solve_implicit_step(750, 1, 1, 20, 1, modified, 10)


def test_exact_solver_reports_multiple_roots_from_low_channel_capacity(parameters):
    modified = replace(parameters, channel_capacity_m3s=3.0)
    with pytest.raises(ValueError, match="Multiple implicit"):
        solve_implicit_step(750, 1, 8, 8, 1, modified, 10)


def test_fixed_iteration_matches_exact_solver_when_iterations_converge(parameters):
    exact = solve_implicit_step(350, 2.5, 5, 5, 1, parameters, 10)
    historical = fixed_point_step(350, 2.5, 5, 5, 1, parameters, 10)
    assert historical == pytest.approx(exact)


def test_fixed_iteration_preserves_the_final_storage_release_stagger(parameters):
    storage, release = fixed_point_step(350, 2.5, 5, 5, 1, parameters, 10, max_iterations=1)
    assert storage == pytest.approx(373.125)
    assert release == pytest.approx(2.73125)
    assert storage - 350 - 5 * (5 + 5 - 2.5 - release) == pytest.approx(-0.71875)


def test_fixed_iteration_retains_negative_raw_storage(parameters):
    assert fixed_point_step(50, 100, 0, 0, 1, parameters, 10) == (-450.0, 0.0)


@pytest.mark.parametrize("dt", [0, -10, float("nan")])
def test_release_rejects_invalid_time_steps(parameters, dt):
    with pytest.raises(ValueError, match="dt_seconds"):
        dztr_release(350, 1, 1, parameters, dt)


@pytest.mark.parametrize("month", [0, 13, 1.5, True])
def test_release_rejects_invalid_month(parameters, month):
    with pytest.raises(ValueError, match="month"):
        dztr_release(350, 1, month, parameters, 10)


def test_solver_rejects_invalid_tolerance_and_iteration_count(parameters):
    with pytest.raises(ValueError, match="atol_storage_m3"):
        solve_implicit_step(350, 2.5, 1, 1, 1, parameters, 10, atol_storage_m3=-1)
    with pytest.raises(ValueError, match="max_iterations"):
        fixed_point_step(350, 2.5, 1, 1, 1, parameters, 10, max_iterations=0)
