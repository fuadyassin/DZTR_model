"""Independent saved-result arithmetic, with no model calls."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import verify_policy_fleet_results as verifier


def test_balance_reconstruction_keeps_trapezoidal_factor_and_spill():
    frame = pd.DataFrame({"storage_m3": [50., 52., 51.], "inflow_m3s": [5., 7., 9.], "release_m3s": [2., 2., 7.], "step_seconds": [2., 2., 2.]})
    # dt/2 = 1, corrected inflow sums are 6 and 8; net increments 2 and -1.
    np.testing.assert_array_equal(verifier.balance(frame, .5), [0., 0.])


def test_independent_front_keeps_equal_tradeoffs_and_rejects_dominated():
    costs = np.array([[1., 2.], [2., 1.], [3., 3.], [1., 2.]])
    assert verifier.nondominated_ids(costs, [10, 20, 30, 40]) == {10, 20, 40}


def test_representatives_use_calibration_costs_and_deterministic_ids():
    front = pd.DataFrame({"release_cost": [-.9, -.7, -.5], "storage_cost": [-.5, -.7, -.9]}, index=[3, 7, 9])
    assert verifier.representatives(front) == {"best_release": 3, "best_storage": 9, "balanced": 7}


def test_nse_uses_finite_paired_observations():
    assert verifier.nse([1., 2., np.nan, 3.], [1., 2., 99., 3.]) == 1
