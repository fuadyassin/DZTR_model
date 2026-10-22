"""Independent verification must keep invalid simulated states on scored support."""
import numpy as np
import pytest
from tools.verify_prepared_handoff_v1 import nse


def test_independent_nse_matches_direct_arithmetic():
    observed = np.array([1., 2., 4., 6.])
    simulated = np.array([1., 3., 5., 6.])
    expected = 1 - np.sum((observed - simulated) ** 2) / np.sum((observed - observed.mean()) ** 2)
    assert nse(observed, simulated) == expected


def test_independent_nse_cannot_drop_bad_simulated_state():
    with pytest.raises(ValueError, match="cannot remove"):
        nse([1., 2., 3.], [1., np.nan, 3.])


def test_independent_nse_uses_observation_support():
    assert nse([1., np.nan, 3.], [1., np.nan, 3.]) == 1.
    assert np.isnan(nse([1., 1.], [1., 2.]))
