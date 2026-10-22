"""Native vector layout and calibration-only Pareto accounting."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from review_critical_release import critical_block,nondominated_mask


def test_critical_release_coordinates_are_fourth_twelve_month_block():
    values=np.arange(144).reshape(2,72)
    np.testing.assert_array_equal(critical_block(values),values[:,36:48])
    with pytest.raises(ValueError):critical_block(np.zeros((12,6)))


def test_pareto_duplicates_and_tradeoffs_are_retained():
    values=np.array([[0,1],[1,0],[1,1],[0,1],[2,2]])
    np.testing.assert_array_equal(nondominated_mask(values),[True,True,False,True,False])
    with pytest.raises(ValueError):nondominated_mask([[np.nan,1]])
