"""The inferred curve stays explicit, causal, bounded and incremental."""
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest

from workflows import aishihik_storage_candidate_v1 as w


def levels(values):
    return pd.Series(values, index=pd.date_range('2000-01-01', periods=len(values), name='date'))


def convert(values, **kwargs):
    return w.incremental_storage_candidate(values, station_number='08AA005', allow_exploratory=True, **kwargs)


def decimal_oracle(elevation):
    with localcontext() as ctx:
        ctx.prec = 40
        x = Decimal(str(elevation)) - Decimal('912')
        coefficients = w.LOWER_COEFFICIENTS if elevation < 915 else w.UPPER_COEFFICIENTS
        # Decimal treats 0**0 as undefined; a polynomial's constant term is v.
        return float(sum(Decimal(str(v)) * (x**power if power else Decimal(1))
                         for v, power in zip(coefficients, range(len(coefficients)-1, -1, -1))))


@pytest.mark.parametrize('elevation', [912., 913., 914.5, 914.9999, 915., 915.6, 916.])
def test_values_match_high_precision_polynomial_with_explicit_datum_and_coordinate(elevation):
    result, proof = convert(levels([elevation - w.GAUGE_ELEVATION_OFFSET_M]))
    assert result.incremental_storage_candidate_m3.iloc[0] == pytest.approx(decimal_oracle(elevation), abs=1e-4, rel=0)
    assert proof['storage_definition'].startswith('incremental_volume_candidate')
    assert not proof['usable_as_certified_model_input'] and not proof['multiplier_3600_applied']


def test_branch_discrepancy_matches_appendix_without_unendorsed_intercept_or_continuity_fix():
    proof = w.branch_diagnostic()
    assert proof['relative_difference_percent'] == pytest.approx(.011417810739, abs=1e-9)
    assert proof['upward_jump_m3'] == pytest.approx(47774.14, abs=1e-4, rel=0)
    assert not proof['branch_jump_corrected']
    assert w.LOWER_COEFFICIENTS[-1] == 0.


def test_valid_domain_is_monotonic_and_missing_or_unsupported_levels_are_not_extrapolated():
    elevation = np.r_[911.9, np.linspace(912., 916., 1001), 916.1, np.nan]
    result, proof = convert(levels(elevation - w.GAUGE_ELEVATION_OFFSET_M))
    valid = result.incremental_storage_candidate_m3.dropna()
    assert len(valid) == 1001 and (np.diff(valid) > 0).all()
    assert result.area_derivative_candidate_m2.dropna().gt(0).all()
    assert result.incremental_storage_candidate_m3.iloc[[0, -2, -1]].isna().all()
    assert proof['outside_domain_days'] == 2 and not proof['extrapolation_applied']


def test_three_day_proxy_is_trailing_and_requires_every_day_without_mutating_input():
    source = levels([1., 2., 3., np.nan, 2., 2., 2.]); before = source.copy(deep=True)
    result, proof = convert(source, smoothing_days=3)
    assert result.trailing_level_proxy_m.notna().tolist() == [False, False, True, False, False, False, True]
    assert result.trailing_level_proxy_m.iloc[2] == 2.
    altered = source.copy(); altered.iloc[4:] = 3.
    future, _ = convert(altered, smoothing_days=3)
    pd.testing.assert_frame_equal(result.iloc[:4], future.iloc[:4])
    pd.testing.assert_series_equal(source, before)
    assert proof['mean_level_storage_proxy'] and not proof['endpoint_storage_verified']


@pytest.mark.parametrize('days', [True, 2, 0, 3.0])
def test_unsupported_smoothing_is_rejected(days):
    with pytest.raises(ValueError): convert(levels([2., 2., 2.]), smoothing_days=days)


def test_other_gauge_cannot_inherit_this_datum_and_exploratory_opt_in_is_required():
    source = levels([2., 2., 2.])
    with pytest.raises(ValueError, match='opt-in'):
        w.incremental_storage_candidate(source, station_number='08AA005')
    with pytest.raises(ValueError, match='only to 08AA005'):
        w.incremental_storage_candidate(source, station_number='08AA012', allow_exploratory=True)


@pytest.mark.parametrize('change', ['infinite', 'gap', 'intraday'])
def test_invalid_levels_or_clock_rejected(change):
    source = levels([1., 2., 3.])
    if change == 'infinite': source.iloc[0] = np.inf
    if change == 'gap': source = source.iloc[[0, 2]]
    if change == 'intraday': source.index += pd.Timedelta(hours=12)
    with pytest.raises(ValueError): convert(source)
