"""Source timing comparisons must not manufacture usable forcing or support."""
import numpy as np
import pandas as pd
import pytest

from workflows import source_clock_review_v1 as w


def case():
    index = pd.date_range('2000-01-01', periods=8, name='date')
    s = pd.Series([100., 100., 100., 99., 99., np.nan, 99., 99.], index=index)
    q = pd.DataFrame({'spill': [0., 0., 0., 0., np.nan, 0., 0., 0.],
                      'power': [0., 0., 0., 0., 0., 0., 0., 0.]}, index=index)
    scenarios = [dict(name='same_day', component_shifts_days={'spill': 0, 'power': 0}, quadrature='current_day'),
                 dict(name='adjacent', component_shifts_days={'spill': 0, 'power': 0}, quadrature='adjacent_day_average')]
    return s, q, scenarios


def test_common_support_preserves_storage_gaps_missing_outlets_and_signed_residuals():
    s, q, scenarios = case(); before = q.copy(deep=True)
    intervals, summary = w.budget_clock_sensitivity(s, q, scenarios)
    assert intervals.index[intervals.common_supported_interval].tolist() == [s.index[i] for i in (1, 2, 3, 7)]
    assert summary.common_intervals.tolist() == [4, 4]
    assert summary.negative_intervals.tolist() == [1, 1]
    np.testing.assert_allclose(summary.negative_net_volume_m3, [1., 1.])
    assert intervals.loc[s.index[3], 'net_flux__same_day'] == -1 / 86400
    assert not summary.automatically_selected.any() and not summary.independent_inflow.any()
    assert not summary.timing_certified.any()
    pd.testing.assert_frame_equal(q, before)


def test_shift_sign_and_adjacent_quadrature_are_explicit():
    s, q, scenarios = case(); s[:] = 1.; q['spill'] = np.arange(8); q['power'] = 10.
    scenarios += [dict(name='lagged_power', component_shifts_days={'spill': 0, 'power': 1}, quadrature='current_day')]
    q['power'] += np.arange(8)
    intervals, _ = w.budget_clock_sensitivity(s, q, scenarios)
    assert intervals['release_proxy__same_day'].iloc[3] == 16.
    assert intervals['release_proxy__adjacent'].iloc[3] == 15.
    assert intervals['release_proxy__lagged_power'].iloc[3] == 15.


def test_constant_storage_offset_does_not_repair_net_flux():
    s, q, scenarios = case()
    a, _ = w.budget_clock_sensitivity(s, q, scenarios)
    b, _ = w.budget_clock_sensitivity(s + 1e9, q, scenarios)
    pd.testing.assert_frame_equal(a, b)
    audit = w.capacity_diagnostic(s, 98.)
    assert audit['above_capacity_days'] == 7
    assert not audit['same_storage_coordinate_verified'] and not audit['records_rescaled_or_clipped']


@pytest.mark.parametrize('change', ['gap', 'mismatch', 'infinity', 'negative_q', 'duplicate', 'missing_outlet', 'bool_shift', 'float_shift', 'quadrature'])
def test_ambiguous_or_invalid_inputs_rejected(change):
    s, q, scenarios = case()
    if change == 'gap': s = s.iloc[[0, 2, 3]]; q = q.reindex(s.index)
    if change == 'mismatch': q = q.iloc[1:]
    if change == 'infinity': s.iloc[0] = np.inf
    if change == 'negative_q': q.iloc[0, 0] = -1.
    if change == 'duplicate': scenarios[1]['name'] = scenarios[0]['name']
    if change == 'missing_outlet': del scenarios[0]['component_shifts_days']['power']
    if change == 'bool_shift': scenarios[0]['component_shifts_days']['power'] = True
    if change == 'float_shift': scenarios[0]['component_shifts_days']['power'] = .5
    if change == 'quadrature': scenarios[0]['quadrature'] = 'choose_best'
    with pytest.raises(ValueError): w.budget_clock_sensitivity(s, q, scenarios)


@pytest.mark.parametrize('capacity', [0., -1., np.inf, np.nan, True])
def test_invalid_capacity_rejected(capacity):
    s, _, _ = case()
    with pytest.raises(ValueError): w.capacity_diagnostic(s, capacity)


def test_no_common_support_is_explicit_and_zero_discharge_is_valid():
    s, q, scenarios = case(); s[:] = np.nan
    _, summary = w.budget_clock_sensitivity(s, q, scenarios)
    assert summary.common_intervals.tolist() == [0, 0]
    assert summary.minimum_net_flux_m3s.isna().all()
