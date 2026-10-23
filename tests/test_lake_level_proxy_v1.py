"""Reference choices must not invent lake depth or change a signed budget."""
import numpy as np
import pandas as pd
import pytest

from workflows import lake_level_proxy_v1 as w


def gauges():
    return pd.DataFrame({'oshkosh': [2., 2.1, np.nan, 2.2], 'stockbridge': [1.84, 1.94, 2., 2.04]},
                        index=pd.date_range('2000-01-01', periods=4, name='date'))


def convert(frame, **kwargs):
    options = dict(reference_elevation_ft=743.91, surface_area_m2=100.,
                   datum_history_assumption='Explicit test assumption only', allow_exploratory=True)
    options.update(kwargs)
    return w.relative_storage_candidate(frame, {'oshkosh': 743.91, 'stockbridge': 744.07}, **options)


def test_datums_are_harmonized_before_averaging_and_every_gauge_is_required():
    source = gauges(); before = source.copy(deep=True)
    result, proof = convert(source)
    np.testing.assert_allclose(result.relative_mean_level_m, np.array([2., 2.1, np.nan, 2.2]) * .3048, rtol=0, atol=2e-14)
    assert result.relative_storage_candidate_m3.iloc[2] != result.relative_storage_candidate_m3.iloc[2]
    assert proof['all_gauges_required'] and not proof['reference_plane_is_lake_bed']
    assert not proof['total_storage_estimated'] and not proof['physical_capacity_estimated']
    pd.testing.assert_frame_equal(source, before)


def test_changing_reference_translates_anomalies_but_preserves_differences_and_signed_values():
    source = gauges().fillna(2.)
    a, _ = convert(source); b, proof = convert(source, reference_elevation_ft=750.)
    assert b.relative_storage_candidate_m3.lt(0).all()
    np.testing.assert_allclose(a.relative_storage_candidate_m3.diff().dropna(), b.relative_storage_candidate_m3.diff().dropna(), rtol=0, atol=1e-10)
    assert not proof['negative_anomalies_clipped']


@pytest.mark.parametrize('change', ['no_optin', 'no_history', 'negative_area', 'bool_reference', 'infinite_level', 'missing_datum', 'gap', 'intraday'])
def test_ambiguous_or_invalid_reference_inputs_are_rejected(change):
    frame = gauges(); options = dict(reference_elevation_ft=743.91, surface_area_m2=100., datum_history_assumption='test', allow_exploratory=True)
    datums = {'oshkosh': 743.91, 'stockbridge': 744.07}
    if change == 'no_optin': options['allow_exploratory'] = False
    if change == 'no_history': options['datum_history_assumption'] = ''
    if change == 'negative_area': options['surface_area_m2'] = -1.
    if change == 'bool_reference': options['reference_elevation_ft'] = True
    if change == 'infinite_level': frame.iloc[0, 0] = np.inf
    if change == 'missing_datum': del datums['stockbridge']
    if change == 'gap': frame = frame.iloc[[0, 2, 3]]
    if change == 'intraday': frame.index += pd.Timedelta(hours=12)
    with pytest.raises(ValueError): w.relative_storage_candidate(frame, datums, **options)


def test_triangular_filter_is_causal_and_does_not_fill_gaps():
    values = pd.Series(np.arange(15.), index=pd.date_range('2000-01-01', periods=15))
    filtered = w.trailing_triangle(values, days=7)
    assert filtered.iloc[:6].isna().all() and filtered.iloc[6] == 3.
    changed = values.copy(); changed.iloc[10:] = 1000.
    pd.testing.assert_series_equal(filtered.iloc[:10], w.trailing_triangle(changed, days=7).iloc[:10])
    values.iloc[7] = np.nan
    assert w.trailing_triangle(values, days=7).iloc[7:14].isna().all()


@pytest.mark.parametrize('days', [True, 3, 7.0])
def test_undeclared_filter_rejected(days):
    values = pd.Series([1., 2.], index=pd.date_range('2000-01-01', periods=2))
    with pytest.raises(ValueError): w.trailing_triangle(values, days=days)


def test_matched_filter_commutes_with_storage_budget_and_retains_negative_flux():
    index = pd.date_range('2000-01-01', periods=30)
    storage = pd.Series(1e8 - np.arange(30.) ** 2 * 1e6, index=index)
    release = pd.Series(50. + 10 * np.cos(np.arange(30.)), index=index)
    frame, proof = w.matched_budget_candidates(storage, release, days=7)
    assert frame.diagnostic_net_flux_m3s.dropna().lt(0).any()
    assert frame.filter_balance_difference_m3.dropna().abs().max() < 1e-4
    assert proof['center_of_weight_delay_days'] == 3 and not proof['uses_future_values']
    assert not proof['independent_inflow'] and not proof['closure_certifies_source_boundary_or_timing']


def test_unmatched_calendars_and_infinity_rejected_and_identity_filter_matches_raw_budget():
    values = pd.Series(np.arange(10.), index=pd.date_range('2000-01-01', periods=10))
    frame, proof = w.matched_budget_candidates(values, values, days=1)
    np.testing.assert_allclose(frame.diagnostic_net_flux_m3s, values.diff() / 86400 + values, rtol=0, atol=0)
    assert proof['maximum_filter_balance_difference_m3'] == 0.
    with pytest.raises(ValueError): w.matched_budget_candidates(values, values.iloc[1:], days=1)
    values.iloc[0] = np.inf
    with pytest.raises(ValueError): w.trailing_triangle(values, days=7)


def test_fitting_windows_exclude_held_out_year_boundary_inputs_and_previous_storage():
    mask = pd.Series([True]*10 + [False]*3 + [True]*10, index=pd.date_range('2000-01-01', periods=23))
    target = w.eligible_source_windows(mask, filter_days=7)
    budget = w.eligible_source_windows(mask, filter_days=7, include_previous_storage=True)
    assert np.flatnonzero(target).tolist() == [6, 7, 8, 9, 19, 20, 21, 22]
    assert np.flatnonzero(budget).tolist() == [7, 8, 9, 20, 21, 22]
    three_day = w.eligible_source_windows(mask, filter_days=3)
    assert not three_day.iloc[13:15].any() and three_day.iloc[15]


def test_ambiguous_phase_masks_and_source_window_lengths_are_rejected():
    mask = pd.Series([True, False], index=pd.date_range('2000-01-01', periods=2))
    with pytest.raises(ValueError): w.eligible_source_windows(mask.astype(int), filter_days=7)
    with pytest.raises(ValueError): w.eligible_source_windows(mask, filter_days=0)
    with pytest.raises(ValueError): w.eligible_source_windows(mask, filter_days=True)
    with pytest.raises(ValueError): w.eligible_source_windows(mask, filter_days=7, include_previous_storage=1)
