"""Boundary corrections must not hide missing stations or invent references."""
import json

import numpy as np
import pandas as pd
import pytest

from workflows import boundary_source_v1 as w


def terms():
    return [{'station': 'downstream', 'coefficient': 1., 'evidence': 'downstream gauge'},
            {'station': 'tributary', 'coefficient': -1., 'evidence': 'subtract intervening creek'}]


def components():
    return pd.DataFrame({'downstream': [10., 1., 0., 2.], 'tributary': [3., 2., 0., np.nan]},
                        index=pd.date_range('2000-01-01', periods=4))


def test_boundary_subtraction_conserves_components_and_retains_signs_gaps_and_zeros():
    frame = components(); original = frame.copy(deep=True)
    result, proof = w.compose_discharge(frame, terms())
    np.testing.assert_array_equal(result.to_numpy(), [7., -1., 0., np.nan])
    pd.testing.assert_frame_equal(frame, original)
    assert proof['negative_days'] == 1 and proof['missing_days'] == 1
    assert proof['all_components_required'] and not proof['boundary_certified_by_arithmetic']
    assert not proof['negative_results_clipped'] and not proof['missing_filled']


@pytest.mark.parametrize('change', ['duplicate', 'infinity', 'zero', 'no_evidence', 'unknown', 'bool'])
def test_invalid_boundary_expression_is_rejected(change):
    spec = terms()
    if change == 'duplicate': spec[1]['station'] = spec[0]['station']
    if change == 'infinity': spec[1]['coefficient'] = np.inf
    if change == 'zero': spec[1]['coefficient'] = 0
    if change == 'bool': spec[1]['coefficient'] = True
    if change == 'no_evidence': spec[1]['evidence'] = ''
    if change == 'unknown': spec[1]['station'] = 'missing'
    with pytest.raises(ValueError): w.compose_discharge(components(), spec)


def test_calendar_and_physical_component_validation():
    frame = components()
    with pytest.raises(ValueError, match='consecutive'): w.compose_discharge(frame.iloc[[0, 2, 3]], terms())
    frame.iloc[0, 0] = -1.
    with pytest.raises(ValueError, match='nonnegative'): w.compose_discharge(frame, terms())


def usgs(tmp_path):
    data = {'value': {'timeSeries': [{'sourceInfo': {'siteCode': [{'value': '01', 'agencyCode': 'USGS'}]},
        'variable': {'variableCode': [{'value': '00065'}], 'options': {'option': [{'name': 'Statistic', 'optionCode': '00003'}]},
                     'unit': {'unitCode': 'ft'}, 'noDataValue': -999.},
        'values': [{'value': [{'dateTime': '2000-01-01T00:00:00.000', 'value': '-1.'},
                             {'dateTime': '2000-01-03T00:00:00.000', 'value': '-999.'},
                             {'dateTime': '2000-01-04T00:00:00.000', 'value': '0.'}]}]}]}}
    path = tmp_path / 'source.json'; path.write_text(json.dumps(data))
    return path, data


def test_usgs_daily_labels_declared_missing_and_valid_negative_heights(tmp_path):
    path, _ = usgs(tmp_path)
    frame = w.read_usgs_daily(path, '01', '00065')
    np.testing.assert_array_equal(frame.to_numpy(), [-1., np.nan, np.nan, 0.])
    assert frame.index.equals(pd.date_range('2000-01-01', periods=4, name='date'))


@pytest.mark.parametrize('change', ['wrong_site', 'wrong_units', 'wrong_stat', 'duplicate_conflict'])
def test_usgs_metadata_and_conflicts_are_checked(tmp_path, change):
    path, data = usgs(tmp_path); series = data['value']['timeSeries'][0]
    if change == 'wrong_site': series['sourceInfo']['siteCode'][0]['value'] = '02'
    if change == 'wrong_units': series['variable']['unit']['unitCode'] = 'm'
    if change == 'wrong_stat': series['variable']['options']['option'][0]['optionCode'] = '00001'
    if change == 'duplicate_conflict': series['values'][0]['value'].append({'dateTime': '2000-01-01', 'value': '2.'})
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): w.read_usgs_daily(path, '01', '00065')


def test_source_comparison_aligns_dates_without_counting_missing_as_zero():
    a = pd.Series([0., 4., np.nan], index=pd.date_range('2000-01-01', periods=3))
    b = pd.Series([4., 0., 2.], index=pd.date_range('2000-01-02', periods=3))
    proof = w.compare(a, b, .001)
    assert proof['paired_days'] == proof['matching_days'] == 1
    assert not proof['source_identity_certified']
    assert w.compare(a.iloc[:0], b, 0)['rms_difference'] is None


def test_usgs_multiple_location_methods_require_explicit_selection(tmp_path):
    path, data = usgs(tmp_path); series = data['value']['timeSeries'][0]
    series['values'][0]['method'] = [{'methodID': 100, 'methodDescription': 'Primary Stream Flow'}]
    series['values'].append({'method': [{'methodID': 200, 'methodDescription': 'UpStream Location'}],
                             'value': [{'dateTime': '2000-01-01', 'value': '100.'}]})
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='explicit method_id'): w.read_usgs_daily(path, '01', '00065')
    result = w.read_usgs_daily(path, '01', '00065', method_id=100)
    np.testing.assert_array_equal(result.to_numpy(), [-1., np.nan, np.nan, 0.])
    with pytest.raises(ValueError, match='selected USGS'): w.read_usgs_daily(path, '01', '00065', method_id=300)


def test_forensic_gauge_relationship_cannot_certify_or_export_a_datum():
    index = pd.date_range('2000-01-01', periods=30)
    x = pd.DataFrame({'a': np.sin(np.arange(30)), 'b': np.cos(np.arange(30))}, index=index)
    y = 3.7112 + .1524 * x.a + .1524 * x.b
    result = w.affine_source_diagnostic(y, x)
    assert result['intercept'] == pytest.approx(3.7112)
    assert result['coefficients'] == pytest.approx({'a': .1524, 'b': .1524})
    assert result['max_absolute_residual'] < 1e-12
    assert not result['used_for_parameterization'] and not result['datum_certified']
    with pytest.raises(ValueError, match='rank deficient'):
        w.affine_source_diagnostic(y, pd.concat([x.a, x.a.rename('c')], axis=1))
