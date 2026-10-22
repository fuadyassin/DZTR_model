"""Strict daily source comparisons and explicit flow-boundary expressions.

These helpers prepare source evidence. A numerical match or a fitted level
relationship does not certify a datum, capacity, outlet boundary or forcing.
"""
from pathlib import Path
import json
from numbers import Real

import numpy as np
import pandas as pd


def _calendar(index):
    if not isinstance(index, pd.DatetimeIndex) or index.empty or index.tz is not None:
        raise ValueError('Use a nonempty daily date-label calendar')
    if not index.equals(pd.date_range(index[0], index[-1], name=index.name)):
        raise ValueError('Daily calendars must be unique, ordered and consecutive')


def compose_discharge(components, terms):
    """Add/subtract declared daily-mean components; require every component.

    Negative results remain signed and flagged. They are never clipped, nor
    does this operation declare them suitable as physical total discharge.
    """
    _calendar(components.index)
    if not isinstance(terms, list) or not terms:
        raise ValueError('Declare at least one boundary component')
    stations = []
    columns = []
    for term in terms:
        if set(term) != {'station', 'coefficient', 'evidence'} or not term['evidence']:
            raise ValueError('Every boundary term needs station, coefficient and evidence')
        station, coefficient = term['station'], term['coefficient']
        if not isinstance(station, str) or station not in components or station in stations:
            raise ValueError('Boundary stations must exist and be unique')
        if isinstance(coefficient, bool) or not isinstance(coefficient, Real) or not np.isfinite(coefficient) or coefficient == 0:
            raise ValueError('Boundary coefficients must be finite and nonzero')
        values = components[station].astype(float)
        if np.isinf(values.to_numpy()).any() or (values.dropna() < 0).any():
            raise ValueError('Gauge discharge components require finite nonnegative values or missing')
        stations.append(station)
        columns.append(values * coefficient)
    result = pd.concat(columns, axis=1).sum(axis=1, min_count=len(columns))
    result.name = 'candidate_release_m3s'
    return result, {
        'terms': terms, 'units': 'm3/s', 'time_basis': 'daily_mean_date_label',
        'all_components_required': True, 'missing_filled': False,
        'negative_results_clipped': False, 'negative_days': int((result < 0).sum()),
        'missing_days': int(result.isna().sum()), 'valid_days': int(result.notna().sum()),
        'boundary_certified_by_arithmetic': False,
    }


def read_usgs_daily(path, site, parameter, method_id=None):
    """Read one USGS daily MEAN series, preserving declared missing values.

    Native feet or ft3/s are returned; callers declare their SI conversion.
    Select a method explicitly when the provider returns multiple methods.
    Daily labels are not converted between time zones.
    """
    if parameter not in ('00060', '00065'):
        raise ValueError('Select discharge 00060 or gage height 00065')
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    selected = []
    for series in payload['value']['timeSeries']:
        info, variable = series['sourceInfo'], series['variable']
        ids = [v['value'] for v in info['siteCode'] if v.get('agencyCode') == 'USGS']
        if ids != [site]:
            raise ValueError('USGS response site differs from requested site')
        if parameter not in [v['value'] for v in variable['variableCode']]:
            continue
        options = variable['options']['option']
        if not any(v.get('name') == 'Statistic' and v.get('optionCode') == '00003' for v in options):
            raise ValueError('Require daily mean statistic 00003')
        expected_unit = 'ft' if parameter == '00065' else 'ft3/s'
        if variable['unit']['unitCode'] != expected_unit:
            raise ValueError('Unexpected USGS native units')
        selected.append(series)
    if len(selected) != 1:
        raise ValueError('Require exactly one matching USGS daily mean series')
    series = selected[0]
    missing = float(series['variable']['noDataValue'])
    blocks = series['values']
    if method_id is None and len(blocks) != 1:
        raise ValueError('Multiple USGS value methods require explicit method_id')
    if method_id is not None:
        blocks = [block for block in blocks if any(
            str(method.get('methodID')) == str(method_id) for method in block.get('method', []))]
        if len(blocks) != 1:
            raise ValueError('Require exactly one selected USGS value method')
    rows = {}
    for block in blocks:
        for row in block['value']:
            day = pd.Timestamp(row['dateTime'][:10])
            value = float(row['value'])
            value = np.nan if value == missing else value
            if np.isinf(value):
                raise ValueError('Nonfinite USGS value')
            if day in rows and not (rows[day] == value or np.isnan(rows[day]) and np.isnan(value)):
                raise ValueError('Conflicting USGS daily values')
            rows[day] = value
    if not rows:
        raise ValueError('USGS series contains no daily values')
    result = pd.Series(rows, name=site, dtype=float).sort_index()
    return result.reindex(pd.date_range(result.index[0], result.index[-1], name='date'))


def compare(left, right, tolerance):
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Declare a finite nonnegative absolute tolerance')
    if not left.index.is_unique or not right.index.is_unique:
        raise ValueError('Comparisons require unique dates')
    paired = pd.concat([left, right], axis=1).dropna()
    if np.isinf(paired.to_numpy()).any():
        raise ValueError('Cannot compare infinite values')
    result = {'paired_days': len(paired), 'absolute_tolerance': tolerance,
              'source_identity_certified': False}
    if paired.empty:
        return dict(result, matching_days=0, mean_difference=None,
                    max_absolute_difference=None, rms_difference=None)
    diff = paired.iloc[:, 0] - paired.iloc[:, 1]
    return dict(result, matching_days=int((abs(diff) <= tolerance).sum()),
                mean_difference=float(diff.mean()),
                max_absolute_difference=float(abs(diff).max()),
                rms_difference=float(np.sqrt(np.mean(diff ** 2))))


def affine_source_diagnostic(response, predictors):
    """Forensic relationship only: NEVER a fitted datum or parameter profile."""
    if not response.index.is_unique or not predictors.index.is_unique:
        raise ValueError('Forensic comparisons require unique dates')
    if not predictors.columns.is_unique or predictors.shape[1] == 0:
        raise ValueError('Provide distinct predictor labels')
    frame = pd.concat([response.rename('response'), predictors], axis=1).dropna()
    if len(frame) <= predictors.shape[1] or not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Insufficient finite paired source records')
    x, y = frame.iloc[:, 1:].to_numpy(), frame.iloc[:, 0].to_numpy()
    xmean, ymean = x.mean(axis=0), y.mean()
    coefficients, _, rank, _ = np.linalg.lstsq(x - xmean, y - ymean, rcond=None)
    if rank != predictors.shape[1]:
        raise ValueError('Source predictors are rank deficient')
    intercept = ymean - xmean @ coefficients
    residual = y - (intercept + x @ coefficients)
    return {'paired_days': len(frame), 'intercept': float(intercept),
            'coefficients': dict(zip(predictors.columns, map(float, coefficients))),
            'rms_residual': float(np.sqrt(np.mean(residual ** 2))),
            'max_absolute_residual': float(abs(residual).max()),
            'estimation_scope': 'all_paired_records_for_source_identification_only',
            'used_for_parameterization': False, 'datum_certified': False}
