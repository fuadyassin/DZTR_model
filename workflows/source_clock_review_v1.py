"""Daily source-budget diagnostics, never an inflow estimator or clock fitter.

Every clock scenario uses identical supported dates for comparison. Signed
residuals are retained; no scenario is selected automatically. In particular,
differencing daily mean storage does not identify midnight storage changes.
"""
from numbers import Integral, Real

import numpy as np
import pandas as pd

from workflows.boundary_source_v1 import _calendar


def _finite_or_missing(values, label):
    result = values.astype(float)
    if np.isinf(result.to_numpy()).any():
        raise ValueError(label + ' must be finite or missing')
    return result


def capacity_diagnostic(storage, capacity_m3):
    """Report exceedances without declaring a coordinate or changing records."""
    _calendar(storage.index)
    if (isinstance(capacity_m3, bool) or not isinstance(capacity_m3, Real)
            or not np.isfinite(capacity_m3) or capacity_m3 <= 0):
        raise ValueError('Capacity must be finite and positive')
    s = _finite_or_missing(storage, 'Storage').dropna()
    return {'valid_days': len(s), 'capacity_reference_m3': float(capacity_m3),
            'above_capacity_days': int((s > capacity_m3).sum()),
            'minimum_m3': float(s.min()) if len(s) else None,
            'maximum_m3': float(s.max()) if len(s) else None,
            'minimum_excess_m3': float((s - capacity_m3).min()) if len(s) else None,
            'same_storage_coordinate_verified': False,
            'records_rescaled_or_clipped': False}


def budget_clock_sensitivity(storage, components, scenarios):
    """Compare declared date shifts and quadratures on common daily support.

    A positive component shift means Q(t-shift) is assigned to row t. The
    adjacent-day average uses (Q(t-1)+Q(t))/2, an explicit mean-to-mean proxy.
    Neither that proxy nor Q(t) certifies daily endpoint timing. Missing
    components and storage gaps remain missing, including shifted boundaries.
    """
    _calendar(storage.index)
    if not storage.index.equals(components.index) or not components.columns.is_unique or components.empty:
        raise ValueError('Use distinct outlet components on the exact storage calendar')
    s = _finite_or_missing(storage, 'Storage')
    q = _finite_or_missing(components, 'Discharge')
    if (q < 0).any().any():
        raise ValueError('Outlet components must be nonnegative or missing')
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError('Declare clock scenarios')
    intervals = pd.DataFrame({'delta_storage_m3': s.diff()}, index=s.index)
    names = []
    for scenario in scenarios:
        if set(scenario) != {'name', 'component_shifts_days', 'quadrature'}:
            raise ValueError('Declare name, component shifts and quadrature')
        name = scenario['name']
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Scenario names must be nonempty and unique')
        shifts = scenario['component_shifts_days']
        if not isinstance(shifts, dict) or set(shifts) != set(q.columns):
            raise ValueError('Declare every outlet shift explicitly')
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in shifts.values()):
            raise ValueError('Clock shifts must be integer days')
        shifted = pd.concat([q[c].shift(int(shifts[c])) for c in q], axis=1)
        total = shifted.sum(axis=1, min_count=len(q.columns))
        if scenario['quadrature'] == 'adjacent_day_average':
            total = (total.shift(1) + total) / 2
        elif scenario['quadrature'] != 'current_day':
            raise ValueError('Unknown daily quadrature')
        intervals['release_proxy__' + name] = total
        intervals['net_flux__' + name] = s.diff() / 86400. + total
        names.append(name)
    net_columns = ['net_flux__' + name for name in names]
    common = intervals[net_columns].notna().all(axis=1)
    intervals['common_supported_interval'] = common
    rows = []
    for name in names:
        values = intervals.loc[common, 'net_flux__' + name]
        rows.append({'scenario': name, 'common_intervals': int(common.sum()),
                     'negative_intervals': int((values < 0).sum()),
                     'minimum_net_flux_m3s': float(values.min()) if len(values) else None,
                     'mean_net_flux_m3s': float(values.mean()) if len(values) else None,
                     'negative_net_volume_m3': float(-values[values < 0].sum() * 86400),
                     'automatically_selected': False, 'independent_inflow': False,
                     'timing_certified': False})
    return intervals, pd.DataFrame(rows)
