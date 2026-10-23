"""Datum-aligned lake storage anomalies and causal matched budget filters.

These are explicit constant-area sensitivity candidates, not surveyed volumes.
A gauge reference is never treated as the bed, total capacity or dead storage.
"""
from numbers import Real, Integral

import numpy as np
import pandas as pd

from workflows.boundary_source_v1 import _calendar


FT_TO_M = .3048
MI_TO_M = 1609.344
SEVEN_DAY_WEIGHTS = np.array([1., 2., 3., 4., 3., 2., 1.])


def _daily(index):
    _calendar(index)
    if index.hasnans or not index.equals(index.normalize()):
        raise ValueError('Use timezone-naive daily midnight date labels')


def _number(value, label, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
            or positive and value <= 0):
        raise ValueError(label + ' must be finite' + (' and positive' if positive else ''))
    return float(value)


def eligible_source_windows(phase_mask, *, filter_days, include_previous_storage=False):
    """Exclude filtered targets whose source days cross a fitting boundary.

    A target needs its entire causal window in the selected phase. A storage
    difference additionally needs the preceding filtered storage window. The
    generic support length also covers the existing three-day Aishihik proxy.
    Coverage requirements must be checked again after these exclusions.
    """
    _daily(phase_mask.index)
    if phase_mask.dtype != bool or phase_mask.isna().any():
        raise ValueError('Use an explicit boolean phase mask without missing roles')
    if isinstance(filter_days, bool) or not isinstance(filter_days, Integral) or filter_days < 1:
        raise ValueError('Declare a positive integer source-window length')
    if not isinstance(include_previous_storage, bool):
        raise ValueError('Declare whether the budget needs the previous storage window')
    required = int(filter_days) + int(include_previous_storage)
    return phase_mask.rolling(required, min_periods=required).sum().eq(required)


def trailing_triangle(values, *, days):
    """One day or a fixed trailing seven-day triangular kernel, no gap fills.

    The seven-day kernel has center of weight at t-3 days; its output remains
    labelled t, with that delay recorded by the caller. No future value enters.
    """
    _daily(values.index)
    if isinstance(days, bool) or not isinstance(days, Integral) or days not in (1, 7):
        raise ValueError('Use one day or the declared seven-day triangular kernel')
    values = values.astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError('Values must be finite or missing')
    if days == 1:
        return values.copy()
    return values.rolling(7, min_periods=7).apply(lambda x: np.dot(x, SEVEN_DAY_WEIGHTS) / 16., raw=True)


def relative_storage_candidate(gauge_heights_ft, gauge_datums_ft, *, reference_elevation_ft,
                               surface_area_m2, datum_history_assumption, allow_exploratory=False):
    """Combine every declared gauge in a common NAVD88 elevation coordinate.

    Every gauge is required on a given day; no one-gauge substitution occurs.
    Negative anomalies remain signed. The constant-area relation is locally
    linear and does not establish the volume beneath the reference plane.
    The caller supplies an explicit historical extension assumption.
    """
    if allow_exploratory is not True:
        raise ValueError('Constant-area storage requires explicit exploratory opt-in')
    _daily(gauge_heights_ft.index)
    if (gauge_heights_ft.empty or not gauge_heights_ft.columns.is_unique
            or not isinstance(gauge_datums_ft, dict) or set(gauge_datums_ft) != set(gauge_heights_ft.columns)):
        raise ValueError('Declare one NAVD88 datum for every distinct gauge')
    if not isinstance(datum_history_assumption, str) or not datum_history_assumption.strip():
        raise ValueError('Declare the historical datum assumption and its limits')
    reference = _number(reference_elevation_ft, 'Reference elevation')
    area = _number(surface_area_m2, 'Surface area', positive=True)
    datums = {name: _number(value, 'Gauge datum') for name, value in gauge_datums_ft.items()}
    heights = gauge_heights_ft.astype(float)
    if np.isinf(heights.to_numpy()).any():
        raise ValueError('Gauge heights must be finite or missing')
    # Subtract the common datum first to retain precision in small changes.
    aligned_relative_ft = heights + pd.Series({name: value - reference for name, value in datums.items()})
    relative_ft = aligned_relative_ft.sum(axis=1, min_count=len(datums)) / len(datums)
    relative_m = relative_ft * FT_TO_M
    result = pd.DataFrame({'relative_mean_level_m': relative_m,
        'mean_surface_elevation_navd88_m': (relative_ft + reference) * FT_TO_M,
        'relative_storage_candidate_m3': area * relative_m}, index=heights.index)
    return result, {'gauge_datums_navd88_ft': datums, 'reference_elevation_navd88_ft': reference,
        'surface_area_m2': area, 'area_assumption': 'constant_lake_surface_area_local_linear_sensitivity',
        'storage_definition': 'volume_anomaly_relative_to_declared_elevation_plane',
        'reference_plane_is_lake_bed': False, 'total_storage_estimated': False,
        'physical_capacity_estimated': False, 'dead_storage_estimated': False,
        'all_gauges_required': True, 'missing_gauges_filled': False, 'negative_anomalies_clipped': False,
        'datum_history_assumption': datum_history_assumption, 'datum_history_verified_for_full_period': False,
        'usable_as_certified_model_input': False}


def matched_budget_candidates(relative_storage, release_proxy, *, days):
    """Filter storage and release consistently and audit finite differences.

    On identical support, dF(S)/dt+F(Q) equals F(dS/dt+Q) for the fixed linear
    filter. This is an algebraic processing check, not independent water-balance
    validation. Source-derived net fluxes and negative anomalies remain signed.
    """
    if not relative_storage.index.equals(release_proxy.index):
        raise ValueError('Use the exact same daily storage and release calendar')
    storage = trailing_triangle(relative_storage, days=1)
    release = trailing_triangle(release_proxy, days=1)
    filtered_storage = trailing_triangle(storage, days=days)
    filtered_release = trailing_triangle(release, days=days)
    raw_net = storage.diff() / 86400. + release
    matched_net = filtered_storage.diff() / 86400. + filtered_release
    filtered_net = trailing_triangle(raw_net, days=days)
    difference = (matched_net - filtered_net) * 86400.
    frame = pd.DataFrame({'relative_storage_candidate_m3': filtered_storage,
        'release_candidate_m3s': filtered_release, 'diagnostic_net_flux_m3s': matched_net,
        'diagnostic_filtered_raw_net_flux_m3s': filtered_net,
        'filter_balance_difference_m3': difference}, index=storage.index)
    proof = {'filter_days': int(days), 'kernel_oldest_to_newest': [1.] if days == 1 else SEVEN_DAY_WEIGHTS.tolist(),
        'kernel_sum': 1. if days == 1 else 16., 'center_of_weight_delay_days': 0 if days == 1 else 3,
        'uses_future_values': False, 'storage_and_release_use_same_kernel': True,
        'maximum_filter_balance_difference_m3': float(difference.abs().max()) if difference.notna().any() else None,
        'checked_intervals': int(difference.notna().sum()), 'independent_inflow': False,
        'closure_certifies_source_boundary_or_timing': False, 'endpoint_storage_verified': False}
    return frame, proof
