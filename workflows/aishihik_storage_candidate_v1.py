"""Explicitly exploratory interpretation of the published Aishihik curve.

The embedded INRS report defines the 08AA005 elevation tie but leaves the
polynomial's reduced coordinate and a printed factor of 3600 ambiguous.
Using x = elevation - 912 reproduces its appendix's stated 0.011% branch
discrepancy. That numerical evidence is an inference, not source certification.
Never substitute this incremental-volume candidate for total lake storage.
"""
from numbers import Integral

import numpy as np
import pandas as pd

from workflows.boundary_source_v1 import _calendar


# Descending powers; the author's clarification specifies no lower intercept.
LOWER_COEFFICIENTS = (-38627.31, 678170.61, -3270008., 6352008.56,
                      -3111511.38, 134383853.50, 0.)
UPPER_COEFFICIENTS = (-50827494.83, 731085628.63, -3932429415.76,
                      9545583654.41, -8448705648.59)
GAUGE_ELEVATION_OFFSET_M = 911.565
REDUCED_ORIGIN_M = 912.
BRANCH_ELEVATION_M = 915.
SUPPORTED_ELEVATION_M = (912., 916.)


def branch_diagnostic():
    """Report both branches without forcing continuity or adding an intercept."""
    x = BRANCH_ELEVATION_M - REDUCED_ORIGIN_M
    lower = float(np.polyval(LOWER_COEFFICIENTS, x))
    upper = float(np.polyval(UPPER_COEFFICIENTS, x))
    return {'lower_limit_m3': lower, 'upper_value_m3': upper,
            'upward_jump_m3': upper - lower,
            'relative_difference_percent': 100 * (upper - lower) / lower,
            'branch_jump_corrected': False, 'coordinate_inferred': True}


def incremental_storage_candidate(levels, *, station_number, smoothing_days=1,
                                  allow_exploratory=False):
    """Convert only 08AA005 daily mean levels under a declared hypothesis.

    Three-day smoothing is trailing and requires all three observations. Gaps
    and values outside 912-916 m remain missing, without clipping. The level
    datum is applied before reducing elevation by 912 m. No factor of 3600 is
    applied, and no bed elevation, total capacity or true daily endpoint is
    inferred. The caller must explicitly opt into the source interpretation.
    """
    if allow_exploratory is not True:
        raise ValueError('An explicit exploratory source-interpretation opt-in is required')
    if station_number != '08AA005':
        raise ValueError('The published 911.565 m elevation tie applies only to 08AA005')
    if isinstance(smoothing_days, bool) or not isinstance(smoothing_days, Integral) or smoothing_days not in (1, 3):
        raise ValueError('Declare unsmoothed or trailing three-day daily means')
    _calendar(levels.index)
    if levels.index.hasnans or not levels.index.equals(levels.index.normalize()):
        raise ValueError('Use daily midnight date labels')
    values = levels.astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError('Levels must be finite or missing')
    smooth = values.rolling(int(smoothing_days), min_periods=int(smoothing_days)).mean()
    elevation = smooth + GAUGE_ELEVATION_OFFSET_M
    x = elevation - REDUCED_ORIGIN_M
    lower = elevation < BRANCH_ELEVATION_M
    supported = elevation.between(*SUPPORTED_ELEVATION_M)
    volume = np.full(len(x), np.nan); derivative = volume.copy()
    for selected, coefficients in ((supported & lower, LOWER_COEFFICIENTS),
                                    (supported & ~lower, UPPER_COEFFICIENTS)):
        volume[selected] = np.polyval(coefficients, x[selected])
        derivative[selected] = np.polyval(np.polyder(coefficients), x[selected])
    status = np.where(smooth.isna(), 'missing_level_or_smoothing_support',
                      np.where(supported, 'exploratory_conversion', 'outside_documented_fit_domain'))
    frame = pd.DataFrame({'gauge_level_m': values, 'trailing_level_proxy_m': smooth,
        'elevation_candidate_m': elevation,
        'incremental_storage_candidate_m3': np.where(supported, volume, np.nan),
        'area_derivative_candidate_m2': np.where(supported, derivative, np.nan),
        'branch': np.where(supported, np.where(lower, 'lower_sixth_order', 'upper_fourth_order'), 'unsupported'),
        'conversion_status': status}, index=levels.index)
    return frame, {'station_number': station_number, 'gauge_elevation_offset_m': GAUGE_ELEVATION_OFFSET_M,
        'reference_evidence': 'Embedded INRS report, printed page 44: 08AA005 reference level 911.565 m.',
        'inferred_reduced_polynomial_origin_m': REDUCED_ORIGIN_M,
        'storage_definition': 'incremental_volume_candidate_above_912m_not_total_lake_storage',
        'smoothing_days': int(smoothing_days), 'smoothing_is_trailing': True,
        'mean_level_storage_proxy': True, 'endpoint_storage_verified': False,
        'multiplier_3600_applied': False, 'coordinate_interpretation_verified': False,
        'published_unit_ambiguity_resolved': False, 'usable_as_certified_model_input': False,
        'datum_history_verified_for_entire_period': False,
        'extrapolation_applied': False, 'missing_days_filled': False,
        'converted_days': int(supported.sum()), 'outside_domain_days': int((smooth.notna() & ~supported).sum()),
        'branch_diagnostic': branch_diagnostic()}
