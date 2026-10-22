"""Datum sign, inverse identifiability, leakage and bounded geometry checks."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from workflows.geometry_v1 import GeometryCurve
from workflows.level_reference_v1 import (DepthReference, anchor_reference, convert,
                                          fit_storage_reference, inverse_depth)
from workflows.level_storage_pilot_v1 import metrics


@pytest.fixture
def example():
    index = pd.date_range("2000-01-01", periods=100, name="date")
    curve = GeometryCurve((0., 5., 10.), (0., 25e6, 100e6), None, "test", "0" * 64)
    depth = pd.Series(np.linspace(1, 9, 100), index=index)
    levels = depth + 493.
    storage = pd.Series(np.interp(depth, curve.depth_m, curve.storage_m3), index=index)
    fit = pd.Series(np.arange(100) < 70, index=index)
    return curve, levels, storage, fit


def test_documented_bottom_and_datum_conversion_sign(example):
    curve, levels, storage, _ = example
    reference = DepthReference(492.687, "PFRA", "CGVD2013", "documented_bottom", "survey",
                               -.313, "documented PFRA to CGVD conversion")
    result, receipt = convert(levels, curve, reference, time_basis="point_level")
    np.testing.assert_allclose(result.storage_m3, storage, atol=2e-6)
    np.testing.assert_array_equal(result.reported_level_m, levels)
    assert not receipt["extrapolation_applied"] and not receipt["capacity_rescaled"]
    # A consistent change of datum for BOTH levels and bed leaves depth unchanged.
    same = replace(reference, zero_level_m=493., reference_datum="PFRA",
                   level_to_reference_offset_m=0., datum_conversion_evidence=None)
    other, _ = convert(levels, curve, same, time_basis="point_level")
    np.testing.assert_allclose(other.depth_m, result.depth_m, atol=1e-12)


@pytest.mark.parametrize("changes", [
    {"level_to_reference_offset_m": float("nan")}, {"zero_level_m": float("inf")},
    {"reference_datum": "CGVD2013"}, {"level_to_reference_offset_m": -.313},
    {"basis": "automatically_guessed_bed"}, {"evidence": ""},
    {"basis": "calibration_storage_alignment"}, {"fitting_signature": "not-a-fit"},
    {"uses_storage_observations": True}])
def test_ambiguous_datum_or_reference_rejected(changes):
    reference = DepthReference(493., "PFRA", "PFRA", "documented_bottom", "survey")
    with pytest.raises(ValueError): replace(reference, **changes)


def test_inverse_alignment_exact_and_held_out_independent(example):
    curve, levels, storage, fit = example
    first, receipt = fit_storage_reference(levels, storage, curve, fit,
                                           fitting_signature="fixed-years", level_datum="PFRA")
    altered_h, altered_v = levels.copy(), storage.copy()
    altered_h.loc[~fit] += 800
    altered_v.loc[~fit] *= 100
    second, other = fit_storage_reference(altered_h, altered_v, curve, fit,
                                           fitting_signature="fixed-years", level_datum="PFRA")
    assert first == second and receipt == other
    assert first.zero_level_m == pytest.approx(493.)
    assert receipt["fitting_pairs"] == 70 and not receipt["assessment_used_for_fitting"]
    assert first.uses_storage_observations and receipt["not_a_surveyed_bed_elevation"]


def test_inverse_outside_volume_not_clipped_into_fit(example):
    curve, levels, storage, fit = example
    storage.iloc[:5] = curve.storage_m3[-1] + 1e9
    reference, receipt = fit_storage_reference(levels, storage, curve, fit,
        fitting_signature="fixed", level_datum="PFRA")
    assert reference.zero_level_m == pytest.approx(493.)
    assert receipt["excluded_outside_curve_pairs"] == 5 and receipt["fitting_pairs"] == 65
    with pytest.raises(ValueError, match="Insufficient"):
        fit_storage_reference(levels, storage, curve, fit, fitting_signature="fixed", level_datum="PFRA", min_pairs=66)
    ambiguous = GeometryCurve((0, 5, 10), (0, 0, 100e6), None, "test", "0" * 64)
    storage.loc[fit] = 0.
    with pytest.raises(ValueError, match="Insufficient"):
        fit_storage_reference(levels, storage, ambiguous, fit, fitting_signature="fixed", level_datum="PFRA")


def test_reference_requires_aligned_masks_and_fitting_signature(example):
    curve, levels, storage, fit = example
    with pytest.raises(ValueError, match="fitting signature"):
        fit_storage_reference(levels, storage, curve, fit, fitting_signature="", level_datum="PFRA")
    with pytest.raises(ValueError, match="exact"):
        fit_storage_reference(levels, storage.iloc[1:], curve, fit, fitting_signature="a", level_datum="PFRA")
    with pytest.raises(ValueError, match="boolean"):
        fit_storage_reference(levels, storage, curve, fit.astype(int), fitting_signature="a", level_datum="PFRA")


def test_no_extrapolation_and_no_mean_to_stock_shortcut(example):
    curve, levels, _, _ = example
    reference = anchor_reference(503., 10., level_datum="PFRA", anchor_datum="PFRA",
                                 basis="documented_level_depth", evidence="survey")
    with pytest.raises(ValueError, match="explicit opt-in"):
        convert(levels, curve, reference, time_basis="daily_mean_level")
    levels.iloc[0], levels.iloc[1], levels.iloc[2] = 600., 490., np.nan
    result, receipt = convert(levels, curve, reference, time_basis="daily_mean_level", allow_mean_level_proxy=True)
    assert result.storage_m3.iloc[:3].isna().all()
    assert receipt["out_of_domain_rows"] == 2 and receipt["missing_level_rows"] == 1
    assert receipt["storage_is_mean_level_proxy"] and not receipt["source_certified"]


def test_fixed_support_metrics_expose_missing_estimates():
    result = metrics([1., 2., 3.], [1., np.nan, 3.], [True] * 3, capacity=10.)
    assert result["nse"] is None and result["supported_nse"] == 1.
    assert result["missing_estimates_on_fixed_support"] == 1
    assert result["conversion_coverage"] == pytest.approx(2 / 3)
    assert not result["fixed_support_complete"]


def test_rounded_curve_plateau_is_ambiguous_only_at_its_volume():
    curve = GeometryCurve((0, 1, 2, 3), (0, 0, 20, 60), None, "test", "0" * 64)
    depth, ambiguous = inverse_depth([-1., 0., 10., 20., 40., 60., 61.], curve)
    np.testing.assert_allclose(depth, [np.nan, np.nan, 1.5, 2., 2.5, 3., np.nan], equal_nan=True)
    assert ambiguous.tolist() == [False, True, False, False, False, False, False]
