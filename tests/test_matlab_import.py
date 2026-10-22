"""Verify MATLAB layout, units and portable reference provenance."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dztr.data import read_reservoir_observations
from dztr.integrations import (
    parameters_from_matlab_vector,
    read_matlab_bounds,
    read_matlab_coefficients,
    read_matlab_observations,
)


REFERENCE = Path(__file__).parent / "reference_cases" / "trinity_matlab_reference.npz"


@pytest.fixture
def matlab_reference():
    with np.load(REFERENCE, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def write_coefficients(tmp_path, values):
    path = tmp_path / "coefficients.txt"
    np.savetxt(path, values, fmt="%.17g")
    return path


def test_coefficient_import_maps_constants_and_monthly_si_targets(tmp_path, matlab_reference):
    values = matlab_reference["coefficients"]
    parameters = read_matlab_coefficients(write_coefficients(tmp_path, values))
    assert parameters.capacity_m3 == 2.5335e9
    assert parameters.initial_release_m3s == 10
    assert parameters.initial_storage_m3 == 1.593e9
    assert parameters.channel_capacity_m3s == 300
    assert parameters.inflow_correction == 1  # Not encoded in the coefficient file.
    assert parameters.dead_storage_fraction == 0.1
    assert parameters.flood_mode == "storage"
    np.testing.assert_array_equal(parameters.monthly_targets.to_numpy(), values[4:].reshape(19, 12)[1:7].T)
    assert parameters.monthly_targets.index.tolist() == list(range(1, 13))
    configured = read_matlab_coefficients(
        write_coefficients(tmp_path, values), inflow_correction=0.92,
        dead_storage_fraction=0.05, flood_mode="inflow",
    )
    assert (configured.inflow_correction, configured.dead_storage_fraction, configured.flood_mode) == (0.92, 0.05, "inflow")


@pytest.mark.parametrize("change,match", [
    ("short", "232"), ("extra", "232"), ("nonnumeric", "numeric"),
    ("nonfinite", "finite"), ("months", "month header"),
])
def test_bad_coefficient_layout_is_rejected(tmp_path, matlab_reference, change, match):
    values = matlab_reference["coefficients"].copy()
    if change == "short":
        values = values[:-1]
    elif change == "extra":
        values = np.append(values, 0)
    elif change == "nonfinite":
        values[-1] = np.inf
    elif change == "months":
        values[4] = 2
    path = write_coefficients(tmp_path, values)
    if change == "nonnumeric":
        text = path.read_text()
        path.write_text("nonsense\n" + text.split("\n", 1)[1])
    with pytest.raises(ValueError, match=match):
        read_matlab_coefficients(path)


def test_bound_rows_are_reordered_from_release_first_to_canonical_si(tmp_path, matlab_reference):
    values = matlab_reference["coefficients"]
    table = values[4:].reshape(19, 12)
    bounds = read_matlab_bounds(write_coefficients(tmp_path, values))
    assert bounds.shape == (12, 12)
    np.testing.assert_array_equal(bounds.lower_storage_critical_m3, table[13])
    np.testing.assert_array_equal(bounds.upper_storage_flood_m3, table[18])
    np.testing.assert_array_equal(bounds.lower_release_normal_m3s, table[8])
    np.testing.assert_array_equal(bounds.upper_release_normal_m3s, table[11])
    reversed_bounds = values.copy()
    reversed_bounds[4 + 13 * 12] = table[16, 0] + 1
    with pytest.raises(ValueError, match="lower <= upper"):
        read_matlab_bounds(write_coefficients(tmp_path, reversed_bounds))


def test_amalgam_vectors_preserve_target_blocks_and_physical_configuration(tmp_path, matlab_reference):
    template = read_matlab_coefficients(
        write_coefficients(tmp_path, matlab_reference["coefficients"]),
        inflow_correction=0.92, dead_storage_fraction=0.05, flood_mode="inflow",
    )
    before = template.monthly_targets.copy(deep=True)
    vector = matlab_reference["parameter_vectors"][1]
    converted = parameters_from_matlab_vector(vector, template)
    expected = vector.reshape(6, 12).T.copy()
    expected[:, :3] *= template.capacity_m3
    np.testing.assert_array_equal(converted.monthly_targets.to_numpy(), expected)
    pd.testing.assert_frame_equal(template.monthly_targets, before)
    assert converted.capacity_m3 == template.capacity_m3
    assert converted.initial_storage_m3 == template.initial_storage_m3
    assert converted.initial_release_m3s == template.initial_release_m3s
    assert converted.channel_capacity_m3s == template.channel_capacity_m3s
    assert (converted.inflow_correction, converted.dead_storage_fraction, converted.flood_mode) == (0.92, 0.05, "inflow")
    for invalid in (vector[:-1], vector.reshape(6, 12), np.full(72, np.nan)):
        with pytest.raises(ValueError, match="72"):
            parameters_from_matlab_vector(invalid, template)
    complex_vector = vector.astype(complex)
    complex_vector[0] += 1j
    with pytest.raises(ValueError, match="real"):
        parameters_from_matlab_vector(complex_vector, template)


def test_headerless_workbook_adapter_matches_repo_observations(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    source = repository / "ResDataInflowStorageRelease" / "Trinity_131.xlsx"
    raw = pd.read_excel(source).iloc[:15].copy()
    raw["configuration"] = np.nan
    raw.loc[:3, "configuration"] = [2.5335e9, 10, 1.593e9, 300]
    path = tmp_path / "res_info.xlsx"
    raw.to_excel(path, header=False, index=False)
    imported = read_matlab_observations(path)
    expected = read_reservoir_observations(source).frame.iloc[:15]
    pd.testing.assert_frame_equal(imported.frame, expected)
    assert imported.source == str(path)


def test_mat_adapter_converts_daily_serial_dates_and_preserves_missing_data(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    path = tmp_path / "res_info_out.mat"
    scipy_io.savemat(path, {"res_info_out": {
        "timed": np.array([719529, 719531]),
        "stoobs": np.array([1.9e9, np.nan]),
        "inflow": np.array([30, 0]), "outflow": np.array([10, 11]),
    }})
    observations = read_matlab_observations(path)
    assert observations.frame.index.equals(pd.DatetimeIndex(["1970-01-01", "1970-01-03"], name="date"))
    assert observations.frame.storage_m3.iloc[0] == 1.9e9
    assert np.isnan(observations.frame.storage_m3.iloc[1])
    assert observations.quality.missing_days == 1
    assert observations.quality.missing_counts["storage_m3"] == 1
    assert observations.source == str(path)


@pytest.mark.parametrize("change,match", [
    ("missing_struct", "res_info_out"), ("missing_field", "res_info_out"),
    ("length", "equal lengths"), ("matrix", "vector"), ("fractional_date", "daily serial"),
])
def test_invalid_mat_observation_layout_is_rejected(tmp_path, change, match):
    scipy_io = pytest.importorskip("scipy.io")
    info = {"timed": [719529, 719530], "stoobs": [10, 20], "inflow": [1, 2], "outflow": [3, 4]}
    if change == "missing_field":
        info.pop("inflow")
    elif change == "length":
        info["inflow"] = [1, 2, 3]
    elif change == "matrix":
        info["inflow"] = [[1, 2], [3, 4]]
    elif change == "fractional_date":
        info["timed"] = [719529.5, 719530]
    path = tmp_path / "invalid.mat"
    scipy_io.savemat(path, {"other": info} if change == "missing_struct" else {"res_info_out": info})
    with pytest.raises(ValueError, match=match):
        read_matlab_observations(path)


def test_portable_reference_arrays_match_repo_data_and_have_provenance(matlab_reference):
    reference = matlab_reference
    assert reference["member_indices"].tolist() == [0, 49, 99]
    assert reference["parameter_vectors"].shape == (3, 72)
    assert reference["storage_m3"].shape == reference["release_m3s"].shape == (730, 3)
    assert reference["dates"][[0, -1]].tolist() == ["1970-01-01", "1971-12-31"]
    assert not any(array.dtype.hasobject for array in reference.values())
    source = Path(__file__).resolve().parents[1] / "ResDataInflowStorageRelease" / "Trinity_131.xlsx"
    observations = read_reservoir_observations(source).frame.iloc[:730]
    np.testing.assert_array_equal(observations.inflow_m3s, reference["inflow_m3s"])
    np.testing.assert_array_equal(observations.storage_m3, reference["observed_storage_m3"])
    np.testing.assert_array_equal(observations.release_m3s, reference["observed_release_m3s"])
    metadata = json.loads(REFERENCE.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["simulation_settings"]["fixed_point_iterations"] == 1000
    assert len(metadata["sources"]["flowSIM_eval.mat"]["sha256"]) == 64
    assert any("S2Q1JAN" in note for note in metadata["provenance_notes"])
