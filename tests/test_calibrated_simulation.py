"""Export identity, exact round-trip and batch selection safeguards."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from simulate_calibrated_reservoirs import (ROOT, ARCHIVE, read, resolve, select_ids,
                                          table_targets, compare_frames)


def test_aliases_deduplicate_and_fort_uses_internal_id():
    registry = read(ROOT / "configs/calibrated_twenty.json")
    assert select_ids(registry, [253], ["ex14_Gardiner", "ex4_Fort_randall"]) == ["ex14_Gardiner", "ex4_Fort_randall"]
    assert len(select_ids(registry, all_reservoirs=True)) == 20
    with pytest.raises(ValueError, match="Unknown calibrated GRAND_ID"):
        select_ids(registry, [2356])
    with pytest.raises(ValueError, match="Use --all"):
        select_ids(registry, [253], all_reservoirs=True)
    with pytest.raises(ValueError, match="Select"):
        select_ids(registry)


def test_all_twenty_exact_export_roundtrips():
    registry = read(ROOT / "configs/calibrated_twenty.json")
    wide = pd.read_csv(resolve(registry["table"]), float_precision="round_trip").set_index("reservoir_id", drop=False)
    for identifier, relative in registry["profiles_by_reservoir_id"].items():
        path = resolve(relative)
        profile = read(path)
        targets = pd.read_csv(path.parent / profile["monthly_targets_file"], index_col="month", float_precision="round_trip")
        loaded = table_targets(wide.loc[identifier], profile, targets)
        np.testing.assert_array_equal(loaded, targets)
    row = wide.loc["ex14_Gardiner"].copy()
    profile = read(resolve(registry["profiles_by_reservoir_id"]["ex14_Gardiner"]))
    targets = pd.read_csv(ARCHIVE / "selected/ex14_Gardiner/monthly_targets_si.csv", index_col="month", float_precision="round_trip")
    row["storage_normal_m3_month01"] += 1
    with pytest.raises(ValueError, match="monthly SI targets differ"):
        table_targets(row, profile, targets)
    row = wide.loc["ex14_Gardiner"].copy()
    row["GRAND_ID"] = 2356
    with pytest.raises(ValueError, match="GRAND_ID differs"):
        table_targets(row, profile, targets)


def test_comparison_rejects_small_numeric_change_and_date_shift():
    frame = pd.DataFrame({"storage_m3": [1., 2.], "release_m3s": [3., 4.],
                          "balance_residual_m3": [0., 0.]}, index=pd.date_range("2000-01-01", periods=2))
    assert compare_frames(frame, frame.copy())["storage_m3"] == 0
    altered = frame.copy()
    altered.iloc[1, 0] += 1e-12
    with pytest.raises(AssertionError):
        compare_frames(altered, frame)
    altered = frame.shift(1, freq="D")
    with pytest.raises(AssertionError):
        compare_frames(altered, frame)
