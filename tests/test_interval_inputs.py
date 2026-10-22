"""Daily mean volume semantics and conservative source-transition quarantine."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from dztr.data.intervals import DailyIntervalPolicy, IntervalPreparation, prepare_daily_intervals


def observations(count=8):
    index = pd.date_range("2009-04-12", periods=count, name="date")
    return pd.DataFrame({
        "storage_m3": np.arange(count, dtype=float) + 1e7,
        "inflow_m3s": np.arange(count, dtype=float) + 1.,
        "release_m3s": np.full(count, 2.),
    }, index=index)


def test_daily_mean_volume_is_current_reported_mean_without_adjacent_averaging():
    frame = observations()
    prepared = prepare_daily_intervals(frame)
    assert isinstance(prepared, IntervalPreparation)
    assert prepared.policy.flow_alignment == "interval_mean_end"
    assert prepared.policy.nominal_step_seconds == 86400
    np.testing.assert_array_equal(prepared.frame.reported_inflow_volume_m3, frame.inflow_m3s * 86400)
    np.testing.assert_array_equal(prepared.frame.usable_inflow_volume_m3.iloc[1:], frame.inflow_m3s.iloc[1:] * 86400)
    assert pd.isna(prepared.frame.usable_inflow_volume_m3.iloc[0])
    assert pd.isna(prepared.frame.interval_start.iloc[0])
    assert np.isnan(prepared.frame.interval_seconds.iloc[0])
    pd.testing.assert_index_equal(pd.DatetimeIndex(prepared.frame.interval_end, name="date"), frame.index)
    np.testing.assert_array_equal(prepared.frame.interval_start.iloc[1:], frame.index[:-1])
    np.testing.assert_array_equal(prepared.frame.interval_seconds.iloc[1:], 86400)
    np.testing.assert_array_equal(prepared.frame.segment_initialization, [True] + [False] * 7)
    np.testing.assert_array_equal(prepared.frame.storage_score_eligible, [False] + [True] * 7)
    assert prepared.segments.iloc[0].usable_intervals == 7
    assert prepared.segments.iloc[0].initial_storage_m3 == frame.storage_m3.iloc[0]
    assert prepared.segments.iloc[0].initialization_status == "provisional_observed_storage"


def test_negative_inflow_needs_explicit_net_semantics_and_remains_signed():
    frame = observations()
    frame.loc[frame.index[3], "inflow_m3s"] = -1.7
    with pytest.raises(ValueError, match="explicit computed_net_inflow"):
        prepare_daily_intervals(frame)
    prepared = prepare_daily_intervals(frame, policy=DailyIntervalPolicy(flow_semantics="computed_net_inflow"))
    assert prepared.frame.inflow_m3s.iloc[3] == -1.7
    assert prepared.frame.reported_inflow_volume_m3.iloc[3] == -1.7 * 86400
    assert prepared.frame.usable_inflow_volume_m3.iloc[3] == -1.7 * 86400
    assert prepared.frame.storage_score_eligible.iloc[3]


def test_flagged_transition_quarantines_both_states_and_all_touching_intervals():
    frame = observations()
    prepared = prepare_daily_intervals(frame, policy=DailyIntervalPolicy(
        quarantined_interval_end_dates=(frame.index[4],),
    ))
    out = prepared.frame
    np.testing.assert_array_equal(out.source_transition_flagged, [False] * 4 + [True] + [False] * 3)
    np.testing.assert_array_equal(out.storage_endpoint_quarantined, [False] * 3 + [True, True] + [False] * 3)
    np.testing.assert_array_equal(out.simulation_interval_eligible, [False, True, True, False, False, False, True, True])
    np.testing.assert_array_equal(out.segment_initialization, [True, False, False, False, False, True, False, False])
    pd.testing.assert_series_equal(out.segment_id, pd.Series(pd.array([1, 1, 1, None, None, 2, 2, 2], dtype="Int64"), index=frame.index, name="segment_id"))
    assert prepared.segments.row_count.tolist() == [3, 3]
    assert prepared.segments.usable_intervals.tolist() == [2, 2]
    assert out.storage_target_eligible.sum() == 6
    pd.testing.assert_frame_equal(out[list(frame.columns)], frame, check_exact=True)
    assert out.loc[out.storage_endpoint_quarantined, "storage_m3"].notna().all()
    assert out.usable_inflow_volume_m3.notna().equals(out.simulation_interval_eligible)


def test_overlapping_quarantines_merge_into_one_block_and_retain_singleton():
    frame = observations(9)
    prepared = prepare_daily_intervals(frame, policy=DailyIntervalPolicy(
        quarantined_interval_end_dates=(frame.index[2], frame.index[3], frame.index[6]),
    ))
    out = prepared.frame
    np.testing.assert_array_equal(out.storage_endpoint_quarantined, [False, True, True, True, False, True, True, False, False])
    assert prepared.segments.row_count.tolist() == [1, 1, 2]
    assert prepared.segments.usable_intervals.tolist() == [0, 0, 1]
    assert out.segment_initialization.sum() == 3
    assert out.storage_score_eligible.sum() == out.release_score_eligible.sum() == 1
    assert len(out) == len(frame)


def test_period_boundary_preserves_simulation_continuity_and_excludes_audit_transition():
    frame = observations()
    labels = pd.Series(["warmup"] * 2 + ["calibration"] * 3 + ["validation"] * 3, index=frame.index)
    prepared = prepare_daily_intervals(frame, period_labels=labels)
    assert prepared.frame.simulation_interval_eligible.iloc[2]
    assert prepared.frame.simulation_interval_eligible.iloc[5]
    assert not prepared.frame.source_audit_interval_eligible.iloc[2]
    assert not prepared.frame.source_audit_interval_eligible.iloc[5]
    assert prepared.frame.source_audit_interval_eligible.sum() == 5
    assert len(prepared.segments) == 1
    assert prepared.segments.iloc[0].period_start == "warmup"
    assert prepared.segments.iloc[0].period_end == "validation"
    pd.testing.assert_series_equal(prepared.frame.period, labels.rename("period"))
    labels.iloc[0] = "changed"
    assert prepared.frame.period.iloc[0] == "warmup"


def test_inputs_extra_objects_and_attributes_are_independent_of_prepared_output():
    frame = observations()
    frame["source_note"] = [{"flags": []} for _ in range(len(frame))]
    frame.attrs["source"] = {"versions": ["original"]}
    original = frame.copy(deep=True)
    prepared = prepare_daily_intervals(frame)
    pd.testing.assert_frame_equal(frame, original, check_exact=True)
    pd.testing.assert_frame_equal(prepared.frame[list(frame.columns)], frame, check_exact=True)
    prepared.frame.iloc[1, prepared.frame.columns.get_loc("storage_m3")] = 0.
    prepared.frame.source_note.iloc[0]["flags"].append("changed")
    prepared.frame.attrs["source"]["versions"].append("changed")
    assert frame.storage_m3.iloc[1] == 10000001.
    assert frame.source_note.iloc[0]["flags"] == []
    assert frame.attrs["source"]["versions"] == ["original"]
    with pytest.raises(FrozenInstanceError):
        prepared.policy = DailyIntervalPolicy()


def test_nullable_string_period_labels_keep_first_audit_transition_ineligible():
    frame = observations()
    labels = pd.Series(["calibration"] * 3 + ["validation"] * 5, index=frame.index, dtype="string")
    prepared = prepare_daily_intervals(frame, period_labels=labels)
    np.testing.assert_array_equal(prepared.frame.source_audit_interval_eligible, [False, True, True, False, True, True, True, True])
    assert prepared.frame.period.dtype == labels.dtype


def test_daily_volume_is_float_and_does_not_overflow_integer_storage():
    frame = observations()
    frame["inflow_m3s"] = np.full(len(frame), 10**15, dtype=np.int64)
    prepared = prepare_daily_intervals(frame)
    np.testing.assert_array_equal(prepared.frame.reported_inflow_volume_m3, 1e15 * 86400)
    assert prepared.frame.reported_inflow_volume_m3.dtype == np.dtype("float64")
    assert prepared.frame.inflow_m3s.dtype == np.dtype("int64")
    frame["inflow_m3s"] = 1e308
    with pytest.raises(ValueError, match="exceed finite SI"):
        prepare_daily_intervals(frame)


def test_policy_dates_are_copied_frozen_and_validated():
    dates = ["2009-04-16", "2009-04-14"]
    policy = DailyIntervalPolicy(quarantined_interval_end_dates=dates)
    dates.append("2009-04-17")
    assert policy.quarantined_interval_end_dates == (pd.Timestamp("2009-04-14"), pd.Timestamp("2009-04-16"))
    with pytest.raises(FrozenInstanceError):
        policy.flow_semantics = "computed_net_inflow"


@pytest.mark.parametrize("dates", [
    ["2009-04-16", "2009-04-16"], ["04/16/2009"], [20090416],
    [pd.NaT], [pd.Timestamp("2009-04-16", tz="UTC")],
    [pd.Timestamp("2009-04-16 01:00")], None, "2009-04-16", ["2009-02-30"],
])
def test_invalid_or_ambiguous_policy_dates_are_rejected(dates):
    with pytest.raises(ValueError):
        DailyIntervalPolicy(quarantined_interval_end_dates=dates)


@pytest.mark.parametrize("flag", ["2009-04-12", "2009-04-11", "2009-04-20"])
def test_first_row_or_outside_quarantine_date_is_rejected(flag):
    with pytest.raises(ValueError, match="preceding storage|input calendar"):
        prepare_daily_intervals(observations(), policy=DailyIntervalPolicy(quarantined_interval_end_dates=(flag,)))


def test_all_rows_can_be_quarantined_without_inventing_a_segment():
    frame = observations(2)
    prepared = prepare_daily_intervals(frame, policy=DailyIntervalPolicy(quarantined_interval_end_dates=(frame.index[1],)))
    assert len(prepared.frame) == 2
    assert prepared.segments.empty
    assert prepared.frame.segment_id.isna().all()
    assert not prepared.frame.segment_initialization.any()
    assert not prepared.frame.storage_target_eligible.any()
    assert not prepared.frame.simulation_interval_eligible.any()


@pytest.mark.parametrize("problem", ["missing_day", "duplicate", "unsorted", "timezone", "hour", "nat", "not_datetime", "singleton"])
def test_invalid_daily_calendar_is_rejected(problem):
    frame = observations()
    if problem == "missing_day":
        frame = frame.drop(frame.index[3])
    elif problem == "duplicate":
        frame.index = pd.DatetimeIndex([frame.index[0], frame.index[0], *frame.index[2:]])
    elif problem == "unsorted":
        frame = frame.iloc[::-1]
    elif problem == "timezone":
        frame.index = frame.index.tz_localize("UTC")
    elif problem == "hour":
        frame.index = frame.index + pd.Timedelta(hours=1)
    elif problem == "nat":
        frame.index = pd.DatetimeIndex([pd.NaT, *frame.index[1:]])
    elif problem == "not_datetime":
        frame.index = pd.RangeIndex(len(frame))
    else:
        frame = frame.iloc[:1]
    with pytest.raises(ValueError):
        prepare_daily_intervals(frame)


@pytest.mark.parametrize("column,value", [
    ("storage_m3", -1.), ("release_m3s", -1.), ("storage_m3", np.nan),
    ("inflow_m3s", np.inf), ("release_m3s", -np.inf),
])
def test_invalid_observations_fail_under_both_flow_semantics(column, value):
    frame = observations()
    frame.loc[frame.index[2], column] = value
    for semantics in ("physical_inflow", "computed_net_inflow"):
        with pytest.raises(ValueError):
            prepare_daily_intervals(frame, policy=DailyIntervalPolicy(flow_semantics=semantics))


@pytest.mark.parametrize("dtype", [str, bool, complex])
def test_non_normalized_numeric_columns_are_rejected(dtype):
    frame = observations()
    frame["inflow_m3s"] = frame.inflow_m3s.astype(dtype)
    with pytest.raises(ValueError, match="normalized real numeric"):
        prepare_daily_intervals(frame)


def test_missing_or_duplicate_columns_and_reserved_masks_are_rejected():
    frame = observations()
    with pytest.raises(ValueError, match="Missing SI"):
        prepare_daily_intervals(frame.drop(columns="release_m3s"))
    with pytest.raises(ValueError, match="Duplicate observation"):
        prepare_daily_intervals(pd.concat([frame, frame[["inflow_m3s"]]], axis=1))
    frame["simulation_interval_eligible"] = True
    with pytest.raises(ValueError, match="reserved preparation"):
        prepare_daily_intervals(frame)


@pytest.mark.parametrize("problem", ["index", "missing", "empty", "nonstrings"])
def test_ambiguous_period_labels_are_rejected(problem):
    frame = observations()
    labels = pd.Series(["calibration"] * len(frame), index=frame.index)
    if problem == "index":
        labels = labels.iloc[::-1]
    elif problem == "missing":
        labels.iloc[2] = None
    elif problem == "empty":
        labels.iloc[2] = "  "
    else:
        labels = pd.Series(np.arange(len(frame)), index=frame.index)
    with pytest.raises(ValueError):
        prepare_daily_intervals(frame, period_labels=labels)


def test_invalid_argument_types_and_unknown_semantics_are_rejected():
    for semantics in ("guess", [], None):
        with pytest.raises(ValueError, match="flow_semantics"):
            DailyIntervalPolicy(flow_semantics=semantics)
    with pytest.raises(TypeError, match="DataFrame"):
        prepare_daily_intervals([1, 2])
    with pytest.raises(TypeError, match="DailyIntervalPolicy"):
        prepare_daily_intervals(observations(), policy={})
    with pytest.raises(TypeError, match="pandas Series"):
        prepare_daily_intervals(observations(), period_labels=["calibration"] * 8)
