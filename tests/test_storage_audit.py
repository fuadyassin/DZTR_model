"""Independent volume arithmetic and declared observation-time hypotheses."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from dztr.evaluation.storage_audit import StorageAuditResult, audit_storage


CONVENTIONS = ("endpoint_trapezoid", "interval_mean_end", "interval_mean_start")


def observations(dates=None, inflow=3., release=1., storage=None):
    dates = pd.date_range("2020-02-27", periods=7, name="date") if dates is None else dates
    count = len(dates)
    inflow = np.broadcast_to(np.asarray(inflow, dtype=float), (count,)).copy()
    release = np.broadcast_to(np.asarray(release, dtype=float), (count,)).copy()
    if storage is None:
        elapsed = np.diff(dates.asi8) / 1e9
        storage = np.r_[1e7, 1e7 + np.cumsum(elapsed * (inflow[1:] - release[1:]))]
    return pd.DataFrame({"storage_m3": storage, "inflow_m3s": inflow,
                         "release_m3s": release}, index=dates)


def hypothesis(result, convention="endpoint_trapezoid", forcing="raw", clock="calendar"):
    name = f"{forcing}__{clock}__{convention}"
    frame = result.intervals.xs(name, level="hypothesis")
    record = next(value for value in result.summary["hypotheses"] if value["hypothesis"] == name)
    return frame, record


@pytest.mark.parametrize("frequency,dates", [
    ("daily", pd.date_range("2019-12-29", periods=8, name="date")),
    ("daily", pd.date_range("2020-02-27", periods=8, name="date")),
    ("monthly", pd.date_range("2020-01-01", periods=7, freq="MS", name="date")),
])
def test_constant_flux_closes_independent_calendar_volumes(frequency, dates):
    frame = observations(dates)
    original = frame.copy(deep=True)
    result = audit_storage(frame, frequency=frequency, capacity_m3=1e8, dead_storage_m3=1e6)
    assert isinstance(result, StorageAuditResult)
    assert result.summary["rows"] == len(frame)
    assert result.summary["intervals"] == len(frame) - 1
    assert set(result.intervals.index.names) == {"hypothesis", "date"}
    durations = np.diff(dates.asi8) / 1e9
    for convention in CONVENTIONS:
        intervals, record = hypothesis(result, convention)
        pd.testing.assert_index_equal(intervals.index, dates[1:], check_names=True)
        np.testing.assert_array_equal(intervals.step_seconds, durations)
        np.testing.assert_array_equal(intervals.storage_change_m3, durations * 2)
        np.testing.assert_array_equal(intervals.inflow_volume_m3, durations * 3)
        np.testing.assert_array_equal(intervals.release_volume_m3, durations)
        assert intervals.valid.all()
        assert not intervals.negative_flow.any()
        np.testing.assert_array_equal(intervals.residual_m3, 0.)
        np.testing.assert_array_equal(intervals.cumulative_residual_m3, 0.)
        assert record["statistics"]["complete_closure"]
        assert record["statistics"]["valid_intervals"] == len(frame) - 1
        assert record["statistics"]["rmse_residual_m3"] == 0
    pd.testing.assert_frame_equal(frame, original, check_exact=True)
    with pytest.raises(FrozenInstanceError):
        result.summary = {}


def test_affine_endpoint_flux_uses_trapezoids_not_invented_interval_means():
    dates = pd.date_range("2020-01-30", periods=7, name="date")
    inflow = np.arange(1., 8.)
    release = np.full(7, 2.)
    increments = 86400 * ((inflow[1:] + inflow[:-1]) / 2 - 2)
    frame = observations(dates, inflow, release, np.r_[1e7, 1e7 + np.cumsum(increments)])
    result = audit_storage(frame)
    endpoint, _ = hypothesis(result)
    ending, _ = hypothesis(result, "interval_mean_end")
    starting, _ = hypothesis(result, "interval_mean_start")
    np.testing.assert_array_equal(endpoint.residual_m3, 0.)
    np.testing.assert_array_equal(ending.residual_m3, -43200.)
    np.testing.assert_array_equal(starting.residual_m3, 43200.)


@pytest.mark.parametrize("mean_label,expected", [
    ("interval_mean_end", {"interval_mean_end": 0., "interval_mean_start": 86400., "endpoint_trapezoid": 43200.}),
    ("interval_mean_start", {"interval_mean_end": -86400., "interval_mean_start": 0., "endpoint_trapezoid": -43200.}),
])
def test_interval_mean_label_is_a_declared_alignment(mean_label, expected):
    inflow = np.arange(1., 8.)
    net = inflow[1:] - 2 if mean_label == "interval_mean_end" else inflow[:-1] - 2
    frame = observations(inflow=inflow, release=2., storage=np.r_[1e7, 1e7 + 86400 * np.cumsum(net)])
    result = audit_storage(frame)
    for convention, residual in expected.items():
        intervals, record = hypothesis(result, convention)
        np.testing.assert_array_equal(intervals.residual_m3, residual)
        np.testing.assert_array_equal(intervals.residual_rate_m3s, residual / 86400)
        assert record["statistics"]["mean_residual_m3"] == residual
        assert record["statistics"]["sum_residual_m3"] == residual * (len(frame) - 1)


def test_monthly_fixed_clock_is_additional_and_does_not_replace_calendar():
    dates = pd.date_range("2020-01-01", periods=5, freq="MS", name="date")
    result = audit_storage(observations(dates), frequency="monthly", fixed_step_days=30.5)
    assert len(result.summary["hypotheses"]) == 6
    elapsed = np.array([31., 29., 31., 30.]) * 86400
    for convention in CONVENTIONS:
        calendar, _ = hypothesis(result, convention)
        fixed, record = hypothesis(result, convention, clock="source_fixed")
        np.testing.assert_array_equal(calendar.step_seconds, elapsed)
        np.testing.assert_array_equal(calendar.residual_m3, 0.)
        np.testing.assert_array_equal(fixed.step_seconds, 30.5 * 86400)
        np.testing.assert_array_equal(fixed.residual_m3, 2 * (elapsed - 30.5 * 86400))
        assert record["clock"] == "source_fixed"


def test_identity_fixed_clock_and_forcing_do_not_duplicate_hypotheses():
    result = audit_storage(observations(), fixed_step_days=1., inflow_factor=1.,
                           inflow_offset_m3s=0., balance_inflow_correction=1.)
    assert len(result.summary["hypotheses"]) == 3
    assert {value["forcing"] for value in result.summary["hypotheses"]} == {"raw"}
    assert {value["clock"] for value in result.summary["hypotheses"]} == {"calendar"}


def test_configured_forcing_is_explicit_composition_and_raw_data_is_preserved():
    frame = observations(inflow=np.arange(1., 8.), release=2.)
    original = frame.copy(deep=True)
    result = audit_storage(frame, inflow_factor=2., inflow_offset_m3s=1., balance_inflow_correction=.5)
    assert len(result.summary["hypotheses"]) == 6
    raw, _ = hypothesis(result, "interval_mean_end")
    configured, record = hypothesis(result, "interval_mean_end", forcing="configured_balance")
    np.testing.assert_array_equal(configured.inflow_volume_m3, 86400 * (frame.inflow_m3s.iloc[1:] + .5))
    np.testing.assert_array_equal(raw.inflow_volume_m3, 86400 * frame.inflow_m3s.iloc[1:])
    np.testing.assert_array_equal(configured.residual_m3 - raw.residual_m3, -43200.)
    assert record["forcing"] == "configured_balance"
    pd.testing.assert_frame_equal(frame, original, check_exact=True)


@pytest.mark.parametrize("column", ["storage_m3", "inflow_m3s", "release_m3s"])
def test_nonfinite_rows_break_affected_intervals_and_cumulative_segments(column):
    frame = observations(inflow=3., release=1.)
    frame.storage_m3 += np.arange(len(frame)) * 10.
    frame.loc[frame.index[3], column] = np.nan
    result = audit_storage(frame)
    endpoint, record = hypothesis(result)
    assert endpoint.valid.tolist() == [True, True, False, False, True, True]
    np.testing.assert_array_equal(endpoint.loc[endpoint.valid, "residual_m3"], 10.)
    np.testing.assert_array_equal(endpoint.loc[endpoint.valid, "cumulative_residual_m3"], [10., 20., 10., 20.])
    assert endpoint.loc[~endpoint.valid, "cumulative_residual_m3"].isna().all()
    assert endpoint.segment_id.iloc[0] == endpoint.segment_id.iloc[1]
    assert endpoint.segment_id.iloc[4] == endpoint.segment_id.iloc[5]
    assert endpoint.segment_id.iloc[0] != endpoint.segment_id.iloc[4]
    assert record["statistics"]["valid_intervals"] == 4
    assert not record["statistics"]["complete_closure"]
    assert result.flags.loc[frame.index[3], "nonfinite_" + column.removesuffix("_m3").removesuffix("_m3s")]


def test_mean_only_intervals_do_not_require_unused_neighbor_flow_values():
    frame = observations()
    frame.loc[frame.index[3], "inflow_m3s"] = np.nan
    result = audit_storage(frame, lag_diagnostics=False)
    ending, _ = hypothesis(result, "interval_mean_end")
    starting, _ = hypothesis(result, "interval_mean_start")
    assert ending.valid.tolist() == [True, True, False, True, True, True]
    assert starting.valid.tolist() == [True, True, True, False, True, True]


def test_signed_fluxes_and_physical_conflicts_are_flags_not_clipped_measurements():
    frame = observations(inflow=[-3., -2., -1., 0., 1., 2., 3.], release=-1.,
                         storage=[50., 55., 60., 65., 70., 75., 80.])
    frame.loc[frame.index[0], "release_m3s"] = 2.
    result = audit_storage(frame, capacity_m3=70., dead_storage_m3=60., lag_diagnostics=False)
    intervals, _ = hypothesis(result, "interval_mean_end")
    np.testing.assert_array_equal(intervals.inflow_volume_m3, 86400 * frame.inflow_m3s.iloc[1:])
    np.testing.assert_array_equal(intervals.release_volume_m3, -86400.)
    assert intervals.valid.all()
    assert intervals.negative_flow.any()
    assert result.flags.below_dead_storage.tolist() == [True, True, False, False, False, False, False]
    assert result.flags.above_capacity.tolist() == [False, False, False, False, False, True, True]
    assert result.flags.positive_release_below_dead.tolist() == [True, False, False, False, False, False, False]
    assert result.flags.negative_inflow.sum() == 3
    assert result.flags.negative_release.sum() == 6


def test_lag_sign_and_shared_support_are_independent_of_best_score():
    frame = observations(inflow=np.arange(1., 8.), release=2.)
    result = audit_storage(frame)
    for convention, count, baseline in (("interval_mean_end", 5, 0.),
                                         ("interval_mean_start", 5, 86400.),
                                         ("endpoint_trapezoid", 4, 43200.)):
        _, record = hypothesis(result, convention)
        lagged = {(entry["inflow_lag_samples"], entry["release_lag_samples"]): entry["statistics"]
                  for entry in record["lag_comparisons"]}
        assert set(lagged) == {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)}
        assert {entry["valid_intervals"] for entry in lagged.values()} == {count}
        assert lagged[0, 0]["mean_residual_m3"] == baseline
        assert lagged[1, 0]["mean_residual_m3"] == baseline + 86400
        assert lagged[-1, 0]["mean_residual_m3"] == baseline - 86400
        assert lagged[0, 1]["mean_residual_m3"] == baseline
        assert lagged[0, -1]["mean_residual_m3"] == baseline
    assert result.summary["settings_selected"] is None
    assert not result.summary["model_ready_claimed"]


def test_period_slice_and_lag_diagnostics_cannot_read_held_out_values():
    frame = observations(pd.date_range("2020-12-25", periods=15, name="date"),
                         inflow=np.arange(1., 16.), release=2.)
    before = audit_storage(frame.loc[:"2020-12-31"])
    frame.loc["2021-01-01":, :] = [-1e9, np.inf, -1e4]
    after = audit_storage(frame.loc[:"2020-12-31"])
    assert before.summary == after.summary
    for field in ("intervals", "monthly", "extremes", "flags"):
        pd.testing.assert_frame_equal(getattr(before, field), getattr(after, field), check_exact=True)
    assert before.intervals.index.get_level_values("date").max() == pd.Timestamp("2020-12-31")


def test_no_lag_option_and_all_missing_data_remain_honest():
    frame = observations()
    frame.loc[:, "storage_m3"] = np.nan
    result = audit_storage(frame, lag_diagnostics=False)
    assert not result.intervals.valid.any()
    assert result.intervals.cumulative_residual_m3.isna().all()
    for record in result.summary["hypotheses"]:
        assert record["lag_comparisons"] == []
        assert record["statistics"]["valid_intervals"] == 0
        assert not record["statistics"]["complete_closure"]


def test_monthly_statistics_weight_rates_by_elapsed_time_and_keep_seasons_non_cumulative():
    dates = pd.date_range("2020-01-01", periods=5, freq="MS", name="date")
    frame = observations(dates)
    frame.storage_m3 += np.arange(len(frame)) * 1000
    result = audit_storage(frame, frequency="monthly", lag_diagnostics=False)
    intervals, record = hypothesis(result)
    statistics = record["statistics"]
    durations = np.array([31., 29., 31., 30.]) * 86400
    assert statistics["valid_duration_seconds"] == durations.sum()
    assert statistics["duration_coverage"] == 1.
    assert statistics["mean_residual_rate_m3s"] == pytest.approx(np.mean(1000 / durations))
    assert statistics["duration_weighted_residual_rate_m3s"] == pytest.approx(4000 / durations.sum())
    assert statistics["rmse_residual_fraction_capacity"] is None
    assert statistics["whole_period_closure_residual_m3"] == 4000
    assert statistics["segments"] == [{"start_date": "2020-01-01", "end_date": "2020-05-01",
        "intervals": 4, "duration_seconds": float(durations.sum()), "closure_residual_m3": 4000.}]
    assert {"complete_closure", "whole_period_closure_residual_m3", "segments"}.isdisjoint(result.monthly.columns)
    monthly = result.monthly.loc[result.monthly.hypothesis.eq(record["hypothesis"])]
    assert monthly.destination_month.tolist() == [2, 3, 4, 5]
    assert monthly.valid_intervals.eq(1).all()
    np.testing.assert_array_equal(intervals.residual_m3, 1000.)


def test_extremes_rank_signed_magnitude_and_preserve_source_context():
    frame = observations()
    residual = np.array([100., -900., 300., -200., 700., 0.])
    frame.storage_m3 += np.r_[0., residual.cumsum()]
    result = audit_storage(frame, capacity_m3=1e8, lag_diagnostics=False)
    _, record = hypothesis(result)
    extremes = result.extremes.loc[result.extremes.hypothesis.eq(record["hypothesis"])]
    assert extremes.residual_m3.tolist() == [-900., 700., 300., -200., 100., 0.]
    assert extremes.date.iloc[0] == frame.index[2]
    assert extremes.start_date.iloc[0] == frame.index[1]
    assert extremes.observed_storage_start_m3.iloc[0] == frame.storage_m3.iloc[1]
    assert extremes.observed_storage_end_m3.iloc[0] == frame.storage_m3.iloc[2]
    assert extremes.raw_inflow_end_m3s.eq(3).all()
    assert record["statistics"]["max_abs_residual_m3"] == 900
    assert record["statistics"]["p95_abs_residual_m3"] == pytest.approx(np.percentile(np.abs(residual), 95))
    assert record["statistics"]["rmse_residual_fraction_capacity"] == pytest.approx(np.sqrt(np.mean(residual ** 2)) / 1e8)


def test_transformed_nonfinite_volumes_cannot_enter_finite_ledgers_or_hide_raw_records():
    frame = observations()
    with np.errstate(over="ignore", invalid="ignore"):
        result = audit_storage(frame, inflow_factor=1e308)
    raw, _ = hypothesis(result)
    configured, record = hypothesis(result, forcing="configured_balance")
    assert raw.valid.all()
    assert not configured.valid.any()
    assert configured.residual_m3.isna().all() and configured.cumulative_residual_m3.isna().all()
    assert record["statistics"]["valid_intervals"] == 0
    assert record["statistics"]["whole_period_closure_residual_m3"] is None
    assert result.summary["observation_flags"]["nonfinite_inflow"] == 0


def test_too_short_common_lag_support_reports_no_evidence_and_selects_no_lag():
    frame = observations(pd.date_range("2020-01-01", periods=2, name="date"))
    result = audit_storage(frame)
    for record in result.summary["hypotheses"]:
        assert record["statistics"]["valid_intervals"] == 1
        for lag in record["lag_comparisons"]:
            assert lag["statistics"]["valid_intervals"] == 0
            assert lag["statistics"]["rmse_residual_m3"] is None
            assert lag["statistics"]["whole_period_closure_residual_m3"] is None
    assert result.summary["settings_selected"] is None


@pytest.mark.parametrize("change", ["duplicate", "reversed", "gap", "nonmidnight", "timezone", "missing_column"])
def test_ambiguous_calendar_or_schema_is_rejected_without_reordering(change):
    frame = observations()
    if change == "duplicate":
        frame.index = pd.DatetimeIndex([frame.index[0], *frame.index[:-1]], name="date")
    elif change == "reversed":
        frame = frame.iloc[::-1]
    elif change == "gap":
        frame = frame.drop(frame.index[3])
    elif change == "nonmidnight":
        frame.index += pd.Timedelta(hours=1)
    elif change == "timezone":
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame = frame.drop(columns="release_m3s")
    with pytest.raises((TypeError, ValueError)):
        audit_storage(frame)


@pytest.mark.parametrize("kwargs", [
    {"frequency": "weekly"}, {"fixed_step_days": 0}, {"fixed_step_days": True},
    {"fixed_step_days": np.inf}, {"inflow_factor": np.nan}, {"inflow_offset_m3s": np.inf},
    {"balance_inflow_correction": np.nan}, {"capacity_m3": 0}, {"dead_storage_m3": -1},
    {"lag_diagnostics": 1},
])
def test_invalid_scientific_settings_are_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        audit_storage(observations(), **kwargs)
