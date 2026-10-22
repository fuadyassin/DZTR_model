"""Literal solver parity, independent volumes, gap starts and fitting isolation."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters
from dztr.data.intervals import prepare_daily_intervals
from dztr.simulation.intervals import IntervalSimulationSettings, simulate_interval_reservoir
from dztr.calibration.candidates import parameters_to_vector
from strategies import interval_scalar_v1 as s
from workflows.extension_assessment_v1 import blocks


def parameters():
    table = pd.DataFrame([[2e8, 5e8, 8e8, 2., 4., 8.]]*12,
        index=pd.Index(range(1, 13), name="month"), columns=MONTHLY_TARGET_COLUMNS)
    return ReservoirParameters(table, 1e9, 5e8, 0., 10., .1)


@pytest.mark.parametrize("rate", [0., 3., 1000.])
def test_array_path_matches_literal_simulator_and_independent_volume(rate):
    index = pd.date_range("2000-01-01", periods=100, name="date")
    inflow = pd.Series(rate, index=index); inflow.iloc[35:38] = np.nan
    params = parameters(); runner = s.ArrayRunner(inflow, np.full(12, 5e8))
    fast = runner.run(params)
    for start, stop in blocks(inflow.notna()):
        section = pd.DataFrame({"inflow_m3s": inflow.iloc[start:stop], "storage_m3": 5e8, "release_m3s": 0.})
        literal = simulate_interval_reservoir(prepare_daily_intervals(section), params,
            settings=IntervalSimulationSettings(root_policy="nearest_previous_storage")).frame
        for column in fast:
            np.testing.assert_array_equal(fast[column][start:stop], literal[column].to_numpy())
        residual = np.diff(fast['storage_m3'][start:stop])-inflow.iloc[start+1:stop].to_numpy()*86400 + fast['interval_mean_total_outflow_m3s'][start+1:stop]*86400
        assert np.max(np.abs(residual)) < 1e-4
    assert np.isnan(fast['storage_m3'][35:38]).all()


def data():
    index = pd.date_range("2000-01-01", "2019-12-31", name="date")
    x = np.arange(len(index))
    return pd.DataFrame({"inflow_m3s": 3+(index.year.to_numpy()-2000)%5+.2*np.sin(x/10),
        "storage_m3": 5e8+1e8*np.sin(x/45)+x*1000,
        "release_m3s": 3+np.cos(x/30)}, index=index)


def test_missing_flows_are_not_filled_and_warmup_is_fixed():
    frame = data(); frame.iloc[100, frame.columns.get_loc('inflow_m3s')] = np.nan
    anchors, score = s.simulation_support(frame, restart_warmup_days=30)
    assert anchors[0] and anchors[101] and not score[100:131].any()
    assert score[131] and np.isnan(frame.inflow_m3s.iloc[100])


def test_selection_keeps_whole_years_and_65_percent_actual_support():
    frame = data(); frame.iloc[2000, frame.columns.get_loc('inflow_m3s')] = np.nan
    _, score = s.simulation_support(frame, restart_warmup_days=30)
    plan, annual = s.select_years(frame, score)
    mask = plan.mask(frame.index, 'calibration') & score
    assert mask.mean() >= .65
    assert set(annual.loc[annual.role=='assessment','regime']) == {'dry','normal','wet'}
    assert annual.loc[2000, 'role'] == 'warmup'
    for year, part in plan.table().groupby('water_year'): assert part.role.nunique() == 1


def test_held_out_responses_do_not_change_targets_bounds_starts_or_objective():
    frame = data(); _, score = s.simulation_support(frame, restart_warmup_days=30)
    plan, _ = s.select_years(frame, score); params = parameters()
    physical = {k: getattr(params, k) for k in ('capacity_m3','initial_storage_m3','initial_release_m3s',
        'channel_capacity_m3s','dead_storage_fraction','inflow_correction','flood_mode')}
    first, bounds = s.fit(frame, plan, score, physical)
    poisoned = frame.copy(); poisoned.loc[plan.mask(frame.index, 'assessment'), ['storage_m3','release_m3s']] = 9e99
    second, other = s.fit(poisoned, plan, score, physical)
    pd.testing.assert_frame_equal(first.monthly_targets, second.monthly_targets)
    pd.testing.assert_frame_equal(bounds.empirical.monthly_bounds, other.empirical.monthly_bounds)
    runner = s.ArrayRunner(frame.inflow_m3s, first.monthly_targets.storage_normal_m3.to_numpy())
    a = s.Objective(frame, plan, score, runner, first, bounds, objective_weights={'storage':.5,'release':.5})
    b = s.Objective(poisoned, plan, score, runner, second, other, objective_weights={'storage':.5,'release':.5})
    vector = parameters_to_vector(first)
    assert a(vector).cost == b(vector).cost
    assert a.observed.loc[~a.mask].isna().all().all()


def test_prefix_does_not_read_future_inflow():
    frame = data().iloc[:60]; params = parameters()
    a = s.ArrayRunner(frame.inflow_m3s, np.full(12, 5e8))
    changed = frame.inflow_m3s.copy(); changed.iloc[30:] = 1e99
    b = s.ArrayRunner(changed, np.full(12, 5e8))
    first, second = a.run(params, end_position=29), b.run(params, end_position=29)
    for column in first: np.testing.assert_array_equal(first[column], second[column])


def test_mutated_parameter_tables_are_revalidated():
    frame = data().iloc[:60]; params = parameters()
    params.monthly_targets.iloc[0,0] = 9e9
    with pytest.raises(ValueError, match='Storage targets'): s.ArrayRunner(frame.inflow_m3s, np.full(12, 5e8)).run(params)
