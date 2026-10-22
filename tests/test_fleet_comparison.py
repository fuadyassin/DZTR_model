"""Fleet split safeguards and exact reuse of the established pilot design."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from strategies.fleet_comparison_v2 import comparison_selections
from strategies.strategy_comparison import comparison_selections as pilot_selections

ROOT = Path(__file__).parents[1]


def test_outcome_summary_distinguishes_tradeoffs_and_failed_assessment(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'tools'))
    from report_comparison_fleet_v2 import direction
    assert direction(.1,.2).startswith('Both improve')
    assert direction(-.1,-.2).startswith('Both decline')
    assert direction(.1,-.2)=='Storage-release trade-off'
    assert direction(0,0)=='No material difference'
    assert direction(np.nan,.2)=='Assessment unavailable'


def test_whole_year_fleet_splits_cover_65_percent_and_exclude_common_assessment():
    source = ROOT/'outputs/representative_years_v1_20261012'
    if not source.exists(): pytest.skip('Prepared reservoir records unavailable')
    ids = json.loads((ROOT/'configs/calibrated_twenty.json').read_text())['profiles_by_reservoir_id']
    for identifier in ids:
        frame = pd.read_csv(source/identifier/'observations_si.csv', index_col='date', parse_dates=True)
        annual = pd.read_csv(source/identifier/'year_selection.csv', index_col='water_year')
        plans, held = comparison_selections(frame, annual)
        assert plans['mixed_pooled'].signature == plans['mixed_balanced'].signature
        chronological = {y for y,r in zip(plans['chronological_pooled'].years, plans['chronological_pooled'].roles) if r=='calibration'}
        assert min(held) > max(chronological)
        counts = []
        for plan in plans.values():
            mask = plan.mask(frame.index, 'calibration')
            assert mask.sum()/len(frame) >= .65
            for col in ('storage_m3', 'release_m3s'):
                assert frame.loc[mask,col].notna().sum()/frame[col].notna().sum() >= .65
            selected = {y for y,r in zip(plan.years, plan.roles) if r=='calibration'}
            assert not selected & set(held)
            counts.append(len(selected))
        assert len(set(counts)) == 1
        later = annual.loc[annual.index > max(chronological)]
        assert set(annual.loc[held, 'regime']) <= set(later.regime)
        if len(later)>1:assert len(held)<len(later)
        if identifier in ('ex5_Garrison', 'ex6_Oahe'):
            original, oldheld = pilot_selections(frame, annual)
            assert held == oldheld
            assert all(plans[k].signature == original[k].signature for k in plans)


def test_completed_fleet_scores_reproduce_from_stored_trajectories():
    from types import SimpleNamespace
    from dztr.calibration.sampled import _observations
    from dztr.evaluation import evaluate_simulation
    out = ROOT/'outputs/strategy_comparison_fleet_v2_20261012'
    if not (out/'pooled_comparison.csv').exists(): pytest.skip('Fleet experiment not completed')
    table = pd.read_csv(out/'pooled_comparison.csv', float_precision='round_trip')
    assert len(table)==54
    for _, row in table.iterrows():
        if not row.assessment_feasible: continue
        folder = out/'assessed'/row.reservoir_id/row.arm/f'pooled_selected_seed{int(row.seed)}_eval{int(row.evaluation_id)}'
        data = pd.read_csv(folder/'simulation.csv', index_col='date', parse_dates=True, float_precision='round_trip')
        part = data[data.role=='assessment']
        assert len(part)==row.common_samples
        obs = part[['observed_storage_m3','observed_release_m3s','inflow_m3s']].rename(columns={'observed_storage_m3':'storage_m3','observed_release_m3s':'release_m3s'})
        scores = evaluate_simulation(_observations(obs), SimpleNamespace(frame=part))
        for var in ('storage','release'):
            np.testing.assert_allclose(scores[var+'_nse'], row['assessment_'+var+'_nse'], rtol=0, atol=1e-12)
