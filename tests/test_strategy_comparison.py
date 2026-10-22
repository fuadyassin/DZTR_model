"""Matched data amounts, common assessment and calibration-only selection."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from strategies.strategy_comparison import comparison_selections,pooled_choice


def test_common_years_are_excluded_and_mixed_arms_share_identical_masks():
    root=Path(__file__).parents[1]/'outputs/representative_years_v1_20261012'
    for identifier in ('ex5_Garrison','ex6_Oahe'):
        frame=pd.read_csv(root/identifier/'observations_si.csv',index_col='date',parse_dates=True)
        annual=pd.read_csv(root/identifier/'year_selection.csv',index_col='water_year')
        plans,held=comparison_selections(frame,annual)
        assert len(held)==4 and set(annual.loc[held,'regime'])=={'dry','normal','wet'}
        assert sum(annual.loc[y,'regime']=='dry' for y in held)==2
        assert plans['mixed_pooled'].signature==plans['mixed_balanced'].signature
        for arm,plan in plans.items():
            assert plan.roles.count('calibration')==240 and plan.roles.count('assessment')==48
            years={y for y,r in zip(plan.years,plan.roles) if r=='calibration'}
            assert not years&set(held)
            if arm=='chronological_pooled':assert years==set(range(1971,1991))
        assert plans['chronological_pooled'].signature!=plans['mixed_pooled'].signature


def test_pooled_choice_ignores_assessment_and_has_stable_seed_ties():
    table=pd.DataFrame({'seed':[43,42,44,42],'evaluation_id':[3,4,5,9],'release_cost':[-.7,-.7,-.9,-.5],'storage_cost':[-.7,-.7,-.5,-.9],'assessment_nse':[999,-999,999,999]})
    first,_=pooled_choice([table])
    table.assessment_nse=-table.assessment_nse
    second,_=pooled_choice([table.sample(frac=1,random_state=1)])
    assert (first.seed,first.evaluation_id)==(42,4)
    assert (first.seed,first.evaluation_id)==(second.seed,second.evaluation_id)


def test_completed_pilot_shared_scores_reproduce_without_new_simulation():
    from types import SimpleNamespace
    from dztr.calibration.sampled import _observations
    from dztr.evaluation import evaluate_simulation
    import json
    root=Path(__file__).parents[1]/'outputs/strategy_comparison_pilot_20261012'
    if not (root/'comparison_scores.csv').exists():pytest.skip('Independent experiment not generated')
    design=json.loads((root/'design.json').read_text())
    assert len(design['jobs'])==18 and sum(j['evaluation_budget'] for j in design['jobs'])==9216
    frozen=json.loads((root/'frozen_selection.json').read_text())
    assert frozen['selection_uses_assessment'] is False and frozen['written_before_assessment']
    pooled=pd.read_csv(root/'pooled_comparison.csv',float_precision='round_trip')
    assert len(pooled)==6 and pooled.assessment_feasible.all()
    for _,row in pooled.iterrows():
        folder=root/'assessed'/row.reservoir_id/row.arm/f'pooled_selected_seed{int(row.seed)}_eval{int(row.evaluation_id)}'
        frame=pd.read_csv(folder/'simulation.csv',index_col='date',parse_dates=True,float_precision='round_trip')
        part=frame[frame.role=='assessment'];assert len(part)==48
        observed=part[['observed_storage_m3','observed_release_m3s','inflow_m3s']].rename(columns={'observed_storage_m3':'storage_m3','observed_release_m3s':'release_m3s'})
        metrics=evaluate_simulation(_observations(observed),SimpleNamespace(frame=part))
        for variable in ('storage','release'):np.testing.assert_allclose(metrics[variable+'_nse'],row['assessment_'+variable+'_nse'],rtol=0,atol=1e-12)
    verification=json.loads((root/'scientific_verification.json').read_text())
    assert verification['search_trials_verified']==9216 and verification['assessment_failures']==0
    assert verification['maximum_balance_residual_m3']<=1e-4
