"""Whole-year coverage, no outcome leakage, continuous state and regime scoring."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from strategies.representative_years import (YearSelection,select_representative_years,fit_selected_years,
                                           RepresentativeYearObjective)
from dztr.calibration.sampled import _observations
from dztr.calibration.candidates import parameters_to_vector
from dztr.schemas import MONTHLY_TARGET_COLUMNS,ReservoirParameters
from dztr.simulation.sampled import SamplingSettings
from dztr.simulation.root_policy import simulate_root_policy_reservoir


def data(frequency='monthly',years=15):
    index=pd.date_range('2000-01-01',f'{1999+years}-12-31',freq='MS' if frequency=='monthly' else 'D',name='date')
    t=index.year-2000
    season=np.sin(index.month*2*np.pi/12)
    frame=pd.DataFrame({'inflow_m3s':.8+t*.025+.1*season,'storage_m3':5e8+t*1e6+3e7*season,'release_m3s':1.+.01*t+.2*season},index=index)
    return _observations(frame),SamplingSettings(frequency=frequency,fixed_step_days=30.5 if frequency=='monthly' else 1.)


def physical():
    return dict(capacity_m3=1e9,initial_storage_m3=5e8,initial_release_m3s=1.,channel_capacity_m3s=20.,dead_storage_fraction=.1,inflow_correction=1.,flood_mode='storage')


@pytest.mark.parametrize('frequency',['daily','monthly'])
def test_whole_year_minimum_includes_warmup_and_is_deterministic(frequency):
    observations,sampling=data(frequency)
    plan,annual,warnings=select_representative_years(observations,sampling)
    repeated,_,_=select_representative_years(observations,sampling)
    assert plan.signature==repeated.signature
    assert sum(r=='calibration' for r in plan.roles)/len(plan.roles)>=.65
    assert annual.role.eq('calibration').sum()/annual.complete.sum()>=.65
    assert all(annual.loc[annual.role=='calibration'].regime.eq(r).any() for r in ('dry','normal','wet'))
    assert annual.loc[2000,'role']=='warmup'
    assert not np.any(plan.mask(observations.frame.index,'calibration') & plan.mask(observations.frame.index,'assessment'))


def test_outcome_values_cannot_select_years_or_fit_targets_bounds():
    observations,sampling=data()
    plan,_,_=select_representative_years(observations,sampling)
    changed=observations.frame.copy();mask=plan.mask(changed.index,'assessment')
    changed.loc[mask,'storage_m3']*=100;changed.loc[mask,'release_m3s']*=1000
    other=_observations(changed)
    repeated,_,_=select_representative_years(other,sampling)
    assert plan.signature==repeated.signature
    template,bounds=fit_selected_years(observations,plan,physical())
    duplicate,otherbounds=fit_selected_years(other,plan,physical())
    pd.testing.assert_frame_equal(template.monthly_targets,duplicate.monthly_targets,check_exact=True)
    np.testing.assert_array_equal(bounds.lower_vector,otherbounds.lower_vector)
    np.testing.assert_array_equal(bounds.upper_vector,otherbounds.upper_vector)


def test_water_year_partial_edges_remain_context():
    observations,sampling=data()
    plan,annual,_=select_representative_years(observations,sampling,water_year_start_month=10)
    assert annual.iloc[0].role=='context' and annual.iloc[-1].role=='context'
    assert annual.loc[2001,'role']=='warmup'
    assert plan.table().iloc[0].role=='context'
    assert plan.table().role.eq('calibration').mean()>=.65


def test_constant_forcing_does_not_manufacture_dry_or_wet_labels():
    observations,sampling=data();frame=observations.frame.copy();frame.inflow_m3s=1.
    _,annual,warnings=select_representative_years(_observations(frame),sampling)
    assert set(annual.loc[annual.role.isin(['calibration','assessment']),'regime'])=={'normal'}
    assert any('no dry' in w for w in warnings)
    assert any('no wet' in w for w in warnings)


def test_invalid_minimum_short_records_and_gaps_are_rejected():
    observations,sampling=data()
    with pytest.raises(ValueError,match='0.65'):select_representative_years(observations,sampling,minimum_fraction=.64)
    short,clock=data(years=5)
    with pytest.raises(ValueError,match='six complete'):select_representative_years(short,clock)
    with pytest.raises(ValueError,match='consecutive'):select_representative_years(_observations(observations.frame.drop(observations.frame.index[3])),sampling)


def test_masked_objective_keeps_state_and_ignores_assessment_outcomes():
    observations,sampling=data(years=10)
    table=pd.DataFrame(np.tile([2e8,6e8,9e8,.5,1.5,3.],(12,1)),index=pd.Index(range(1,13),name='month'),columns=MONTHLY_TARGET_COLUMNS)
    template=ReservoirParameters(table,**physical());vector=parameters_to_vector(template)
    roles=tuple('warmup' if d.year==2000 else 'assessment' if d.year in (2003,2009) else 'calibration' for d in observations.frame.index)
    plan=YearSelection(tuple(d.date().isoformat() for d in observations.frame.index),roles,tuple(observations.frame.index.year),tuple('normal' for d in observations.frame.index),.65,42,1,1)
    objective=RepresentativeYearObjective(observations,plan,template,sampling=sampling,cost_mode='pooled_nse')
    result=objective.evaluate(vector)
    assert result.feasible,result.reason
    assert result.simulation.frame.index[-1]==pd.Timestamp('2008-12-01')
    direct=simulate_root_policy_reservoir(_observations(observations.frame.loc[:'2008-12-01']),template,sampling=sampling)
    pd.testing.assert_frame_equal(result.simulation.frame,direct.frame,check_exact=True)
    assert pd.Timestamp('2003-06-01') in result.simulation.frame.index
    assert result.paired_counts['storage']==7*12
    changed=observations.frame.copy();mask=plan.mask(changed.index,'assessment')
    changed.loc[mask,'storage_m3']*=100;changed.loc[mask,'release_m3s']*=100
    changed.loc[changed.index.year==2009,'inflow_m3s']=np.nan
    other=RepresentativeYearObjective(_observations(changed),plan,template,sampling=sampling,cost_mode='pooled_nse').evaluate(vector)
    assert other.feasible,other.reason
    np.testing.assert_array_equal(result.objectives,other.objectives)


def test_regime_macro_costs_are_distinct_from_pooled_nse(monkeypatch):
    observations,sampling=data(years=12);plan,_,_=select_representative_years(observations,sampling)
    template,_=fit_selected_years(observations,plan,physical())
    import strategies.representative_years as strategy
    def fake(obs,params,**kwargs):
        frame=obs.frame.copy();frame.storage_m3=frame.storage_m3+np.arange(len(frame))*1e5
        frame.release_m3s=frame.release_m3s+.1;frame['balance_residual_m3']=0.
        return SimpleNamespace(frame=frame,parameters=params)
    # dataclasses.replace requires a dataclass result, as the real simulator supplies.
    from dztr.schemas import SimulationResult
    def wrapped(obs,params,**kwargs):return SimulationResult(fake(obs,params).frame,params,'test','test',())
    monkeypatch.setattr(strategy,'simulate_root_policy_reservoir',wrapped)
    objective=RepresentativeYearObjective(observations,plan,template,sampling=sampling)
    result=objective.evaluate(parameters_to_vector(template));assert result.feasible,result.reason
    for i,v in enumerate(('release','storage')):
        average=np.mean([result.metrics[r+'_'+v+'_nse'] for r in ('dry','normal','wet')])
        assert result.objectives[i]==-average
    assert result.metrics['regime_balanced_storage_nse']!=result.metrics['storage_nse']


def test_changed_selection_rejects_other_bounds_and_changes_identity():
    observations,sampling=data()
    plan,_,_=select_representative_years(observations,sampling)
    other,_,_=select_representative_years(observations,sampling,seed=43)
    template,bounds=fit_selected_years(observations,plan,physical())
    with pytest.raises(ValueError,match='different year selection'):RepresentativeYearObjective(observations,other,template,sampling=sampling,bounds=bounds)
    a=RepresentativeYearObjective(observations,plan,template,sampling=sampling)
    b=RepresentativeYearObjective(observations,other,template,sampling=sampling)
    assert a.identity!=b.identity


def test_invalid_raw_targets_remain_unchanged_with_separate_feasible_seed():
    observations,sampling=data()
    plan,annual,_=select_representative_years(observations,sampling)
    frame=observations.frame.copy()
    low_years=annual.index[annual.role=='calibration'][:3]
    frame.loc[frame.index.year.isin(low_years)&(frame.index.month==4),'storage_m3']=5e7
    altered=_observations(frame)
    with pytest.raises(ValueError,match='Storage targets'):fit_selected_years(altered,plan,physical())
    template,bounds=fit_selected_years(altered,plan,physical(),allow_feasible_initialization=True)
    assert bounds.generalized_targets.loc[4,'storage_critical_m3']<=1e8
    assert template.monthly_targets.loc[4,'storage_critical_m3']>1e8
    bounds.validate(parameters_to_vector(template),template)
    assert bounds.initialization_kind.startswith('feasible_bounds_midpoint')
    np.testing.assert_array_equal(altered.frame.storage_m3,frame.storage_m3)


def test_checkpoint_identity_rejects_a_different_scoring_mask(tmp_path):
    from dztr.calibration.optimizer import NSGA2Settings,optimize_nsga2
    from dztr.calibration.checkpoint import save_optimizer_checkpoint,load_optimizer_checkpoint
    observations,sampling=data()
    first,_,_=select_representative_years(observations,sampling,seed=42)
    second,_,_=select_representative_years(observations,sampling,seed=43)
    objective=SimpleNamespace(bounds=SimpleNamespace(lower_vector=np.zeros(72),upper_vector=np.ones(72)),evaluate=lambda vector:SimpleNamespace(feasible=True,objectives=(float(vector.sum()),float(np.square(vector-1).sum())),reason=None))
    states=[]
    optimize_nsga2(objective,np.full(72,.5),NSGA2Settings(seed=42,population_size=4,evaluation_budget=8),checkpoint_callback=states.append)
    path=tmp_path/'checkpoint.npz';identity={'strategy':'representative_years_v1','selection':first.signature}
    save_optimizer_checkpoint(path,states[-1],identity)
    load_optimizer_checkpoint(path,identity)
    with pytest.raises(ValueError):load_optimizer_checkpoint(path,dict(identity,selection=second.signature))


def test_prepared_cli_job_uses_new_objective_without_launching_search(tmp_path,monkeypatch):
    import calibrate_representative_years as cli
    from simulate_calibrated_reservoirs import ROOT,read
    prepared=ROOT/'outputs/representative_years_v1_20261012'
    if not (prepared/'preparation_manifest.json').exists():pytest.skip('Prepared fleet snapshot not generated')
    def fake_optimizer(objective,baseline,settings,**kwargs):
        assert isinstance(objective,RepresentativeYearObjective)
        assert objective.cost_mode=='regime_balanced_nse'
        assert kwargs['bounds'].selection_signature==objective.selection.signature
        candidate=SimpleNamespace(evaluation_id=0,vector=baseline,objectives=(0.,0.),feasible=True,reason=None)
        return SimpleNamespace(evaluated=(candidate,),front=(candidate,),history=(),evaluations=1,termination='mocked_no_evaluations')
    monkeypatch.setattr(cli,'optimize_nsga2',fake_optimizer)
    output=tmp_path/'job'
    monkeypatch.setattr(sys,'argv',['calibrate',str(prepared),'--reservoir-id','ex5_Garrison','--output',str(output),'--evaluation-budget','4','--population','4'])
    cli.main()
    result=read(output/'result.json')
    assert result['assessment_evaluations']==0 and not result['profile_promoted']
    assert result['objective']['strategy']=='representative_years_v1'
