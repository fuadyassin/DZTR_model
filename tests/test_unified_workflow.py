"""Real optimization continuation, causal fitting, registry and input safeguards."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from workflows import reservoir as w


@pytest.fixture
def inputs(tmp_path):
    dates=pd.date_range('2000-01-01','2009-12-01',freq='MS')
    x=np.arange(len(dates));phase=2*np.pi*x/12
    data=pd.DataFrame({'date':dates,'inflow_m3s':80+15*np.sin(phase)+8*np.sin(x/19),
        'storage_m3':500e6+80e6*np.sin(phase-.8)+2e6*(dates.year-2000),
        'release_m3s':80+12*np.sin(phase-.3)+3*np.sin(x/15)})
    data.to_csv(tmp_path/'observations.csv',index=False,float_format='%.17g')
    model={'physical_constants':{'capacity_m3':1e9,'initial_storage_m3':500e6,'initial_release_m3s':80.,'channel_capacity_m3s':500.,'dead_storage_fraction':.05,'inflow_correction':1.,'flood_mode':'storage'},
        'sampling':{'frequency':'monthly','fixed_step_days':30.5,'month_clock':'observed'},'root_policy':'nearest_previous_storage','atol_storage_m3':1e-4}
    (tmp_path/'model.json').write_text(json.dumps(model))
    cfg={'schema_version':1,'optimization':{'seeds':[7,8],'population':4,'evaluation_budget':8,'workers':1},
        'reservoirs':[{'reservoir_id':'synthetic','grand_id':123,'observations_csv':'observations.csv','model_config':'model.json'}]}
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg));return path,tmp_path


def test_defaults_units_and_unsafe_configuration(inputs):
    path,base=inputs;_,cfg=w.config(path)
    assert (cfg['year_selection'],cfg['scoring'])==('mixed','pooled')
    frame=w.observations(base/'observations.csv').frame
    converted=pd.read_csv(base/'observations.csv');converted.inflow_m3s*=86400;converted.release_m3s*=86400;converted.storage_m3/=1e6
    converted.to_csv(base/'scaled.csv',index=False,float_format='%.17g')
    actual=w.observations(base/'scaled.csv',units={'inflow':'m3/day','release':'m3/day','storage':'Mm3'}).frame
    np.testing.assert_allclose(actual,frame,rtol=1e-14)
    cfg['reservoirs'][0]['reservoir_id']='../escape';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='Unsafe'):w.config(path)


@pytest.mark.parametrize('years,cost',[('mixed','pooled'),('mixed','equal_regime'),('chronological','pooled'),('chronological','equal_regime')])
def test_preparation_excludes_assessment_from_targets(inputs,years,cost):
    path,base=inputs;cfg=json.loads(path.read_text());cfg.update(year_selection=years,scoring=cost);path.write_text(json.dumps(cfg))
    out=w.prepare(path,base/'first');_,template,bounds,plan,frame,_=w.objective_for(out,'synthetic')
    assert plan.roles.count('calibration')/len(plan.roles)>=.65
    held=plan.mask(frame.index,'assessment')
    changed=pd.read_csv(base/'observations.csv',float_precision='round_trip');changed.loc[held,['storage_m3','release_m3s']]*=1.5;changed.to_csv(base/'observations.csv',index=False,float_format='%.17g')
    second=w.prepare(path,base/'second');_,new,newbounds,newplan,_,_=w.objective_for(second,'synthetic')
    assert newplan.signature==plan.signature
    np.testing.assert_array_equal(template.monthly_targets,new.monthly_targets)
    np.testing.assert_array_equal(bounds.lower_vector,newbounds.lower_vector)
    np.testing.assert_array_equal(bounds.upper_vector,newbounds.upper_vector)


def test_continuation_matches_uninterrupted_search_and_exports_versioned_registry(inputs):
    path,base=inputs;out=w.prepare(path,base/'continued')
    for seed in (7,8):w.calibrate(out,'synthetic',seed)
    registry=w.finish(out);oldmanifest=w.sha(registry.parent/'manifest.json')
    w.verify_registry(registry)
    profile,parameters=w.load_profile(registry,grand_id=123)
    assert profile['selection_uses_assessment'] is False
    assert profile['year_selection']=='mixed' and profile['scoring']=='pooled'
    assert profile['assessment']['feasible']
    for seed in (7,8):w.calibrate(out,'synthetic',seed,16)
    later=w.finish(out,16);assert later!=registry
    assert w.sha(registry.parent/'manifest.json')==oldmanifest
    direct=w.prepare(path,base/'direct')
    w.calibrate(direct,'synthetic',7,16)
    for name in ('trials.csv','vectors.csv'):
        a=pd.read_csv(out/'jobs/synthetic/seed_7/budget_16'/name,float_precision='round_trip')
        b=pd.read_csv(direct/'jobs/synthetic/seed_7/budget_16'/name,float_precision='round_trip')
        pd.testing.assert_frame_equal(a,b)
    # Selecting/finishing a completed budget reuses its verified release.
    assert w.finish(out,16)==later
    with pytest.raises(ValueError,match='cannot shrink'):w.calibrate(out,'synthetic',7,8)
    forcing=base/'forcing.csv';pd.read_csv(base/'observations.csv')[['date','inflow_m3s']].to_csv(forcing,index=False)
    result=w.simulate(later,forcing,base/'simulation',identifier='synthetic')
    assert w.read(result/'receipt.json')['maximum_balance_residual_m3']<=1e-4
    altered=pd.read_csv(forcing).iloc[1:];altered.to_csv(base/'shifted.csv',index=False)
    with pytest.raises(ValueError,match='explicit initial'):w.simulate(later,base/'shifted.csv',base/'bad',identifier='synthetic')


def test_modified_preparation_and_profile_are_rejected(inputs):
    path,base=inputs;out=w.prepare(path,base/'prepared')
    model=out/'prepared/synthetic/model_config.json';model.write_text(model.read_text()+' ')
    with pytest.raises(ValueError,match='Prepared input changed'):w.calibrate(out,'synthetic',7)


def test_cli_runs_independent_jobs_and_reuses_completed_release(inputs):
    import subprocess,sys
    path,base=inputs;out=base/'cli';script=Path(__file__).parents[1]/'tools/reservoir_workflow.py'
    command=[sys.executable,'-B',str(script),'run','--config',str(path),'--output',str(out)]
    first=subprocess.run(command,capture_output=True,text=True)
    assert first.returncode==0,first.stdout+first.stderr
    assert w.read(out/'status.json')['stage']=='completed'
    registry=out/'releases/budget_8/registry.json';digest=w.sha(registry.parent/'manifest.json')
    second=subprocess.run(command,capture_output=True,text=True)
    assert second.returncode==0,second.stdout+second.stderr
    assert w.sha(registry.parent/'manifest.json')==digest
    seeds=pd.read_csv(registry.parent/'seed_scores.csv')
    assert set(seeds.seed)=={7,8} and seeds.assessment_feasible.all()


def test_archive_import_reproduces_calibration_selection_without_model_calls(inputs,monkeypatch):
    root=Path(__file__).parents[1];archive=root/'outputs/strategy_comparison_pilot_20261012'
    if not archive.exists():pytest.skip('Independent comparison archive unavailable')
    path,base=inputs
    cfg={'schema_version':1,'archives':[str(archive)],'reservoirs':[{'reservoir_id':'ex5_Garrison','grand_id':753}]}
    path.write_text(json.dumps(cfg))
    def forbidden(*args,**kwargs):raise AssertionError('Archive import must not rerun simulations or optimization')
    monkeypatch.setattr(w,'simulate_root_policy_reservoir',forbidden);monkeypatch.setattr(w,'optimize_nsga2',forbidden)
    registry=w.adopt(path,base/'release')
    profile,params=w.load_profile(registry,grand_id=753)
    assert profile['source_kind']=='verified_comparison_archive'
    receipt=w.read(registry.parent/'import_receipt.json');assert receipt['new_optimization_evaluations']==receipt['new_simulations']==0
    expected=pd.read_csv(archive/f'assessed/ex5_Garrison/mixed_pooled/pooled_selected_seed{profile["seed"]}_eval{profile["evaluation_id"]}/monthly_targets_si.csv',index_col='month',float_precision='round_trip')
    pd.testing.assert_frame_equal(params.monthly_targets,expected)
    targets=registry.parent/'profiles/ex5_Garrison/monthly_targets_si.csv';targets.write_text(targets.read_text()+'\n')
    with pytest.raises(ValueError,match='Release artifact changed'):w.load_profile(registry,identifier='ex5_Garrison')


def test_archive_import_rejects_different_numerical_equations(inputs,monkeypatch):
    root=Path(__file__).parents[1];archive=root/'outputs/strategy_comparison_pilot_20261012'
    if not archive.exists():pytest.skip('Independent comparison archive unavailable')
    path,base=inputs;path.write_text(json.dumps({'schema_version':1,'archives':[str(archive)],'reservoirs':[{'reservoir_id':'ex5_Garrison','grand_id':753}]}))
    real_sha=w.sha
    def changed_numeric_hash(p):
        return '0'*64 if Path(p).resolve()==root/'src/dztr/model/release.py' else real_sha(p)
    monkeypatch.setattr(w,'sha',changed_numeric_hash)
    with pytest.raises(ValueError,match='Archived numerical source changed'):w.adopt(path,base/'not_created')
    assert not (base/'not_created').exists()
