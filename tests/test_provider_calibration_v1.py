"""Preserve provider signs, verify literal solver parity and checkpoint replay."""
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
from dztr.schemas import MONTHLY_TARGET_COLUMNS,ReservoirParameters
from dztr.calibration.candidates import parameters_to_vector
from strategies import provider_interval_v1 as s
from workflows import provider_calibration_v1 as w,minidoka_source_v1 as source


def parameters():
    table=pd.DataFrame([[2e8,5e8,8e8,2.,4.,8.]]*12,index=pd.Index(range(1,13),name='month'),columns=MONTHLY_TARGET_COLUMNS)
    return ReservoirParameters(table,1e9,5e8,0.,10.,.1)


@pytest.mark.parametrize('rate',[-3.,0.,20.])
def test_signed_driver_exactly_matches_canonical_net_policy_and_preserves_gaps(rate):
    index=pd.date_range('2000-01-01',periods=70,name='date');inflow=pd.Series(rate,index=index);inflow.iloc[30:33]=np.nan
    runner=s.SignedNetRunner(inflow,np.full(12,5e8));p=parameters();sim=runner.run(p)
    assert w.shared.canonical_check(runner,p,sim)<1e-4
    np.testing.assert_array_equal(runner.inflow,inflow.to_numpy())
    assert not runner.inflow.flags.writeable and np.isnan(sim['storage_m3'][30:33]).all()
    with pytest.raises(ValueError,match='unit correction'):runner.run(replace(p,inflow_correction=1.1))


def test_signed_insufficient_water_fails_without_flooring_storage():
    inflow=pd.Series([-1e9]*3,index=pd.date_range('2000-01-01',periods=3))
    runner=s.SignedNetRunner(inflow,np.full(12,5e8))
    with pytest.raises(ValueError,match='negative storage'):runner.run(parameters())
    assert (runner.inflow==-1e9).all()


def test_provider_parser_preserves_numeric_signs_and_missing_cells(tmp_path):
    path=tmp_path/'provider.html';path.write_text('<table><tr>'+''.join('<th>'+x+'</th>' for x in ['DateTime','min_af','min_id','min_qt','min_fb'])+'</tr><tr><td>2000-01-01</td><td>-7550</td><td>-99.99</td><td>100</td><td>4235.28</td></tr><tr><td>2000-01-02</td><td>0</td><td></td><td>200</td><td>4236</td></tr></table>')
    frame=source.read_provider(path);assert frame.min_af.iloc[0]==-7550 and frame.min_id.iloc[0]==-99.99 and np.isnan(frame.min_id.iloc[1])
    normalized=source.normalize(frame);assert normalized.storage_m3.iloc[0]>0
    np.testing.assert_allclose(normalized.storage_m3.diff(),frame.min_af.diff()*w.d.ACRE_FOOT_M3,rtol=0,atol=1e-7,equal_nan=True)
    np.testing.assert_allclose(normalized.storage_m3/w.d.ACRE_FOOT_M3-115000,frame.min_af,rtol=0,atol=1e-10)
    path.write_text(path.read_text().replace('min_id','min_in'))
    with pytest.raises(ValueError,match='Unexpected provider fields'):source.read_provider(path)


@pytest.fixture
def workspace():
    folder=Path(tempfile.mkdtemp(prefix='test_provider_calibration_',dir=w.ROOT/'outputs')).resolve();yield folder
    assert folder.is_relative_to((w.ROOT/'outputs').resolve()) and folder.name.startswith('test_provider_calibration_');shutil.rmtree(folder)


def config(folder):
    index=pd.date_range('2000-01-01','2012-12-31',name='date');x=np.arange(len(index))
    storage=5e8+1e8*np.sin(x/45);q=60+10*np.cos(x/30)
    frame=pd.DataFrame({'inflow_m3s':q+pd.Series(storage).diff().to_numpy()/86400,'storage_m3':storage,'release_m3s':q},index=index);frame.to_csv(folder/'observations.csv')
    w.write(folder/'source.json',{'fixture':'known reference; computed signed forcing'})
    w.write(folder/'catalog.json',{'reservoirs':[{'reservoir_id':'etienne_Minidoka'}]});pd.DataFrame([{'reservoir_id':'etienne_Minidoka','GRAND_ID':None}]).to_csv(folder/'inventory.csv',index=False)
    rel=lambda name:str((folder/name).relative_to(w.ROOT))
    cfg={'kind':'provider_endpoint_interval_calibration_v1','reservoir_id':'etienne_Minidoka','GRAND_ID':None,'observations_csv':rel('observations.csv'),'source_review':rel('source.json'),
        'base_catalog':rel('catalog.json'),'base_inventory':rel('inventory.csv'),'start':'2000-01-01','end':'2012-12-31','balance_tolerance_m3':1e-4,'restart_warmup_days':30,
        'physical_constants':{'capacity_m3':1e9,'dead_storage_fraction':.1,'initial_storage_m3':5e8,'initial_release_m3s':0.,'inflow_correction':1.,'flood_mode':'storage'},
        'objective_weights':{'storage':1.,'release':0.},'optimization':{'seeds':[7],'horizon':4,'evaluation_budget':4,'workers':1,'perturbation':.2}}
    path=folder/'config.json';w.write(path,cfg);return path


def test_exact_checkpoint_resume_and_corrupted_identity_rejected(workspace):
    cfg=config(workspace);a=w.prepare(cfg,workspace/'direct');b=w.prepare(cfg,workspace/'resumed')
    w.worker(a,7);w.worker(b,7,2);w.worker(b,7,4)
    assert w.d.read(a/'jobs/seed_7/checkpoint.json')['state']==w.d.read(b/'jobs/seed_7/checkpoint.json')['state']
    checkpoint=w.d.read(b/'jobs/seed_7/checkpoint.json');checkpoint['identity']['seed']=99;w.write(b/'jobs/seed_7/checkpoint.json',checkpoint)
    with pytest.raises(ValueError,match='identity'):w.worker(b,7)
    assert not (b/'jobs/seed_7/running.lock').exists()


def test_publication_registry_exact_replay_and_unresolved_release_timing(workspace):
    out=w.prepare(config(workspace),workspace/'run');w.worker(out,7);report=w.report(out)
    profile,p=w.native.load_profile(report/'registry.json',identifier='etienne_Minidoka')
    assert profile['scoring']=='storage_nse' and not profile['release_performance_accepted'] and not profile['release_timing_validated']
    assert profile['grand_id'] is None and not profile['selection_uses_assessment']
    assert w.replay(out)['new_optimizer_evaluations']==0 and w.report(out)==report


def test_prepared_tampering_is_rejected_and_calibration_views_hide_assessment(workspace):
    cfg=config(workspace);out=w.prepare(cfg,workspace/'run');assert w.prepare(cfg,out)==out
    obj,frame,plan,support=w.objective(out);assert obj.observed.loc[~obj.mask].isna().all().all()
    poisoned=frame.copy();poisoned.loc[plan.mask(frame.index,'assessment'),['storage_m3','release_m3s']]=9e99
    other,bounds=w.fit(poisoned,plan,support,w.d.read(out/'prepared/physical_constants.json'));pd.testing.assert_frame_equal(other.monthly_targets,obj.template.monthly_targets)
    path=out/'prepared/observations_si.csv';path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='Prepared cohort changed'):w.worker(out,7)


def test_instant_total_release_objective_rejects_unresolved_spill(workspace):
    out=w.prepare(config(workspace),workspace/'run');obj,frame,plan,support=w.objective(out)
    original=obj.runner
    class SpillRunner:
        tolerance=1e-4
        def run(self,p,**kwargs):
            result=original.run(p,**kwargs);result['spill_volume_m3'][obj.mask]=100.;return result
    joint=s.EndpointObjective(frame,plan,support,SpillRunner(),obj.template,obj.bounds,objective_weights={'storage':.5,'release':.5})
    result=joint(parameters_to_vector(obj.template));assert not result.feasible and 'Instant total release' in result.reason
    storage=s.EndpointObjective(frame,plan,support,SpillRunner(),obj.template,obj.bounds,objective_weights={'storage':1.,'release':0.})
    assert storage(parameters_to_vector(obj.template)).feasible
