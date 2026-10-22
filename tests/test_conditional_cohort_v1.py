"""Net-policy parity, honest short-record selection and resumable publication."""
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters
from strategies.conditional_interval_v1 import NonnegativeNetRunner, ShortRecordSelection, short_record_selection
from strategies import interval_scalar_v1 as scalar
from workflows import conditional_cohort_v1 as w


def parameters():
    table=pd.DataFrame([[2e8,5e8,8e8,2.,4.,8.]]*12,index=pd.Index(range(1,13),name='month'),columns=MONTHLY_TARGET_COLUMNS)
    return ReservoirParameters(table,1e9,5e8,0.,10.,.1)


@pytest.mark.parametrize('rate',[0.,3.,1000.])
def test_nonnegative_net_path_is_exact_canonical_net_policy_with_gaps(rate):
    index=pd.date_range('2000-01-01',periods=100,name='date');inflow=pd.Series(rate,index=index);inflow.iloc[35:38]=np.nan
    runner=NonnegativeNetRunner(inflow,np.full(12,5e8));p=parameters()
    assert w.canonical_check(runner,p,runner.run(p))<1e-4
    with pytest.raises(ValueError,match='storage flood mode'):runner.run(replace(p,inflow_correction=1.1))


def test_signed_net_is_never_clipped_or_silently_admitted():
    inflow=pd.Series([1.,-1.,2.],index=pd.date_range('2000-01-01',periods=3))
    with pytest.raises(ValueError,match='nonnegative'):NonnegativeNetRunner(inflow,np.full(12,5e8))
    assert inflow.iloc[1]==-1.


def short_frame():
    index=pd.date_range('2018-05-03','2019-12-31',name='date');x=np.arange(len(index))
    frame=pd.DataFrame({'inflow_m3s':np.nan,'storage_m3':5e8+1e8*np.sin(x/45),'release_m3s':60+10*np.cos(x/30)},index=index)
    return w.d.conditional_inflow(frame)


def test_short_selection_has_actual_65_percent_all_months_and_separate_assessment():
    frame=short_frame();_,support=scalar.simulation_support(frame,restart_warmup_days=30)
    plan=short_record_selection(frame.index,support);mask=plan.mask(frame.index,'calibration')
    assert mask.mean()>=.65 and not (mask&~support).any()
    assert set(frame.index[mask].month)==set(range(1,13))
    assert set(plan.table().regime)=={'unclassified_short_record'}
    assert frame.index[mask][-1]<frame.index[plan.mask(frame.index,'assessment')][0]
    assert w.selection(plan.as_dict())==plan
    with pytest.raises(ValueError,match='complete daily'):ShortRecordSelection((),())
    with pytest.raises(ValueError,match='calendar changed'):plan.mask(frame.index[:-1],'calibration')
    with pytest.raises(ValueError,match='Insufficient'):short_record_selection(frame.index,np.zeros(len(frame),bool))


@pytest.fixture
def workspace():
    path=Path(tempfile.mkdtemp(prefix='test_conditional_cohort_',dir=w.ROOT/'outputs')).resolve()
    yield path
    assert path.is_relative_to((w.ROOT/'outputs').resolve()) and path.name.startswith('test_conditional_cohort_')
    shutil.rmtree(path)


def config(folder):
    short_frame().to_csv(folder/'source.csv')
    ids=['etienne_sample']+[f'etienne_deferred_{i}' for i in range(19)]
    w.write(folder/'catalog.json',{'reservoirs':[{'reservoir_id':rid} for rid in ids]})
    pd.DataFrame([{'reservoir_id':rid,'GRAND_ID':None,'parameter_status':'descriptive'} for rid in ids]).to_csv(folder/'inventory.csv',index=False)
    rel=lambda name:str((folder/name).relative_to(w.ROOT))
    cfg={'kind':'conditional_extension_cohort_v1','base_catalog':rel('catalog.json'),'base_inventory':rel('inventory.csv'),
        'cases':[{'reservoir_id':ids[0],'GRAND_ID':None,'observations_csv':rel('source.csv'),'start':'2018-05-03','end':'2019-12-31',
            'selection':'short_record_chronological','restart_warmup_days':30,'allow_feasible_initialization':True,
            'physical_constants':{'capacity_m3':1e9,'dead_storage_fraction':.1,'initial_storage_m3':5e8,'initial_release_m3s':0.,'inflow_correction':1.,'flood_mode':'storage'},'source_decision':'conditional fixture'}],
        'decisions':{rid:'source evidence required' for rid in ids},'balance_tolerance_m3':1e-4,'objective_weights':{'storage':.5,'release':.5},
        'optimization':{'seeds':[7],'horizon':4,'evaluation_budget':4,'workers':1,'perturbation':.2}}
    path=folder/'config.json';w.write(path,cfg);return path


def test_checkpoint_continuation_exact_and_identity_guard(workspace):
    cfg=config(workspace);a=w.prepare(cfg,workspace/'direct');b=w.prepare(cfg,workspace/'continued')
    w.worker(a,'etienne_sample',7);w.worker(b,'etienne_sample',7,2);w.worker(b,'etienne_sample',7,4)
    first=w.d.read(a/'jobs/etienne_sample/seed_7/checkpoint.json');last=w.d.read(b/'jobs/etienne_sample/seed_7/checkpoint.json')
    assert first['state']==last['state']
    last['identity']['seed']=999;w.write(b/'jobs/etienne_sample/seed_7/checkpoint.json',last)
    with pytest.raises(ValueError,match='identity'):w.worker(b,'etienne_sample',7)
    assert not (b/'jobs/etienne_sample/seed_7/running.lock').exists()


def test_publication_retains_all_decisions_and_null_grand_and_replays(workspace,monkeypatch):
    cfg=config(workspace);out=w.prepare(cfg,workspace/'run');w.worker(out,'etienne_sample',7)
    original=w.metrics
    def guarded(*args,**kwargs):
        assert w.d.read(out/'selection_frozen.json')['assessment_responses_enter_forcing']
        return original(*args,**kwargs)
    monkeypatch.setattr(w,'metrics',guarded)
    report=w.report(out);assert w.replay(out,'etienne_sample')['new_optimizer_evaluations']==0
    profile,p=w.native.load_profile(report/'registry.json',identifier='etienne_sample')
    assert profile['assessment_responses_enter_forcing'] and not profile['independent_prediction_validated']
    assert profile['grand_id'] is None and not profile['selection_uses_assessment']
    status=pd.read_csv(report/'remaining_20_status.csv');assert len(status)==20 and (status.status=='deferred').sum()==19
    assert pd.read_csv(report/'reservoir_parameterization_all_62_by_GRAND_ID.csv').GRAND_ID.isna().all()
    assert w.report(out)==report


def test_source_and_prepared_tampering_fail_before_new_evaluations(workspace):
    cfg=config(workspace);out=w.prepare(cfg,workspace/'run');assert w.prepare(cfg,out)==out
    path=out/'prepared/etienne_sample/observations_si.csv';path.write_text(path.read_text()+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Prepared cohort changed'):w.worker(out,'etienne_sample',7)
    path=workspace/'inventory.csv';path.write_text(path.read_text()+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Cohort input changed'):w.checked(out)


def test_calibration_views_hide_direct_assessment_but_reconstruction_is_conditional(workspace):
    out=w.prepare(config(workspace),workspace/'run');obj,frame,plan,support=w.objective(out,'etienne_sample')
    assert obj.observed.loc[~obj.mask].isna().all().all()
    poisoned=frame.copy();assessment=plan.mask(frame.index,'assessment');poisoned.loc[assessment,'storage_m3']+=1e6
    second,bounds=w._fit(poisoned,plan,support,w.d.read(out/'prepared/etienne_sample/physical_constants.json'),True)
    pd.testing.assert_frame_equal(obj.template.monthly_targets,second.monthly_targets)
    assert not np.array_equal(w.d.conditional_inflow(poisoned).inflow_m3s[assessment],frame.inflow_m3s[assessment])
