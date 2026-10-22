"""Persistent jobs, frozen source decisions, publication and correct-clock replay."""
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from workflows import daily_interval_calibration_v1 as w


@pytest.fixture
def workspace():
    path=Path(tempfile.mkdtemp(prefix='test_daily_interval_calibration_',dir=w.ROOT/'outputs')).resolve()
    yield path
    assert path.is_relative_to((w.ROOT/'outputs').resolve()) and path.name.startswith('test_daily_interval_calibration_')
    shutil.rmtree(path)


def config(folder,budget=4):
    index=pd.date_range('2000-01-01','2019-12-31',name='date');x=np.arange(len(index))
    frame=pd.DataFrame({'inflow_m3s':3+(index.year.to_numpy()-2000)%5+.2*np.sin(x/10),
        'storage_m3':5e8+1e8*np.sin(x/45)+x*1000,'release_m3s':3+np.cos(x/30)},index=index)
    frame.to_csv(folder/'source.csv')
    pd.DataFrame([{'reservoir_id':'sample','GRAND_ID':299,'parameter_status':'descriptive'}]).to_csv(folder/'inventory.csv',index=False)
    cfg={'kind':'daily_interval_scalar_calibration_v1','reservoir_id':'sample','GRAND_ID':299,
        'storage_csv':str((folder/'source.csv').relative_to(w.ROOT)),'flow_csv':str((folder/'source.csv').relative_to(w.ROOT)),
        'inventory_csv':str((folder/'inventory.csv').relative_to(w.ROOT)),
        'start':'2000-01-01','end':'2019-12-31','excluded_inflow_dates':['2006-08-23'],
        'restart_warmup_days':30,'minimum_calibration_fraction':.65,'year_selection_seed':42,
        'balance_tolerance_m3':1e-4,'physical_constants':{'capacity_m3':1e9,'initial_storage_m3':5e8,
            'initial_release_m3s':0.,'channel_capacity_m3s':10.,'dead_storage_fraction':.1,'inflow_correction':1.,'flood_mode':'storage'},
        'objective_weights':{'storage':.5,'release':.5},'optimization':{'seeds':[7],'horizon':budget,'evaluation_budget':budget,'workers':1,'perturbation':.2},
        'source_decisions':{'fixture':'known conditional scenario'}}
    path=folder/'config.json';w.write(path,cfg)
    return path


def test_quarantine_and_verified_reuse_do_not_change_source_records(workspace):
    cfg=config(workspace);out=w.prepare(cfg,workspace/'run')
    source=w.data.frame_read(out/'prepared/source_observations_si.csv')
    prepared=w.data.frame_read(out/'prepared/observations_si.csv')
    assert source.loc['2006-08-23','inflow_m3s']>0 and np.isnan(prepared.loc['2006-08-23','inflow_m3s'])
    assert prepared.drop(index=pd.Timestamp('2006-08-23')).equals(source.drop(index=pd.Timestamp('2006-08-23')))
    before=(w.data.sha(out/'protocol.json'),(out/'protocol.json').stat().st_mtime_ns)
    assert w.prepare(cfg,out)==out
    assert before==(w.data.sha(out/'protocol.json'),(out/'protocol.json').stat().st_mtime_ns)
    selection=w.data.frame_read(out/'prepared/selection.csv')
    assert selection.fitting_eligible.mean()>=.65


def test_fresh_and_continued_jobs_are_exact_and_checkpoint_identity_is_verified(workspace):
    cfg=config(workspace)
    direct=w.prepare(cfg,workspace/'direct');continued=w.prepare(cfg,workspace/'continued')
    w.calibrate(direct,7);w.calibrate(continued,7,2);w.calibrate(continued,7,4)
    a=w.data.read(direct/'jobs/seed_7/checkpoint.json');b=w.data.read(continued/'jobs/seed_7/checkpoint.json')
    assert a['state']==b['state']
    b['identity']['seed']=99;w.write(continued/'jobs/seed_7/checkpoint.json',b)
    with pytest.raises(ValueError,match='Checkpoint identity'):w.calibrate(continued,7,4)
    assert not (continued/'jobs/seed_7/running.lock').exists()


def test_complete_publication_loads_shared_grand_profile_and_replays_mean_clock(workspace):
    cfg=config(workspace,budget=2);out=w.prepare(cfg,workspace/'run');w.calibrate(out,7)
    published=w.report(out)
    metadata,parameters=w.native.load_profile(published/'registry.json',grand_id=299)
    assert metadata['route']=='daily_mean_interval_scalar_v1' and not metadata['selection_uses_assessment']
    assert parameters.monthly_targets.shape==(12,6)
    assert w.replay(out)['status']=='exact_shared_registry_replay_passed'
    assert w.data.read(published/'scientific_checks.json')['canonical_trajectories_verified']==2
    assert not w.data.read(out/'selection_frozen.json')['selection_uses_assessment']
    assert w.report(out)==published


def test_tampered_prepared_forcing_is_rejected_before_new_evaluations(workspace):
    cfg=config(workspace);out=w.prepare(cfg,workspace/'run')
    path=out/'prepared/observations_si.csv';path.write_text(path.read_text()+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Prepared data changed'):w.calibrate(out,7)
