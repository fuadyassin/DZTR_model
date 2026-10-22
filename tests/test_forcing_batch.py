"""Supplied forcing validation, units, clocks, observations and batch failures."""
import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).parents[1] / "tools"))
from forcing_batch import FrozenProfiles, load_forcing, initial_parameters, simulate_job, run_batch
from simulate_calibrated_reservoirs import write, read


@pytest.fixture(scope="module")
def store():
    return FrozenProfiles()


def request(tmp_path,store,*,observations=False):
    profile,parameters,config=store.load({"grand_id":253})
    columns={"date":"date","inflow":"inflow"}
    units={"inflow":"m3/s"}
    data={"date":["2040-01-01","2040-01-02","2040-01-03"],"inflow":[171.,171.,171.]}
    if observations:
        columns["storage"]="storage";units["storage"]="Mm3"
        data["storage"]=[7900.,7901.,7902.]
    pd.DataFrame(data).to_csv(tmp_path / "forcing.csv",index=False)
    return {"job_id":"future_gardiner","grand_id":253,
            "forcing":{"csv":"forcing.csv","stage":"model_input","columns":columns,"units":units},
            "initial":{"storage":parameters.initial_storage_m3,"storage_unit":"m3","release":parameters.initial_release_m3s,"release_unit":"m3/s"},"sampling":profile["sampling"]},profile,parameters,config


def test_future_initial_state_and_optional_storage_only_metrics(tmp_path,store):
    job,profile,parameters,config=request(tmp_path,store,observations=True)
    job["initial"]["storage"]=7900.;job["initial"]["storage_unit"]="Mm3"
    frame,obs,params,summary=simulate_job(job,tmp_path,store)
    assert frame.index[0]==pd.Timestamp("2040-01-01")
    assert frame.storage_m3.iloc[0]==7.9e9
    assert obs.frame.storage_m3.iloc[0]==7.9e9
    assert summary["metrics"][0]["storage_pairs"]==2
    assert summary["metrics"][0]["release_pairs"]==0
    assert summary["metrics"][0]["release_nse"] is None
    assert summary["selection_uses_supplied_observations"] is False
    # Changing measurements must not change any simulated state.
    data=pd.read_csv(tmp_path/'forcing.csv');data.storage+=100
    data.to_csv(tmp_path/'forcing.csv',index=False)
    second=simulate_job(job,tmp_path,store)[0]
    np.testing.assert_array_equal(frame.storage_m3,second.storage_m3)
    np.testing.assert_array_equal(frame.release_m3s,second.release_m3s)


def test_unit_conversion_raw_adjustment_applied_once(tmp_path,store):
    job,profile,parameters,config=request(tmp_path,store)
    data=pd.read_csv(tmp_path/'forcing.csv');data.inflow*=86400
    data.to_csv(tmp_path/'forcing.csv',index=False)
    job['forcing']['units']['inflow']='m3/day'
    adjusted=copy.deepcopy(config);adjusted['inflow_preprocessing']={'factor':2.,'offset_m3s':3.}
    obs,_,_=load_forcing(job,tmp_path,profile,adjusted)
    np.testing.assert_array_equal(obs.frame.inflow_m3s,[171.]*3)
    job['forcing']['stage']='raw_source'
    obs,_,_=load_forcing(job,tmp_path,profile,adjusted)
    np.testing.assert_array_equal(obs.frame.inflow_m3s,[345.]*3)


@pytest.mark.parametrize('bad',['gap','duplicate','negative','nan','infinity','unit','clock','stage'])
def test_invalid_forcing_is_rejected(tmp_path,store,bad):
    job,profile,parameters,config=request(tmp_path,store)
    data=pd.read_csv(tmp_path/'forcing.csv')
    if bad=='gap': data.loc[2,'date']='2040-01-04'
    if bad=='duplicate': data.loc[2,'date']='2040-01-02'
    if bad=='negative': data.loc[2,'inflow']=-1
    if bad=='nan': data.loc[2,'inflow']=np.nan
    if bad=='infinity': data.loc[2,'inflow']=np.inf
    if bad=='unit': job['forcing']['units']['inflow']='acre-feet'
    if bad=='clock': job['sampling']['fixed_step_days']=2
    if bad=='stage': job['forcing']['stage']='guess'
    data.to_csv(tmp_path/'forcing.csv',index=False)
    with pytest.raises(ValueError): load_forcing(job,tmp_path,profile,config)


def test_invalid_initial_state_and_metric_period(tmp_path,store):
    job,_,parameters,_=request(tmp_path,store)
    job['initial']['storage']=parameters.capacity_m3+1
    with pytest.raises(ValueError,match='Initial storage'): initial_parameters(parameters,job['initial'])
    job['initial']['storage']=parameters.initial_storage_m3
    job['initial']['release']=True
    with pytest.raises(ValueError,match='finite numbers'): initial_parameters(parameters,job['initial'])
    job['initial']['release']=parameters.initial_release_m3s
    job['evaluation_periods']={'wrong':{'start':'2040-01-01','end':'2040-01-04'}}
    with pytest.raises(ValueError,match='exact existing date boundaries'): simulate_job(job,tmp_path,store)


def test_batch_isolates_failure_and_protects_output(tmp_path,store):
    job,_,_,_=request(tmp_path,store)
    invalid=copy.deepcopy(job);invalid['job_id']='bad';invalid['grand_id']=2356
    manifest=tmp_path/'batch.json'
    write(manifest,{'schema_version':1,'kind':'dztr_forcing_batch','jobs':[invalid,job]})
    output=tmp_path/'run'
    assert not run_batch(manifest,output)
    summary=read(output/'batch_summary.json')
    assert [r['status'] for r in summary['jobs']]==['failed','passed']
    assert (output/job['job_id']/'normalized_inputs_si.csv').exists()
    with pytest.raises(ValueError,match='new directory'): run_batch(manifest,output)
    invalid['job_id']=job['job_id'].upper()
    write(manifest,{'schema_version':1,'kind':'dztr_forcing_batch','jobs':[invalid,job]})
    with pytest.raises(ValueError,match='unique path-safe'): run_batch(manifest,tmp_path/'next')


def test_monthly_forcing_preserves_native_fixed_clock(tmp_path,store):
    job,profile,parameters,config=request(tmp_path,store)
    sampling={'frequency':'monthly','fixed_step_days':30.5,'month_clock':'observed','legacy_start':None,'allow_signed_inflow':False}
    profile=copy.deepcopy(profile);profile['sampling']=sampling
    job['sampling']=sampling
    pd.DataFrame({'date':['2040-01-01','2040-02-01','2040-03-01'],'inflow':[171.,171.,171.]}).to_csv(tmp_path/'forcing.csv',index=False)
    obs,clock,_=load_forcing(job,tmp_path,profile,config)
    from dztr.simulation.root_policy import simulate_root_policy_reservoir
    frame=simulate_root_policy_reservoir(obs,parameters,sampling=clock,policy='reject').frame
    assert len(frame)==3
    np.testing.assert_array_equal(frame.step_seconds,[30.5*86400]*3)
    np.testing.assert_array_equal(frame.rule_month,[1,2,3])
