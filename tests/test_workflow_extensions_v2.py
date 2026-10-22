"""Versioned initializer validation and short-record source-quality safeguards."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from workflows import reservoir_v2 as v2
from workflows import reservoir as v1
from strategies.short_record import select_short_record
from dztr.calibration.sampled import _observations
from dztr.calibration.candidates import parameters_to_vector
from dztr.simulation.sampled import SamplingSettings

ROOT=Path(__file__).parents[1]


def mcgregor():
    path=ROOT/'data/reservoirs/ex22_Mcgregor/observations_si.csv'
    return _observations(pd.read_csv(path,index_col='date',parse_dates=True,float_precision='round_trip'))


def test_short_record_keeps_normal_year_and_exposes_constant_storage():
    obs=mcgregor();plan,metadata=select_short_record(obs,SamplingSettings())
    assert metadata['calibration_years']==[2003,2004,2005,2007]
    assert metadata['assessment_years']==[2006] and metadata['calibration_fraction']>=.65
    assert metadata['constant_observation_years']['storage']==[2005]
    assert metadata['undefined_calibration_regime_nse']==['dry_storage']
    assert metadata['source_review_required']
    held=plan.mask(obs.frame.index,'assessment');changed=obs.frame.copy();changed.loc[held,['storage_m3','release_m3s']]*=2
    altered,newmeta=select_short_record(_observations(changed),SamplingSettings())
    assert altered.signature==plan.signature
    assert newmeta['selection_uses_response_values'] is False
    with pytest.raises(ValueError,match='Cannot retain'):select_short_record(obs,SamplingSettings(),'chronological')
    with pytest.raises(ValueError,match='exactly six'):select_short_record(_observations(obs.frame.loc['2003-01-01':]),SamplingSettings())


def extension_config(tmp_path, identifier):
    cfg=v1.read(ROOT/'configs/workflow_deferred_two_v2.json');cfg['reservoirs']=[e for e in cfg['reservoirs'] if e['reservoir_id']==identifier]
    # Resolve paths before moving the configuration into its test directory.
    entry=cfg['reservoirs'][0]
    for key in ('observations_csv','model_config'):entry[key]=str((ROOT/'configs'/entry[key]).resolve())
    if 'initializer' in entry:entry['initializer']['csv']=str((ROOT/'configs'/entry['initializer']['csv']).resolve())
    cfg['optimization']={'seeds':[42],'population':4,'evaluation_budget':8,'workers':1}
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg));return path,cfg


def test_explicit_witness_is_used_and_v1_registry_remains_readable(tmp_path):
    path,cfg=extension_config(tmp_path,'ex42_Tyuyamuyun');out=v2.prepare(path,tmp_path/'prepared')
    objective,template,_,plan,_,_=v2.objective_for(out,'ex42_Tyuyamuyun')
    expected=pd.read_csv(cfg['reservoirs'][0]['initializer']['csv'],index_col='month',float_precision='round_trip')
    np.testing.assert_allclose(template.monthly_targets,expected,rtol=1e-14,atol=1e-6)
    assert plan.signature==cfg['reservoirs'][0]['initializer']['selection_sha256']
    evaluated=objective.evaluate(parameters_to_vector(template));assert evaluated.feasible
    assert v2.independent_balance(evaluated.simulation.frame,1.)<=1e-4
    snapshot=v2.calibrate(out,'ex42_Tyuyamuyun',42)
    vectors=pd.read_csv(snapshot/'vectors.csv',index_col='evaluation_id',float_precision='round_trip')
    np.testing.assert_array_equal(vectors.loc[0],parameters_to_vector(template))
    registry=v2.finish(out)
    profile,_=v2.load_profile(registry,identifier='ex42_Tyuyamuyun')
    assert profile['initialization']['kind']=='explicit_calibration_initializer'
    assert profile['selection_protocol']['kind']=='standard'
    old=ROOT/'outputs/unified_workflow_v1_20261013/registry.json'
    assert len(v2.verify_registry(old)['profiles_by_reservoir_id'])==20
    prior=tmp_path/'v1';prior.mkdir();v1.write(prior/'protocol.json',{'kind':'unified_reservoir_workflow_v1'})
    with pytest.raises(ValueError,match='original v1 runner'):
        v2.check_protocol(prior)


def test_initializer_signature_and_bounds_cannot_be_bypassed(tmp_path):
    path,cfg=extension_config(tmp_path,'ex42_Tyuyamuyun');cfg['reservoirs'][0]['initializer']['selection_sha256']='0'*64;path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='different calibration selection'):v2.prepare(path,tmp_path/'wrong_mask')
    path,cfg=extension_config(tmp_path,'ex42_Tyuyamuyun');targets=pd.read_csv(cfg['reservoirs'][0]['initializer']['csv'],index_col='month',float_precision='round_trip')
    targets['release_critical_m3s']=0;bad=tmp_path/'outside.csv';targets.to_csv(bad,float_format='%.17g')
    cfg['reservoirs'][0]['initializer'].update(csv=str(bad),sha256=v2.sha(bad));path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):v2.prepare(path,tmp_path/'outside_bounds')


def test_short_record_prepares_but_blocks_unreviewed_optimization(tmp_path):
    path,cfg=extension_config(tmp_path,'ex22_Mcgregor');out=v2.prepare(path,tmp_path/'pooled')
    assert v2.read(out/'preparation.json')['reservoirs'][0]['source_review_required']
    with pytest.raises(ValueError,match='Constant short-record'):v2.calibrate(out,'ex22_Mcgregor',42)
    assert not (out/'jobs').exists()
    cfg['scoring']='equal_regime';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='Equal-regime scoring is undefined'):v2.prepare(path,tmp_path/'equal')
