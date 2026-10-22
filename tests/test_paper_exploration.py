"""Paper membership and scientific safeguards for exploratory admission."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from workflows import paper_exploration as p
from workflows import reservoir_v2 as w
from strategies.representative_years import select_representative_years
from strategies.fleet_comparison_v2 import comparison_selections
from dztr.calibration.sampled import _observations
from dztr.calibration.candidates import parameters_to_vector
from dztr.simulation.sampled import SamplingSettings

ROOT=Path(__file__).parents[1]

def tok():
    identifier='ex41_Toktogul'
    cfg=w.read(ROOT/'configs/reservoirs'/f'{identifier}.json')
    frame=pd.read_csv(ROOT/'data/reservoirs'/identifier/'observations_si.csv',index_col='date',parse_dates=True,float_precision='round_trip')
    obs=_observations(frame)
    _,annual,_=select_representative_years(obs,SamplingSettings(**cfg['sampling']))
    plans,_=comparison_selections(obs.frame,annual)
    return obs,cfg,plans['mixed_pooled'],next(r for r in p.paper_cohort() if r['reservoir_id']==identifier)

def test_published_table_has_37_unique_case_studies_and_one_falcon():
    records=p.paper_cohort()
    ids={r['reservoir_id'] for r in records}
    assert len(records)==len(ids)==37
    assert sum(r['reservoir_id']=='ex10_Falcon' for r in records)==1
    assert 'ex40_Semione' in ids and 'ex30_Big_horn' in ids
    active={r['reservoir_id'] for r in w.read(ROOT/'data/reservoirs/manifest.json')['reservoirs']}
    assert ids<=active
    assert active-ids=={'ex21_Travers','ex22_Mcgregor','ex27_Glenmore','ex31_Brazeau'}
    assert 'ex43_TuyenQuang' not in ids

def test_floor_hypotheses_are_declared_lower_values_not_fitted_scores():
    assert p.floor_candidates(.1)==[.1,.05,0.]
    assert p.floor_candidates(.4714937391304348)==[.4714937391304348,.1,.05,0.]
    assert p.floor_candidates(0.)==[0.]

def test_above_nominal_observations_do_not_raise_capacity_or_leak_assessment():
    obs,cfg,plan,record=tok()
    assert obs.frame.storage_m3.gt(cfg['model']['capacity_m3']).any()
    original=obs.frame.copy(deep=True)
    initial,_,attempts=p.choose_treatment(obs,cfg,plan,record)
    assert initial is not None
    assert initial.capacity_m3==cfg['model']['capacity_m3']
    assert attempts[-1]['feasible'] and not attempts[-1]['assessment_used']
    assert attempts[-1]['independent_balance_residual_m3']<=cfg['atol_storage_m3']
    pd.testing.assert_frame_equal(original,obs.frame)
    changed=obs.frame.copy()
    changed.loc[plan.mask(changed.index,'assessment'),['storage_m3','release_m3s']]*=100
    other,_,other_attempts=p.choose_treatment(_observations(changed),cfg,plan,record)
    assert np.array_equal(parameters_to_vector(initial),parameters_to_vector(other))
    assert initial.dead_storage_fraction==other.dead_storage_fraction
    assert [(r['capacity_m3'],r['dead_storage_fraction'],r.get('initializer_kind'),r['feasible']) for r in attempts]==[
        (r['capacity_m3'],r['dead_storage_fraction'],r.get('initializer_kind'),r['feasible']) for r in other_attempts]

def test_signed_forcing_requires_an_adapter_without_clipping():
    obs,cfg,plan,record=tok()
    changed=obs.frame.copy();changed.iloc[2,changed.columns.get_loc('inflow_m3s')]=-1.
    with pytest.raises(ValueError,match='Signed forcing'):
        p.choose_treatment(_observations(changed),cfg,plan,record)
    assert changed.iloc[2].inflow_m3s==-1.
