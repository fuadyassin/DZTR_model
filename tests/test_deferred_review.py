"""Saved review accountability and calibration-only candidate evidence."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
from workflows import reservoir as w
from dztr.calibration.sampled import _observations
from dztr.evaluation import evaluate_simulation

ROOT=Path(__file__).parents[1]


def test_review_accounts_for_every_active_reservoir_and_preserves_gates():
    folder=ROOT/'outputs/deferred_readiness_20261013'
    if not folder.exists():pytest.skip('Independent readiness review not produced')
    review=json.loads((folder/'readiness_review.json').read_text())
    manifest=json.loads((ROOT/'data/reservoirs/manifest.json').read_text())
    reference=set(json.loads((ROOT/'configs/calibrated_twenty.json').read_text())['profiles_by_reservoir_id'])
    records={r['reservoir_id']:r for r in review['records']}
    assert len(records)==21 and not reference&set(records)
    assert reference|set(records)=={e['reservoir_id'] for e in manifest['reservoirs']}
    assert 'ex43_TuyenQuang' not in records and review['new_calibration_admissions']==0
    assert all(r['standard_calibration_admitted'] is False for r in records.values())
    for r in records.values():
        if r['calibration_fraction'] is not None:assert r['calibration_fraction']>=.65
    assert records['ex35_Bull_Lake']['current_status']=='calibrated_separate_interval_workflow'
    assert records['ex22_Mcgregor']['current_status']=='short_record_protocol_required'
    for identifier,count in [('ex8_Flaming_gorge',4),('ex33_Navajo',21),('ex40_Semione',2)]:
        assert records[identifier]['full_record_screen_audit_only']['negative_inflow']==count
        assert records[identifier]['preparation_status']=='year_selection_or_input_gate_failed'


def test_policy_preflight_scores_only_selected_calibration_years():
    folder=ROOT/'outputs/deferred_policy_preflight_20261013'
    if not folder.exists():pytest.skip('Independent candidate preflight not produced')
    summary=json.loads((folder/'summary.json').read_text());record=summary['results'][0]
    assert record['feasible'] and record['assessment_used'] is False
    assert summary['optimization_evaluations']==summary['held_out_evaluations']==0
    frame=pd.read_csv(folder/'ex42_Tyuyamuyun_calibration_simulation.csv',index_col='date',parse_dates=True,float_precision='round_trip')
    selected=json.loads((ROOT/'outputs/deferred_readiness_20261013/ex42_Tyuyamuyun/proposed_year_selection.json').read_text())
    dates=pd.DatetimeIndex(pd.to_datetime(selected['dates']))
    mask=np.asarray(selected['roles'])=='calibration'
    source=pd.read_csv(ROOT/'data/reservoirs/ex42_Tyuyamuyun/observations_si.csv',index_col='date',parse_dates=True,float_precision='round_trip')
    metrics=evaluate_simulation(_observations(source.loc[dates[mask]]),SimpleNamespace(frame=frame.loc[dates[mask]]))
    for variable in ('storage','release'):
        np.testing.assert_allclose(metrics[variable+'_nse'],record['calibration_metrics'][variable+'_nse'],rtol=0,atol=1e-12)
    assert int(frame.root_count.gt(1).sum())==5
    assert w.independent_balance(frame,1.)<=1e-4
    assert frame.index.max().year==2010 and 2009 in frame.index.year
    assert not mask[dates.year==2009].any()  # Carry forcing/state, never score held-out observations.
