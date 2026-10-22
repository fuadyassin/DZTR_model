"""Isolation of target intervention, conservation and calibration-only outputs."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from run_critical_interventions import validate_intervention,independent_balance,OUT
from simulate_calibrated_reservoirs import ROOT,read,sha
from dztr.schemas import MONTHLY_TARGET_COLUMNS


def test_intervention_changes_only_critical_release_to_exact_baseline():
    index=pd.Index(range(1,13),name='month')
    selected=pd.DataFrame(np.ones((12,6)),index=index,columns=MONTHLY_TARGET_COLUMNS)
    baseline=selected.copy();baseline.release_critical_m3s=.5
    restored=selected.copy();restored.release_critical_m3s=.5
    validate_intervention(restored,selected,baseline)
    restored.loc[1,'release_normal_m3s']=1.1
    with pytest.raises(AssertionError):validate_intervention(restored,selected,baseline)
    restored=selected.copy()
    with pytest.raises(AssertionError):validate_intervention(restored,selected,baseline)


def test_balance_uses_correction_and_rejects_falsified_diagnostic():
    frame=pd.DataFrame({'storage_m3':[100.,110.],'release_m3s':[1.,1.],'inflow_m3s':[1.,1.],'step_seconds':[10.,10.],'balance_residual_m3':[0.,0.]})
    assert independent_balance(frame,2.)==0
    with pytest.raises(AssertionError):independent_balance(frame,1.)


def test_fixed_experiment_has_two_calls_and_no_heldout_rows():
    if not OUT.exists():pytest.skip('Independent experiment not generated')
    summary=read(OUT/'experiment_summary.json')
    assert summary['status']=='passed' and summary['forward_simulation_calls']==2
    assert summary['optimization_evaluations']==0 and not summary['heldout_simulated']
    for job in summary['jobs']:
        folder=OUT/job['reservoir_id'];frame=pd.read_csv(folder/'intervention_simulation.csv',index_col='date',parse_dates=True,float_precision='round_trip')
        assert len(frame)==180 and frame.index[-1]==pd.Timestamp('1984-12-01')
        assert set(frame.period)=={'warmup','calibration'} and not job['profile_promoted']
        assert job['maximum_balance_residual_m3']<=1e-4
        assert job['calibration_storage_nse_change_vs_selected']<0 and job['calibration_release_nse_change_vs_selected']<0
    proof=read(OUT/'execution_manifest.json')
    for relative,digest in proof['artifact_sha256'].items():assert sha(OUT/relative)==digest
    for relative,digest in proof['source_sha256'].items():assert sha(ROOT/relative)==digest
