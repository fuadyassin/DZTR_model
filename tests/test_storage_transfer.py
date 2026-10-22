"""Independent conservation identity and declared divergence screen."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from diagnose_storage_transfer import difference_budget, first_sustained, OUT
from simulate_calibrated_reservoirs import ROOT, read, sha


def test_release_difference_integrates_to_storage_separation():
    dates=pd.date_range('2000-01-01',periods=4,freq='MS',name='date')
    base=pd.DataFrame({'storage_m3':[100.,100.,100.,100.],'release_m3s':[1.,1.,1.,1.],
                       'inflow_m3s':[1.,1.,1.,1.],'step_seconds':[10.]*4,'balance_residual_m3':[0.]*4},index=dates)
    selected=base.copy();selected.release_m3s=[1.,3.,3.,1.];selected.storage_m3=[100.,90.,70.,60.]
    out=difference_budget(selected,base)
    np.testing.assert_array_equal(out.storage_difference_m3,[0.,-10.,-30.,-40.])
    np.testing.assert_array_equal(out.cumulative_extra_release_volume_m3,[0.,10.,30.,40.])
    selected.inflow_m3s+=1
    with pytest.raises(AssertionError):difference_budget(selected,base)


def test_initial_carryover_and_residual_are_not_attributed_to_release():
    dates=pd.date_range('2000-01-01',periods=2,name='date')
    base=pd.DataFrame({'storage_m3':[100.,100.],'release_m3s':[1.,1.],'inflow_m3s':[1.,1.],
                       'step_seconds':[10.,10.],'balance_residual_m3':[0.,0.]},index=dates)
    selected=base.copy();selected.storage_m3=[90.,90.00001];selected.balance_residual_m3=[0.,.00001]
    out=difference_budget(selected,base)
    np.testing.assert_allclose(out.budget_reconstructed_difference_m3,[-10.,-9.99999],rtol=0,atol=1e-10)


def test_sustained_screen_requires_consecutive_months():
    dates=pd.date_range('2000-01-01',periods=9,freq='MS')
    assert first_sustained(pd.Series([True,True,False,True,True,True,True,True,True],index=dates))==dates[3]
    assert first_sustained(pd.Series([True]*5,index=dates[:5])) is None


def test_saved_diagnosis_receipt_and_zero_new_simulations():
    if not OUT.exists():pytest.skip('Independent analysis artifact not generated')
    manifest=read(OUT/'analysis_manifest.json')
    for relative,digest in manifest['artifact_sha256'].items():assert sha(OUT/relative)==digest
    # The external MATLAB writer was checked at analysis time and its excerpt
    # is pinned above. Repository regression must remain portable without E:.
    for path,digest in manifest['consumed_sha256'].items():
        if Path(path).is_relative_to(ROOT):assert sha(Path(path))==digest
    assert read(OUT/'summary.json')['simulations']==0
    for item in read(OUT/'summary.json')['reservoirs']:
        assert item['maximum_budget_reconstruction_error_m3']<1e-3
        assert item['demonstrated_unit_or_row_shift_error'] is False
