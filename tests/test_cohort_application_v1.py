"""Application dispatch, preprocessing, state isolation and diagnostic semantics."""
from dataclasses import asdict
import numpy as np
import pandas as pd
import pytest
from dztr.schemas import ReservoirParameters,MONTHLY_TARGET_COLUMNS
from dztr.simulation.signed_net import SignedNetPolicy
from dztr.simulation.sampled import SamplingSettings
from workflows import profile_application_v1 as app
from workflows.cohort_assessment_v1 import pair_stats,within_block_error_slope,baseline_metrics
from workflows import reservoir_v2 as w


def model(signed=False):
    targets=pd.DataFrame(np.tile([2e8,5e8,8e8,1.,2.,3.],(12,1)),
        columns=MONTHLY_TARGET_COLUMNS,index=pd.Index(range(1,13),name="month"))
    parameters=ReservoirParameters(targets,1e9,4e8,1.,20.,.1,1.,"storage")
    metadata={"initial_date":"2000-01-01","sampling":asdict(SamplingSettings()),
        "root_policy":"nearest_previous_storage","atol_storage_m3":1e-4}
    if signed: metadata.update(route="signed_net_native_v1",forcing_policy=asdict(SignedNetPolicy()))
    return metadata,parameters


def forcing(values=(1.,2.,3.)):
    return pd.DataFrame({"inflow_m3s":values},index=pd.date_range("2000-01-01",periods=len(values),name="date"))


def test_canonical_transform_is_applied_once_and_prepared_input_is_unchanged():
    raw=forcing(); original=raw.copy(deep=True)
    prepared=app.normalize_forcing(raw,stage="canonical",factor=1.45,offset=8.)
    np.testing.assert_array_equal(prepared.inflow_m3s,raw.inflow_m3s*1.45+8.)
    untouched=app.normalize_forcing(prepared,stage="prepared",factor=1.45,offset=8.)
    pd.testing.assert_frame_equal(prepared,untouched)
    pd.testing.assert_frame_equal(raw,original)
    metadata,parameters=model()
    one,_=app.run_frame(raw,metadata,parameters,forcing_stage="canonical",preprocessing={"factor":1.45,"offset_m3s":8.})
    two,_=app.run_frame(prepared,metadata,parameters,forcing_stage="prepared",preprocessing={"factor":1.45,"offset_m3s":8.})
    pd.testing.assert_frame_equal(one.frame,two.frame)


@pytest.mark.parametrize("bad_stage",[None,"automatic","raw",True])
def test_input_stage_cannot_be_guessed(bad_stage):
    with pytest.raises(ValueError,match="Declare"):
        app.normalize_forcing(forcing(),stage=bad_stage,factor=1.,offset=0.)


def test_signed_route_accepts_net_losses_and_standard_route_rejects_them():
    frame=forcing((1.,-2.,3.)); metadata,parameters=model(True)
    result,residual=app.run_frame(frame,metadata,parameters,forcing_stage="prepared",preprocessing={"factor":1.,"offset_m3s":0.})
    np.testing.assert_array_equal(frame.inflow_m3s,result.frame.inflow_m3s)
    assert residual<=1e-4
    standard,_=model()
    with pytest.raises(ValueError,match="nonnegative"):
        app.run_frame(frame,standard,parameters,forcing_stage="prepared",preprocessing={"factor":1.,"offset_m3s":0.})


def test_observed_responses_never_initialize_or_reset_application_state():
    metadata,parameters=model()
    frame=forcing(); frame["storage_m3"]=[1.,2.,3.]; frame["release_m3s"]=[100.,200.,300.]
    actual,_=app.run_frame(frame,metadata,parameters,forcing_stage="prepared",preprocessing={"factor":1.,"offset_m3s":0.})
    frame[["storage_m3","release_m3s"]]*=1e6
    changed,_=app.run_frame(frame,metadata,parameters,forcing_stage="prepared",preprocessing={"factor":1.,"offset_m3s":0.})
    pd.testing.assert_frame_equal(actual.frame,changed.frame)
    assert actual.frame.storage_m3.iloc[0]==parameters.initial_storage_m3


def test_changed_start_requires_explicit_initial_state_and_si_validation():
    metadata,parameters=model(); frame=forcing(); frame.index+=pd.Timedelta(days=5)
    options={"forcing_stage":"prepared","preprocessing":{"factor":1.,"offset_m3s":0.}}
    with pytest.raises(ValueError,match="explicit initial"):
        app.run_frame(frame,metadata,parameters,**options)
    result,_=app.run_frame(frame,metadata,parameters,initial={"storage_m3":3e8,"release_m3s":1.},**options)
    assert result.frame.storage_m3.iloc[0]==3e8
    with pytest.raises(ValueError,match="finite SI"):
        app.run_frame(frame,metadata,parameters,initial={"storage_m3":True,"release_m3s":1.},**options)


def test_resolver_uses_verified_ids_and_refuses_ambiguity_or_unknown_ids():
    rows=[{"reservoir_id":"one","GRAND_ID":753},{"reservoir_id":"two","GRAND_ID":None}]
    catalog={"reservoirs":rows}
    assert app.resolve_entry(catalog,grand_id=753)["reservoir_id"]=="one"
    assert app.resolve_entry(catalog,reservoir_id="two")["reservoir_id"]=="two"
    for kwargs in ({},{"grand_id":753,"reservoir_id":"one"},{"grand_id":True},{"grand_id":999},{"reservoir_id":"guessed_name"}):
        with pytest.raises(ValueError): app.resolve_entry(catalog,**kwargs)


def test_source_gated_and_conditional_interval_cases_cannot_be_simulated_as_native(monkeypatch):
    for status in ("source_definition_required","calibrated_separate_interval_workflow"):
        entry={"reservoir_id":"case","GRAND_ID":None,"monthly_profile_available":False,
            "status":status,"next_action":"Use its separate source/interval route"}
        monkeypatch.setattr(app,"read_catalog",lambda path:{"reservoirs":[entry]})
        with pytest.raises(ValueError,match=status): app.load_profile("unused",reservoir_id="case")


def test_missing_forcing_and_calendar_gaps_are_not_filled():
    metadata,parameters=model()
    with pytest.raises(ValueError,match="finite"):
        app.run_frame(forcing((1.,np.nan,2.)),metadata,parameters,forcing_stage="prepared",preprocessing={"factor":1.,"offset_m3s":0.})
    frame=forcing().iloc[[0,2]]
    with pytest.raises(ValueError,match="consecutive"):
        app.run_frame(frame,metadata,parameters,forcing_stage="prepared",preprocessing={"factor":1.,"offset_m3s":0.})


def test_pair_metrics_handle_constant_fractional_values_and_missing_pairs():
    result=pair_stats([1.,2.,np.nan],[2.,4.,999.])
    assert result["pairs"]==2 and result["pbias"]==100.
    assert result["nse"]==-9.
    assert np.isnan(pair_stats([.1]*10,[.2]*10)["nse"])
    assert pair_stats([np.nan],[1.])["pairs"]==0


def test_storage_error_trends_do_not_connect_unscored_gaps():
    index=pd.date_range("2000-01-01",periods=8,name="date")
    error=np.array([0.,1.,2.,np.nan,np.nan,1000.,1001.,1002.])
    mask=np.array([1,1,1,0,0,1,1,1],bool)
    slope=within_block_error_slope(index,error,mask)
    assert slope==pytest.approx(365.2425)
    # A context gap must not turn an intercept change into a large drift rate.
    assert slope < 400.


def test_infeasible_archived_baseline_is_not_replaced_or_assigned_nse(tmp_path):
    parent=tmp_path/"assessed/case/mixed_pooled";parent.mkdir(parents=True)
    jobs=tmp_path/"jobs/case/mixed_pooled";(jobs/"seed_42").mkdir(parents=True)
    pd.DataFrame([{"evaluation_id":0,"feasible":False,"reason":"No feasible implicit root"}]).to_csv(jobs/"seed_42/trials.csv",index=False)
    kind,metrics,calls,path=baseline_metrics({"identifier":"case","legacy":True,"archive":tmp_path,"jobs":jobs},None,None,None)
    assert kind=="initial_reference_infeasible" and calls==0
    assert np.isnan(metrics["calibration"]["storage_nse"])
    assert np.isnan(metrics["assessment"]["release_nse"])
    assert metrics["calibration"]["reason"]=="No feasible implicit root"


def test_catalog_checksum_and_source_drift_are_refused(tmp_path):
    path=tmp_path/"catalog.json"
    w.write(path,{"kind":"paper_profile_application_v1","application_source_sha256":{},"reservoirs":[]})
    w.write(tmp_path/"manifest.json",{"artifact_sha256":{"catalog.json":"0"*64}})
    with pytest.raises(ValueError,match="checksum"): app.read_catalog(path)
    w.write(path,{"kind":"paper_profile_application_v1","application_source_sha256":{"workflows/profile_application_v1.py":"0"*64},"reservoirs":[]})
    w.write(tmp_path/"manifest.json",{"artifact_sha256":{"catalog.json":w.sha(path)}})
    with pytest.raises(ValueError,match="source changed"): app.read_catalog(path)


def test_external_forcing_requires_explicit_stage_before_any_simulation(monkeypatch,tmp_path):
    metadata,parameters=model()
    monkeypatch.setattr(app,"load_profile",lambda *a,**kw:({},metadata,parameters))
    with pytest.raises(ValueError,match="Explicit forcing requires"):
        app.simulate("unused",tmp_path/"output",reservoir_id="case",forcing_csv="unused.csv")
