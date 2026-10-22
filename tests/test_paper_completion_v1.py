"""Independent conservation, source masks, year selection and resume invariants."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from dztr.schemas import ReservoirParameters, MONTHLY_TARGET_COLUMNS
from dztr.calibration.sampled import _observations
from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.optimizer import optimize_nsga2, NSGA2Settings
from dztr.calibration.checkpoint import save_optimizer_checkpoint, load_optimizer_checkpoint
from dztr.simulation.root_policy import simulate_root_policy_reservoir
from dztr.simulation.signed_net import SignedNetPolicy, simulate_signed_net_reservoir
from dztr.simulation.sampled import SamplingSettings
from strategies.signed_years_v1 import SignedYearObjective, select_signed_years
from strategies.representative_years import select_representative_years, fit_selected_years
from strategies.fleet_comparison_v2 import comparison_selections
from workflows.paper_completion_v1 import canyon_profile, trinity_quality_profile, require_calibration_ready


def parameters(initial=4e8,release=1.,floor=.1,flood_mode="storage"):
    targets = pd.DataFrame(np.tile([2e8,5e8,8e8,1.,2.,3.],(12,1)),
        index=pd.Index(range(1,13),name="month"),columns=MONTHLY_TARGET_COLUMNS)
    return ReservoirParameters(targets,1e9,initial,release,20.,floor,1.,flood_mode)


def observations(net,index=None):
    index = pd.date_range("2000-01-01",periods=len(net),name="date") if index is None else index
    return _observations(pd.DataFrame({"storage_m3":np.linspace(3e8,5e8,len(net)),
        "release_m3s":np.linspace(1.,2.,len(net)),"inflow_m3s":net},index=index))


@pytest.mark.parametrize("sampling,index",[
    (SamplingSettings(),pd.date_range("2000-01-01",periods=12,name="date")),
    (SamplingSettings(frequency="monthly"),pd.date_range("2000-01-01",periods=12,freq="MS",name="date")),
    (SamplingSettings(frequency="monthly",fixed_step_days=30.5),pd.date_range("2000-01-01",periods=12,freq="MS",name="date"))])
def test_nonnegative_native_solver_arithmetic_is_preserved(sampling,index):
    obs = observations(np.linspace(0,3,len(index)),index)
    expected = simulate_root_policy_reservoir(obs,parameters(),sampling=sampling,policy="nearest_previous_storage")
    actual = simulate_signed_net_reservoir(obs,parameters(),sampling=sampling,forcing_policy=SignedNetPolicy())
    for col in ("storage_m3","release_m3s","balance_residual_m3","overflow_volume_m3","zone"):
        np.testing.assert_array_equal(expected.frame[col],actual.frame[col])


def test_signed_losses_preserve_literal_trapezoidal_volume_below_floor():
    # Below the operating floor Q=0: exact solution is cumulative signed volume.
    net = np.array([1.,-2.,-3.,4.,-1.])
    params = parameters(initial=8e7,release=0.)
    result = simulate_signed_net_reservoir(observations(net),params,forcing_policy=SignedNetPolicy())
    expected = np.r_[params.initial_storage_m3,params.initial_storage_m3+np.cumsum(43200*(net[1:]+net[:-1]))]
    np.testing.assert_array_equal(result.frame.storage_m3,expected)
    np.testing.assert_array_equal(result.frame.inflow_m3s,net)
    np.testing.assert_array_equal(result.frame.positive_net_component_m3s-result.frame.negative_net_component_m3s,net)
    assert not result.frame.release_m3s.any()


def test_conservative_spill_and_negative_flux_independent_balance():
    obs = observations([4000.,4000.,-1.,2.])
    params = parameters(initial=9e8)
    result = simulate_signed_net_reservoir(obs,params,forcing_policy=SignedNetPolicy()).frame
    residual = np.diff(result.storage_m3)-43200*(np.asarray(obs.frame.inflow_m3s)[1:]+
        np.asarray(obs.frame.inflow_m3s)[:-1]-np.asarray(result.release_m3s)[1:]-np.asarray(result.release_m3s)[:-1])
    assert result.overflow_volume_m3.sum()>0
    assert np.abs(residual).max()<=1e-4
    assert result.storage_m3.max()<=params.capacity_m3


def test_inflow_driven_rule_needs_explicit_net_proxy_and_no_loss_rescaling():
    obs = observations([0.,-1.,2.]); params = parameters(flood_mode="inflow")
    with pytest.raises(ValueError,match="proxy hypothesis"):
        simulate_signed_net_reservoir(obs,params,forcing_policy=SignedNetPolicy())
    simulate_signed_net_reservoir(obs,params,forcing_policy=SignedNetPolicy(flood_rule_driver="signed_net_proxy"))
    with pytest.raises(ValueError,match="rescaled"):
        simulate_signed_net_reservoir(obs,replace(params,inflow_correction=1.1),forcing_policy=SignedNetPolicy(flood_rule_driver="signed_net_proxy"))


def test_impossible_net_volume_remains_a_failure():
    with pytest.raises(ValueError,match="negative storage"):
        simulate_signed_net_reservoir(observations([-1e9,-1e9]),parameters(),forcing_policy=SignedNetPolicy())
    with pytest.raises(ValueError,match="Missing net forcing"):
        simulate_signed_net_reservoir(observations([0.,np.nan]),parameters(),forcing_policy=SignedNetPolicy())


def test_canyon_calendar_reconstruction_leaves_gap_and_unverified_release():
    cross = pd.DataFrame({"provider_calendar_month":["2000-09-01","2001-01-01"],
        "provider_volume_calendar_mean_inflow_m3s":[12.,14.],
        "canonical_storage_m3_unmodified":[2e9,2.1e9],
        "canonical_release_m3s_unmodified_not_reinterpreted":[11.,13.]})
    result = canyon_profile(cross)
    assert len(result)==5
    assert result.index[~result.source_row_available].strftime("%Y-%m").tolist()==["2000-10","2000-11","2000-12"]
    assert result.release_m3s.isna().all() and not result.calibration_eligible.any()
    assert result.loc["2001-01-01","release_m3s_unverified"]==13.
    assert result.loc["2000-10-01","inflow_m3s"]!=result.loc["2000-10-01","inflow_m3s"]


def test_trinity_invalid_storage_masks_preserve_source_flags_signed_flux_and_surcharge():
    index = pd.date_range("2000-01-01",periods=5,name="date")
    raw = pd.DataFrame({"storage_m3":[3.2e9,-1.,np.nan,3e9,3e9+86400],
        "inflow_m3s":[2.,-2.,np.nan,2.,2.],"release_m3s":[1.,1.,1.,1.,1.],
        "storage_m3_data_flag":["r","","","e",""]},index=index)
    original = raw.copy(deep=True); result = trinity_quality_profile(raw)
    pd.testing.assert_frame_equal(raw,original)
    np.testing.assert_array_equal(result.source_storage_m3,raw.storage_m3)
    assert result.loc[index[1],"inflow_m3s"]==-2.
    assert result.storage_m3.iloc[0]==3.2e9  # Above normal pool stays an observation.
    assert result.storage_m3.iloc[1:3].isna().all()
    assert result.source_daily_mean_closure_residual_m3.iloc[4]==0.
    assert result.storage_m3_data_flag.tolist()==raw.storage_m3_data_flag.tolist()


def monthly_history(signed=False):
    index = pd.date_range("2000-01-01","2014-12-01",freq="MS",name="date")
    y = index.year.to_numpy()-2000; m = index.month.to_numpy()
    net = 3.+np.sin(y)+.1*m
    if signed: net[25]=-2.
    return _observations(pd.DataFrame({"storage_m3":(0.4+0.08*np.sin(y)+.005*m)*1e9,
        "release_m3s":2.+.5*np.sin(y)+.03*m,"inflow_m3s":net},index=index))


def test_signed_year_selector_matches_existing_nonnegative_protocol_exactly():
    obs = monthly_history(); sampling = SamplingSettings(frequency="monthly",fixed_step_days=30.5)
    _,annual,_ = select_representative_years(obs,sampling)
    expected,_ = comparison_selections(obs.frame,annual)
    actual,_,_ = select_signed_years(obs,sampling)
    assert actual.as_dict()==expected["mixed_pooled"].as_dict()
    signed = monthly_history(True); plan,table,_ = select_signed_years(signed,sampling)
    part = signed.frame.loc["2002"]
    assert table.loc[2002,"inflow_mean_m3s"]==pytest.approx(np.average(part.inflow_m3s,weights=part.index.days_in_month))
    assert plan.roles.count("calibration")/len(plan.roles)>=.65
    assert set(plan.table().loc[plan.table().role.eq("calibration"),"regime"])=={"dry","normal","wet"}


def make_objective(obs=None):
    obs = monthly_history(True) if obs is None else obs
    sampling = SamplingSettings(frequency="monthly",fixed_step_days=30.5)
    plan,_,_ = select_signed_years(obs,sampling)
    constants = {k:getattr(parameters(),k) for k in ("capacity_m3","initial_storage_m3","initial_release_m3s",
        "channel_capacity_m3s","dead_storage_fraction","inflow_correction","flood_mode")}
    template,bounds = fit_selected_years(obs,plan,constants,allow_feasible_initialization=True)
    return SignedYearObjective(obs,plan,template,bounds=bounds,sampling=sampling,
        cost_mode="pooled_nse",forcing_policy=SignedNetPolicy(),policy="nearest_previous_storage")


def test_held_out_responses_cannot_change_preparation_or_calibration_costs():
    first = make_objective(); frame = first._source.frame.copy()
    mask = first.selection.mask(frame.index,"assessment")
    frame.loc[mask,["storage_m3","release_m3s"]]*=100.
    other = make_objective(_observations(frame))
    np.testing.assert_array_equal(parameters_to_vector(first.template),parameters_to_vector(other.template))
    assert first.selection.signature==other.selection.signature
    assert first.evaluate(parameters_to_vector(first.template)).objectives==other.evaluate(parameters_to_vector(other.template)).objectives


def test_fresh_signed_checkpoint_continuation_matches_uninterrupted_stream(tmp_path):
    objective = make_objective(); baseline = parameters_to_vector(objective.template)
    assert objective.evaluate(baseline).feasible
    identity = {"objective":objective.identity,"seed":42,"population":4}
    path = tmp_path/"checkpoint.npz"
    optimize_nsga2(objective,baseline,NSGA2Settings(seed=42,population_size=4,evaluation_budget=8),
        checkpoint_callback=lambda state:save_optimizer_checkpoint(path,state,identity))
    state = load_optimizer_checkpoint(path,identity)
    resumed = optimize_nsga2(objective,baseline,NSGA2Settings(seed=42,population_size=4,evaluation_budget=12),resume_state=state)
    full = optimize_nsga2(objective,baseline,NSGA2Settings(seed=42,population_size=4,evaluation_budget=12))
    np.testing.assert_array_equal([v.vector for v in resumed.evaluated],[v.vector for v in full.evaluated])
    np.testing.assert_array_equal([v.objectives for v in resumed.evaluated],[v.objectives for v in full.evaluated])


def test_source_gate_is_enforced_before_calibration():
    with pytest.raises(ValueError,match="not calibration ready"):
        require_calibration_ready({"reservoir_id":"ex26_Ghost","calibration_ready":False,"gates":["Storage clock"]})
