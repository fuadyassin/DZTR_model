"""Paper-source completion routes; immutable archives and explicit admission gates.

Builds source profiles rather than silently changing canonical data. Numerical
admission and source certification are separate. No search runs during prepare.
"""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd

from workflows import reservoir_v2 as w
from workflows.paper_exploration import paper_cohort, floor_candidates
from strategies.representative_years import YearSelection, select_representative_years, fit_selected_years
from strategies.fleet_comparison_v2 import comparison_selections
from strategies.signed_years_v1 import SignedYearObjective, select_signed_years
from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.sampled import _observations
from dztr.integrations.world_reservoirs import _forcing
from dztr.schemas import ReservoirParameters
from dztr.simulation.sampled import SamplingSettings
from dztr.simulation.signed_net import SignedNetPolicy
from tools.fleet_admission import ordered_candidate

ROOT = w.ROOT
SIGNED_IDS = ("ex40_Semione", "ex8_Flaming_gorge", "ex33_Navajo")
ADDITIONS = ("workflows/paper_completion_v1.py", "strategies/signed_years_v1.py",
             "tools/paper_completion_v1.py")


def code_hashes():
    return {**w.source_hashes(), **{p:w.sha(ROOT/p) for p in ADDITIONS}}


def canonical(identifier):
    cfg_path = ROOT/"configs/reservoirs"/(identifier+".json")
    cfg = w.read(cfg_path)
    path = ROOT/"data/reservoirs"/identifier/"observations_si.csv"
    if w.sha(path) != cfg["observations_sha256"]:
        raise ValueError("Canonical observations changed: "+identifier)
    frame = pd.read_csv(path,index_col="date",parse_dates=True,float_precision="round_trip")
    return cfg, _forcing(_observations(frame), cfg["inflow_preprocessing"]), [cfg_path,path]


def verified_artifacts(base, mapping):
    base = Path(base).resolve(strict=True)
    for relative,digest in mapping.items():
        path = (base/relative).resolve(strict=True)
        if not path.is_relative_to(base) or w.sha(path) != digest:
            raise ValueError("Archived evidence changed: "+relative)


def signed_preflight(identifier, folder):
    cfg, obs, consumed = canonical(identifier)
    folder.mkdir()
    sampling = SamplingSettings(**cfg["sampling"])
    plan, annual, _ = select_signed_years(obs,sampling)
    forcing_policy = SignedNetPolicy(flood_rule_driver=("signed_net_proxy" if cfg["model"]["flood_mode"] == "inflow" else "storage_only"))
    frame = obs.frame
    frame.to_csv(folder/"observations_si.csv",index_label="date",float_format="%.17g")
    ledger = frame.loc[frame.inflow_m3s.lt(0)].copy()
    ledger["year_role"] = plan.table().loc[ledger.index,"role"]
    ledger.to_csv(folder/"signed_records.csv",index_label="date",float_format="%.17g")
    w.write(folder/"selection.json",plan.as_dict())
    annual.to_csv(folder/"annual_net_flux_regimes.csv",float_format="%.17g")
    plan.table().to_csv(folder/"row_selection.csv",index_label="date")
    attempts = []; chosen = None; calls = 0
    for floor in floor_candidates(cfg["model"]["dead_storage_fraction"]):
        constants = dict(cfg["model"],dead_storage_fraction=floor)
        try:
            template,bounds = fit_selected_years(obs,plan,constants,allow_feasible_initialization=True)
        except ValueError as error:
            attempts.append({"floor":floor,"stage":"preparation","feasible":False,"reason":str(error)})
            continue
        points = [("prepared_initializer",template)]
        monthly = bounds.empirical.monthly_bounds
        gap = ordered_candidate(monthly.iloc[:,:6].to_numpy(),monthly.iloc[:,6:].to_numpy(),
            template.capacity_m3,floor,(sampling.fixed_step_days or 1.)*86400.,close_critical_gap=True)
        if gap["feasible"]:
            candidate = ReservoirParameters(pd.DataFrame(gap["monthly_targets_si"],index=template.monthly_targets.index,
                                                          columns=template.monthly_targets.columns),**constants)
            if not np.array_equal(parameters_to_vector(candidate),parameters_to_vector(template)):
                points.append(("critical_gap_witness",bounds.validate(parameters_to_vector(candidate),template)))
        for kind, initial in points:
            objective = SignedYearObjective(obs,plan,initial,sampling=sampling,bounds=bounds,
                policy="nearest_previous_storage",atol_storage_m3=cfg["atol_storage_m3"],
                cost_mode="pooled_nse",forcing_policy=forcing_policy)
            evaluation = objective.evaluate(parameters_to_vector(initial)); calls += 1
            attempt = {"floor":floor,"initializer":kind,"feasible":evaluation.feasible,
                       "reason":evaluation.reason,"assessment_used":False}
            if evaluation.feasible:
                attempt["maximum_independent_balance_residual_m3"] = w.independent_balance(evaluation.simulation.frame,1.)
                attempt["calibration_metrics"] = evaluation.metrics
            attempts.append(attempt)
            if evaluation.feasible:
                chosen = initial
                initial.monthly_targets.to_csv(folder/"initial_targets_si.csv",index_label="month",float_format="%.17g")
                bounds.empirical.monthly_bounds.to_csv(folder/"bounds_si.csv",float_format="%.17g")
                bounds.generalized_targets.to_csv(folder/"raw_generalized_targets_si.csv",float_format="%.17g")
                model = {"physical_constants":constants,"sampling":asdict(sampling),
                    "forcing_policy":asdict(forcing_policy),"root_policy":"nearest_previous_storage",
                    "atol_storage_m3":cfg["atol_storage_m3"],"cost_mode":"pooled_nse"}
                w.write(folder/"model.json",model)
                break
        if chosen is not None:
            break
    fraction = plan.roles.count("calibration")/len(plan.roles)
    if fraction < .65:
        raise ValueError("Calibration support below 65 percent")
    profile = {"reservoir_id":identifier,"route":"signed_net_native_v1",
        "status":"exploratory_calibration_ready" if chosen is not None else "numerical_preflight_failed",
        "calibration_ready":chosen is not None,"source_certified":False,
        "negative_inflow_records":len(ledger),"calibration_fraction":fraction,
        "forcing_policy":asdict(forcing_policy),"observed_storage_outside_model_limits_retained":True,
        "net_components_are_measured_gross_flows":False,"selection_uses_assessment":False,
        "clock_is_provisional":True,"source_derivation_is_unresolved":True,
        "new_preflight_model_calls":calls,"attempts":attempts,
        "next_action":"Run declared checkpointed searches, retaining source/clock limitations." if chosen is not None else "Review recorded failed timestep and physical parameters; no root is fabricated."}
    w.write(folder/"source_profile.json",profile)
    return profile, consumed


def canyon_profile(crosswalk):
    """Reconstruct calendar and verified inflow rates; refuse unknown releases."""
    required = {"provider_calendar_month","provider_volume_calendar_mean_inflow_m3s",
        "canonical_storage_m3_unmodified","canonical_release_m3s_unmodified_not_reinterpreted"}
    if not required <= set(crosswalk):
        raise ValueError("Incomplete Canyon Ferry crosswalk")
    source = crosswalk.copy()
    source.index = pd.DatetimeIndex(pd.to_datetime(source.provider_calendar_month),name="date")
    if not source.index.is_unique or not source.index.is_monotonic_increasing:
        raise ValueError("Calendar crosswalk must be one-to-one and ordered")
    index = pd.date_range(source.index[0],source.index[-1],freq="MS",name="date")
    result = source.reindex(index)
    result["source_row_available"] = index.isin(source.index)
    result["storage_m3"] = result.canonical_storage_m3_unmodified
    result["inflow_m3s"] = result.provider_volume_calendar_mean_inflow_m3s
    # This is a preserved source value, explicitly not a verified model flux.
    result["release_m3s_unverified"] = result.canonical_release_m3s_unmodified_not_reinterpreted
    result["release_m3s"] = np.nan
    result["calibration_eligible"] = False
    factor = 1233.48183754752
    result["declared_curve_capacity_m3"] = np.where(index < pd.Timestamp("1998-10-01"),2052723*factor,1992977*factor)
    result["curve_coordinate_verified"] = False
    return result


def trinity_quality_profile(reference):
    """Keep source values/flags and define explicit, unfilled score support."""
    result = reference.copy(deep=True)
    required = ["storage_m3","inflow_m3s","release_m3s"]
    if not set(required) <= set(result) or not result.index.is_unique:
        raise ValueError("Invalid provider reference")
    for column in required:
        result["source_"+column] = result[column]
    result["storage_score_eligible"] = np.isfinite(result.storage_m3) & result.storage_m3.ge(0)
    result["release_score_eligible"] = np.isfinite(result.release_m3s) & result.release_m3s.ge(0)
    result["forcing_available"] = np.isfinite(result.inflow_m3s)
    result.loc[~result.storage_score_eligible,"storage_m3"] = np.nan
    result.loc[~result.release_score_eligible,"release_m3s"] = np.nan
    result["signed_net_inflow_retained"] = result.inflow_m3s.lt(0)
    previous = result.storage_score_eligible.shift(1,fill_value=False)
    consecutive = result.index.to_series().diff().eq(pd.Timedelta(days=1))
    result["daily_mean_closure_eligible"] = previous & result.storage_score_eligible & result.release_score_eligible & result.forcing_available & consecutive
    mask = result.daily_mean_closure_eligible
    result["source_daily_mean_closure_residual_m3"] = np.nan
    result.loc[mask,"source_daily_mean_closure_residual_m3"] = (
        result.storage_m3.diff() - 86400*(result.inflow_m3s-result.release_m3s)).loc[mask]
    result["calibration_eligible"] = False  # Adapter/clock/anchor decisions still open.
    return result


def source_profiles(out):
    consumed = []
    base = ROOT/"outputs/source_limit_review_v1_20261013"
    cross = base/"canyon_calendar_crosswalk.csv"; evidence = base/"canyon_calendar_evidence.json"
    folder = out/"source_profiles/ex7_CanyonFerry"; folder.mkdir(parents=True)
    frame = canyon_profile(pd.read_csv(cross,float_precision="round_trip"))
    frame.to_csv(folder/"calendar_rate_profile.csv",index_label="date",float_format="%.17g")
    consumed += [cross,evidence]
    records = [{"reservoir_id":"ex7_CanyonFerry","route":"calendar_rate_source_v1",
        "calibration_ready":False,"status":"calendar_repaired_remaining_source_gates",
        "missing_months":[d.date().isoformat() for d in frame.index[~frame.source_row_available]],
        "gates":["Missing October-December 2000 storage and releases", "Release units and aggregation", "Storage stock timing", "Pre/post survey coordinate compatibility and time-dependent capacity adapter"],
        "next_action":"Supply missing source records and flux/stock definitions; execute gate validation before fitting."}]
    w.write(folder/"source_profile.json",records[-1])
    base = ROOT/"outputs/source_reconciliation_20261008"
    reference = base/"trinity_cdec_reference_si.csv"; provenance = base/"trinity_cdec_reference_provenance.json"
    if w.sha(reference) != w.read(provenance)["si_sha256"]:
        raise ValueError("Pinned Trinity provider reference changed")
    folder = out/"source_profiles/ex12_Trinity"; folder.mkdir()
    raw = pd.read_csv(reference,index_col="date",parse_dates=True,float_precision="round_trip")
    frame = trinity_quality_profile(raw)
    frame.to_csv(folder/"provider_quality_profile.csv",index_label="date",float_format="%.17g")
    consumed += [reference,provenance,base/"calibration_gates.json"]
    records.append({"reservoir_id":"ex12_Trinity","route":"provider_daily_mean_source_v1",
        "status":"provider_quality_profile_staged","calibration_ready":False,
        "source_version":w.read(provenance)["retrieval_label"],"storage_coordinate":"total_contents",
        "normal_pool_m3":2447700*1233.48183754752,"physical_dead_storage_m3":10000*1233.48183754752,
        "normal_pool_is_hard_surcharge_limit":False,"storage_clock_supported":"2400 hours historical definition",
        "integration_hypothesis":"Named-day mean fluxes between preceding and current end-day storage",
        "computed_inflow_independent_of_observed_storage_release":False,
        "invalid_storage_scores_masked":int((~frame.storage_score_eligible).sum()),
        "missing_inflow_records":int((~frame.forcing_available).sum()),
        "signed_inflow_records_retained":int(frame.signed_net_inflow_retained.sum()),
        "missing_release_scores":int((~frame.release_score_eligible).sum()),
        "provider_flags_retained":True,"gates":["Quality-block/anchor and missing forcing policy", "Historical feed clock applicability", "Surcharge and operating-floor declaration", "Computed inflow accounting including evaporation"],
        "next_action":"Declare a separate segmented conditional daily-mean route; re-estimate targets on calibration-only valid support."})
    w.write(folder/"source_profile.json",records[-1])
    for identifier,gates in {
        "ex26_Ghost":["Storage datum and level-volume curve/total-live-active coordinate", "Storage observation hour", "Boundary flow coverage and tailrace components", "64.02 versus 65.02 million m3 capacity version"],
        "ex15_Bennett":["39.471 versus 74.3 billion m3 active/gross coordinate", "Operating floor in the observed coordinate"]}.items():
        cfg,obs,paths = canonical(identifier); consumed += paths
        folder = out/"source_profiles"/identifier; folder.mkdir()
        obs.frame.to_csv(folder/"source_observations_si.csv",index_label="date",float_format="%.17g")
        profile = {"reservoir_id":identifier,"route":"unresolved_coordinate_source_v1",
            "status":"source_definition_required","calibration_ready":False,
            "inherited_model_constants_for_reference_only":cfg["model"],"gates":gates,
            "next_action":"Resolve coordinate/clock from source evidence; no capacity substitution or fitted closure correction."}
        records.append(profile); w.write(folder/"source_profile.json",profile)
    return records,consumed


def require_calibration_ready(profile):
    if profile.get("calibration_ready") is not True:
        raise ValueError("Source profile is not calibration ready: "+profile["reservoir_id"]+"; "+str(profile.get("gates",profile.get("attempts"))))


def bull_route(out):
    base = ROOT/"outputs/bull_lake_interval_population_20261009"
    proof = w.read(base/"verification.json")
    verified_artifacts(base,proof["artifact_sha256"])
    checks = w.read(base/"scientific_checks.json")
    verified_artifacts(base/"scientific_checks",checks["artifact_sha256"])
    consumed = [base/"verification.json",base/"scientific_checks.json",base/"job_metrics.csv"]
    cfg,_,paths = canonical("ex35_Bull_Lake"); consumed += paths
    science = pd.read_csv(base/"scientific_checks.csv",float_precision="round_trip")
    consumed.append(base/"scientific_checks.csv")
    folder = out/"interval_profiles/ex35_Bull_Lake"; folder.mkdir(parents=True)
    records = []
    # Preserve each archived calibration representative; never rank on held-out fit.
    for seed in (42,43,44):
        name = f"population_32_seed_{seed}_targets.csv"
        source = base/"scientific_checks"/name; consumed.append(source)
        (folder/name).write_bytes(source.read_bytes())
        trajectory_name = f"population_32_seed_{seed}.csv"
        trajectory = base/"scientific_checks"/trajectory_name; consumed.append(trajectory)
        (folder/trajectory_name).write_bytes(trajectory.read_bytes())
        summary_path = base/f"runs/population_32/seed_{seed}/summary.json"; consumed.append(summary_path)
        summary = w.read(summary_path)
        records.append({"seed":seed,"targets_file":name,"targets_sha256":w.sha(source),
                        "trajectory_file":trajectory_name,"trajectory_sha256":w.sha(trajectory),
                        "physical_constants":cfg["model"],
                        "calibration_metrics":science.loc[(science.population_size==32)&(science.seed==seed)].iloc[0].to_dict(),
                        "summary_reference":str(summary_path.relative_to(ROOT)),"archived_summary":summary})
    route = {"reservoir_id":"ex35_Bull_Lake","route":"segmented_conditional_interval_v1",
        "status":"calibrated_separate_interval_workflow","calibration_ready":False,
        "calibration_completed":True,"standard_profile_available":False,
        "population":32,"seeds":[42,43,44],"evaluations_per_seed":512,
        "held_out_validation_calls":0,"independent_predictive_validation":False,
        "observed_block_anchors":True,"forcing_depends_on_observed_storage_release":True,
        "convergence_established":False,"representatives":records,
        "maximum_interval_balance_residual_m3":checks["maximum_interval_balance_residual_m3"],
        "next_action":"Use archived interval representatives through this explicit route; independent forcing/clock evidence needed for predictive validation."}
    w.write(folder/"route.json",route)
    pd.read_csv(base/"job_metrics.csv",float_precision="round_trip").to_csv(folder/"archived_job_metrics.csv",index=False,float_format="%.17g")
    return route, consumed


def report_standard(out):
    table_path = ROOT/"outputs/paper_exploration_v1_20261013/report/reservoir_parameterization_by_GRAND_ID.csv"
    table = pd.read_csv(table_path,float_precision="round_trip")
    table = table.loc[table.in_published_paper].copy()
    rows = []; registries = {}
    for _,row in table.iterrows():
        path = ROOT/row.profile_registry
        if str(path) not in registries:
            registries[str(path)] = w.verify_registry(path)
        profile = w.read(path.parent/registries[str(path)]["profiles_by_reservoir_id"][row.reservoir_id])
        profile_folder = path.parent/Path(registries[str(path)]["profiles_by_reservoir_id"][row.reservoir_id]).parent
        targets = pd.read_csv(profile_folder/profile["monthly_targets_file"],index_col="month",float_precision="round_trip")
        for month,values in targets.iterrows():
            for name,value in values.items():
                if row[f"{name}_month_{month:02d}"] != value:
                    raise ValueError("Combined table disagrees with frozen monthly targets")
        scores = profile["assessment"]; metrics = scores["metrics"]
        s,q = metrics.get("storage_nse"),metrics.get("release_nse")
        positive = scores["feasible"] and s is not None and q is not None and s > 0 and q > 0
        rows.append({"reservoir_id":row.reservoir_id,"GRAND_ID":row.GRAND_ID,
            "calibration_storage_nse":profile["calibration"].get("storage_nse"),
            "calibration_release_nse":profile["calibration"].get("release_nse"),
            "assessment_storage_nse":s,"assessment_release_nse":q,
            "assessment_feasible":scores["feasible"],"both_assessment_nse_positive":positive,
            "quality_status":"positive_both_assessment_nse" if positive else "performance_review_required",
            "source_certified":profile.get("physical_source_certified",False),
            "convergence_established":profile.get("convergence_established",False),
            "profile_registry":row.profile_registry,"assumptions_file":row.assumptions_file})
    if len(rows) != 29:
        raise ValueError("Expected 29 completed standard paper profiles")
    pd.DataFrame(rows).to_csv(out/"standard_profile_performance.csv",index=False,float_format="%.17g")
    return rows,[table_path,*map(Path,registries)]


def prepare(output):
    out = w.fresh_output(output)
    inventory_path = ROOT/"outputs/paper_parameterization_inventory_v1_20261013/paper_reservoir_parameterization_all_37.csv"
    inventory = pd.read_csv(inventory_path,float_precision="round_trip")
    consumed = [inventory_path]
    standard,paths = report_standard(out); consumed += paths
    bull,paths = bull_route(out); consumed += paths
    source,paths = source_profiles(out); consumed += paths
    signed = []
    (out/"signed_profiles").mkdir()
    for identifier in SIGNED_IDS:
        profile,paths = signed_preflight(identifier,out/"signed_profiles"/identifier)
        signed.append(profile); consumed += paths
    routes = {p["reservoir_id"]:p for p in [bull,*source,*signed]}
    for i,row in inventory.iterrows():
        identifier = row.reservoir_id
        if identifier in routes:
            record = routes[identifier]
            inventory.loc[i,"simulation_route"] = record["route"]
            inventory.loc[i,"programme_status"] = record["status"]
            inventory.loc[i,"next_action"] = record["next_action"]
        else:
            inventory.loc[i,"simulation_route"] = "standard_native_v1"
        inventory.loc[i,"new_calibration_ready"] = routes.get(identifier,{}).get("calibration_ready",False)
    inventory.to_csv(out/"paper_reservoir_inventory_all_37.csv",index=False,float_format="%.17g")
    record = {"kind":"paper_completion_routes_v1","paper_reservoirs":37,
        "standard_completed":29,"separate_interval_completed":1,
        "new_signed_calibration_ready":sum(p["calibration_ready"] for p in signed),
        "new_signed_preflight_calls":sum(p["new_preflight_model_calls"] for p in signed),
        "new_optimizer_evaluations":0,"new_held_out_model_calls":0,
        "completed_profiles_both_held_out_nse_positive":sum(p["both_assessment_nse_positive"] for p in standard),
        "source_profiles":source,"signed_profiles":signed,
        "prior_standard_parameters_changed":False,"Bull_Lake_has_no_held_out_validation":True,
        "exact_existing_paper_monthly_targets_verified":29*72,
        "optimization_protocol":{"seeds":[42,43,44],"population":32,"budget":512,"scoring":"pooled_nse"}}
    w.write(out/"preparation.json",record)
    w.write(out/"protocol.json",{"kind":"paper_completion_routes_v1","code_sha256":code_hashes(),
        "input_sha256":{str(p.relative_to(ROOT)):w.sha(p) for p in consumed},
        "prepared_sha256":{p.relative_to(out).as_posix():w.sha(p) for p in out.rglob("*") if p.is_file()}})
    return out


def check_protocol(output):
    out = Path(output).resolve(strict=True)
    if not out.is_relative_to(ROOT/"outputs"):
        raise ValueError("Study must be inside outputs")
    proof = w.read(out/"protocol.json")
    if proof.get("kind") != "paper_completion_routes_v1":
        raise ValueError("Wrong workflow route")
    for name in ("code_sha256","input_sha256"):
        for rel,digest in proof[name].items():
            if w.sha(ROOT/rel) != digest:
                raise ValueError("Pinned input/source changed: "+rel)
    verified_artifacts(out,proof["prepared_sha256"])
    return out,proof


def signed_objective(output,identifier):
    out,_ = check_protocol(output)
    if identifier not in SIGNED_IDS:
        raise ValueError("Unknown signed reservoir")
    folder = out/"signed_profiles"/identifier
    require_calibration_ready(w.read(folder/"source_profile.json"))
    model = w.read(folder/"model.json"); plan = YearSelection(**w.read(folder/"selection.json"))
    obs = _observations(pd.read_csv(folder/"observations_si.csv",index_col="date",parse_dates=True,float_precision="round_trip"))
    template,bounds = fit_selected_years(obs,plan,model["physical_constants"],allow_feasible_initialization=True)
    initial = ReservoirParameters(pd.read_csv(folder/"initial_targets_si.csv",index_col="month",float_precision="round_trip"),**model["physical_constants"])
    template = bounds.validate(parameters_to_vector(initial),template)
    return SignedYearObjective(obs,plan,template,sampling=SamplingSettings(**model["sampling"]),
        bounds=bounds,policy=model["root_policy"],atol_storage_m3=model["atol_storage_m3"],
        cost_mode=model["cost_mode"],forcing_policy=SignedNetPolicy(**model["forcing_policy"]))


def load_paper_route(output, identifier, *, seed=None, budget=512):
    """One loader, preserving standard, signed and conditional interval meanings.

    Returns metadata and parameters, plus Bull Lake's already verified trajectory.
    Bull Lake requires an explicit seed because no across-seed profile is chosen.
    Source-stage and preflight targets can never masquerade as calibrated profiles.
    """
    out,_ = check_protocol(output)
    inventory = pd.read_csv(out/"paper_reservoir_inventory_all_37.csv")
    row = inventory.loc[inventory.reservoir_id.eq(identifier)]
    if len(row) != 1: raise ValueError("Unknown paper reservoir")
    route = row.iloc[0].simulation_route
    if route == "standard_native_v1":
        reference = pd.read_csv(out/"standard_profile_performance.csv").set_index("reservoir_id").loc[identifier,"profile_registry"]
        metadata,parameters = w.load_profile(ROOT/reference,identifier=identifier)
        return {"route":route,"metadata":metadata,"parameters":parameters,"cached_trajectory":None}
    if identifier == "ex35_Bull_Lake":
        if seed not in (42,43,44): raise ValueError("Select an explicit archived Bull Lake seed: 42, 43 or 44")
        folder = out/"interval_profiles"/identifier
        metadata = w.read(folder/"route.json")
        record = next(v for v in metadata["representatives"] if v["seed"] == seed)
        targets = pd.read_csv(folder/record["targets_file"],index_col="month",float_precision="round_trip")
        parameters = ReservoirParameters(targets,**record["physical_constants"])
        trajectory = pd.read_csv(folder/record["trajectory_file"],index_col=0,parse_dates=True,float_precision="round_trip")
        return {"route":route,"metadata":metadata,"parameters":parameters,"cached_trajectory":trajectory}
    if identifier in SIGNED_IDS:
        path = out/"releases"/f"budget_{budget}"/"registry.json"
        if not path.exists(): raise ValueError("Signed forcing is prepared but calibration is not complete")
        metadata,parameters = w.load_profile(path,identifier=identifier)
        return {"route":route,"metadata":metadata,"parameters":parameters,"cached_trajectory":None}
    folder = out/"source_profiles"/identifier
    require_calibration_ready(w.read(folder/"source_profile.json"))
    raise ValueError("No completed parameter profile for this source route")
