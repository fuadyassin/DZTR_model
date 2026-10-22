"""Consolidated, calibration-frozen assessment of the completed paper cohort."""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

from workflows import reservoir_v2 as w
from workflows import profile_application_v1 as app
from strategies.representative_years import YearSelection
from dztr.schemas import ReservoirParameters
from dztr.calibration.checkpoint import load_optimizer_checkpoint
from tools.assess_paper_completion_v1 import normalized_hypervolume, front_distance

ROOT=w.ROOT
SOURCE_TABLE=ROOT/"outputs/paper_forcing_metadata_v1_20261017/reservoir_parameterization_all_37_by_GRAND_ID.csv"


def csv(path,dates=False):
    return pd.read_csv(path,index_col="date" if dates else None,parse_dates=dates,float_precision="round_trip")


def pair_stats(observed,simulated):
    x,y=np.asarray(observed,float),np.asarray(simulated,float)
    if x.shape!=y.shape or x.ndim!=1: raise ValueError("Paired vectors must have equal one-dimensional shape")
    if np.isinf(x).any() or np.isinf(y).any(): raise ValueError("Infinite observation or simulation")
    keep=np.isfinite(x)&np.isfinite(y); x,y=x[keep],y[keep]
    if not len(x): return dict(pairs=0,nse=np.nan,mean_bias=np.nan,pbias=np.nan,rmse=np.nan)
    variance=np.sum((x-x.mean())**2)
    return dict(pairs=len(x),nse=float(1-np.sum((y-x)**2)/variance) if variance>0 and np.ptp(x)>0 else np.nan,
        mean_bias=float(np.mean(y-x)),pbias=float(100*np.sum(y-x)/np.sum(x)) if np.sum(x)!=0 else np.nan,
        rmse=float(np.sqrt(np.mean((y-x)**2))))


def within_block_error_slope(index,error,mask):
    """Common error slope with separate intercepts for contiguous scored blocks.

    Context/other-phase gaps are never treated as consecutive observations.
    Return SI error per year; this is descriptive drift, not a fitted loss term.
    """
    positions=np.flatnonzero(mask); numerator=denominator=0.
    for block in np.split(positions,np.flatnonzero(np.diff(positions)>1)+1):
        if len(block)<2: continue
        values=np.asarray(error)[block]; good=np.isfinite(values)
        if good.sum()<2: continue
        time=(index[block]-index[block[0]]).total_seconds().to_numpy()/86400/365.2425
        time=time[good]; values=values[good]
        numerator+=np.sum((time-time.mean())*(values-values.mean()))
        denominator+=np.sum((time-time.mean())**2)
    return float(numerator/denominator) if denominator>0 else np.nan


def describe(row):
    registry=app.reference(row.profile_registry)
    metadata,parameters=w.load_profile(registry,identifier=row.reservoir_id)
    legacy=metadata.get("source_kind")=="verified_comparison_archive"
    if legacy:
        archive=ROOT/"outputs"/str(metadata["source_archive"]).replace("\\","/").split("/")[-1]
        prepared=archive/"prepared"/row.reservoir_id/"mixed_pooled"
        jobs=archive/"jobs"/row.reservoir_id/"mixed_pooled"
    else:
        archive=registry.parent.parent.parent
        prepared=archive/("signed_profiles" if metadata.get("route")=="signed_net_native_v1" else "prepared")/row.reservoir_id
        jobs=archive/"jobs"/row.reservoir_id
    return dict(identifier=row.reservoir_id,registry=registry,release=registry.parent,metadata=metadata,
        parameters=parameters,archive=archive,prepared=prepared,jobs=jobs,legacy=legacy,
        route="signed_net_native_v1" if metadata.get("route")=="signed_net_native_v1" else "standard_native_v1")


def build_catalog(table,descriptions):
    entries=[]
    for _,row in table.iterrows():
        cfg_path=ROOT/"configs/reservoirs"/(row.reservoir_id+".json"); cfg=w.read(cfg_path)
        entry={"reservoir_id":row.reservoir_id,"GRAND_ID":int(row.GRAND_ID) if pd.notna(row.GRAND_ID) else None,
            "name":row.paper_name,"monthly_profile_available":bool(row.parameter_values_available),
            "status":row.programme_status,"next_action":row.next_action,
            "route":row.simulation_route,"inflow_preprocessing":cfg["inflow_preprocessing"],
            "input_sha256":{cfg_path.relative_to(ROOT).as_posix():w.sha(cfg_path)}}
        if entry["monthly_profile_available"]:
            d=descriptions[row.reservoir_id]
            entry.update(registry=d["registry"].relative_to(ROOT).as_posix(),route=d["route"],
                         prepared_forcing=(d["prepared"]/"observations_si.csv").relative_to(ROOT).as_posix())
            for path in (d["registry"],d["prepared"]/"observations_si.csv",d["prepared"]/"selection.json"):
                entry["input_sha256"][path.relative_to(ROOT).as_posix()]=w.sha(path)
        entries.append(entry)
    return {"kind":"paper_profile_application_v1","source_table":SOURCE_TABLE.relative_to(ROOT).as_posix(),
        "source_table_sha256":w.sha(SOURCE_TABLE),"application_source_sha256":app.application_sources(),
        "reservoirs":entries,"selection_uses_assessment":False,"source_certified":False}


def diagnostic_simulation(d,source,plan,cache,*,initial=False):
    """Reuse complete cached trajectories; replay only an uncovered scored suffix."""
    profile,parameters=d["metadata"],d["parameters"]
    if initial:
        targets=pd.read_csv(d["prepared"]/"initial_targets_si.csv",index_col="month",float_precision="round_trip")
        parameters=ReservoirParameters(targets,**profile["physical_constants"])
    roles=np.asarray(plan.roles); dates=source.index[np.isin(roles,["calibration","assessment"])]
    end=dates[-1]
    archived=d["release"]/"diagnostics"/d["identifier"]/("assessment_simulation.csv" if d["route"]=="signed_net_native_v1" else "simulation.csv")
    if not initial:
        frame=csv(archived,True)
        if frame.index[-1]>=end: return frame.loc[:end],0,archived
    key={"source_sha256":w.sha(d["prepared"]/"observations_si.csv"),"end":end.date().isoformat(),
        "targets":parameters.monthly_targets.to_numpy().tolist(),"constants":profile["physical_constants"],
        "route":d["route"],"root_policy":profile["root_policy"],"sampling":profile["sampling"],
        "forcing_policy":profile.get("forcing_policy"),"atol_storage_m3":profile["atol_storage_m3"],
        "application_sha256":app.application_sources(),"initial":initial}
    folder=cache/d["identifier"]/("initializer" if initial else "selected")
    if (folder/"receipt.json").exists():
        receipt=w.read(folder/"receipt.json")
        if receipt["identity"]!=key or w.sha(folder/"simulation.csv")!=receipt["simulation_sha256"]:
            raise ValueError("Diagnostic replay cache changed")
        return csv(folder/"simulation.csv",True),0,folder/"simulation.csv"
    result,residual=app.run_frame(source.loc[:end],profile,parameters,forcing_stage="prepared",
        preprocessing={"factor":1.,"offset_m3s":0.})
    folder.mkdir(parents=True,exist_ok=False)
    result.frame.to_csv(folder/"simulation.csv",index_label="date",float_format="%.17g")
    w.write(folder/"receipt.json",{"identity":key,"simulation_sha256":w.sha(folder/"simulation.csv"),
        "independent_balance_residual_m3":residual,"purpose":"Fixed-profile diagnostic replay; no optimization or reselection"})
    return result.frame,1,folder/"simulation.csv"


def baseline_metrics(d,source,plan,cache):
    identifier=d["identifier"]
    if d["legacy"]:
        parent=d["archive"]/"assessed"/identifier/"mixed_pooled"
        matches=[path for path in parent.glob("*/scores.json") if path.parent.name.startswith(("generalized_baseline_","feasible_initialization_reference_"))]
        if not matches:
            path=d["jobs"]/"seed_42/trials.csv"
            trial=csv(path).iloc[0]
            if trial.feasible: raise ValueError("Feasible baseline has no archived assessment")
            return "initial_reference_infeasible",{phase:{"storage_nse":np.nan,"release_nse":np.nan,
                "reason":str(trial.reason)} for phase in ("calibration","assessment")},0,path
        if len(matches)!=1: raise ValueError("Ambiguous archived baseline")
        path=matches[0]
        saved=w.read(path)
        kind="generalized" if path.parent.name.startswith("generalized") else "feasible_initialization_reference"
        return kind,{period:{v+"_nse":saved[period+"_"+v+"_nse"] for v in ("storage","release")}
                              for period in ("calibration","assessment")},0,path
    if d["route"]=="signed_net_native_v1":
        path=d["release"]/"scores.json"; saved=next(r for r in w.read(path) if r["reservoir_id"]==identifier)
        initial=pd.read_csv(d["prepared"]/"initial_targets_si.csv",index_col="month",float_precision="round_trip")
        raw=pd.read_csv(d["prepared"]/"raw_generalized_targets_si.csv",index_col="month",float_precision="round_trip")
        kind="generalized" if np.array_equal(initial.to_numpy(),raw.to_numpy()) else "feasible_initializer"
        return kind,{"calibration":saved["fixed_initializer_calibration"],"assessment":saved["fixed_initializer_assessment"]["metrics"]},0,path
    if identifier=="ex37_Charvak":
        path=ROOT/"outputs/source_limit_review_v1_20261013/charvak_baseline_comparison.csv"
        values=csv(path); selected=values.loc[values.profile.eq("generalized_source_capacity_baseline")]
        return "generalized_source_capacity",{r.period:r.to_dict() for _,r in selected.iterrows()},0,path
    if identifier=="ex42_Tyuyamuyun":
        frame,calls,path=diagnostic_simulation(d,source,plan,cache,initial=True)
        metrics={}
        for phase in ("calibration","assessment"):
            dates=source.index[plan.mask(source.index,phase)]
            metrics[phase]={v+"_nse":pair_stats(source.loc[dates,c],frame.loc[dates,c])["nse"]
                for v,c in (("storage","storage_m3"),("release","release_m3s"))}
        return "sufficient_gap_initializer",metrics,calls,path
    path=ROOT/"outputs/paper_exploration_v1_20261013/report/baseline_comparison.csv"
    values=csv(path); values=values.loc[values.reservoir_id.eq(identifier)&values.profile.eq("fixed_initializer")]
    kinds=csv(ROOT/"outputs/paper_exploration_v1_20261013/report/new_reservoir_scores.csv").set_index("reservoir_id")
    return kinds.loc[identifier,"fixed_initializer_kind"],{r.period:r.to_dict() for _,r in values.iterrows()},0,path


def checkpoints(d):
    records=[]; fronts={}; consumed=[]
    for seed in (42,43,44):
        job=d["jobs"]/f"seed_{seed}"; snap=job if d["legacy"] else job/"budget_512"
        proof=job/"completion_manifest.json" if d["legacy"] else snap/"manifest.json"
        for relative,digest in w.read(proof)["artifact_sha256"].items():
            if w.sha((job if d["legacy"] else snap)/relative)!=digest: raise ValueError("Search snapshot changed")
        identity=job/("checkpoint_identity.json" if d["legacy"] else "identity.json")
        state=load_optimizer_checkpoint(job/"checkpoint.npz",w.read(identity))
        trials=csv(snap/"trials.csv"); vectors=pd.read_csv(snap/"vectors.csv",index_col="evaluation_id",float_precision="round_trip")
        if len(state.evaluated)!=len(trials) or len(trials)!=512: raise ValueError("Incomplete search")
        np.testing.assert_array_equal(vectors.to_numpy(),[r.vector for r in state.evaluated])
        costs=trials[["release_cost","storage_cost"]].to_numpy()
        np.testing.assert_array_equal(costs,[r.objectives for r in state.evaluated])
        front=csv(snap/("pareto_front.csv" if d["legacy"] else "front.csv"))
        fronts[seed]=front[["release_cost","storage_cost"]].to_numpy()
        hv256,hv512=normalized_hypervolume(costs[:256]),normalized_hypervolume(costs)
        if hv512<hv256-1e-14: raise ValueError("Cumulative archive hypervolume declined")
        records.append({"reservoir_id":d["identifier"],"seed":seed,"evaluations":512,
            "population":state.settings.population_size,"generation":state.generation,"feasible_trials":int(trials.feasible.sum()),
            "hypervolume_256":hv256,"hypervolume_512":hv512,"hypervolume_gain":hv512-hv256,
            "checkpoint_exact":True,"convergence_established":False})
        consumed.extend([proof,identity,job/"checkpoint.npz",snap/"trials.csv",snap/"vectors.csv"])
    agreement=[{"reservoir_id":d["identifier"],"left_seed":a,"right_seed":b,
        "front_distance_nse_linf":front_distance(fronts[a],fronts[b])} for a,b in combinations((42,43,44),2)]
    return records,agreement,consumed


def trajectory_diagnostics(identifier,source,simulation,plan,parameters):
    frame=simulation.reindex(source.index)
    roles=np.asarray(plan.roles); regimes=np.asarray(plan.regimes)
    summaries={}; monthly=[]; zones=[]; regime_scores=[]
    calibration=roles=="calibration"
    releases=source.loc[calibration,"release_m3s"].dropna().to_numpy()
    peak_threshold=float(np.quantile(releases,.9,method="hazen"))
    for phase in ("calibration","assessment"):
        mask=roles==phase; summary={}
        for variable,column in (("storage","storage_m3"),("release","release_m3s")):
            stats=pair_stats(source.loc[mask,column],frame.loc[mask,column])
            summary.update({variable+"_"+k:v for k,v in stats.items()})
            for regime in ("dry","normal","wet"):
                sub=mask&(regimes==regime)
                stats=pair_stats(source.loc[sub,column],frame.loc[sub,column])
                regime_scores.append({"reservoir_id":identifier,"period":phase,"regime":regime,"variable":variable,**stats})
            for month in range(1,13):
                sub=mask&(source.index.month==month)
                stats=pair_stats(source.loc[sub,column],frame.loc[sub,column])
                monthly.append({"reservoir_id":identifier,"period":phase,"month":month,"variable":variable,**stats,
                    "storage_bias_pct_capacity":100*stats["mean_bias"]/parameters.capacity_m3 if variable=="storage" else np.nan})
            for zone in range(5):
                sub=mask&frame.zone.eq(zone).to_numpy()
                zones.append({"reservoir_id":identifier,"period":phase,"simulated_zone":zone,"variable":variable,
                    **pair_stats(source.loc[sub,column],frame.loc[sub,column])})
        error=(frame.storage_m3-source.storage_m3).to_numpy()
        slope=within_block_error_slope(source.index,error,mask)
        summary["within_block_storage_error_trend_pct_capacity_per_year"]=100*slope/parameters.capacity_m3
        summary["storage_mean_bias_pct_capacity"]=100*summary["storage_mean_bias"]/parameters.capacity_m3
        peak=mask&source.release_m3s.ge(peak_threshold).to_numpy()
        summary.update({"peak_release_"+k:v for k,v in pair_stats(source.loc[peak,"release_m3s"],frame.loc[peak,"release_m3s"]).items()})
        summary["peak_release_threshold_m3s_calibration_p90_hazen"]=peak_threshold
        summary["missing_regimes"]=",".join(r for r in ("dry","normal","wet") if not (mask&(regimes==r)).any())
        summaries[phase]=summary
    return summaries,monthly,zones,regime_scores


def assess(output,cache):
    table=csv(SOURCE_TABLE)
    source_manifest=w.read(SOURCE_TABLE.parent/"manifest.json")
    if source_manifest["artifact_sha256"][SOURCE_TABLE.name]!=w.sha(SOURCE_TABLE): raise ValueError("Input table changed")
    out=w.fresh_output(output); cache=Path(cache).resolve()
    if not cache.is_relative_to(ROOT/"outputs") or cache==ROOT/"outputs": raise ValueError("Use a versioned diagnostic cache in outputs")
    cache.mkdir(exist_ok=True)
    available=table.loc[table.parameter_values_available]
    if len(table)!=37 or len(available)!=32: raise ValueError("Expected 32 completed profiles in the 37-paper inventory")
    descriptions={row.reservoir_id:describe(row) for _,row in available.iterrows()}
    catalog=build_catalog(table,descriptions); w.write(out/"catalog.json",catalog)
    selected_rows=[]; seed_rows=[]; month_rows=[]; zone_rows=[]; regime_rows=[]; checkpoint_rows=[]; agreement_rows=[]
    consumed={SOURCE_TABLE.relative_to(ROOT).as_posix():w.sha(SOURCE_TABLE)}
    new_calls=0; reused_simulations=0; baseline_types={}
    for _,row in available.iterrows():
        identifier=row.reservoir_id; d=descriptions[identifier]; profile=d["metadata"]; parameters=d["parameters"]
        print("Assessing "+identifier,flush=True)
        plan=YearSelection(**w.read(d["prepared"]/"selection.json")); source=csv(d["prepared"]/"observations_si.csv",True)
        before=w.sha(d["release"]/"selection.json")
        for month,values in parameters.monthly_targets.iterrows():
            for name,value in values.items():
                if row[f"{name}_month_{month:02d}"]!=value: raise ValueError("Catalog and parameter export disagree")
        simulation,calls,trajectory=diagnostic_simulation(d,source,plan,cache)
        new_calls+=calls; reused_simulations+=not calls
        base_kind,base,calls,base_path=baseline_metrics(d,source,plan,cache); new_calls+=calls
        baseline_types[identifier]=base_kind
        diagnostics,months,zones,regimes=trajectory_diagnostics(identifier,source,simulation,plan,parameters)
        for phase in ("calibration","assessment"):
            frozen=profile["calibration"] if phase=="calibration" else profile["assessment"]["metrics"]
            for v in ("storage","release"):
                np.testing.assert_allclose(diagnostics[phase][v+"_nse"],frozen[v+"_nse"],rtol=0,atol=1e-10)
        month_rows+=months; zone_rows+=zones; regime_rows+=regimes
        jobs,pairs,paths=checkpoints(d); checkpoint_rows+=jobs; agreement_rows+=pairs
        paths += [d["registry"],d["release"]/"manifest.json",d["release"]/"selection.json",d["release"]/"seed_scores.csv",d["prepared"]/"selection.json",d["prepared"]/"observations_si.csv",trajectory,base_path]
        for path in paths: consumed[path.relative_to(ROOT).as_posix()]=w.sha(path)
        seeds=csv(d["release"]/"seed_scores.csv"); seeds=seeds.loc[seeds.reservoir_id.eq(identifier)]
        if len(seeds)!=3 or set(seeds.seed)!={42,43,44}: raise ValueError("Missing seed assessments")
        for _,seed in seeds.iterrows():
            for phase in ("calibration","assessment"):
                seed_rows.append({"reservoir_id":identifier,"seed":int(seed.seed),"evaluation_id":int(seed.evaluation_id),
                    "period":phase,"assessment_feasible":seed.assessment_feasible,"baseline_kind":base_kind,
                    "baseline_reason":base[phase].get("reason"),
                    **{v+"_nse":seed.get(phase+"_"+v+"_nse",np.nan) for v in ("storage","release")},
                    **{v+"_nse_gain":seed.get(phase+"_"+v+"_nse",np.nan)-base[phase][v+"_nse"] for v in ("storage","release")}})
        result={"reservoir_id":identifier,"name":row.paper_name,"GRAND_ID":row.GRAND_ID,"route":d["route"],
            "seed":profile["seed"],"evaluation_id":profile["evaluation_id"],"baseline_kind":base_kind,
            "baseline_reason":base["calibration"].get("reason"),
            "source_certified":False,"convergence_established":False,"maximum_seed_front_distance":max(v["front_distance_nse_linf"] for v in pairs),
            "all_seed_assessments_feasible":bool(seeds.assessment_feasible.all()),
            "calibration_fraction":plan.roles.count("calibration")/len(plan.roles)}
        for phase,metrics in diagnostics.items():
            result.update({phase+"_"+k:v for k,v in metrics.items()})
            for v in ("storage","release"):
                result[phase+"_baseline_"+v+"_nse"]=base[phase][v+"_nse"]
                result[phase+"_"+v+"_nse_gain"]=metrics[v+"_nse"]-base[phase][v+"_nse"]
        residual=w.independent_balance(simulation,parameters.inflow_correction)
        if residual>profile["atol_storage_m3"] or not simulation.storage_m3.between(0,parameters.capacity_m3).all(): raise ValueError("Physical audit failed")
        result["maximum_independent_balance_residual_m3"]=residual
        gains=[result["assessment_"+v+"_nse_gain"] for v in ("storage","release")]
        result["assessment_baseline_effect"]=("baseline_unavailable" if not np.isfinite(gains).all() else
            "both_improve" if min(gains)>0 else "both_worsen" if max(gains)<0 else "tradeoff")
        held_regimes=[v for v in regimes if v["period"]=="assessment" and v["pairs"]>0]
        flags={"nonpositive_pooled_nse":min(result["assessment_storage_nse"],result["assessment_release_nse"])<=0,
            "baseline_unavailable":result["assessment_baseline_effect"]=="baseline_unavailable",
            "nonpositive_regime_nse":any(not np.isfinite(v["nse"]) or v["nse"]<=0 for v in held_regimes),
            "missing_assessment_regimes":bool(result["assessment_missing_regimes"]),
            "release_bias_over_20pct":abs(result["assessment_release_pbias"])>20,
            "peak_release_bias_over_20pct":abs(result["assessment_peak_release_pbias"])>20,
            "storage_error_trend_over_2pct_capacity_per_year":abs(result["assessment_within_block_storage_error_trend_pct_capacity_per_year"])>2,
            "seasonal_storage_bias_over_10pct_capacity":any(abs(v["storage_bias_pct_capacity"])>10 for v in months if v["period"]=="assessment" and v["variable"]=="storage"),
            "seed_front_distance_over_0_01":result["maximum_seed_front_distance"]>.01}
        result.update({"flag_"+k:v for k,v in flags.items()}); result["quality_flags"]=";".join(k for k,v in flags.items() if v)
        selected_rows.append(result)
        if w.sha(d["release"]/"selection.json")!=before: raise ValueError("Assessment changed frozen selection")
    frames={"cohort_summary.csv":selected_rows,"seed_baseline_comparison.csv":seed_rows,
        "seasonal_bias.csv":month_rows,"zone_diagnostics.csv":zone_rows,"regime_diagnostics.csv":regime_rows,
        "checkpoint_diagnostics.csv":checkpoint_rows,"seed_front_agreement.csv":agreement_rows}
    for name,rows in frames.items(): pd.DataFrame(rows).to_csv(out/name,index=False,float_format="%.17g")
    # Preserve the complete 37-row parameter inventory, including forcing assumptions.
    table.to_csv(out/"reservoir_parameterization_all_37_by_GRAND_ID.csv",index=False,float_format="%.17g")
    pd.testing.assert_frame_equal(csv(out/"reservoir_parameterization_all_37_by_GRAND_ID.csv"),table)
    summary=pd.DataFrame(selected_rows); counts=summary.assessment_baseline_effect.value_counts().to_dict()
    receipt={"status":"passed","paper_inventory_rows":37,"completed_profiles":32,"separate_interval_studies":1,"source_gated":4,
        "seed_searches_verified":len(checkpoint_rows),"cached_evaluations_verified":len(checkpoint_rows)*512,
        "seed_phase_baseline_comparisons":len(seed_rows),"new_diagnostic_simulation_calls":int(new_calls),
        "seed_phase_comparisons_with_available_baseline":sum(np.isfinite(r["storage_nse_gain"]) and np.isfinite(r["release_nse_gain"]) for r in seed_rows),
        "selected_trajectories_reused":int(reused_simulations),"new_optimizer_evaluations":0,
        "all_selected_calibration_and_assessment_nse_reproduced":True,"prior_parameters_preserved":True,
        "monthly_parameter_values_preserved":32*72,"assessment_baseline_outcomes":counts,
        "both_pooled_assessment_nse_positive":int(((summary.assessment_storage_nse>0)&(summary.assessment_release_nse>0)).sum()),
        "maximum_balance_residual_m3":float(summary.maximum_independent_balance_residual_m3.max()),
        "jobs_with_positive_hypervolume_gain":sum(r["hypervolume_gain"]>0 for r in checkpoint_rows),
        "convergence_established":False,"source_certified":False,"consumed_sha256":consumed,
        "report_source_sha256":{name:w.sha(ROOT/name) for name in ("workflows/cohort_assessment_v1.py","tools/assess_calibrated_cohort_v1.py")}}
    w.write(out/"verification.json",receipt)
    write_report(out,summary,receipt)
    write_diagnostic_findings(out,summary)
    plot_summary(out,summary)
    w.write(out/"manifest.json",{"artifact_sha256":{p.relative_to(out).as_posix():w.sha(p) for p in out.rglob("*") if p.is_file()}})
    return out


def write_report(out,summary,receipt):
    lines=["# Consolidated assessment of 32 calibrated paper reservoirs","",
        "The 37-reservoir inventory contains 29 standard native profiles, three exploratory signed-native profiles, Bull Lake's separate conditional interval study, and four source-gated cases. The assessment below covers the 32 completed monthly profiles. Travers is outside the published cohort.","",
        "## Calibration and held-out baseline comparisons","",
        f"All 32 selected profiles reproduce their archived calibration/held-out NSE. All 96 seed checkpoints match their 512-trial snapshots exactly ({receipt['cached_evaluations_verified']:,} attempts). There are 192 seed/phase rows, including {receipt['seed_phase_comparisons_with_available_baseline']} with numerical baseline comparisons. Yellowtail and Nurek retain their archived initial-reference failures; their baseline NSE/gains remain blank. Baseline types are explicit: generalized estimates, source-capacity generalized estimates, or feasible/sufficient-gap initializers. Physical assumptions and year masks are held fixed within each comparison; cross-reservoir absolute skill is not a controlled comparison of source quality.","",
        "Held-out outcomes: "+", ".join(f"{value} {key.replace('_',' ')}" for key,value in receipt["assessment_baseline_outcomes"].items())+".",
        f"{receipt['both_pooled_assessment_nse_positive']} of 32 have positive pooled held-out NSE for storage and release. Positive pooled NSE does not establish adequate seasonal skill, independent forcing, or source certification.","",
        "| Reservoir | Baseline | Held-out storage NSE | Release NSE | Storage gain | Release gain |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for _,r in summary.iterrows():
        gains=[f"{v:+.3f}" if np.isfinite(v) else "unavailable" for v in (r.assessment_storage_nse_gain,r.assessment_release_nse_gain)]
        lines.append(f"| {r['name']} | {r.baseline_kind} | {r.assessment_storage_nse:.3f} | {r.assessment_release_nse:.3f} | {gains[0]} | {gains[1]} |")
    lines += ["","![Held-out NSE gains](figures/held_out_gains.png)","",
        "## Diagnostic interpretation","",
        "[Per-reservoir findings](diagnostic_findings.md), [seasonal bias](seasonal_bias.csv), [dry/normal/wet regime results](regime_diagnostics.csv), [simulated-zone diagnostics](zone_diagnostics.csv) and [per-seed baseline comparisons](seed_baseline_comparison.csv) retain paired support counts and undefined NSE. Zone groups use simulated DZTR zones; zone-conditioned NSE has restricted variance and does not diagnose causality. Release peaks use the calibration-only Hazen 90th percentile, applied unchanged to assessment. Storage error trends use a separate intercept per contiguous scored block and never join gaps as consecutive time steps. They are descriptive error trends, not inferred evaporation or a mass-balance correction. A within-year error trend can reflect seasonal/timing mismatch rather than persistent long-term storage drift.","",
        "Quality flags are transparent post-fit screening criteria: nonpositive NSE, missing assessment regimes, absolute release/peak bias over 20%, seasonal storage bias over 10% of capacity, within-block storage-error trend over 2% of capacity/year, and seed-front distance over 0.01 NSE. These thresholds do not change calibration, select parameters or certify application suitability. A short or low-variance assessment can amplify NSE and error-trend estimates.","",
        "Seminoe worsens both pooled held-out objectives against its initializer and has negative storage NSE within all held-out regimes. Navajo's pooled release NSE remains near zero with large release bias. Flaming Gorge improves both pooled objectives but retains dry/normal release and normal-storage weaknesses. Existing source assumptions for Big Horn, Chardara, Toktogul, Kayrakkum, Falcon and Charvak remain visible rather than being interpreted as optimizer problems.","",
        "## Convergence evidence","",
        f"{receipt['jobs_with_positive_hypervolume_gain']} of 96 cumulative calibration fronts improve between 256 and 512 attempts. [Checkpoint diagnostics](checkpoint_diagnostics.csv) use a fixed negative-NSE reference (1,1), normalized by area four. [Seed-front distances](seed_front_agreement.csv) use symmetric Hausdorff distance in raw NSE L-infinity coordinates. The 0.01 distance is a reporting tolerance, not a predeclared stopping policy. A completed budget, small seed spread, or improving objective front does not prove global or parameter convergence. No extension was launched.","",
        "## Reproducible application","",
        "The [profile catalog](catalog.json) resolves verified GRAND IDs or explicit reservoir IDs and dispatches standard versus signed-net solvers. Archived prepared forcing is the default and is not transformed again. External inputs require `--forcing-stage canonical` or `prepared`; a different start needs explicit SI initial storage and release. The simulator uses no observed response to initialize or reset state. Bull Lake and the four source-gated cases cannot be run as ordinary native profiles through this catalog.","",
        "```powershell","& .\\tools\\Set-ProjectCache.ps1",
        f"python -B examples/simulate_profile.py --catalog {out.relative_to(ROOT).as_posix()}/catalog.json --grand-id 753 --output outputs/example_garrison_new",
        f"python -B examples/simulate_profile.py --catalog {out.relative_to(ROOT).as_posix()}/catalog.json --reservoir-id ex40_Semione --output outputs/example_seminoe_new","```","",
        "[All-37 parameter table](reservoir_parameterization_all_37_by_GRAND_ID.csv) retains all 2,304 completed monthly targets and Bennett's documented 1.45 preprocessing multiplier. Bennett remains source-gated. Canyon Ferry, Trinity and Ghost retain their source requirements; Bull Lake retains conditional interval metadata and zero independent held-out claims.","",
        f"Assessment required {receipt['new_diagnostic_simulation_calls']} fixed-profile diagnostic calls where the stored trajectory did not cover calibration support or an initializer assessment was missing. Those trajectories are stored in a versioned cache and reused on subsequent reports. No optimization was performed and original parameter choices, sources and archives are preserved. Maximum independent selected-trajectory mass residual: {receipt['maximum_balance_residual_m3']:.9g} m3. [Verification receipt](verification.json).",""]
    (out/"README.md").write_text("\n".join(lines),encoding="utf-8")


def plot_summary(out,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values=summary[["assessment_storage_nse_gain","assessment_release_nse_gain"]].to_numpy()
    fig,ax=plt.subplots(figsize=(8,13),layout="constrained")
    im=ax.imshow(values,cmap="RdYlGn",vmin=-.5,vmax=.5,aspect="auto")
    ax.set_yticks(range(len(summary)),summary.name); ax.set_xticks([0,1],["Storage NSE gain","Release NSE gain"])
    for i in range(len(summary)):
        for j in range(2): ax.text(j,i,f"{values[i,j]:+.3f}" if np.isfinite(values[i,j]) else "unavailable",ha="center",va="center",fontsize=8)
    ax.set_title("Held-out gain against each reservoir's declared baseline\nColors saturate at +/-0.5; annotations show actual gains")
    fig.colorbar(im,ax=ax,label="NSE gain")
    folder=out/"figures"; folder.mkdir()
    fig.savefig(folder/"held_out_gains.png",dpi=180); fig.savefig(folder/"held_out_gains.pdf"); plt.close(fig)


def write_diagnostic_findings(out,summary):
    lines=["# Diagnostic findings for each calibrated paper reservoir","",
        "These post-fit descriptions preserve frozen calibration selections. Flags identify follow-up questions; they are not source certifications, causal diagnoses or parameter-selection rules. Missing/undefined metrics retain their support counts in the CSV tables.",""]
    labels={"nonpositive_pooled_nse":"Review weak pooled assessment fit before application.",
        "baseline_unavailable":"The saved initial reference is infeasible; no numerical improvement claim can be made against it.",
        "nonpositive_regime_nse":"Inspect the regime scores and within-regime variance; pooled skill hides a weak/undefined regime.",
        "missing_assessment_regimes":"Assessment does not represent every inflow regime; generalization evidence is incomplete.",
        "release_bias_over_20pct":"Review total release bias and forcing/outflow boundaries.",
        "peak_release_bias_over_20pct":"Inspect the calibration-threshold release-peak subset, including paired support.",
        "storage_error_trend_over_2pct_capacity_per_year":"Inspect within-block storage-error slopes alongside seasonal bias and storage/flow timing.",
        "seasonal_storage_bias_over_10pct_capacity":"Review month-specific storage bias relative to the declared model capacity.",
        "seed_front_distance_over_0_01":"Seed fronts differ beyond the reporting tolerance; convergence is not established."}
    for _,r in summary.iterrows():
        lines += ["## "+str(r["name"])+" ("+r.reservoir_id+")","",
            f"Held-out storage/release NSE: {r.assessment_storage_nse:.3f}/{r.assessment_release_nse:.3f}. Baseline: {r.baseline_kind}; outcome: {r.assessment_baseline_effect}. "
            f"Release bias: {r.assessment_release_pbias:+.2f}%; peak-subset bias: {r.assessment_peak_release_pbias:+.2f}%. "
            f"Within-block storage-error trend: {r.assessment_within_block_storage_error_trend_pct_capacity_per_year:+.2f}% of capacity/year.",""]
        if pd.notna(r.baseline_reason): lines += ["Archived baseline failure: "+str(r.baseline_reason),""]
        for flag in str(r.quality_flags).split(";"):
            if flag in labels: lines.append("- "+labels[flag])
        lines += ["", "Source definitions remain exploratory. Additional evaluations alone do not resolve forcing, timing, storage-coordinate or release-rule limitations.",""]
    (out/"diagnostic_findings.md").write_text("\n".join(lines),encoding="utf-8")
