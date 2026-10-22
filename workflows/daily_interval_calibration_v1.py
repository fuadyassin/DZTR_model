"""Checkpointed DDS jobs and assessment for a frozen daily-mean scenario."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

import numpy as np
import pandas as pd

from dztr.calibration.candidates import parameters_to_vector
from dztr.data.intervals import prepare_daily_intervals
from dztr.model.release import dztr_release
from dztr.simulation.intervals import simulate_interval_reservoir, IntervalSimulationSettings
from strategies import dds_v1
from strategies import interval_scalar_v1 as strategy
from strategies.flow_only_v1 import nse
from strategies.representative_years import YearSelection
from workflows import extension_assessment_v1 as data
from workflows import reservoir_v2 as native
from workflows.atomic_io_v1 import write
from workflows.geometry_v1 import digest

ROOT = native.ROOT
FILES = ("strategies/interval_scalar_v1.py", "strategies/dds_v1.py", "strategies/flow_only_v1.py",
    "strategies/full_data_scalar_v1.py", "strategies/representative_years.py", "workflows/extension_assessment_v1.py",
    "workflows/daily_interval_calibration_v1.py", "workflows/atomic_io_v1.py", "tools/calibrate_daily_intervals_v1.py")


def sources(): return {**native.source_hashes(), **{p: data.sha(ROOT/p) for p in FILES}}


def checked(output):
    out = Path(output).resolve(strict=True); protocol = data.read(out/"protocol.json")
    for name, expected in protocol["source_sha256"].items():
        if data.sha(ROOT/name) != expected: raise ValueError("Calibration source changed: "+name)
    for name, expected in protocol["prepared_sha256"].items():
        path = (out/name).resolve(strict=True)
        if not path.is_relative_to(out) or data.sha(path) != expected: raise ValueError("Prepared data changed: "+name)
    return out, protocol


def prepare(config_path, output):
    cfg = data.read(config_path)
    if cfg.get("kind") != "daily_interval_scalar_calibration_v1": raise ValueError("Wrong scenario kind")
    out = Path(output).resolve()
    if out.exists():
        _, proof = checked(out)
        if proof["config"] != cfg: raise ValueError("Frozen scenario changed")
        return out
    if not out.is_relative_to(ROOT/"outputs") or out == ROOT/"outputs": raise ValueError("Use a fresh project output directory")
    for name in ("storage_csv", "flow_csv", "inventory_csv"):
        if not (ROOT/cfg[name]).is_file(): raise ValueError("Missing input: "+name)
    optimization = cfg["optimization"]
    if (set(optimization) != {"seeds","horizon","evaluation_budget","workers","perturbation"}
        or not optimization["seeds"] or len(set(optimization["seeds"])) != len(optimization["seeds"])
        or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in optimization["seeds"])
        or any(isinstance(optimization[k],bool) or not isinstance(optimization[k],int) for k in ("horizon","evaluation_budget","workers"))
        or not 1 <= optimization["evaluation_budget"] <= optimization["horizon"] or optimization["horizon"] < 2
        or optimization["workers"] < 1 or not 0 < optimization["perturbation"] <= 1):
        raise ValueError("Invalid fixed DDS settings")
    storage = data.frame_read(ROOT/cfg["storage_csv"])
    flows = data.frame_read(ROOT/cfg["flow_csv"])
    frame = flows.loc[cfg["start"]:cfg["end"], data.COLUMNS].copy()
    frame["storage_m3"] = storage.storage_m3.reindex(frame.index)
    if frame.index[0] != pd.Timestamp(cfg["start"]) or frame.index[-1] != pd.Timestamp(cfg["end"]): raise ValueError("Incomplete scenario calendar")
    raw_frame = frame.copy()
    for date in cfg["excluded_inflow_dates"]:
        stamp = pd.Timestamp(date)
        if stamp not in frame.index: raise ValueError("Excluded inflow date is outside scenario")
        frame.loc[stamp,"inflow_m3s"] = np.nan
    anchors, eligible = strategy.simulation_support(frame, restart_warmup_days=cfg["restart_warmup_days"])
    plan, annual = strategy.select_years(frame, eligible, minimum_fraction=cfg["minimum_calibration_fraction"], seed=cfg["year_selection_seed"])
    physical = dict(cfg["physical_constants"])
    fit_mask = plan.mask(frame.index, "calibration") & eligible
    physical["channel_capacity_m3s"] = float(np.percentile(frame.release_m3s[fit_mask].dropna(),99,method="hazen"))
    template, bounds = strategy.fit(frame, plan, eligible, physical)
    initials = template.monthly_targets.storage_normal_m3.to_numpy()
    runner = strategy.ArrayRunner(frame.inflow_m3s, initials, tolerance=cfg["balance_tolerance_m3"])
    obj = strategy.Objective(frame, plan, eligible, runner, template, bounds, objective_weights=cfg["objective_weights"])
    vector = parameters_to_vector(template)
    evaluation = obj(vector)
    if not evaluation.feasible: raise ValueError("Generalized initializer failed: "+str(evaluation.reason))
    out.mkdir(parents=True); prepared = out/"prepared"; prepared.mkdir()
    evidence = prepared/"evidence"; evidence.mkdir()
    for name in cfg.get("evidence_files",[]):
        path = (ROOT/name).resolve(strict=True)
        if not path.is_relative_to(ROOT): raise ValueError("Evidence must be in the project")
        if (evidence/path.name).exists(): raise ValueError("Evidence names must be unique")
        shutil.copyfile(path,evidence/path.name)
    raw_frame.to_csv(prepared/"source_observations_si.csv")
    frame.to_csv(prepared/"observations_si.csv")
    table = plan.table(); table["segment_initialization"] = anchors; table["score_eligible"] = eligible
    table["fitting_eligible"] = fit_mask; table.to_csv(prepared/"selection.csv")
    annual.to_csv(prepared/"annual_year_selection.csv")
    template.monthly_targets.to_csv(prepared/"generalized_targets_si.csv")
    bounds.empirical.monthly_bounds.to_csv(prepared/"bounds_si.csv")
    write(prepared/"selection.json", plan.as_dict())
    write(prepared/"physical_constants.json", physical)
    write(prepared/"initialization.json", {"monthly_initial_storage_m3": initials.tolist(),
        "method":"fixed calibration-only generalized monthly normal target at every finite-forcing block start",
        "observed_assessment_storage_used":False,"restart_warmup_days":cfg["restart_warmup_days"]})
    write(prepared/"admission.json", {"status":"calibration_ready_under_explicit_exploratory_scenario",
        "reservoir_id":cfg["reservoir_id"],"GRAND_ID":cfg["GRAND_ID"],"source_decisions":cfg["source_decisions"],
        "calibration_calendar_fraction":float(plan.mask(frame.index,"calibration").mean()),
        "actual_fitting_calendar_fraction":float(fit_mask.mean()),"assessment_responses_used_by_objective":False,
        "inflow_independence_verified":False,"provider_clock_verified":False,"baseline_calibration_cost":evaluation.cost,
        "physical_constants":physical,"excluded_inflow_dates":cfg["excluded_inflow_dates"]})
    pins = {p.relative_to(out).as_posix():data.sha(p) for p in prepared.rglob('*') if p.is_file()}
    inputs = {name:data.sha(ROOT/cfg[name]) for name in ("storage_csv","flow_csv","inventory_csv")}
    write(out/"protocol.json", {"kind":cfg["kind"],"config":cfg,"prepared_sha256":pins,
        "input_sha256":inputs,"source_sha256":sources(),"config_sha256":data.sha(config_path),
        "runtime":{"python":sys.version,"numpy":np.__version__,"pandas":pd.__version__}})
    return out


def objective(output):
    out, protocol = checked(output); cfg = protocol["config"]
    frame = data.frame_read(out/"prepared/observations_si.csv")
    plan = YearSelection(**data.read(out/"prepared/selection.json"))
    support = data.frame_read(out/"prepared/selection.csv").score_eligible.to_numpy(bool)
    physical = data.read(out/"prepared/physical_constants.json")
    template, bounds = strategy.fit(frame,plan,support,physical)
    pd.testing.assert_frame_equal(template.monthly_targets,pd.read_csv(out/"prepared/generalized_targets_si.csv",index_col="month",float_precision="round_trip"))
    initials = data.read(out/"prepared/initialization.json")["monthly_initial_storage_m3"]
    runner = strategy.ArrayRunner(frame.inflow_m3s,initials,tolerance=cfg["balance_tolerance_m3"])
    return strategy.Objective(frame,plan,support,runner,template,bounds,objective_weights=cfg["objective_weights"]), frame, plan, support


def calibrate(output, seed, budget=None):
    out, protocol = checked(output); cfg = protocol["config"]; opt = cfg["optimization"]
    budget = opt["evaluation_budget"] if budget is None else budget
    if seed not in opt["seeds"] or not 1 <= budget <= opt["horizon"]: raise ValueError("Undeclared seed/budget")
    folder = out/"jobs"/f"seed_{seed}"; folder.mkdir(parents=True,exist_ok=True)
    with (folder/"running.lock").open("x") as lock: lock.write(str(os.getpid()))
    try:
        obj, _, _, _ = objective(out)
        identity = {"protocol_sha256":data.sha(out/"protocol.json"),"seed":seed}
        state = None
        if (folder/"checkpoint.json").exists():
            checkpoint = data.read(folder/"checkpoint.json")
            if checkpoint["identity"] != identity or checkpoint["state_sha256"] != digest(checkpoint["state"]): raise ValueError("Checkpoint identity/content changed")
            state = checkpoint["state"]
        def save(current, force=False):
            count = len(current["trials"])
            if force or count%16==0 or count==budget:
                write(folder/"checkpoint.json",{"identity":identity,"state":current,"state_sha256":digest(current)})
                costs = [t["cost"] for t in current["trials"] if t["cost"] is not None]
                write(folder/"progress.json",{"evaluations":count,"budget":budget,"best_cost":min(costs) if costs else None})
        start = time.perf_counter()
        result = dds_v1.search(obj,parameters_to_vector(obj.template),obj.bounds.lower_vector,obj.bounds.upper_vector,
            seed=seed,horizon=opt["horizon"],budget=budget,perturbation=opt["perturbation"],state=state,checkpoint=save)
        save(result,True)
        pd.DataFrame([{k:v for k,v in row.items() if k!='vector'} for row in result["trials"]]).to_csv(folder/"trials.csv",index=False)
        write(folder/"completed.json",{"evaluations":len(result["trials"]),"checkpoint_sha256":data.sha(folder/"checkpoint.json"),"elapsed_seconds_this_invocation":time.perf_counter()-start})
        return folder
    finally: (folder/"running.lock").unlink(missing_ok=True)


def run(config_path, output):
    out = prepare(config_path, output); _, proof = checked(out); opt = proof["config"]["optimization"]
    if (out/"report/manifest.json").exists(): native.verify_registry(out/"report/registry.json"); return out/"report"
    queued = iter(opt["seeds"]); active = []; handles = []
    try:
        while True:
            while len(active) < opt["workers"]:
                seed = next(queued,None)
                if seed is None: break
                folder = out/"jobs"/f"seed_{seed}"; folder.mkdir(parents=True,exist_ok=True)
                handle = (folder/"execution.log").open("a",encoding="utf-8"); handles.append(handle)
                process = subprocess.Popen([sys.executable,"-B",str(ROOT/"tools/calibrate_daily_intervals_v1.py"),"worker","--output",str(out),"--seed",str(seed)],
                    cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
                active.append((seed,process)); print(f"DDS seed {seed} started",flush=True)
            if not active: break
            time.sleep(1.)
            for seed,process in active[:]:
                code = process.poll()
                if code is not None:
                    if code: raise ValueError(f"DDS seed {seed} failed; inspect its execution.log")
                    print(f"DDS seed {seed} completed",flush=True); active.remove((seed,process))
        return report(out)
    finally:
        for _,process in active:
            if process.poll() is None: process.terminate(); process.wait()
        for handle in handles: handle.close()


def _scores(frame, plan, support, simulation):
    result = {}
    for role in ("calibration","assessment"):
        mask = plan.mask(frame.index,role)&support
        for variable, observed, modeled in (("storage","storage_m3","storage_m3"),("release","release_m3s","interval_mean_total_outflow_m3s")):
            o, p = frame[observed].to_numpy()[mask], simulation[modeled][mask]
            result[f"{role}_{variable}_nse"] = nse(o,p)
            valid = np.isfinite(o)
            result[f"{role}_{variable}_bias"] = float(np.mean(p[valid]-o[valid]))
            result[f"{role}_{variable}_samples"] = int(valid.sum())
    return result


def canonical_replay(runner, parameters, fast):
    maximum = 0.
    for start,stop in data.blocks(np.isfinite(runner.inflow)):
        index = runner.index[start:stop]
        drivers = pd.DataFrame({"inflow_m3s":runner.inflow[start:stop],
            "storage_m3":runner.initials[runner.months[start]],"release_m3s":0.},index=index)
        # The canonical API requires finite storage/Q fields, but only its first
        # storage value initializes this block. These are declared driver fields,
        # never replacement observations and never supplied to target estimation.
        literal = simulate_interval_reservoir(prepare_daily_intervals(drivers),parameters,
            settings=IntervalSimulationSettings(root_policy=runner.policy),atol_storage_m3=runner.tolerance).frame
        for column in fast: np.testing.assert_array_equal(fast[column][start:stop],literal[column].to_numpy())
        residual = np.diff(fast["storage_m3"][start:stop])-runner.inflow[start+1:stop]*86400.+fast["interval_mean_total_outflow_m3s"][start+1:stop]*86400.
        if len(residual): maximum = max(maximum,float(np.max(np.abs(residual))))
    if maximum > runner.tolerance: raise ValueError("Independent representative volume check failed")
    return maximum


def report(output):
    out, protocol = checked(output); cfg = protocol["config"]
    destination = out/"report"
    if (destination/"manifest.json").exists(): native.verify_registry(destination/"registry.json"); return destination
    if destination.exists(): raise ValueError("Partial report exists; preserve it and choose a fresh publication directory")
    obj, frame, plan, support = objective(out)
    choices, states = [], {}
    for seed in cfg["optimization"]["seeds"]:
        folder = out/"jobs"/f"seed_{seed}"; checkpoint = data.read(folder/"checkpoint.json"); state = checkpoint["state"]
        if checkpoint["state_sha256"] != digest(state) or checkpoint["identity"] != {"protocol_sha256":data.sha(out/"protocol.json"),"seed":seed}: raise ValueError("Unverified seed checkpoint")
        if len(state["trials"]) != cfg["optimization"]["evaluation_budget"]: raise ValueError("Campaign is incomplete")
        finite = [r for r in state["trials"] if r["cost"] is not None]
        if not finite: raise ValueError("Seed has no feasible calibration result")
        choice = {**min(finite,key=lambda r:(r["cost"],r["evaluation_id"])),"seed":seed}
        choices.append(choice); states[seed] = state
    selected = min(choices,key=lambda r:(r["cost"],r["seed"],r["evaluation_id"]))
    # Freeze every choice before inspecting any held-out model performance.
    write(out/"selection_frozen.json",{"selected":selected,"per_seed":choices,"selection_uses_assessment":False,
        "rule":"minimum normalized weighted pooled calibration NSE loss; tie: seed then evaluation id"})
    destination.mkdir(); rows=[]; maximum=0.; simulations={}; trajectories=[]; zones=[]
    assessments = [("generalized_baseline",None,parameters_to_vector(obj.template))]+[(f"seed_{r['seed']}",r,r["vector"]) for r in choices]
    for label,choice,vector in assessments:
        parameters = obj.bounds.validate(vector,obj.template)
        calibration = obj(vector)
        if not calibration.feasible: raise ValueError("Frozen calibration representative did not replay")
        if choice is not None and abs(calibration.cost-choice["cost"]) > 1e-10: raise ValueError("Checkpoint objective replay differs")
        record = {"profile":label,"seed":None if choice is None else choice["seed"],"calibration_cost":calibration.cost,
            "selected":choice is not None and choice["seed"]==selected["seed"],"assessment_feasible":False}
        try:
            fast = obj.runner.run(parameters)
            residual = canonical_replay(obj.runner,parameters,fast); maximum=max(maximum,residual)
            record.update(_scores(frame,plan,support,fast),assessment_feasible=True,maximum_balance_residual_m3=residual)
            replayed_cost=sum(obj.weights[v]*(1-record[f"calibration_{v}_nse"]) for v in obj.weights if obj.weights[v])
            if abs(replayed_cost-calibration.cost)>1e-10: raise ValueError("Full and prefix calibration costs differ")
            sim=pd.DataFrame(fast,index=frame.index).join(plan.table())
            for column in data.COLUMNS: sim["observed_"+column]=frame[column]
            sim["score_eligible"]=support;sim.to_csv(destination/f"simulation_{label}.csv")
            simulations[label]=sim
            for role in ("calibration","assessment"):
                for zone in range(5):
                    zone_mask=plan.mask(frame.index,role)&support&(fast['zone']==zone)
                    o=frame.release_m3s.to_numpy()[zone_mask];p=fast['interval_mean_total_outflow_m3s'][zone_mask];valid=np.isfinite(o)
                    zones.append({'profile':label,'role':role,'zone':zone,'samples':int(valid.sum()),
                        'release_mean_bias_m3s':float(np.mean(p[valid]-o[valid])) if valid.any() else None})
                for month in range(1,13):
                    mask=plan.mask(frame.index,role)&support&(frame.index.month==month)
                    for variable,observed,modeled in (("storage","storage_m3","storage_m3"),("release","release_m3s","interval_mean_total_outflow_m3s")):
                        o=frame[observed].to_numpy()[mask];p=fast[modeled][mask];valid=np.isfinite(o)
                        trajectories.append({"profile":label,"role":role,"month":month,"variable":variable,"samples":int(valid.sum()),"mean_bias":float(np.mean(p[valid]-o[valid]))})
        except ValueError as exc: record["assessment_reason"]=str(exc)
        rows.append(record)
    scores=pd.DataFrame(rows);baseline=rows[0]
    for column in [c for c in scores if c.endswith("_nse")]: scores["delta_"+column]=scores[column]-baseline.get(column,np.nan)
    scores.to_csv(destination/"scores_vs_baseline.csv",index=False)
    pd.DataFrame(trajectories).to_csv(destination/"seasonal_bias.csv",index=False)
    pd.DataFrame(zones).to_csv(destination/"zone_diagnostics.csv",index=False)
    chosen_parameters=obj.bounds.validate(selected["vector"],obj.template)
    physical={key:getattr(chosen_parameters,key) for key in ("capacity_m3","initial_storage_m3","initial_release_m3s","channel_capacity_m3s","dead_storage_fraction","inflow_correction","flood_mode")}
    first=int(np.flatnonzero(np.isfinite(obj.runner.inflow))[0])
    physical['initial_storage_m3']=float(obj.runner.initials[obj.runner.months[first]])
    physical['initial_release_m3s']=dztr_release(physical['initial_storage_m3'],float(obj.runner.inflow[first]),int(frame.index[first].month),chosen_parameters)
    selected_scores=next(r for r in rows if r["selected"])
    profile={"reservoir_id":cfg["reservoir_id"],"grand_id":cfg["GRAND_ID"],"seed":selected["seed"],"evaluation_id":selected["evaluation_id"],
        "route":"daily_mean_interval_scalar_v1","year_selection":"mixed_whole_years","scoring":"pooled_weighted_nse",
        "physical_constants":physical,"sampling":{"frequency":"daily","storage_semantics":"interval_end"},
        "root_policy":"nearest_previous_storage","atol_storage_m3":cfg["balance_tolerance_m3"],
        "objective_weights":obj.weights,"evaluation_budget":cfg["optimization"]["evaluation_budget"],
        "status":"calibrated_exploratory_source_and_clock_assumptions","physical_source_certified":False,
        "inflow_independence_verified":False,"provider_clock_verified":False,
        "initialization":data.read(out/"prepared/initialization.json"),"source_decisions":cfg["source_decisions"],
        "assessment":selected_scores,"selection_sha256":plan.signature,"prepared_scenario":str(out.relative_to(ROOT))}
    # Shared registry exporter and parameter loader remain usable; simulation
    # requires this daily-mean adapter rather than the native endpoint adapter.
    inventory=pd.read_csv(ROOT/cfg["inventory_csv"],float_precision="round_trip")
    target=inventory.reservoir_id==cfg["reservoir_id"]
    if target.sum()!=1: raise ValueError("Reservoir must have exactly one inventory row")
    updates={"GRAND_ID":cfg["GRAND_ID"],"parameter_status":"calibrated_daily_interval_exploratory",
        "calibration_status":"completed_dds_under_frozen_source_scenario","calibration_ready":True,"parameters_ready":True,
        "simulation_ready":selected_scores["assessment_feasible"],"capacity_m3":physical["capacity_m3"],
        "channel_capacity_m3s":physical["channel_capacity_m3s"],"monthly_targets_file":str((destination/"profiles"/cfg["reservoir_id"]/"monthly_targets_si.csv").relative_to(ROOT)),
        "target_fitting_scope":"calibration-only mixed years; fixed exclusions and gap warmup", "availability_route":"observed_monthly", "next_action":"Retain exploratory profile; verify provider inflow independence and clock before predictive use"}
    for month,row in chosen_parameters.monthly_targets.iterrows():
        for column,value in row.items():updates[f"m{month:02d}_{column}"]=value
    for key,value in updates.items(): inventory.loc[target,key]=value
    inventory.to_csv(destination/"reservoir_parameterization_all_62_by_GRAND_ID.csv",index=False)
    convergence=[]
    for seed,state in states.items():
        for budget in (128,256,cfg["optimization"]["evaluation_budget"]):
            if budget<=len(state["trials"]):
                finite=[r["cost"] for r in state["trials"][:budget] if r["cost"] is not None]
                convergence.append({"seed":seed,"evaluations":budget,"best_calibration_cost":min(finite) if finite else None})
    pd.DataFrame(convergence).to_csv(destination/"convergence_checkpoints.csv",index=False)
    seed_scores=scores[scores.seed.notna()]
    write(destination/'seed_agreement.json',{'calibration_cost_range':float(seed_scores.calibration_cost.max()-seed_scores.calibration_cost.min()),
        'normalized_parameter_standard_deviation':np.std(np.array([c['vector'] for c in choices])/np.where(obj.bounds.upper_vector>obj.bounds.lower_vector,obj.bounds.upper_vector-obj.bounds.lower_vector,1.),axis=0).tolist(),
        'seed_count':len(choices),'convergence_established':False})
    _plots(destination,simulations,states,selected["seed"],scores)
    comparison=['| Period | Variable | Generalized NSE | Selected calibrated NSE | Change |','|---|---|---:|---:|---:|']
    for role in ('calibration','assessment'):
        for variable in ('storage','release'):
            key=f'{role}_{variable}_nse';before=baseline.get(key);after=selected_scores.get(key)
            comparison.append(f'| {role} | {variable} | {before:.4f} | {after:.4f} | {after-before:+.4f} |' if before is not None and after is not None else f'| {role} | {variable} | unavailable | unavailable | see failure reason |')
    comparison_text='\n'.join(comparison)
    text=f"""# Hungry Horse calibration: GRAND {cfg['GRAND_ID']}

{len(cfg['optimization']['seeds'])} DDS seeds, {cfg['optimization']['evaluation_budget']} evaluations each; all 72 monthly targets fitted using calibration observations only. The selected seed is **{selected['seed']}**, chosen by calibration cost before held-out assessment. Objective weights: {obj.weights}.

This is an exploratory, segmented daily-mean scenario. Storage uses the whole USBRPN archive; inflow/release use the separate USBR archive. The excluded inflow date is {cfg['excluded_inflow_dates']}. Missing inflows remain missing. Each finite-forcing block starts from the frozen calibration-only monthly normal storage; the first {cfg['restart_warmup_days']} days of each block are excluded from fitting/scoring. Held-out storage never initializes a block. Annual inflow regimes use available-day annual means with at least 95% inflow coverage; no missing daily inflows are filled.

Source timing is an explicit interval-end convention, not a verified provider clock. Reported inflow independence remains unverified. Capacity includes documented surcharge storage, separately from normal full pool; the dead threshold uses documented physical dead storage. These model settings are fixed, not optimized.

{comparison_text}

[Admission and assumptions](../prepared/admission.json), [annual year selection](../prepared/annual_year_selection.csv), [seed comparison against baseline](scores_vs_baseline.csv), [seasonal bias](seasonal_bias.csv), [convergence checkpoints](convergence_checkpoints.csv), [all 62 reservoirs](reservoir_parameterization_all_62_by_GRAND_ID.csv), [shared profile registry](registry.json).

![Calibration and assessment](performance.png)

![Time series](timeseries.png)

![Search histories](convergence.png)

All four full trajectories were checked against the unchanged canonical simulator where assessment succeeded. Maximum independent mass-balance residual: {maximum:.9g} m3, tolerance {cfg['balance_tolerance_m3']} m3. The 512-evaluation checkpoint does not establish convergence. Every seed is assessed; held-out scores do not choose or replace the selected seed. Assessment failures remain recorded.

Use `python -B tools/calibrate_daily_intervals_v1.py replay --output {out.relative_to(ROOT).as_posix()}` to replay the exported GRAND profile through the proper mean-flow adapter. Re-running `run` verifies and reuses completed jobs. Checkpoints can resume within their frozen horizon; increasing that horizon changes the DDS experiment.
"""
    (destination/"README.md").write_text(text,encoding="utf-8")
    write(destination/"scientific_checks.json",{"maximum_independent_balance_residual_m3":maximum,
        "canonical_trajectories_verified":len(simulations),"selection_uses_assessment":False,
        "assessment_storage_used_for_starts":False,"optimizer_evaluations":sum(len(s["trials"]) for s in states.values()),
        "observations_modified_outside_declared_inflow_exclusions":False,"convergence_established":False})
    native.export_release(destination,[(profile,chosen_parameters.monthly_targets)],{"protocol_sha256":data.sha(out/"protocol.json"),"scenario":"daily_mean_interval_scalar_v1"})
    return destination


def replay(output):
    out,_=checked(output); obj,_,_,_=objective(out)
    profile,parameters=native.load_profile(out/"report/registry.json",grand_id=data.read(out/"protocol.json")["config"]["GRAND_ID"])
    result=obj.runner.run(parameters)
    saved=data.frame_read(out/f"report/simulation_seed_{profile['seed']}.csv")
    for column in result:np.testing.assert_array_equal(result[column],saved[column].to_numpy())
    return {"status":"exact_shared_registry_replay_passed","optimizer_evaluations":0}


def _plots(folder,simulations,states,selected_seed,scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,variable in zip(axes,("storage","release")):
        x=np.arange(len(scores));width=.36
        ax.bar(x-width/2,scores.get(f"calibration_{variable}_nse",pd.Series(np.nan,index=scores.index)),width,label="Calibration")
        ax.bar(x+width/2,scores.get(f"assessment_{variable}_nse",pd.Series(np.nan,index=scores.index)),width,label="Held-out")
        ax.set(xticks=x,xticklabels=scores.profile,ylabel="NSE",title=variable.title());ax.tick_params(axis='x',rotation=25);ax.axhline(0,color='gray',lw=.7)
    axes[0].legend(frameon=False);fig.savefig(folder/"performance.png",dpi=150);plt.close(fig)
    sim=simulations.get(f"seed_{selected_seed}")
    fig,axes=plt.subplots(2,1,figsize=(12,6),constrained_layout=True)
    if sim is not None:
        for ax,observed,modeled,scale,label in ((axes[0],"observed_storage_m3","storage_m3",1e9,"Storage (billion m3)"),(axes[1],"observed_release_m3s","interval_mean_total_outflow_m3s",1.,"Release (m3/s)")):
            # Monthly means are only for display; every model step remains daily.
            monthly=sim[[observed,modeled]].resample("MS").mean()/scale
            ax.plot(monthly.index,monthly[observed],color='black',lw=1,label="Observed monthly mean")
            ax.plot(monthly.index,monthly[modeled],lw=1,label="Calibrated monthly mean")
            for year in sorted(set(sim.index.year[sim.role=='assessment'])):ax.axvspan(pd.Timestamp(year,1,1),pd.Timestamp(year+1,1,1),color='gray',alpha=.12)
            ax.set(ylabel=label);ax.grid(alpha=.2)
        axes[0].legend(frameon=False,ncol=2);axes[0].set_title("Hungry Horse: daily model, monthly display; gray years are held out")
    fig.savefig(folder/"timeseries.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    for seed,state in states.items():
        costs=np.array([np.inf if t['cost'] is None else t['cost'] for t in state['trials']])
        ax.plot(np.arange(1,len(costs)+1),np.minimum.accumulate(costs),label=f"Seed {seed}")
    ax.set(xlabel="Evaluations",ylabel="Best weighted calibration loss",title="DDS checkpoint histories; convergence is not presumed")
    ax.legend(frameon=False);ax.grid(alpha=.2);fig.savefig(folder/"convergence.png",dpi=150);plt.close(fig)
