"""Frozen provider-net scenario, storage-objective DDS and honest diagnostics."""
from pathlib import Path
import os
import subprocess
import sys
import numpy as np
import pandas as pd
from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.sampled import _observations
from dztr.model.release import dztr_release
from strategies import dds_v1,interval_scalar_v1 as old,provider_interval_v1 as strategy
from strategies.flow_only_v1 import nse
from strategies.representative_years import YearSelection,fit_selected_years
from workflows import conditional_cohort_v1 as shared,extension_assessment_v1 as d,reservoir_v2 as native
from workflows.atomic_io_v1 import write
from workflows.geometry_v1 import digest
ROOT=native.ROOT
FILES=(*shared.FILES,'strategies/provider_interval_v1.py','workflows/provider_calibration_v1.py',
    'workflows/minidoka_source_v1.py','tools/calibrate_provider_interval_v1.py','workflows/geometry_v1.py')


def checked(out):return shared.checked(out)


def fit(frame,plan,eligible,physical):
    view=frame.copy();view.loc[~eligible,['storage_m3','release_m3s']]=np.nan
    return fit_selected_years(_observations(view,'provider calibration-only target view'),plan,physical,quantile_method='hazen',allow_feasible_initialization=True)


def prepare(config,output):
    cfg=d.read(config);out=Path(output).resolve()
    if cfg['kind']!='provider_endpoint_interval_calibration_v1':raise ValueError('Wrong provider scenario')
    if cfg['objective_weights']!={'storage':1.,'release':0.}:raise ValueError('This published scenario calibrates storage only; QT spill timing remains unresolved')
    if out.exists():
        _,proof=checked(out)
        if proof['config']!=cfg:raise ValueError('Frozen source decision changed')
        return out
    if not out.is_relative_to(ROOT/'outputs') or out==ROOT/'outputs':raise ValueError('Use a fresh project output')
    opt=cfg['optimization']
    if (not opt['seeds'] or len(set(opt['seeds']))!=len(opt['seeds']) or not 1<=opt['evaluation_budget']<=opt['horizon'] or opt['horizon']<2
        or isinstance(opt['workers'],bool) or not isinstance(opt['workers'],int) or opt['workers']<1):raise ValueError('Invalid DDS settings')
    frame=d.frame_read(ROOT/cfg['observations_csv']).loc[cfg['start']:cfg['end']].copy()
    anchors,eligible=strategy.support(frame,cfg['restart_warmup_days']);plan,annual=old.select_years(frame,eligible)
    mask=plan.mask(frame.index,'calibration')&eligible
    if mask.mean()<.65:raise ValueError('Require 65% actual fitting support')
    physical=cfg['physical_constants'].copy();physical['channel_capacity_m3s']=float(np.percentile(frame.release_m3s[mask].dropna(),99,method='hazen'))
    template,bounds=fit(frame,plan,eligible,physical);initials=template.monthly_targets.storage_normal_m3.to_numpy()
    runner=strategy.SignedNetRunner(frame.inflow_m3s,initials,tolerance=cfg['balance_tolerance_m3'])
    objective=strategy.EndpointObjective(frame,plan,eligible,runner,template,bounds,objective_weights=cfg['objective_weights'])
    result=objective(parameters_to_vector(template))
    if not result.feasible:raise ValueError('Initializer fails: '+str(result.reason))
    out.mkdir(parents=True);folder=out/'prepared';folder.mkdir()
    frame.to_csv(folder/'observations_si.csv');annual.to_csv(folder/'annual_selection.csv');write(folder/'selection.json',plan.as_dict())
    table=plan.table();table['score_eligible']=eligible;table['segment_initialization']=anchors;table['fitting_eligible']=mask;table.to_csv(folder/'selection.csv')
    template.monthly_targets.to_csv(folder/'baseline_targets_si.csv');bounds.generalized_targets.to_csv(folder/'raw_generalized_targets_si.csv');bounds.empirical.monthly_bounds.to_csv(folder/'bounds_si.csv')
    write(folder/'physical_constants.json',physical);write(folder/'initialization.json',{'monthly_initial_storage_m3':initials.tolist(),'method':'fixed calibration-only monthly normal storage at every forcing-block start','direct_assessment_storage_used':False})
    write(folder/'source_decisions.json',d.read(ROOT/cfg['source_review']))
    write(folder/'admission.json',{'status':'conditional_storage_objective_scenario','fitting_fraction':float(mask.mean()),'baseline_cost':result.cost,
        'inflow_is_provider_computed':True,'inflow_independence_verified':False,'QT_is_instantaneous_total':True,'release_objective_enabled':False,'initialization_kind':bounds.initialization_kind})
    inputs={str(Path(config).resolve().relative_to(ROOT)):d.sha(config)}
    for key in ('observations_csv','source_review','base_inventory','base_catalog'):inputs[cfg[key]]=d.sha(ROOT/cfg[key])
    evidence=(ROOT/cfg['source_review']).parent/'evidence'
    for p in evidence.glob('*'):
        if p.is_file():inputs[str(p.relative_to(ROOT))]=d.sha(p)
    write(out/'protocol.json',{'kind':cfg['kind'],'config':cfg,'source_sha256':{**native.source_hashes(),**{p:d.sha(ROOT/p) for p in FILES}},'input_sha256':inputs,
        'prepared_sha256':{p.relative_to(out).as_posix():d.sha(p) for p in folder.rglob('*') if p.is_file()},'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}})
    return out


def objective(output):
    out,proof=checked(output);cfg=proof['config'];frame=d.frame_read(out/'prepared/observations_si.csv');plan=YearSelection(**d.read(out/'prepared/selection.json'))
    eligible=d.frame_read(out/'prepared/selection.csv').score_eligible.to_numpy(bool)
    template,bounds=fit(frame,plan,eligible,d.read(out/'prepared/physical_constants.json'))
    pd.testing.assert_frame_equal(template.monthly_targets,pd.read_csv(out/'prepared/baseline_targets_si.csv',index_col='month',float_precision='round_trip'))
    runner=strategy.SignedNetRunner(frame.inflow_m3s,d.read(out/'prepared/initialization.json')['monthly_initial_storage_m3'],tolerance=cfg['balance_tolerance_m3'])
    return strategy.EndpointObjective(frame,plan,eligible,runner,template,bounds,objective_weights=cfg['objective_weights']),frame,plan,eligible


def worker(output,seed,budget=None):
    out,proof=checked(output);cfg=proof['config'];opt=cfg['optimization'];budget=opt['evaluation_budget'] if budget is None else budget
    if seed not in opt['seeds'] or not 1<=budget<=opt['horizon']:raise ValueError('Undeclared DDS seed/budget')
    obj,_,_,_=objective(out);folder=out/'jobs'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'running.lock').open('x') as f:f.write(str(os.getpid()))
    try:
        identity={'protocol_sha256':d.sha(out/'protocol.json'),'seed':seed};state=None
        if (folder/'checkpoint.json').exists():
            checkpoint=d.read(folder/'checkpoint.json')
            if checkpoint['identity']!=identity or checkpoint['state_sha256']!=digest(checkpoint['state']):raise ValueError('Checkpoint identity changed')
            state=checkpoint['state']
        def save(state,force=False):
            count=len(state['trials'])
            if force or count%16==0 or count==budget:
                write(folder/'checkpoint.json',{'identity':identity,'state':state,'state_sha256':digest(state)})
                finite=[t['cost'] for t in state['trials'] if t['cost'] is not None]
                write(folder/'progress.json',{'evaluations':count,'budget':budget,'best_cost':min(finite) if finite else None})
        result=dds_v1.search(obj,parameters_to_vector(obj.template),obj.bounds.lower_vector,obj.bounds.upper_vector,seed=seed,horizon=opt['horizon'],budget=budget,perturbation=opt['perturbation'],state=state,checkpoint=save)
        save(result,True);pd.DataFrame([{k:v for k,v in t.items() if k!='vector'} for t in result['trials']]).to_csv(folder/'trials.csv',index=False)
        write(folder/'completed.json',{'evaluations':len(result['trials']),'checkpoint_sha256':d.sha(folder/'checkpoint.json')});return folder
    finally:(folder/'running.lock').unlink(missing_ok=True)


def run(config,output):
    out=prepare(config,output);_,proof=checked(out)
    if (out/'report/manifest.json').exists():native.verify_registry(out/'report/registry.json');return out/'report'
    processes=[];handles=[];active=[]
    try:
        for seed in proof['config']['optimization']['seeds']:
            if len(active)>=proof['config']['optimization']['workers']:
                if active.pop(0).wait()!=0:raise ValueError('DDS job failed; inspect persistent execution logs')
            folder=out/'jobs'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True);log=(folder/'execution.log').open('a',encoding='utf-8');handles.append(log)
            processes.append(subprocess.Popen([sys.executable,'-B',str(ROOT/'tools/calibrate_provider_interval_v1.py'),'worker','--output',str(out),'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
            active.append(processes[-1])
        if any(p.wait()!=0 for p in processes):raise ValueError('DDS job failed; inspect persistent execution logs')
        return report(out)
    finally:
        for p in processes:
            if p.poll() is None:p.terminate();p.wait()
        for log in handles:log.close()


def report(output):
    out,proof=checked(output);cfg=proof['config'];folder=out/'report'
    if (folder/'manifest.json').exists():native.verify_registry(folder/'registry.json');return folder
    options=[];curves=[]
    for seed in cfg['optimization']['seeds']:
        checkpoint=d.read(out/'jobs'/f'seed_{seed}'/'checkpoint.json');state=checkpoint['state']
        if checkpoint['identity']!={'protocol_sha256':d.sha(out/'protocol.json'),'seed':seed} or checkpoint['state_sha256']!=digest(state):raise ValueError('Checkpoint corrupted')
        if len(state['trials'])!=cfg['optimization']['evaluation_budget']:raise ValueError('Incomplete search')
        options.append({**min((t for t in state['trials'] if t['cost'] is not None),key=lambda t:(t['cost'],t['evaluation_id'])),'seed':seed})
        for budget in sorted(set([b for b in [128,256,cfg['optimization']['evaluation_budget']] if b<=len(state['trials'])])):
            curves.append({'seed':seed,'evaluations':budget,'best_cost':min(t['cost'] for t in state['trials'][:budget] if t['cost'] is not None)})
    selected=min(options,key=lambda t:(t['cost'],t['seed'],t['evaluation_id']));write(out/'selection_frozen.json',{'selection_uses_assessment':False,'selected':selected,'per_seed':options})
    folder.mkdir(exist_ok=True);obj,frame,plan,eligible=objective(out);rows=[];checks=[]
    for label,trial,vector in [('baseline',None,parameters_to_vector(obj.template))]+[(f"seed_{t['seed']}",t,t['vector']) for t in options]:
        parameters=obj.bounds.validate(vector,obj.template);evaluation=obj(vector)
        if not evaluation.feasible or (trial is not None and abs(evaluation.cost-trial['cost'])>1e-10):raise ValueError('Calibration replay failed')
        row={'profile':label,'seed':None if trial is None else trial['seed'],'selected':trial is not None and trial['seed']==selected['seed'],'calibration_cost':evaluation.cost,'assessment_feasible':False}
        try:
            sim=obj.runner.run(parameters);maximum=shared.canonical_check(obj.runner,parameters,sim)
            saved=pd.DataFrame(sim,index=frame.index).join(plan.table());saved['score_eligible']=eligible
            for c in d.COLUMNS:saved['observed_'+c]=frame[c]
            saved.to_csv(folder/f'simulation_{label}.csv')
            for role in ['calibration','assessment']:
                mask=plan.mask(frame.index,role)&eligible
                row[role+'_storage_nse']=nse(frame.storage_m3.to_numpy()[mask],sim['storage_m3'][mask])
                row[role+'_operating_endpoint_vs_instant_total_diagnostic_nse']=nse(frame.release_m3s.to_numpy()[mask],sim['release_endpoint_m3s'][mask])
                row[role+'_interval_mean_vs_instant_total_proxy_nse']=nse(frame.release_m3s.to_numpy()[mask],sim['interval_mean_total_outflow_m3s'][mask])
                row[role+'_spill_days']=int(((sim['spill_volume_m3']>obj.runner.tolerance)&mask).sum())
            row.update(assessment_feasible=True,maximum_balance_residual_m3=maximum);checks.append({'profile':label,'canonical_exact_match':True,'maximum_balance_residual_m3':maximum})
        except ValueError as exc:row['assessment_reason']=str(exc)
        rows.append(row)
    for row in rows:
        for role in ['calibration','assessment']:
            if role+'_storage_nse' in row:row['delta_'+role+'_storage_nse']=row[role+'_storage_nse']-rows[0][role+'_storage_nse']
    pd.DataFrame(rows).to_csv(folder/'scores_vs_baseline.csv',index=False);pd.DataFrame(curves).to_csv(folder/'convergence.csv',index=False)
    parameters=obj.bounds.validate(selected['vector'],obj.template);physical={k:getattr(parameters,k) for k in cfg['physical_constants']};physical['channel_capacity_m3s']=parameters.channel_capacity_m3s
    first=int(np.flatnonzero(np.isfinite(obj.runner.inflow))[0]);physical['initial_storage_m3']=float(obj.runner.initials[obj.runner.months[first]])
    physical['initial_release_m3s']=dztr_release(physical['initial_storage_m3'],0.,int(frame.index[first].month),parameters)
    profile={'reservoir_id':cfg['reservoir_id'],'grand_id':cfg['GRAND_ID'],'seed':selected['seed'],'evaluation_id':selected['evaluation_id'],
        'physical_constants':physical,'route':'provider_signed_net_storage_objective_v1','year_selection':'mixed_whole_years','scoring':'storage_nse',
        'objective_weights':cfg['objective_weights'],'sampling':{'frequency':'daily','inflow_semantics':'provider_computed_interval_mean_net','release_semantics':'instant_total_QT; diagnostic_only'},
        'status':'conditional_exploration_unverified_historical_coordinate_and_clock','physical_source_certified':False,'inflow_independence_verified':False,
        'independent_prediction_validated':False,'release_timing_validated':False,'release_performance_accepted':False,'root_policy':obj.runner.policy,'atol_storage_m3':obj.runner.tolerance,
        'source_decisions':d.read(out/'prepared/source_decisions.json'),'initialization':d.read(out/'prepared/initialization.json'),
        'assessment':next(r for r in rows if r['selected']),'evaluation_budget':cfg['optimization']['evaluation_budget'],'prepared_scenario':str(out.relative_to(ROOT))}
    write(folder/'scientific_checks.json',{'representatives':checks,'optimizer_evaluations':sum(len(d.read(out/'jobs'/f'seed_{seed}'/'checkpoint.json')['state']['trials']) for seed in cfg['optimization']['seeds']),
        'selection_uses_assessment':False,'reported_computed_inflow_is_not_independent':True,'instant_release_objective_enabled':False})
    _plots(folder,frame,plan,selected['seed'],rows)
    lines=['# Minidoka provider-record exploratory calibration','','Storage-only objective; three fixed DDS seeds. Provider-computed signed mean inflow is retained. Daily instantaneous total QT is not silently treated as an observed daily mean; release scores below are timing/operator diagnostics, not accepted release-performance scores.','',
        'Historical +115000 acre-foot storage offset and 222240 acre-foot gross ceiling are explicit sensitivity assumptions, not certified 1985-2017 physical metadata. No records are clipped or filled. This is a conditional assessment, not independent validation.','',
        '[Scores](scores_vs_baseline.csv), [convergence](convergence.csv), [registry](registry.json).','','| Profile | Fit storage NSE | Conditional assessment storage NSE |','|---|---:|---:|']
    for row in rows:lines.append(f"| {row['profile']} | {row.get('calibration_storage_nse',float('nan')):.4f} | {row.get('assessment_storage_nse',float('nan')):.4f} |")
    lines+=['','![Storage performance](performance.png)','','![Conditional trajectories](timeseries.png)','']
    (folder/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    native.export_release(folder,[(profile,parameters.monthly_targets)],{'protocol_sha256':d.sha(out/'protocol.json'),'scenario':'provider_net_storage_objective'});return folder


def replay(output):
    out,_=checked(output);obj,_,_,_=objective(out);profile,parameters=native.load_profile(out/'report/registry.json',identifier='etienne_Minidoka')
    saved=d.frame_read(out/f"report/simulation_seed_{profile['seed']}.csv");sim=obj.runner.run(parameters)
    for c in sim:np.testing.assert_array_equal(sim[c],saved[c].to_numpy())
    return {'status':'exact_provider_net_profile_replay','new_optimizer_evaluations':0}


def _plots(folder,frame,plan,seed,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=pd.DataFrame(rows);fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True);x=np.arange(len(table))
    ax.bar(x-.18,table.calibration_storage_nse,.36,label='Calibration');ax.bar(x+.18,table.assessment_storage_nse,.36,label='Conditional assessment');ax.set(xticks=x,xticklabels=table.profile,ylabel='Storage NSE',title='Minidoka - provider-computed forcing');ax.legend();fig.savefig(folder/'performance.png',dpi=150);plt.close(fig)
    sim=d.frame_read(folder/f'simulation_seed_{seed}.csv');fig,axes=plt.subplots(2,1,figsize=(13,6),constrained_layout=True)
    monthly=sim[['storage_m3','observed_storage_m3','interval_mean_total_outflow_m3s','observed_release_m3s']].resample('MS').mean()
    axes[0].plot(monthly.index,monthly.observed_storage_m3/1e6,color='black',label='Observed, historical gross coordinate');axes[0].plot(monthly.index,monthly.storage_m3/1e6,label='Modeled');axes[0].set(ylabel='Storage (million m3)');axes[0].legend()
    axes[1].plot(monthly.index,monthly.observed_release_m3s,color='black',label='Instant total QT, monthly sample mean');axes[1].plot(monthly.index,monthly.interval_mean_total_outflow_m3s,label='Modeled interval mean, monthly mean');axes[1].set(ylabel='Release (m3/s)',title='Release timing proxy only - not calibration objective');axes[1].legend()
    for ax in axes:
        for a,b in d.blocks(plan.mask(frame.index,'assessment')):ax.axvspan(frame.index[a],frame.index[b-1],color='gray',alpha=.12)
    fig.savefig(folder/'timeseries.png',dpi=150);plt.close(fig)
