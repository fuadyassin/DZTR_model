"""Source-declared cohort admission, checkpointed DDS and conditional assessment."""
from pathlib import Path
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd

from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.sampled import _observations
from dztr.model.release import dztr_release
from dztr.data.intervals import DailyIntervalPolicy, prepare_daily_intervals
from dztr.simulation.intervals import IntervalSimulationSettings, simulate_interval_reservoir
from strategies import dds_v1, interval_scalar_v1 as scalar
from strategies.conditional_interval_v1 import NonnegativeNetRunner, ShortRecordSelection, short_record_selection
from strategies.representative_years import YearSelection, fit_selected_years
from workflows import extension_assessment_v1 as d
from workflows import reservoir_v2 as native
from workflows.atomic_io_v1 import write
from workflows.geometry_v1 import digest

ROOT=native.ROOT
FILES=('strategies/conditional_interval_v1.py','workflows/conditional_cohort_v1.py','tools/calibrate_conditional_cohort_v1.py',
    'strategies/interval_scalar_v1.py','strategies/dds_v1.py','strategies/representative_years.py',
    'strategies/flow_only_v1.py','strategies/full_data_scalar_v1.py','workflows/extension_assessment_v1.py','workflows/atomic_io_v1.py')


def checked(output):
    out=Path(output).resolve(strict=True);proof=d.read(out/'protocol.json')
    for name,expected in proof['source_sha256'].items():
        if d.sha(ROOT/name)!=expected:raise ValueError('Cohort source changed: '+name)
    for name,expected in proof['input_sha256'].items():
        if d.sha(ROOT/name)!=expected:raise ValueError('Cohort input changed: '+name)
    for name,expected in proof['prepared_sha256'].items():
        p=(out/name).resolve(strict=True)
        if not p.is_relative_to(out) or d.sha(p)!=expected:raise ValueError('Prepared cohort changed: '+name)
    return out,proof


def selection(value):
    value=dict(value);kind=value.pop('kind',None)
    return ShortRecordSelection(**value) if kind=='short_record_chronological_v1' else YearSelection(**value)


def _fit(frame,plan,support,physical,allow_feasible):
    view=frame.copy();view.loc[~support,['storage_m3','release_m3s']]=np.nan
    return fit_selected_years(_observations(view,'conditional calibration-only target view'),plan,physical,
        quantile_method='hazen',allow_feasible_initialization=allow_feasible)


def prepare(config,output):
    cfg=d.read(config);out=Path(output).resolve()
    if cfg.get('kind')!='conditional_extension_cohort_v1':raise ValueError('Wrong cohort config')
    if out.exists():
        _,proof=checked(out)
        if cfg!=proof['config']:raise ValueError('Cohort decisions changed')
        return out
    if not out.is_relative_to(ROOT/'outputs') or out==ROOT/'outputs':raise ValueError('Use a fresh project output')
    ids=[c['reservoir_id'] for c in cfg['cases']]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate case')
    if set(cfg['decisions'])!={r['reservoir_id'] for r in d.read(ROOT/cfg['base_catalog'])['reservoirs']
        if r['reservoir_id'].startswith('etienne_') and r['reservoir_id']!='etienne_Hungry_Horse'}:
        raise ValueError('Keep all 20 remaining additions in the decision inventory')
    if not set(ids)<=set(cfg['decisions']):raise ValueError('Every case needs a source decision')
    opt=cfg['optimization']
    if (opt['horizon']<2 or not 1<=opt['evaluation_budget']<=opt['horizon'] or opt['workers']<1
        or len(set(opt['seeds']))!=len(opt['seeds']) or not opt['seeds']):raise ValueError('Invalid DDS settings')
    out.mkdir(parents=True);rows=[];inputs={str(Path(config).resolve().relative_to(ROOT)):d.sha(config)}
    for key in ('base_catalog','base_inventory'):inputs[cfg[key]]=d.sha(ROOT/cfg[key])
    for case in cfg['cases']:
        rid=d.collection.safe_id(case['reservoir_id']);folder=out/'prepared'/rid;folder.mkdir(parents=True)
        source=(ROOT/case['observations_csv']).resolve(strict=True);inputs[str(source.relative_to(ROOT))]=d.sha(source)
        original=d.frame_read(source)
        # The entire reconstruction is frozen before selection; it necessarily
        # contains assessment responses. Never describe its closure as validation.
        reconstructed=d.conditional_inflow(original)
        frame=reconstructed.loc[case['start']:case['end'],d.COLUMNS].copy()
        frame.to_csv(folder/'observations_si.csv')
        write(folder/'source_decision.json',{**case,'assessment_responses_enter_forcing':True,
            'inflow_independent':False,'formula':'I_net(t)=Q_mean(t)+(S(t)-S(t-1))/86400',
            'source_sha256':d.sha(source),'clock_verified':False,'capacity_verified':False})
        try:
            anchors,support=scalar.simulation_support(frame,restart_warmup_days=case['restart_warmup_days'])
            if case['selection']=='mixed_whole_years':
                plan,annual=scalar.select_years(frame,support,minimum_fraction=.65,seed=42)
                annual.to_csv(folder/'annual_selection.csv')
            elif case['selection']=='short_record_chronological':plan=short_record_selection(frame.index,support)
            else:raise ValueError('Unknown selection route')
            mask=plan.mask(frame.index,'calibration')&support
            if mask.mean()<.65:raise ValueError('Actual fitting support is below 65%')
            physical=dict(case['physical_constants'])
            physical['channel_capacity_m3s']=float(np.percentile(frame.release_m3s[mask].dropna(),99,method='hazen'))
            template,bounds=_fit(frame,plan,support,physical,case['allow_feasible_initialization'])
            runner=NonnegativeNetRunner(frame.inflow_m3s,template.monthly_targets.storage_normal_m3.to_numpy(),tolerance=cfg['balance_tolerance_m3'])
            obj=scalar.Objective(frame,plan,support,runner,template,bounds,objective_weights=cfg['objective_weights'])
            result=obj(parameters_to_vector(template))
            if not result.feasible:raise ValueError('Baseline initialization fails: '+str(result.reason))
            table=plan.table();table['score_eligible']=support;table['segment_initialization']=anchors;table['fitting_eligible']=mask
            table.to_csv(folder/'selection.csv');write(folder/'selection.json',plan.as_dict())
            write(folder/'physical_constants.json',physical)
            write(folder/'initialization.json',{'monthly_initial_storage_m3':runner.initials.tolist(),
                'method':'fixed calibration-only monthly normal storage at every forcing block start',
                'direct_assessment_storage_anchor_used':False})
            template.monthly_targets.to_csv(folder/'baseline_targets_si.csv')
            bounds.generalized_targets.to_csv(folder/'raw_generalized_targets_si.csv')
            bounds.empirical.monthly_bounds.to_csv(folder/'bounds_si.csv')
            row={'reservoir_id':rid,'status':'admitted_conditional_exploration','selection':case['selection'],
                'fitting_fraction':float(mask.mean()),'baseline_kind':bounds.initialization_kind,
                'baseline_calibration_cost':result.cost,'assessment_responses_enter_forcing':True,
                'direct_assessment_storage_anchor_used':False}
        except ValueError as exc:row={'reservoir_id':rid,'status':'preparation_blocked','reason':str(exc)}
        rows.append(row);write(folder/'admission.json',row)
    write(out/'admission.json',{'cases':rows,'all_remaining_additions':cfg['decisions']})
    pins={p.relative_to(out).as_posix():d.sha(p) for p in (out/'prepared').rglob('*') if p.is_file()}
    pins['admission.json']=d.sha(out/'admission.json')
    write(out/'protocol.json',{'kind':cfg['kind'],'config':cfg,'input_sha256':inputs,'prepared_sha256':pins,
        'source_sha256':{**native.source_hashes(),**{p:d.sha(ROOT/p) for p in FILES}},
        'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}})
    return out


def objective(output,rid):
    out,proof=checked(output);cfg=proof['config'];case=next(c for c in cfg['cases'] if c['reservoir_id']==rid)
    folder=out/'prepared'/rid;admission=d.read(folder/'admission.json')
    if admission['status']!='admitted_conditional_exploration':raise ValueError('Case is source/preparation gated')
    frame=d.frame_read(folder/'observations_si.csv');plan=selection(d.read(folder/'selection.json'))
    support=d.frame_read(folder/'selection.csv').score_eligible.to_numpy(bool)
    template,bounds=_fit(frame,plan,support,d.read(folder/'physical_constants.json'),case['allow_feasible_initialization'])
    pd.testing.assert_frame_equal(template.monthly_targets,pd.read_csv(folder/'baseline_targets_si.csv',index_col='month',float_precision='round_trip'))
    runner=NonnegativeNetRunner(frame.inflow_m3s,d.read(folder/'initialization.json')['monthly_initial_storage_m3'],tolerance=cfg['balance_tolerance_m3'])
    return scalar.Objective(frame,plan,support,runner,template,bounds,objective_weights=cfg['objective_weights']),frame,plan,support


def worker(output,rid,seed,budget=None):
    out,proof=checked(output);cfg=proof['config'];opt=cfg['optimization'];budget=opt['evaluation_budget'] if budget is None else budget
    if seed not in opt['seeds'] or not 1<=budget<=opt['horizon']:raise ValueError('Undeclared seed/budget')
    obj,_,_,_=objective(out,rid)
    folder=out/'jobs'/rid/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'running.lock').open('x') as stream:stream.write(str(os.getpid()))
    try:
        identity={'protocol_sha256':d.sha(out/'protocol.json'),'reservoir_id':rid,'seed':seed}
        state=None
        if (folder/'checkpoint.json').exists():
            checkpoint=d.read(folder/'checkpoint.json')
            if checkpoint['identity']!=identity or checkpoint['state_sha256']!=digest(checkpoint['state']):raise ValueError('Checkpoint identity changed')
            state=checkpoint['state']
        def save(current,force=False):
            count=len(current['trials'])
            if force or count%16==0 or count==budget:
                write(folder/'checkpoint.json',{'identity':identity,'state':current,'state_sha256':digest(current)})
                costs=[t['cost'] for t in current['trials'] if t['cost'] is not None]
                write(folder/'progress.json',{'evaluations':count,'budget':budget,'best_cost':min(costs) if costs else None})
        result=dds_v1.search(obj,parameters_to_vector(obj.template),obj.bounds.lower_vector,obj.bounds.upper_vector,
            seed=seed,horizon=opt['horizon'],budget=budget,perturbation=opt['perturbation'],state=state,checkpoint=save)
        save(result,True)
        pd.DataFrame([{k:v for k,v in t.items() if k!='vector'} for t in result['trials']]).to_csv(folder/'trials.csv',index=False)
        write(folder/'completed.json',{'evaluations':len(result['trials']),'checkpoint_sha256':d.sha(folder/'checkpoint.json')})
        return folder
    finally:(folder/'running.lock').unlink(missing_ok=True)


def run(config,output):
    out=prepare(config,output);_,proof=checked(out);cfg=proof['config']
    if (out/'report/manifest.json').exists():native.verify_registry(out/'report/registry.json');return out/'report'
    admitted=[r['reservoir_id'] for r in d.read(out/'admission.json')['cases'] if r['status']=='admitted_conditional_exploration']
    pending=iter([(rid,seed) for rid in admitted for seed in cfg['optimization']['seeds']]);active=[];handles=[]
    try:
        while True:
            while len(active)<cfg['optimization']['workers']:
                task=next(pending,None)
                if task is None:break
                rid,seed=task;folder=out/'jobs'/rid/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
                handle=(folder/'execution.log').open('a',encoding='utf-8');handles.append(handle)
                p=subprocess.Popen([sys.executable,'-B',str(ROOT/'tools/calibrate_conditional_cohort_v1.py'),'worker','--output',str(out),'--reservoir-id',rid,'--seed',str(seed)],
                    cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                active.append((task,p));print(f'Started {rid} seed {seed}',flush=True)
            if not active:break
            time.sleep(1)
            for task,p in active[:]:
                if p.poll() is not None:
                    if p.returncode:raise ValueError(f'Job failed: {task}; inspect its execution.log')
                    print(f'Completed {task}',flush=True);active.remove((task,p))
        return report(out)
    finally:
        for _,p in active:
            if p.poll() is None:p.terminate();p.wait()
        for handle in handles:handle.close()


def canonical_check(runner,parameters,fast):
    maximum=0.
    for a,b in d.blocks(np.isfinite(runner.inflow)):
        frame=pd.DataFrame({'inflow_m3s':runner.inflow[a:b],'storage_m3':runner.initials[runner.months[a]],'release_m3s':0.},index=runner.index[a:b])
        literal=simulate_interval_reservoir(prepare_daily_intervals(frame,policy=DailyIntervalPolicy('computed_net_inflow')),
            parameters,settings=IntervalSimulationSettings(root_policy=runner.policy),atol_storage_m3=runner.tolerance).frame
        for column in fast:np.testing.assert_array_equal(fast[column][a:b],literal[column].to_numpy())
        residual=np.diff(fast['storage_m3'][a:b])-runner.inflow[a+1:b]*86400+fast['interval_mean_total_outflow_m3s'][a+1:b]*86400
        if len(residual):maximum=max(maximum,float(np.max(np.abs(residual))))
    if maximum>runner.tolerance:raise ValueError('Independent volume check failed')
    return maximum


def metrics(frame,plan,support,sim):
    from strategies.flow_only_v1 import nse
    result={}
    for role in ('calibration','assessment'):
        mask=plan.mask(frame.index,role)&support
        for variable,observed,predicted in (('storage','storage_m3','storage_m3'),('release','release_m3s','interval_mean_total_outflow_m3s')):
            result[f'{role}_{variable}_nse']=nse(frame[observed].to_numpy()[mask],sim[predicted][mask])
    return result


def report(output):
    out,proof=checked(output);cfg=proof['config'];folder=out/'report'
    if folder.exists():native.verify_registry(folder/'registry.json');return folder
    admissions=d.read(out/'admission.json')['cases'];chosen=[];states={}
    for admission in admissions:
        if admission['status']!='admitted_conditional_exploration':continue
        rid=admission['reservoir_id'];options=[]
        for seed in cfg['optimization']['seeds']:
            checkpoint=d.read(out/'jobs'/rid/f'seed_{seed}'/'checkpoint.json');state=checkpoint['state']
            if checkpoint['state_sha256']!=digest(state) or checkpoint['identity']!={'protocol_sha256':d.sha(out/'protocol.json'),'reservoir_id':rid,'seed':seed}:raise ValueError('Unverified checkpoint')
            if len(state['trials'])!=cfg['optimization']['evaluation_budget']:raise ValueError('Cohort incomplete')
            finite=[t for t in state['trials'] if t['cost'] is not None]
            if not finite:raise ValueError('No feasible candidate')
            options.append({**min(finite,key=lambda t:(t['cost'],t['evaluation_id'])),'seed':seed,'reservoir_id':rid});states[(rid,seed)]=state
        chosen.append({'reservoir_id':rid,'per_seed':options,'selected':min(options,key=lambda t:(t['cost'],t['seed'],t['evaluation_id']))})
    write(out/'selection_frozen.json',{'selection_uses_assessment_metrics':False,'assessment_responses_enter_forcing':True,'choices':chosen})
    if not chosen:raise ValueError('No admitted cases; inspect admission.json before launching jobs')
    folder.mkdir();scores=[];profiles=[];checks=[];curves=[]
    for item in chosen:
        rid=item['reservoir_id'];obj,frame,plan,support=objective(out,rid);case=next(c for c in cfg['cases'] if c['reservoir_id']==rid)
        destination=folder/rid;destination.mkdir();sims={}
        evaluations=[('baseline',None,parameters_to_vector(obj.template))]+[(f"seed_{t['seed']}",t,t['vector']) for t in item['per_seed']]
        for label,t,vector in evaluations:
            parameters=obj.bounds.validate(vector,obj.template);evaluation=obj(vector)
            if not evaluation.feasible or (t is not None and abs(evaluation.cost-t['cost'])>1e-10):raise ValueError('Calibration cost failed exact replay')
            row={'reservoir_id':rid,'profile':label,'seed':None if t is None else t['seed'],'calibration_cost':evaluation.cost,
                'selected':t is not None and t['seed']==item['selected']['seed'],'selection':case['selection'],
                'assessment_responses_enter_forcing':True,'assessment_kind':'conditional_response_derived_net_flux','assessment_feasible':False}
            try:
                fast=obj.runner.run(parameters);balance=canonical_check(obj.runner,parameters,fast)
                row.update(metrics(frame,plan,support,fast),assessment_feasible=True,maximum_balance_residual_m3=balance)
                sim=pd.DataFrame(fast,index=frame.index).join(plan.table());sim['score_eligible']=support
                for column in d.COLUMNS:sim['observed_'+column]=frame[column]
                sim.to_csv(destination/f'simulation_{label}.csv');sims[label]=sim
                checks.append({'reservoir_id':rid,'profile':label,'canonical_exact_match':True,'maximum_balance_residual_m3':balance})
            except ValueError as exc:row['assessment_reason']=str(exc)
            scores.append(row)
        selected=item['selected'];parameters=obj.bounds.validate(selected['vector'],obj.template)
        physical={key:getattr(parameters,key) for key in ('capacity_m3','dead_storage_fraction','initial_storage_m3','initial_release_m3s','channel_capacity_m3s','inflow_correction','flood_mode')}
        first=int(np.flatnonzero(np.isfinite(obj.runner.inflow))[0])
        physical['initial_storage_m3']=float(obj.runner.initials[obj.runner.months[first]])
        physical['initial_release_m3s']=dztr_release(physical['initial_storage_m3'],0.,int(frame.index[first].month),parameters)
        profile={'reservoir_id':rid,'grand_id':case['GRAND_ID'],'seed':selected['seed'],'evaluation_id':selected['evaluation_id'],
            'year_selection':case['selection'],'scoring':'pooled_weighted_nse','physical_constants':physical,
            'route':'conditional_response_net_daily_v1','sampling':{'frequency':'daily','flow_semantics':'computed_net_inflow'},
            'root_policy':obj.runner.policy,'atol_storage_m3':obj.runner.tolerance,'selection_sha256':plan.signature,
            'status':'conditional_exploratory_unverified_storage_capacity_and_clock','assessment_responses_enter_forcing':True,
            'independent_prediction_validated':False,'physical_source_certified':False,'objective_weights':obj.weights,
            'initialization':d.read(out/'prepared'/rid/'initialization.json'),'source_decision':case,
            'assessment':next(r for r in scores if r['reservoir_id']==rid and r['selected']),
            'evaluation_budget':cfg['optimization']['evaluation_budget'],'prepared_scenario':str(out.relative_to(ROOT))}
        profiles.append((profile,parameters.monthly_targets))
        _plots(destination,rid,sims,item['selected']['seed'],[r for r in scores if r['reservoir_id']==rid])
        for seed in cfg['optimization']['seeds']:
            state=states[(rid,seed)]
            for budget in (128,256,cfg['optimization']['evaluation_budget']):
                if budget<=len(state['trials']):curves.append({'reservoir_id':rid,'seed':seed,'evaluations':budget,'best_cost':min(t['cost'] for t in state['trials'][:budget] if t['cost'] is not None)})
    for row in scores:
        baseline=next(r for r in scores if r['reservoir_id']==row['reservoir_id'] and r['profile']=='baseline')
        for key in ('calibration_storage_nse','calibration_release_nse','assessment_storage_nse','assessment_release_nse'):
            if key in row and key in baseline:row['delta_'+key]=row[key]-baseline[key]
    pd.DataFrame(scores).to_csv(folder/'scores_vs_baseline.csv',index=False);pd.DataFrame(curves).to_csv(folder/'convergence.csv',index=False)
    write(folder/'scientific_checks.json',{'representatives':checks,'optimizer_evaluations':sum(len(s['trials']) for s in states.values()),
        'selection_uses_assessment_metrics':False,'assessment_responses_enter_forcing':True,'independent_validation_established':False})
    inventory=pd.read_csv(ROOT/cfg['base_inventory'],float_precision='round_trip')
    for profile,targets in profiles:
        mask=inventory.reservoir_id==profile['reservoir_id']
        if mask.sum()!=1:raise ValueError('Inventory identity is ambiguous')
        updates={'parameter_status':'calibrated_conditional_exploration','calibration_status':'completed_dds_conditional_net',
            'active_profile_route':profile['route'],'active_profile_registry':str((folder/'registry.json').relative_to(ROOT)),
            'parameters_ready':True,'calibration_ready':True,'simulation_ready':profile['assessment']['assessment_feasible'],
            'capacity_m3':profile['physical_constants']['capacity_m3'],'capacity_verified':False,'provider_clock_verified':False,
            'channel_capacity_m3s':profile['physical_constants']['channel_capacity_m3s'],
            'monthly_calibrated_profile_available':True,'target_fitting_scope':'calibration_only_supported_rows',
            'monthly_targets_file':str((folder/'profiles'/profile['reservoir_id']/'monthly_targets_si.csv').relative_to(ROOT)),
            'independent_forcing_verified':False,'assessment_responses_enter_forcing':True,'selection_strategy':profile['year_selection'],
            'source_decision':profile['source_decision']['source_decision'],'active_profile_observations':str((out/'prepared'/profile['reservoir_id']/'observations_si.csv').relative_to(ROOT))}
        for month,values in targets.iterrows():
            for column,value in values.items():updates[f'm{month:02d}_{column}']=value
        for key,value in updates.items():inventory.loc[mask,key]=value
    for rid,decision in cfg['decisions'].items():
        mask=inventory.reservoir_id==rid
        if not any(p['reservoir_id']==rid for p,_ in profiles):inventory.loc[mask,'next_action']=decision
    inventory.to_csv(folder/'reservoir_parameterization_all_62_by_GRAND_ID.csv',index=False)
    pd.DataFrame([{'reservoir_id':rid,'status':'completed_conditional_calibration' if any(p['reservoir_id']==rid for p,_ in profiles) else 'deferred',
        'next_action':decision} for rid,decision in cfg['decisions'].items()]).to_csv(folder/'remaining_20_status.csv',index=False)
    lines=['# Conditional reservoir extension cohort','',
        f"{len(profiles)} new conditional profiles; all 20 remaining additions retained in the decision report. Existing profiles, including Hungry Horse, are unchanged.",'',
        '**These are response-conditioned experiments, not independent validation.** Net forcing is computed from the observed storage changes and observed release, including assessment responses. Numeric closure is therefore not proof of predictive skill. The zero dead-storage reference and catalog capacity are explicit sensitivity assumptions, not reconciled physical metadata.','',
        '[All 62 parameterizations](reservoir_parameterization_all_62_by_GRAND_ID.csv), [all 20 decisions](remaining_20_status.csv), [scores](scores_vs_baseline.csv), [registry](registry.json), [convergence](convergence.csv).','',
        '| Reservoir | Split | Calibration storage NSE | Calibration release NSE | Conditional assessment storage NSE | Conditional assessment release NSE |','|---|---|---:|---:|---:|---:|']
    for profile,_ in profiles:
        row=profile['assessment'];values=[f"{row[k]:.4f}" if k in row else 'failed' for k in ('calibration_storage_nse','calibration_release_nse','assessment_storage_nse','assessment_release_nse')]
        lines.append('| '+ ' | '.join([profile['reservoir_id'],profile['year_selection'],*values])+' |')
    lines+=['','Hugh Keenleyside uses whole mixed years classified by reconstructed net-flux annual means; those regimes are proxies, not independently observed naturalized wet/dry years. Revelstoke has fewer than two storage years and uses a chronological short-record split with all 12 fitting months and at least 65% actual fitting support; its later assessment omits part of the seasonal cycle and cannot establish climatic transferability.','',
        'Each seed is selected using calibration cost before assessment metrics are calculated. Missing forcing is not filled; gap starts are fixed calibration-only monthly normal storage with 30 excluded warmup days. Capacity/dead-storage assumptions, target fitting and source gates remain attached to every profile.','']
    for profile,_ in profiles:
        rid=profile['reservoir_id'];lines += [f'## {rid}','',f'![Performance]({rid}/performance.png)',f'![Time series]({rid}/timeseries.png)','']
    (folder/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    native.export_release(folder,profiles,{'protocol_sha256':d.sha(out/'protocol.json'),'assessment_kind':'conditional_net_flux'})
    return folder


def replay(output,rid):
    out,_=checked(output);obj,_,_,_=objective(out,rid)
    profile,parameters=native.load_profile(out/'report/registry.json',identifier=rid)
    result=obj.runner.run(parameters);saved=d.frame_read(out/'report'/rid/f"simulation_seed_{profile['seed']}.csv")
    for column in result:np.testing.assert_array_equal(result[column],saved[column].to_numpy())
    return {'reservoir_id':rid,'status':'exact_conditional_replay','new_optimizer_evaluations':0}


def _plots(folder,rid,simulations,seed,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scores=pd.DataFrame(records);x=np.arange(len(scores));fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,var in zip(axes,('storage','release')):
        ax.bar(x-.18,scores.get(f'calibration_{var}_nse',np.full(len(scores),np.nan)),.36,label='Calibration')
        ax.bar(x+.18,scores.get(f'assessment_{var}_nse',np.full(len(scores),np.nan)),.36,label='Conditional assessment')
        ax.set(xticks=x,xticklabels=scores.profile,ylabel='NSE',title=var.title());ax.tick_params(axis='x',rotation=25)
    axes[0].legend(frameon=False);fig.suptitle(rid+' — response-derived forcing');fig.savefig(folder/'performance.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,6),constrained_layout=True);sim=simulations.get(f'seed_{seed}')
    if sim is not None:
        for ax,observed,modeled,scale,label in ((axes[0],'observed_storage_m3','storage_m3',1e9,'Storage (billion m3)'),(axes[1],'observed_release_m3s','interval_mean_total_outflow_m3s',1.,'Release (m3/s)')):
            m=sim[[observed,modeled]].resample('MS').mean()/scale
            ax.plot(m.index,m[observed],color='black',label='Observed monthly mean');ax.plot(m.index,m[modeled],label='Modeled monthly mean')
            assessment=sim.index[sim.role=='assessment']
            for a,b in d.blocks(sim.role.eq('assessment').to_numpy()):ax.axvspan(sim.index[a],sim.index[b-1],alpha=.12,color='gray')
            ax.set(ylabel=label);ax.grid(alpha=.2)
        axes[0].legend(frameon=False);axes[0].set_title(rid+' — conditional reconstruction, monthly display')
    fig.savefig(folder/'timeseries.png',dpi=150);plt.close(fig)
