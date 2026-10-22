"""Version two: explicit calibration initializers and named short-record protocols."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import re

import numpy as np
import pandas as pd

from strategies.representative_years import (YearSelection, RepresentativeYearObjective,
    select_representative_years, fit_selected_years)
from strategies.fleet_comparison_v2 import comparison_selections
from strategies.short_record import select_short_record
from strategies.strategy_comparison import pooled_choice
from dztr.calibration.sampled import _observations
from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.optimizer import NSGA2Settings, optimize_nsga2
from dztr.calibration.checkpoint import load_optimizer_checkpoint, save_optimizer_checkpoint
from dztr.evaluation import evaluate_simulation
from dztr.schemas import ReservoirParameters
from dztr.simulation.sampled import SamplingSettings, validate_sampling_index
from dztr.simulation.root_policy import simulate_root_policy_reservoir, POLICIES

ROOT = Path(__file__).resolve().parents[1]
ARMS = {('mixed','pooled'): 'mixed_pooled', ('mixed','equal_regime'): 'mixed_balanced',
        ('chronological','pooled'): 'chronological_pooled'}
FLOW = {'m3/s': 1., 'm3/day': 1./86400.}
STORAGE = {'m3': 1., 'Mm3': 1e6}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    def clean(x):
        if isinstance(x, dict): return {k: clean(v) for k,v in x.items()}
        if isinstance(x, (tuple,list)): return [clean(v) for v in x]
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(base, value): return (Path(base)/value).resolve(strict=True)


def fresh_output(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT/'outputs') or path==ROOT/'outputs' or path.exists():
        raise ValueError('Use a fresh directory inside repository outputs')
    path.mkdir(parents=True)
    return path


def config(path):
    path=Path(path).resolve(strict=True);value=read(path)
    allowed={'schema_version','year_selection','scoring','optimization','reservoirs','archives'}
    if set(value)-allowed or value.get('schema_version') not in (1,2):raise ValueError('Invalid workflow configuration schema')
    value.setdefault('year_selection','mixed');value.setdefault('scoring','pooled')
    if value['year_selection'] not in ('mixed','chronological') or value['scoring'] not in ('pooled','equal_regime'):
        raise ValueError('Choose mixed/chronological years and pooled/equal_regime scoring')
    settings=dict(seeds=[42,43,44],population=32,evaluation_budget=512,workers=3)
    if set(value.get('optimization',{}))-set(settings):raise ValueError('Unknown optimization setting')
    settings.update(value.get('optimization',{}))
    seeds=settings['seeds']
    if not isinstance(seeds,list) or not seeds or len(set(seeds))!=len(seeds):raise ValueError('Declare unique seeds')
    for seed in seeds:NSGA2Settings(seed=seed,population_size=settings['population'],evaluation_budget=settings['evaluation_budget'])
    if settings['evaluation_budget']%settings['population']:raise ValueError('Budget must contain complete populations for reproducible continuation')
    if isinstance(settings['workers'],bool) or not isinstance(settings['workers'],int) or settings['workers']<1:raise ValueError('workers must be a positive integer')
    value['optimization']=settings
    entries=value.get('reservoirs',[])
    if not entries:raise ValueError('Declare at least one reservoir')
    identifiers=[];grand_ids=[]
    for entry in entries:
        if set(entry)-{'reservoir_id','grand_id','observations_csv','model_config','columns','units','initializer','selection_protocol'}:raise ValueError('Unknown reservoir configuration field')
        if value['schema_version']==1 and set(entry)&{'initializer','selection_protocol'}:raise ValueError('Initializer/short-record extensions require schema_version 2')
        if entry.get('selection_protocol','standard') not in ('standard','short_record_v1'):raise ValueError('Unknown selection protocol')
        if 'initializer' in entry:
            spec=entry['initializer']
            if set(spec)!={'csv','sha256','basis','selection_sha256','provenance'} or spec['basis']!='calibration_only':raise ValueError('Initializer requires pinned CSV, calibration-only basis, selection signature and provenance')
            if not all(isinstance(spec[k],str) and spec[k] for k in spec):raise ValueError('Invalid initializer metadata')
            if not re.fullmatch('[0-9a-f]{64}',spec['sha256']) or not re.fullmatch('[0-9a-f]{64}',spec['selection_sha256']):raise ValueError('Initializer signatures must be SHA256')
        identifier=entry['reservoir_id']
        if not isinstance(identifier,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',identifier):raise ValueError('Unsafe reservoir_id')
        identifiers.append(identifier)
        gid=entry.get('grand_id')
        if gid is not None:
            if isinstance(gid,bool) or not isinstance(gid,int) or gid<=0:raise ValueError('GRAND_ID must be a positive integer or null')
            grand_ids.append(gid)
    if len(set(identifiers))!=len(identifiers) or len(set(grand_ids))!=len(grand_ids):raise ValueError('Duplicate reservoir or GRAND_ID')
    return path,value


def observations(path, columns=None, units=None):
    columns=columns or {'date':'date','inflow':'inflow_m3s','storage':'storage_m3','release':'release_m3s'}
    units=units or {'inflow':'m3/s','storage':'m3','release':'m3/s'}
    if set(columns)-{'date','inflow','storage','release'} or not {'date','inflow'}<=set(columns):raise ValueError('Declare date/inflow and optional storage/release columns')
    if set(units)!=set(columns)-{'date'} or len(set(columns.values()))!=len(columns):raise ValueError('Declare explicit units for each unique measurement column')
    import csv
    with Path(path).open(encoding='utf-8-sig',newline='') as stream: header=next(csv.reader(stream))
    if len(set(header))!=len(header):raise ValueError('Duplicate CSV columns')
    raw=pd.read_csv(path,float_precision='round_trip')
    index=pd.DatetimeIndex(pd.to_datetime(raw[columns['date']],errors='raise'),name='date')
    if index.tz is not None or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:raise ValueError('Dates must be unique, sorted and timezone-naive')
    frame=pd.DataFrame(index=index)
    for name,target,factors in [('inflow','inflow_m3s',FLOW),('storage','storage_m3',STORAGE),('release','release_m3s',FLOW)]:
        if name not in columns:frame[target]=np.nan;continue
        if units[name] not in factors:raise ValueError(f'Unsupported {name} unit')
        values=pd.to_numeric(raw[columns[name]],errors='raise').to_numpy()*factors[units[name]]
        if np.isinf(values).any() or (values[np.isfinite(values)]<0).any():raise ValueError(f'{name} must be nonnegative or missing')
        frame[target]=values
    if not np.isfinite(frame.inflow_m3s).all():raise ValueError('Model-input inflows must be finite and nonnegative')
    return _observations(frame,str(path))


def source_hashes():
    paths=list((ROOT/'src/dztr').rglob('*.py'))
    paths += [ROOT/p for p in ['strategies/representative_years.py','strategies/strategy_comparison.py','strategies/fleet_comparison_v2.py','workflows/__init__.py','workflows/reservoir.py','tools/reservoir_workflow.py','workflows/reservoir_v2.py','tools/reservoir_workflow_v2.py','strategies/short_record.py']]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}


def check_protocol(out, identifier=None):
    out=Path(out).resolve()
    if not out.is_relative_to(ROOT/'outputs') or out==ROOT/'outputs':raise ValueError('Workflow must be inside repository outputs')
    protocol=read(out/'protocol.json')
    if protocol.get('kind')!='unified_reservoir_workflow_v2':raise ValueError('Use the original v1 runner for v1 optimizer preparations; checkpoint identities are not migrated')
    for rel,digest in protocol['code_sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError(f'Workflow/numerical source changed: {rel}; prepare a new version')
    for rel,digest in protocol['prepared_sha256'].items():
        if identifier is not None and not rel.startswith(f'prepared/{identifier}/'):continue
        if sha(out/rel)!=digest:raise ValueError(f'Prepared input changed: {rel}')
    if sha(out/'config.json')!=protocol['config_sha256']:raise ValueError('Prepared configuration changed')
    return out,protocol,read(out/'config.json')


def prepare(config_path, output):
    path,cfg=config(config_path);out=fresh_output(output);write(out/'config.json',cfg)
    inputs={str(path):sha(path)};records=[]
    for entry in cfg['reservoirs']:
        identifier=entry['reservoir_id'];folder=out/'prepared'/identifier;folder.mkdir(parents=True)
        data=resolve(path.parent,entry['observations_csv']);model_path=resolve(path.parent,entry['model_config'])
        inputs.update({str(data):sha(data),str(model_path):sha(model_path)})
        model=read(model_path)
        sampling=SamplingSettings(**model['sampling'])
        obs=observations(data,entry.get('columns'),entry.get('units'))
        validate_sampling_index(obs.frame.index,sampling.frequency)
        protocol_kind=entry.get('selection_protocol','standard')
        if protocol_kind=='short_record_v1':
            plan,selection_metadata=select_short_record(obs,sampling,cfg['year_selection'])
            held=selection_metadata['assessment_years']
        else:
            _,annual,warnings=select_representative_years(obs,sampling)
            plans,held=comparison_selections(obs.frame,annual)
            plan=plans['mixed_pooled' if cfg['year_selection']=='mixed' else 'chronological_pooled']
            selection_metadata={'kind':'standard','assessment_years':held,'missing_assessment_regimes':sorted({'dry','normal','wet'}-set(annual.loc[held,'regime']))}
        mode='pooled_nse' if cfg['scoring']=='pooled' else 'regime_balanced_nse'
        if mode=='regime_balanced_nse' and selection_metadata.get('undefined_calibration_regime_nse'):
            raise ValueError('Equal-regime scoring is undefined for '+', '.join(selection_metadata['undefined_calibration_regime_nse'])+'; review observations, not parameter choices')
        model=dict(model,cost_mode=mode)
        if not np.isfinite(model.get('atol_storage_m3',1e-4)) or model.get('atol_storage_m3',1e-4)<=0:raise ValueError('Invalid mass-balance tolerance')
        model.setdefault('atol_storage_m3',1e-4);model.setdefault('root_policy','reject')
        if model['root_policy'] not in POLICIES:raise ValueError('Unknown root policy')
        if model.get('source_preprocessing_already_applied') is False:raise ValueError('Normalize forcing before this model-input workflow')
        if model.get('quantile_method','hazen')!='hazen':raise ValueError('This workflow uses the verified Hazen percentile formulation')
        template,bounds=fit_selected_years(obs,plan,model['physical_constants'],allow_feasible_initialization=True)
        initialization_metadata={'kind':bounds.initialization_kind,'basis':'calibration_only'}
        if 'initializer' in entry:
            spec=entry['initializer'];initial_path=resolve(path.parent,spec['csv'])
            if sha(initial_path)!=spec['sha256']:raise ValueError('Initializer CSV changed')
            if plan.signature!=spec['selection_sha256']:raise ValueError('Initializer belongs to a different calibration selection')
            targets=pd.read_csv(initial_path,index_col='month',float_precision='round_trip')
            if not targets.index.equals(template.monthly_targets.index) or list(targets.columns)!=list(template.monthly_targets.columns):raise ValueError('Initializer must have exact native SI columns and months 1-12')
            explicit=ReservoirParameters(targets,**model['physical_constants'])
            template=bounds.validate(parameters_to_vector(explicit),template)
            inputs[str(initial_path)]=sha(initial_path)
            initialization_metadata=dict(spec,kind='explicit_calibration_initializer')
        write(folder/'initialization.json',initialization_metadata)
        write(folder/'selection_metadata.json',selection_metadata)
        # Construction validates solver policies and all physical/sampling inputs.
        RepresentativeYearObjective(obs,plan,template,sampling=sampling,bounds=bounds,policy=model['root_policy'],atol_storage_m3=model['atol_storage_m3'],cost_mode=mode)
        obs.frame.to_csv(folder/'observations_si.csv',index_label='date',float_format='%.17g')
        write(folder/'selection.json',plan.as_dict());write(folder/'model_config.json',model)
        plan.table().to_csv(folder/'row_selection.csv',index_label='date')
        bounds.generalized_targets.to_csv(folder/'raw_generalized_targets_si.csv',float_format='%.17g')
        template.monthly_targets.to_csv(folder/'initial_targets_si.csv',float_format='%.17g')
        bounds.empirical.monthly_bounds.to_csv(folder/'bounds_si.csv',float_format='%.17g')
        records.append({'reservoir_id':identifier,'grand_id':entry.get('grand_id'),'calibration_fraction':plan.roles.count('calibration')/len(plan.roles),'assessment_years':held,
                        'missing_assessment_regimes':selection_metadata['missing_assessment_regimes'],'selection_protocol':protocol_kind,'initialization_kind':initialization_metadata['kind'],
                        'source_review_required':selection_metadata.get('source_review_required',False)})
    write(out/'preparation.json',{'reservoirs':records,'new_evaluations':0,'prior_profiles_changed':False})
    write(out/'protocol.json',{'kind':'unified_reservoir_workflow_v2','code_sha256':source_hashes(),'input_sha256':inputs,'config_sha256':sha(out/'config.json'),
        'prepared_sha256':{p.relative_to(out).as_posix():sha(p) for p in sorted((out/'prepared').rglob('*')) if p.is_file()}})
    return out


def objective_for(out, identifier):
    folder=Path(out)/'prepared'/identifier;model=read(folder/'model_config.json');plan=YearSelection(**read(folder/'selection.json'))
    frame=pd.read_csv(folder/'observations_si.csv',index_col='date',parse_dates=True,float_precision='round_trip')
    obs=_observations(frame)
    template,bounds=fit_selected_years(obs,plan,model['physical_constants'],allow_feasible_initialization=True)
    initialization=read(folder/'initialization.json')
    if initialization['kind']=='explicit_calibration_initializer':
        targets=pd.read_csv(folder/'initial_targets_si.csv',index_col='month',float_precision='round_trip')
        template=bounds.validate(parameters_to_vector(ReservoirParameters(targets,**model['physical_constants'])),template)
    objective=RepresentativeYearObjective(obs,plan,template,sampling=SamplingSettings(**model['sampling']),bounds=bounds,
        policy=model['root_policy'],atol_storage_m3=model['atol_storage_m3'],cost_mode=model['cost_mode'])
    return objective,template,bounds,plan,frame,model


def calibrate(output, identifier, seed, budget=None):
    out,protocol,cfg=check_protocol(output,identifier)
    if identifier not in {e['reservoir_id'] for e in cfg['reservoirs']} or seed not in cfg['optimization']['seeds']:raise ValueError('Job not declared in configuration')
    selection_metadata=read(out/'prepared'/identifier/'selection_metadata.json')
    if selection_metadata.get('source_review_required'):raise ValueError('Constant short-record observation years require a source review before optimization; mathematical preparation is retained')
    folder=out/'jobs'/identifier/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    lock=folder/'calibration.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(__import__('os').getpid()))
    try:return _calibrate(output,identifier,seed,budget,validated=(out,protocol,cfg))
    finally:lock.unlink(missing_ok=True)


def _calibrate(output, identifier, seed, budget=None, validated=None):
    out,protocol,cfg=validated or check_protocol(output,identifier);settings=cfg['optimization']
    if identifier not in {e['reservoir_id'] for e in cfg['reservoirs']} or seed not in settings['seeds']:raise ValueError('Job not declared in configuration')
    budget=settings['evaluation_budget'] if budget is None else budget
    nsga=NSGA2Settings(seed=seed,population_size=settings['population'],evaluation_budget=budget)
    if budget%settings['population']:raise ValueError('Continuation budget must contain complete populations')
    folder=out/'jobs'/identifier/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    objective,template,bounds,_,_,_=objective_for(out,identifier)
    identity={'kind':'unified_reservoir_workflow_v2','protocol_sha256':sha(out/'protocol.json'),'reservoir_id':identifier,'seed':seed,'population':settings['population'],
        'objective':objective.identity,'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}}
    state=None
    if (folder/'checkpoint.npz').exists():
        if read(folder/'identity.json')!=identity:raise ValueError('Checkpoint identity changed')
        state=load_optimizer_checkpoint(folder/'checkpoint.npz',identity)
        if budget<len(state.evaluated):raise ValueError('Budget cannot shrink; existing budget snapshots are immutable')
    snapshot=folder/f'budget_{budget}'
    if (snapshot/'manifest.json').exists():
        if state is None or len(state.evaluated)<budget:raise ValueError('Incomplete checkpoint for snapshot')
        for rel,digest in read(snapshot/'manifest.json')['artifact_sha256'].items():
            if sha(snapshot/rel)!=digest:raise ValueError('Completed search snapshot changed')
        return snapshot
    write(folder/'identity.json',identity)
    def checkpoint(current):
        save_optimizer_checkpoint(folder/'checkpoint.npz',current,identity)
        write(folder/'progress.json',{'evaluations':len(current.evaluated),'generation':current.generation})
    result=optimize_nsga2(objective,parameters_to_vector(template),nsga,bounds=bounds,resume_state=state,checkpoint_callback=checkpoint)
    if result.evaluations!=budget:raise ValueError(f'Search stopped before requested budget: {result.termination}')
    snapshot.mkdir(exist_ok=True)
    pd.DataFrame([{'evaluation_id':p.evaluation_id,'generation':p.generation,'feasible':p.feasible,'release_cost':p.objectives[0],'storage_cost':p.objectives[1],'reason':p.reason} for p in result.evaluated]).to_csv(snapshot/'trials.csv',index=False,float_format='%.17g')
    pd.DataFrame([p.vector for p in result.evaluated],index=[p.evaluation_id for p in result.evaluated]).to_csv(snapshot/'vectors.csv',index_label='evaluation_id',float_format='%.17g')
    pd.DataFrame([{'evaluation_id':p.evaluation_id,'release_cost':p.objectives[0],'storage_cost':p.objectives[1]} for p in result.front],columns=['evaluation_id','release_cost','storage_cost']).to_csv(snapshot/'front.csv',index=False,float_format='%.17g')
    pd.DataFrame([asdict(g) for g in result.history]).to_csv(snapshot/'history.csv',index=False,float_format='%.17g')
    write(snapshot/'result.json',{'status':'completed','evaluations':result.evaluations,'termination':result.termination,'assessment_used':False})
    write(snapshot/'manifest.json',{'artifact_sha256':{p.name:sha(p) for p in snapshot.iterdir() if p.is_file()}})
    return snapshot


def selected(output,budget):
    out,_,cfg=check_protocol(output);records=[]
    for entry in cfg['reservoirs']:
        parts=[]
        for seed in cfg['optimization']['seeds']:
            folder=out/'jobs'/entry['reservoir_id']/f'seed_{seed}'/f'budget_{budget}'
            for rel,digest in read(folder/'manifest.json')['artifact_sha256'].items():
                if sha(folder/rel)!=digest:raise ValueError('Search snapshot changed')
            table=pd.read_csv(folder/'front.csv',float_precision='round_trip');table['seed']=seed;parts.append(table)
        if not any(len(part) for part in parts):raise ValueError(f'No feasible calibration front for {entry["reservoir_id"]}')
        point,_=pooled_choice(parts)
        records.append({'reservoir_id':entry['reservoir_id'],'grand_id':entry.get('grand_id'),'seed':int(point.seed),'evaluation_id':int(point.evaluation_id),
            'release_cost':float(point.release_cost),'storage_cost':float(point.storage_cost)})
    return records


def seed_selections(out,cfg,budget):
    records=[]
    for entry in cfg['reservoirs']:
        for seed in cfg['optimization']['seeds']:
            folder=out/'jobs'/entry['reservoir_id']/f'seed_{seed}'/f'budget_{budget}'
            table=pd.read_csv(folder/'front.csv',float_precision='round_trip');table['seed']=seed
            if table.empty:continue  # Explicitly recorded as unavailable below, not replaced.
            point,_=pooled_choice([table])
            records.append({'reservoir_id':entry['reservoir_id'],'seed':seed,'evaluation_id':int(point.evaluation_id)})
    return records


def independent_balance(frame, correction):
    s,q,i,dt=[frame[c].to_numpy() for c in ('storage_m3','release_m3s','inflow_m3s','step_seconds')]
    residual=np.diff(s)-dt[1:]/2*(correction*(i[1:]+i[:-1])-q[1:]-q[:-1])
    if not np.array_equal(residual,frame.balance_residual_m3.to_numpy()[1:]):raise ValueError('Independent balance disagrees with reported residual')
    return float(np.abs(residual).max()) if len(residual) else 0.


def assessment(parameters, sampling, policy, tolerance, source, plan):
    dates=source.index[plan.mask(source.index,'assessment')]
    try:
        result=simulate_root_policy_reservoir(_observations(source.loc[:dates[-1]]),parameters,sampling=sampling,policy=policy,atol_storage_m3=tolerance)
        residual=independent_balance(result.frame,parameters.inflow_correction)
        if result.frame.storage_m3.lt(0).any() or residual>tolerance:raise ValueError('Mass-balance/physical gate failed')
        metrics=evaluate_simulation(_observations(source.loc[dates]),replace(result,frame=result.frame.loc[dates]))
        labels=plan.table().loc[dates,'regime']
        for regime in sorted(set(labels)):
            group=dates[labels==regime]
            scores=evaluate_simulation(_observations(source.loc[group]),replace(result,frame=result.frame.loc[group]))
            for var in ('storage','release'):metrics[regime+'_'+var+'_nse']=scores[var+'_nse']
        for var in ('storage','release'):metrics['regime_balanced_'+var+'_nse']=np.mean([metrics[r+'_'+var+'_nse'] for r in sorted(set(labels))])
        frame=result.frame.copy()
        for col in ('storage_m3','release_m3s'):frame['observed_'+col]=source.loc[frame.index,col]
        return {'feasible':True,'metrics':metrics,'reason':None,'samples':len(dates),'maximum_balance_residual_m3':residual},frame.join(plan.table())
    except (ValueError,TypeError) as error:return {'feasible':False,'metrics':{},'reason':str(error),'samples':len(dates)},None


def export_release(release, profiles, provenance):
    registry={'schema_version':1,'kind':'unified_dztr_profiles','parameter_status':'exploratory','selection_uses_assessment':False,
              'profiles_by_reservoir_id':{},'profiles_by_GRAND_ID':{},'table':'reservoir_parameterization_by_GRAND_ID.csv','provenance':provenance}
    rows=[]
    for profile,targets in profiles:
        identifier=profile['reservoir_id'];folder=release/'profiles'/identifier;folder.mkdir(parents=True,exist_ok=True)
        targets.to_csv(folder/'monthly_targets_si.csv',index_label='month',float_format='%.17g')
        profile=dict(profile,monthly_targets_file='monthly_targets_si.csv',monthly_targets_sha256=sha(folder/'monthly_targets_si.csv'),selection_uses_assessment=False,convergence_established=False)
        write(folder/'parameters.json',profile)
        relative=(folder/'parameters.json').relative_to(release).as_posix()
        registry['profiles_by_reservoir_id'][identifier]=relative
        if profile.get('grand_id') is not None:registry['profiles_by_GRAND_ID'][str(profile['grand_id'])]=relative
        row={'reservoir_id':identifier,'GRAND_ID':profile.get('grand_id'),'seed':profile['seed'],'evaluation_id':profile['evaluation_id'],'year_selection':profile['year_selection'],'scoring':profile['scoring']}
        for month,values in targets.iterrows():
            for column,value in values.items():row[f'{column}_month_{int(month):02d}']=value
        rows.append(row)
    wide=pd.DataFrame(rows);wide['GRAND_ID']=wide['GRAND_ID'].astype('Int64')
    wide.to_csv(release/registry['table'],index=False,float_format='%.17g')
    write(release/'registry.json',registry)
    write(release/'manifest.json',{'code_sha256':source_hashes(),'artifact_sha256':{p.relative_to(release).as_posix():sha(p) for p in sorted(release.rglob('*')) if p.is_file() and p.name!='manifest.json'}})
    return release/'registry.json'


def finish(output,budget=None):
    out,_,cfg=check_protocol(output);budget=cfg['optimization']['evaluation_budget'] if budget is None else budget
    release=out/'releases'/f'budget_{budget}'
    if (release/'registry.json').exists() and (release/'manifest.json').exists():verify_registry(release/'registry.json');return release/'registry.json'
    records=selected(out,budget);seed_records=seed_selections(out,cfg,budget);release.mkdir(parents=True,exist_ok=True)
    frozen={'selection_uses_assessment':False,'selection_rule':'pooled nondominated calibration front; normalized distance to ideal','records':records,'seed_records':seed_records}
    if (release/'selection.json').exists():
        if read(release/'selection.json')!=frozen:raise ValueError('Frozen calibration selection changed')
    else:write(release/'selection.json',frozen)
    profiles=[];scores=[];seed_scores=[];cache={}
    for record in records:
        identifier=record['reservoir_id'];objective,_,_,plan,source,model=objective_for(out,identifier)
        vectors=pd.read_csv(out/'jobs'/identifier/f'seed_{record["seed"]}'/f'budget_{budget}'/'vectors.csv',index_col='evaluation_id',float_precision='round_trip')
        evaluation=objective.evaluate(vectors.loc[record['evaluation_id']].to_numpy())
        if not evaluation.feasible:raise ValueError(f'Selected calibration replay failed: {evaluation.reason}')
        if independent_balance(evaluation.simulation.frame,evaluation.parameters.inflow_correction)>objective.atol_storage_m3:raise ValueError('Selected calibration fails independent mass balance')
        np.testing.assert_allclose(evaluation.objectives,[record['release_cost'],record['storage_cost']],rtol=0,atol=1e-10)
        assessed,frame=assessment(evaluation.parameters,objective.sampling,objective.root_policy,objective.atol_storage_m3,source,plan)
        cache[(identifier,record['seed'],record['evaluation_id'])]=(evaluation,assessed)
        folder=release/'diagnostics'/identifier;folder.mkdir(parents=True,exist_ok=True)
        write(folder/'scores.json',{'calibration':evaluation.metrics,'assessment':assessed})
        if frame is not None:frame.to_csv(folder/'simulation.csv',index_label='date',float_format='%.17g')
        profile={**record,'year_selection':cfg['year_selection'],'scoring':cfg['scoring'],'physical_constants':model['physical_constants'],'sampling':model['sampling'],
            'root_policy':model['root_policy'],'atol_storage_m3':model['atol_storage_m3'],'initial_date':source.index[0].date().isoformat(),'selection_sha256':plan.signature,'initialization':read(out/'prepared'/identifier/'initialization.json'),'selection_protocol':read(out/'prepared'/identifier/'selection_metadata.json'),
            'calibration':evaluation.metrics,'assessment':assessed,'evaluation_budget':budget,'source_kind':'new_calibration','physical_source_certified':False}
        profiles.append((profile,evaluation.parameters.monthly_targets))
        scores.append({'reservoir_id':identifier,'assessment_feasible':assessed['feasible'],**{'calibration_'+k:v for k,v in evaluation.metrics.items()},**{'assessment_'+k:v for k,v in assessed['metrics'].items()}})
    for entry in cfg['reservoirs']:
        identifier=entry['reservoir_id'];objective,_,_,plan,source,_=objective_for(out,identifier)
        for seed in cfg['optimization']['seeds']:
            selection=next((r for r in seed_records if r['reservoir_id']==identifier and r['seed']==seed),None)
            if selection is None:
                seed_scores.append({'reservoir_id':identifier,'seed':seed,'assessment_feasible':False,'reason':'No feasible calibration front'});continue
            key=(identifier,seed,selection['evaluation_id'])
            if key in cache:evaluation,assessed=cache[key]
            else:
                vectors=pd.read_csv(out/'jobs'/identifier/f'seed_{seed}'/f'budget_{budget}'/'vectors.csv',index_col='evaluation_id',float_precision='round_trip')
                evaluation=objective.evaluate(vectors.loc[selection['evaluation_id']].to_numpy())
                if not evaluation.feasible:raise ValueError('Frozen seed calibration replay failed')
                assessed,_=assessment(evaluation.parameters,objective.sampling,objective.root_policy,objective.atol_storage_m3,source,plan)
            seed_scores.append({**selection,'assessment_feasible':assessed['feasible'],'reason':assessed['reason'],**{'calibration_'+k:v for k,v in evaluation.metrics.items()},**{'assessment_'+k:v for k,v in assessed['metrics'].items()}})
    pd.DataFrame(seed_scores).to_csv(release/'seed_scores.csv',index=False,float_format='%.17g')
    pd.DataFrame(scores).to_csv(release/'scores.csv',index=False,float_format='%.17g')
    (release/'README.md').write_text(f'# Versioned calibration release\n\n{len(profiles)} calibration-selected profiles; {cfg["year_selection"]} years, {cfg["scoring"]} scoring.\n\nAssessment failures: {sum(not p[0]["assessment"]["feasible"] for p in profiles)}. No profile was selected or replaced using assessment scores.\n\n[Registry](registry.json), [scores](scores.csv), [seed scores](seed_scores.csv), [parameter table](reservoir_parameterization_by_GRAND_ID.csv).\n',encoding='utf-8')
    return export_release(release,profiles,{'protocol_sha256':sha(out/'protocol.json'),'budget':budget})


def verify_registry(path):
    path=Path(path).resolve(strict=True);release=path.parent;proof=read(release/'manifest.json')
    for rel,digest in proof['code_sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError(f'Profile numerical/workflow source changed: {rel}')
    for rel,digest in proof['artifact_sha256'].items():
        target=(release/rel).resolve(strict=True)
        if not target.is_relative_to(release) or sha(target)!=digest:raise ValueError(f'Release artifact changed: {rel}')
    registry=read(path)
    if registry.get('kind')!='unified_dztr_profiles' or registry.get('selection_uses_assessment') is not False:raise ValueError('Invalid profile registry')
    return registry


def load_profile(registry_path, identifier=None, grand_id=None):
    if (identifier is None)==(grand_id is None):raise ValueError('Select exactly one reservoir_id or grand_id')
    path=Path(registry_path).resolve(strict=True);registry=verify_registry(path)
    relative=registry['profiles_by_reservoir_id'][identifier] if identifier is not None else registry['profiles_by_GRAND_ID'][str(grand_id)]
    target=(path.parent/relative).resolve(strict=True)
    if not target.is_relative_to(path.parent):raise ValueError('Profile path escapes release')
    profile=read(target);targets=pd.read_csv(target.parent/profile['monthly_targets_file'],index_col='month',float_precision='round_trip')
    if sha(target.parent/profile['monthly_targets_file'])!=profile['monthly_targets_sha256']:raise ValueError('Profile target table changed')
    return profile,ReservoirParameters(targets,**profile['physical_constants'])


def adopt(config_path, output):
    """Reuse completed comparison fronts/trajectories; run no new simulations."""
    path,cfg=config(config_path)
    if any(set(entry)&{'initializer','selection_protocol'} for entry in cfg['reservoirs']):raise ValueError('Archive import preserves recorded initializers and selections; extensions require fresh preparation')
    arm=ARMS.get((cfg['year_selection'],cfg['scoring']))
    if arm is None:raise ValueError('No archived chronological/equal-regime comparison; use a new calibration')
    if not cfg.get('archives'):raise ValueError('Declare completed comparison archive directories')
    sources={};consumed={str(path):sha(path)}
    def checked(base, proof, relative):
        target=(base/relative).resolve(strict=True)
        if not target.is_relative_to(base) or sha(target)!=proof['artifact_sha256'].get(relative):raise ValueError(f'Archive artifact changed: {relative}')
        consumed[str(target)]=sha(target)
        return target
    for relative in cfg['archives']:
        base=resolve(path.parent,relative);proof=read(base/'final_verification.json')
        consumed[str(base/'final_verification.json')]=sha(base/'final_verification.json')
        for rel,digest in proof['source_sha256'].items():
            if sha(ROOT/rel)!=digest:raise ValueError(f'Archived scientific source changed: {rel}')
        protocol=read(checked(base,proof,'protocol.json'))
        for rel,digest in protocol['source_sha256'].items():
            if rel.startswith('src/') and sha(ROOT/rel)!=digest:raise ValueError(f'Archived numerical source changed: {rel}')
        design=read(checked(base,proof,'design.json'))
        for spec in design['reservoirs']:
            identifier=spec['reservoir_id']
            if identifier in sources:raise ValueError('Duplicate reservoir across archives')
            sources[identifier]=(base,proof,design)
    selected_records=[];profiles=[];stored=[];seed_tables=[]
    # Freeze/recompute all choices from calibration fronts before consuming scores.
    for entry in cfg['reservoirs']:
        identifier=entry['reservoir_id'];base,proof,design=sources[identifier]
        specs=[j for j in design['jobs'] if j['reservoir_id']==identifier and j['arm']==arm]
        settings=cfg['optimization']
        if {j['seed'] for j in specs}!=set(settings['seeds']) or any(j['population']!=settings['population'] or j['evaluation_budget']!=settings['evaluation_budget'] for j in specs):raise ValueError('Configuration differs from completed search settings; use a new calibration')
        parts=[]
        for seed in settings['seeds']:
            rel=f'jobs/{identifier}/{arm}/seed_{seed}/pareto_front.csv'
            table=pd.read_csv(checked(base,proof,rel),float_precision='round_trip');table['seed']=seed;parts.append(table)
        point,_=pooled_choice(parts)
        record={'reservoir_id':identifier,'grand_id':entry.get('grand_id'),'seed':int(point.seed),'evaluation_id':int(point.evaluation_id),'release_cost':float(point.release_cost),'storage_cost':float(point.storage_cost)}
        frozen=read(checked(base,proof,'frozen_selection.json'))
        if frozen['selection_uses_assessment'] is not False:raise ValueError('Archived selection used assessment')
        archived=next(s for s in frozen['selections'] if s['reservoir_id']==identifier and s['arm']==arm and s['role']=='pooled_selected')
        for key in ('seed','evaluation_id','release_cost','storage_cost'):
            if archived[key]!=record[key]:raise ValueError('Archived selection differs from pooled calibration rule')
        selected_records.append(record);stored.append((entry,record,base,proof))
    out=fresh_output(output);write(out/'config.json',cfg)
    write(out/'selection.json',{'selection_uses_assessment':False,'records':selected_records,'selection_rule':'pooled nondominated calibration front; normalized distance to ideal'})
    for base,proof,design in {str(base):(base,proof,design) for base,proof,design in sources.values()}.values():
        table=pd.read_csv(checked(base,proof,'comparison_scores.csv'),float_precision='round_trip')
        table=table[(table.arm==arm)&(table.role=='seed_balanced')&table.reservoir_id.isin([r['reservoir_id'] for r in selected_records])]
        seed_tables.append(table)
    pd.concat(seed_tables,ignore_index=True).to_csv(out/'seed_scores.csv',index=False,float_format='%.17g')
    for entry,record,base,proof in stored:
        identifier=entry['reservoir_id'];prefix=f'assessed/{identifier}/{arm}/pooled_selected_seed{record["seed"]}_eval{record["evaluation_id"]}'
        targets=pd.read_csv(checked(base,proof,prefix+'/monthly_targets_si.csv'),index_col='month',float_precision='round_trip')
        model=read(checked(base,proof,f'prepared/{identifier}/{arm}/config.json'))
        plan=YearSelection(**read(checked(base,proof,f'prepared/{identifier}/{arm}/selection.json')))
        data=checked(base,proof,f'prepared/{identifier}/{arm}/observations_si.csv')
        initial_date=str(pd.read_csv(data,nrows=1)['date'].iloc[0])
        parameters=ReservoirParameters(targets,**model['physical_constants'])
        vectors=pd.read_csv(checked(base,proof,f'jobs/{identifier}/{arm}/seed_{record["seed"]}/vectors.csv'),index_col='evaluation_id',float_precision='round_trip')
        np.testing.assert_allclose(parameters_to_vector(parameters),vectors.loc[record['evaluation_id']].to_numpy(),rtol=1e-13,atol=1e-13)
        scores=read(checked(base,proof,prefix+'/scores.json'))
        profile={**record,'year_selection':cfg['year_selection'],'scoring':cfg['scoring'],'physical_constants':model['physical_constants'],'sampling':model['sampling'],
            'root_policy':model['root_policy'],'atol_storage_m3':model['atol_storage_m3'],'initial_date':initial_date,'selection_sha256':plan.signature,'initialization':{'kind':'archived_selected_profile','basis':'calibration_only'},'selection_protocol':{'kind':'archived_matched_comparison'},
            'calibration':{k.removeprefix('calibration_').replace('macro_','regime_balanced_'):v for k,v in scores.items() if k.startswith('calibration_')},
            'assessment':{'feasible':scores['assessment_feasible'],'reason':scores['assessment_reason'],'metrics':{k.removeprefix('assessment_'):v for k,v in scores.items() if k.startswith('assessment_') and k not in ('assessment_feasible','assessment_reason')},'samples':scores['common_samples']},
            'evaluation_budget':cfg['optimization']['evaluation_budget'],'source_kind':'verified_comparison_archive','source_archive':str(base),'physical_source_certified':False}
        profiles.append((profile,targets));folder=out/'diagnostics'/identifier;folder.mkdir(parents=True)
        write(folder/'scores.json',scores);write(folder/'year_selection.json',plan.as_dict())
        simulation=prefix+'/simulation.csv'
        if simulation in proof['artifact_sha256']:
            import shutil
            shutil.copyfile(checked(base,proof,simulation),folder/'simulation.csv')
            frame=pd.read_csv(folder/'simulation.csv',index_col='date',parse_dates=True,float_precision='round_trip')
            residual=independent_balance(frame,parameters.inflow_correction)
            if residual>model['atol_storage_m3'] or frame.storage_m3.lt(0).any():raise ValueError('Archived profile fails independent physical gate')
            profile['assessment']['maximum_balance_residual_m3']=residual
    write(out/'import_receipt.json',{'new_optimization_evaluations':0,'new_simulations':0,'selection_reproduced_from_calibration_fronts':True,'input_sha256':consumed})
    (out/'README.md').write_text(f'# Versioned calibration profile release\n\n{len(profiles)} verified profiles adopted with {cfg["year_selection"]} years and {cfg["scoring"]} scoring. No optimization or simulations were repeated.\n\nThe original registry is preserved. These profiles remain exploratory; assessment results are attached, not used for selection.\n\n[Registry](registry.json), [GRAND_ID parameters](reservoir_parameterization_by_GRAND_ID.csv), [import receipt](import_receipt.json).\n',encoding='utf-8')
    return export_release(out,profiles,{'kind':'verified_archive_import','configuration_sha256':sha(path),'archive_inputs':consumed})


def simulate(registry_path, forcing_csv, output, *, identifier=None, grand_id=None, initial=None):
    profile,parameters=load_profile(registry_path,identifier,grand_id)
    obs=observations(forcing_csv,{'date':'date','inflow':'inflow_m3s'},{'inflow':'m3/s'})
    if initial is None and obs.frame.index[0].date().isoformat()!=profile['initial_date']:raise ValueError('A different forcing start requires explicit initial storage_m3 and release_m3s')
    if initial is not None:
        if set(initial)!={'storage_m3','release_m3s'}:raise ValueError('Declare initial storage_m3 and release_m3s')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in initial.values()):raise ValueError('Initial states must be finite numeric SI values')
        parameters=replace(parameters,initial_storage_m3=initial['storage_m3'],initial_release_m3s=initial['release_m3s'])
    sampling=SamplingSettings(**profile['sampling']);validate_sampling_index(obs.frame.index,sampling.frequency)
    if sampling.month_clock=='legacy' and obs.frame.index[0].date().isoformat()!=profile['initial_date']:raise ValueError('A legacy target clock requires a dedicated profile for a different start date')
    result=simulate_root_policy_reservoir(obs,parameters,sampling=sampling,policy=profile['root_policy'],atol_storage_m3=profile['atol_storage_m3'])
    residual=independent_balance(result.frame,parameters.inflow_correction)
    if result.frame.storage_m3.lt(0).any() or residual>profile['atol_storage_m3']:raise ValueError('Simulation failed independent physical gate')
    out=fresh_output(output)
    result.frame.to_csv(out/'simulation.csv',index_label='date',float_format='%.17g')
    write(out/'receipt.json',{'reservoir_id':profile['reservoir_id'],'registry_sha256':sha(registry_path),'forcing_sha256':sha(forcing_csv),'initial':initial,
        'maximum_balance_residual_m3':residual})
    return out
