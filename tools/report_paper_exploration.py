"""Finalize completed paper-first studies using frozen selections and cached diagnostics."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from workflows import reservoir_v2 as w
from dztr.calibration.checkpoint import load_optimizer_checkpoint
from dztr.calibration.candidates import parameters_to_vector

def report(output):
    out=Path(output).resolve(strict=True)
    if not out.is_relative_to(ROOT/'outputs'):raise ValueError('Study must be inside outputs')
    review=w.read(out/'review.json');execution=w.read(out/'execution.json')
    if execution['stage']!='completed':raise ValueError('Complete study jobs before final reporting; keep checkpoints/logs')
    for rel,digest in review['source_sha256'].items():
        if w.sha(ROOT/rel)!=digest:raise ValueError('Study controller changed')
    dest=out/'report';dest.mkdir(exist_ok=True)
    if (dest/'manifest.json').exists():raise ValueError('Preserve the final report')
    old_folders=[ROOT/'outputs/unified_workflow_v1_20261013',
                 ROOT/'outputs/tyuyamuyun_calibration_v2_20261013/releases/budget_512',
                 ROOT/'outputs/charvak_calibration_source_v2_20261013/releases/budget_512']
    folders=old_folders+[ROOT/j['output']/'releases/budget_512' for j in review['ready_jobs']]
    summaries=[];comparisons=[];seed_rows=[];receipts=[];new_baseline_calls=0
    for job in review['ready_jobs']:
        identifier=job['reservoir_id'];prepared=ROOT/job['output'];release=prepared/'releases/budget_512'
        w.check_protocol(prepared);w.verify_registry(release/'registry.json')
        profile,parameters=w.load_profile(release/'registry.json',identifier=identifier)
        objective,initial,_,plan,source,_=w.objective_for(prepared,identifier)
        frozen_sha=w.sha(release/'selection.json')
        assert w.read(release/'selection.json')['selection_uses_assessment'] is False
        baseline_vector=parameters_to_vector(initial);jobs=[]
        for seed in (42,43,44):
            folder=prepared/'jobs'/identifier/f'seed_{seed}'
            state=load_optimizer_checkpoint(folder/'checkpoint.npz',w.read(folder/'identity.json'))
            snapshot=folder/'budget_512'
            vectors=pd.read_csv(snapshot/'vectors.csv',index_col='evaluation_id',float_precision='round_trip')
            trials=pd.read_csv(snapshot/'trials.csv',float_precision='round_trip')
            assert len(state.evaluated)==len(vectors)==len(trials)==512
            assert np.array_equal(state.baseline_vector,baseline_vector)
            assert np.array_equal(vectors.to_numpy(),np.asarray([p.vector for p in state.evaluated]))
            assert np.array_equal(trials[['release_cost','storage_cost']].to_numpy(),np.asarray([p.objectives for p in state.evaluated]))
            for rel,digest in w.read(snapshot/'manifest.json')['artifact_sha256'].items():assert w.sha(snapshot/rel)==digest
            jobs.append({'seed':seed,'evaluations':512,'feasible_trials':int(trials.feasible.sum()),
                         'checkpoint_sha256':w.sha(folder/'checkpoint.npz'),'state_snapshot_exact':True})
        baseline_path=dest/f'{identifier}_baseline_assessment.json'
        if baseline_path.exists():
            baseline=w.read(baseline_path)
            if baseline['selection_sha256_before_assessment']!=frozen_sha:raise ValueError('Cached baseline belongs to a different frozen selection')
        else:
            baseline,frame=w.assessment(initial,objective.sampling,objective.root_policy,objective.atol_storage_m3,source,plan)
            baseline.update({'selection_sha256_before_assessment':frozen_sha})
            if frame is not None:frame.to_csv(dest/f'{identifier}_baseline_simulation.csv',index_label='date',float_format='%.17g')
            w.write(baseline_path,baseline);new_baseline_calls+=1
        assert w.sha(release/'selection.json')==frozen_sha
        assumption=w.read(prepared.parent/'assumptions.json');preflight=w.read(prepared.parent/'preflight.json')
        selected_frame_path=release/'diagnostics'/identifier/'simulation.csv'
        residual=None;root_selections=None
        if selected_frame_path.exists():
            frame=pd.read_csv(selected_frame_path,index_col='date',parse_dates=True,float_precision='round_trip')
            residual=w.independent_balance(frame,parameters.inflow_correction)
            assert residual<=profile['atol_storage_m3'] and frame.storage_m3.between(0,parameters.capacity_m3).all()
            root_selections=int(frame.root_policy_applied.sum())
        for kind,period,metrics in [('fixed_initializer','calibration',preflight['attempts'][-1]['metrics']),
                                    ('selected','calibration',profile['calibration']),
                                    ('fixed_initializer','assessment',baseline['metrics']),
                                    ('selected','assessment',profile['assessment']['metrics'])]:
            comparisons.append({'reservoir_id':identifier,'profile':kind,'period':period,**metrics})
        for row in pd.read_csv(release/'seed_scores.csv',float_precision='round_trip').to_dict('records'):seed_rows.append(row)
        def gain(key):
            return profile['assessment']['metrics'].get(key,np.nan)-baseline['metrics'].get(key,np.nan)
        row={'reservoir_id':identifier,'in_published_paper':True,'capacity_m3':parameters.capacity_m3,
             'assumed_dead_storage_fraction':parameters.dead_storage_fraction,
             'calibration_fraction':preflight['calibration_fraction'],'selected_seed':profile['seed'],
             'selected_evaluation_id':profile['evaluation_id'],'assessment_feasible':profile['assessment']['feasible'],
             'baseline_assessment_feasible':baseline['feasible'],'calibration_storage_nse':profile['calibration'].get('storage_nse'),
             'calibration_release_nse':profile['calibration'].get('release_nse'),
             'baseline_calibration_storage_nse':preflight['attempts'][-1]['metrics'].get('storage_nse'),
             'baseline_calibration_release_nse':preflight['attempts'][-1]['metrics'].get('release_nse'),
             'calibration_storage_nse_gain':profile['calibration'].get('storage_nse',np.nan)-preflight['attempts'][-1]['metrics'].get('storage_nse',np.nan),
             'calibration_release_nse_gain':profile['calibration'].get('release_nse',np.nan)-preflight['attempts'][-1]['metrics'].get('release_nse',np.nan),
             'assessment_storage_nse':profile['assessment']['metrics'].get('storage_nse'),
             'assessment_release_nse':profile['assessment']['metrics'].get('release_nse'),
             'assessment_storage_nse_gain':gain('storage_nse'),'assessment_release_nse_gain':gain('release_nse'),
             'independent_assessment_balance_residual_m3':residual,
             'root_selections_in_stored_assessment_prefix':root_selections,
             'missing_assessment_regimes':','.join(profile['selection_protocol']['missing_assessment_regimes']),
             'source_definitions_certified':False,'release_registry':str((release/'registry.json').relative_to(ROOT))}
        fitting=source.loc[plan.mask(source.index,'calibration')]
        row['calibration_observations_above_model_capacity']=int(fitting.storage_m3.gt(parameters.capacity_m3).sum())
        row['calibration_observations_below_assumed_floor']=int(fitting.storage_m3.lt(parameters.capacity_m3*parameters.dead_storage_fraction).sum())
        raw_generalized=pd.read_csv(prepared/'prepared'/identifier/'raw_generalized_targets_si.csv',index_col='month',float_precision='round_trip')
        row['fixed_initializer_kind']='generalized' if np.array_equal(initial.monthly_targets.to_numpy(),raw_generalized.to_numpy()) else 'feasible_bounds_initializer'
        summaries.append(row);receipts.append({'reservoir_id':identifier,'jobs':jobs,
            'assumptions_sha256':w.sha(prepared.parent/'assumptions.json'),'assumptions':assumption,
            'selection_sha256_unchanged_after_baseline':frozen_sha,'independent_assessment_balance_residual_m3':residual,
            'assessment_failure_reason':profile['assessment']['reason'],'baseline_assessment_failure_reason':baseline['reason']})
    tables=[]
    paper_records={r['reservoir_id']:r for r in w.read(out/'cohort.json')['records']}
    assumption_files={j['reservoir_id']:str((Path(j['output']).parent/'assumptions.json').as_posix()) for j in review['ready_jobs']}
    for folder in folders:
        w.verify_registry(folder/'registry.json')
        table=pd.read_csv(folder/'reservoir_parameterization_by_GRAND_ID.csv',float_precision='round_trip')
        for column in ('flood_mode','root_policy','sampling_frequency','profile_registry','assumptions_file'):
            table[column]=pd.Series('',index=table.index,dtype='object')
        table['in_published_paper']=table.reservoir_id.isin(paper_records)
        for _,row in table.iterrows():
            saved_profile,params=w.load_profile(folder/'registry.json',identifier=row.reservoir_id)
            for month,values in params.monthly_targets.iterrows():
                for name,value in values.items():assert row[f'{name}_month_{month:02d}']==value
            for name,value in saved_profile['physical_constants'].items():table.loc[table.reservoir_id==row.reservoir_id,name]=value
            table.loc[table.reservoir_id==row.reservoir_id,'root_policy']=saved_profile['root_policy']
            table.loc[table.reservoir_id==row.reservoir_id,'sampling_frequency']=saved_profile['sampling']['frequency']
            table.loc[table.reservoir_id==row.reservoir_id,'fixed_step_days']=saved_profile['sampling'].get('fixed_step_days')
            table.loc[table.reservoir_id==row.reservoir_id,'profile_registry']=str((folder/'registry.json').relative_to(ROOT))
            table.loc[table.reservoir_id==row.reservoir_id,'in_published_paper']=row.reservoir_id in paper_records
            table.loc[table.reservoir_id==row.reservoir_id,'assumptions_file']=assumption_files.get(row.reservoir_id,'')
        tables.append(table)
    combined=pd.concat(tables,ignore_index=True)
    assert combined.reservoir_id.is_unique and combined.GRAND_ID.dropna().is_unique
    combined.to_csv(dest/'reservoir_parameterization_by_GRAND_ID.csv',index=False,float_format='%.17g')
    summary=pd.DataFrame(summaries);summary.to_csv(dest/'new_reservoir_scores.csv',index=False,float_format='%.17g')
    pd.DataFrame(comparisons).to_csv(dest/'baseline_comparison.csv',index=False,float_format='%.17g')
    pd.DataFrame(seed_rows).to_csv(dest/'seed_scores.csv',index=False,float_format='%.17g')
    programme=pd.read_csv(out/'prioritized_programme.csv')
    for row in summaries:
        programme.loc[programme.reservoir_id==row['reservoir_id'],['programme_status','next_action']]=[
            'calibrated_assumption_based_exploratory',
            'Retain frozen three-seed profile, explicit assumptions, baseline comparison and assessment failures/limitations; source definitions and convergence remain unresolved.']
    programme.to_csv(dest/'prioritized_programme.csv',index=False)
    paper_ids=set(w.read(out/'cohort.json')['records'][i]['reservoir_id'] for i in range(37))
    paper_standard=sum(identifier in paper_ids for identifier in combined.reservoir_id)
    paired=summary.loc[summary.assessment_feasible & summary.baseline_assessment_feasible]
    both=((paired.assessment_storage_nse_gain>0)&(paired.assessment_release_nse_gain>0)).sum()
    worse=((paired.assessment_storage_nse_gain<0)&(paired.assessment_release_nse_gain<0)).sum()
    equal=((paired.assessment_storage_nse_gain==0)&(paired.assessment_release_nse_gain==0)).sum()
    tests=[]
    for name in ('focused','full'):
        log=ROOT/f'outputs/paper_exploration_v1_{name}_tests_20261013.log'
        raw=log.read_bytes();text=raw.decode('utf-16') if raw.startswith(b'\xff\xfe') else raw.decode('utf-8')
        final=text.strip().splitlines()[-1]
        if 'failed' in final or 'passed' not in final:raise ValueError('Tests have not passed')
        tests.append({'suite':name,'result':final,'log_sha256':w.sha(log)})
    receipt={'status':'passed','paper_count':37,'standard_profile_count':len(combined),
        'paper_standard_profiles':paper_standard,'paper_separate_interval_studies':1,
        'paper_pending':37-paper_standard-1,'active_reservoir_count':41,'Tuyen_excluded':True,
        'new_calibrated_reservoirs':len(summaries),'new_optimization_evaluations':len(summaries)*1536,
        'new_diagnostic_baseline_assessment_calls':new_baseline_calls,
        'assessment_failures':int((~summary.assessment_feasible).sum()),
        'baseline_assessment_failures':int((~summary.baseline_assessment_feasible).sum()),
        'paired_baseline_comparisons':len(paired),'both_assessment_objectives_improved':int(both),
        'both_assessment_objectives_worse':int(worse),'both_assessment_objectives_equal':int(equal),
        'assessment_tradeoffs':int(len(paired)-both-worse-equal),
        'exact_parameter_values_verified':len(combined)*72,'verified_GRAND_ID_mappings':int(combined.GRAND_ID.notna().sum()),
        'selection_uses_assessment':False,'canonical_inputs_unchanged':True,'numerical_equations_unchanged':True,
        'convergence_demonstrated':False,'jobs_and_assumptions':receipts,'tests':tests,
        'source_sha256':{**review['source_sha256'],'tools/report_paper_exploration.py':w.sha(Path(__file__))},
        'profile_registries':[str((folder/'registry.json').relative_to(ROOT)) for folder in folders]}
    w.write(dest/'verification.json',receipt)
    return dest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(report(parser.parse_args().output))
