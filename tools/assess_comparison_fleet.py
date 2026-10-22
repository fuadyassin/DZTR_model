"""Freeze calibration-only winners, then assess identical excluded years."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulate_calibrated_reservoirs import ROOT,read,write,sha
from comparison_fleet_job import OUT,objective_for
sys.path.insert(0,str(ROOT))
from strategies.strategy_comparison import pooled_choice
from dztr.calibration.optimizer import EvaluatedCandidate
from dztr.calibration.selection import select_pareto_representatives
from dztr.calibration.sampled import _observations
from dztr.calibration.checkpoint import load_optimizer_checkpoint
from dztr.evaluation import evaluate_simulation
from dztr.simulation.root_policy import simulate_root_policy_reservoir
from run_critical_interventions import independent_balance


def main():
    if (OUT/'comparison_scores.csv').exists():raise ValueError('Assessment completed; reuse stored results')
    assert read(OUT/'execution_summary.json')['all_completed']
    design=read(OUT/'design.json');protocol=read(OUT/'protocol.json')
    for relative,digest in {**protocol['source_sha256'],**protocol['code_sha256']}.items():assert sha(ROOT/relative)==digest
    selections=[];vectors={};checks=0
    for spec in design['reservoirs']:
        identifier=spec['reservoir_id']
        for arm in spec['arms']:
            parts=[]
            for seed in (42,43,44):
                folder=OUT/'jobs'/identifier/arm/f'seed_{seed}'
                for name,digest in read(folder/'completion_manifest.json')['artifact_sha256'].items():assert sha(folder/name)==digest
                state=load_optimizer_checkpoint(folder/'checkpoint.npz',read(folder/'checkpoint_identity.json'))
                assert len(state.evaluated)==512
                table=pd.read_csv(folder/'vectors.csv',index_col='evaluation_id',float_precision='round_trip')
                trials=pd.read_csv(folder/'trials.csv',index_col='evaluation_id',float_precision='round_trip')
                np.testing.assert_array_equal(table.to_numpy(),[p.vector for p in state.evaluated])
                np.testing.assert_array_equal(trials[['release_cost','storage_cost']].to_numpy(),[p.objectives for p in state.evaluated])
                vectors[(identifier,arm,seed)]=table
                front=pd.read_csv(folder/'pareto_front.csv',float_precision='round_trip');front['seed']=seed
                candidates=tuple(EvaluatedCandidate(int(r.evaluation_id),0,table.loc[int(r.evaluation_id)].to_numpy(),(r.release_cost,r.storage_cost),True,None) for _,r in front.iterrows())
                winner=select_pareto_representatives(candidates)['balanced']
                point=front.set_index('evaluation_id').loc[winner]
                selections.append({'reservoir_id':identifier,'arm':arm,'role':'seed_balanced','seed':seed,'evaluation_id':winner,'release_cost':float(point.release_cost),'storage_cost':float(point.storage_cost)})
                parts.append(front);checks+=1
            point,_=pooled_choice(parts)
            selections.append({'reservoir_id':identifier,'arm':arm,'role':'pooled_selected','seed':int(point.seed),'evaluation_id':int(point.evaluation_id),'release_cost':float(point.release_cost),'storage_cost':float(point.storage_cost)})
            if arm=='mixed_balanced':
                for seed in (42,43,44):
                    np.testing.assert_array_equal(vectors[(identifier,'mixed_pooled',seed)].iloc[:32],vectors[(identifier,arm,seed)].iloc[:32])
    frozen={'selection_uses_assessment':False,'written_before_assessment':True,'selections':selections,'checkpoint_trials_verified':checks*512,'mixed_objective_initial_population_parity':True}
    if (OUT/'frozen_selection.json').exists():
        assert read(OUT/'frozen_selection.json')==frozen,'Calibration-only selection changed on restart'
    else:write(OUT/'frozen_selection.json',frozen)
    cache={};rows=[];physical_checks=[];cal_calls=0;assessment_calls=0
    for spec in design['reservoirs']:
        identifier=spec['reservoir_id']
        for arm in spec['arms']:
            objective,template,bounds,plan,source=objective_for(identifier,arm)
            choices=[s for s in selections if s['reservoir_id']==identifier and s['arm']==arm]
            choices.append({'reservoir_id':identifier,'arm':arm,'role':'generalized_baseline' if bounds.initialization_kind=='empirical_generalized' else 'feasible_initialization_reference','seed':42,'evaluation_id':0})
            for selection in choices:
                seed=selection['seed'];evaluation=selection['evaluation_id']
                vector=vectors[(identifier,arm,seed)].loc[evaluation].to_numpy()
                key=(identifier,plan.signature,vector.tobytes())
                if key not in cache:
                    calibration=objective.evaluate(vector);cal_calls+=1
                    if not calibration.feasible:
                        if selection['role'] not in ('generalized_baseline','feasible_initialization_reference'):raise ValueError(f'Chosen calibration replay failed: {selection}: {calibration.reason}')
                        rows.append({**selection,'assessment_feasible':False,'assessment_reason':'Baseline initialization failed calibration replay: '+str(calibration.reason),'common_samples':int(plan.roles.count('assessment'))});continue
                    mask=plan.mask(source.index,'assessment');dates=source.index[mask]
                    try:
                        if calibration.simulation.frame.index[-1]>=dates[-1]:common=replace(calibration.simulation,frame=calibration.simulation.frame.loc[:dates[-1]].copy())
                        else:
                            assessment_calls+=1
                            common=simulate_root_policy_reservoir(_observations(source.loc[:dates[-1]]),calibration.parameters,sampling=objective.sampling,policy=objective.root_policy,atol_storage_m3=objective.atol_storage_m3)
                        residual=independent_balance(common.frame,calibration.parameters.inflow_correction)
                        assert residual<=objective.atol_storage_m3
                        measured=_observations(source.loc[dates]);scored=replace(common,frame=common.frame.loc[dates])
                        metrics=evaluate_simulation(measured,scored)
                        labels=plan.table().loc[dates,'regime']
                        for regime in sorted(set(labels)):
                            group=labels==regime
                            groupmetrics=evaluate_simulation(_observations(measured.frame.loc[group]),replace(scored,frame=scored.frame.loc[group]))
                            for var in ('release','storage'):metrics[regime+'_'+var+'_nse']=groupmetrics[var+'_nse']
                        for var in ('release','storage'):metrics['regime_balanced_'+var+'_nse']=float(np.mean([metrics[r+'_'+var+'_nse'] for r in sorted(set(labels))]))
                        error=common.frame.loc[dates].storage_m3-source.loc[dates].storage_m3
                        metrics['storage_mean_bias_m3']=float(error.mean())
                        assessment={'feasible':True,'metrics':metrics,'max_balance_residual_m3':residual,'reason':None}
                    except (TypeError,ValueError,AssertionError) as error:
                        common=None;assessment={'feasible':False,'metrics':{},'reason':str(error)}
                    cache[key]=(calibration,common,assessment)
                calibration,common,assessment=cache[key]
                prefix='regime_balanced_' if objective.cost_mode=='regime_balanced_nse' else ''
                costs=(-calibration.metrics[prefix+'release_nse'],-calibration.metrics[prefix+'storage_nse'])
                if selection['role'] not in ('generalized_baseline','feasible_initialization_reference'):np.testing.assert_allclose(costs,[selection['release_cost'],selection['storage_cost']],rtol=0,atol=1e-10)
                row={**selection,'calibration_release_nse':calibration.metrics['release_nse'],'calibration_storage_nse':calibration.metrics['storage_nse'],
                     'calibration_macro_release_nse':calibration.metrics['regime_balanced_release_nse'],'calibration_macro_storage_nse':calibration.metrics['regime_balanced_storage_nse'],
                     'assessment_feasible':assessment['feasible'],'assessment_reason':assessment['reason'],'common_samples':int(plan.roles.count('assessment'))}
                for name,value in assessment['metrics'].items():row['assessment_'+name]=value
                rows.append(row)
                output=OUT/'assessed'/identifier/arm/f"{selection['role']}_seed{seed}_eval{evaluation}"
                output.mkdir(parents=True)
                write(output/'selection.json',selection)
                calibration.parameters.monthly_targets.to_csv(output/'monthly_targets_si.csv',float_format='%.17g')
                if common is not None:
                    frame=common.frame.copy()
                    for col in ('storage_m3','release_m3s'):frame['observed_'+col]=source.loc[frame.index,col]
                    frame=frame.join(plan.table())
                    frame.to_csv(output/'simulation.csv',index_label='date',float_format='%.17g')
                    physical_checks.append({'reservoir_id':identifier,'arm':arm,'role':selection['role'],'maximum_balance_residual_m3':assessment['max_balance_residual_m3']})
                write(output/'scores.json',{k:v if not isinstance(v,(float,np.floating)) or np.isfinite(v) else None for k,v in row.items()})
    # Historical 512-evaluation seed-balanced fits are explicitly unmatched references.
    old=ROOT/'outputs/policy_fleet_20261012';oldproof=read(old/'execution_verification.json')
    for spec in design['reservoirs']:
        identifier=spec['reservoir_id'];plan=objective_for(identifier,'chronological_pooled')[3]
        for seed in (42,43,44):
            folder=old/'pilots'/identifier/f'seed_{seed}';metadata=folder/'result.json'
            assert sha(metadata)==oldproof['artifact_sha256'][metadata.relative_to(old).as_posix()]
            selected=next(r for r in read(metadata)['representatives'] if r['role']=='balanced')
            path=folder/f"selected_{selected['evaluation_id']}_simulation.csv"
            assert sha(path)==oldproof['artifact_sha256'][path.relative_to(old).as_posix()]
            frame=pd.read_csv(path,index_col='date',parse_dates=True,float_precision='round_trip')
            dates=frame.index[plan.mask(frame.index,'assessment')]
            source=frame[['observed_storage_m3','observed_release_m3s','inflow_m3s']].rename(columns={'observed_storage_m3':'storage_m3','observed_release_m3s':'release_m3s'})
            metrics=evaluate_simulation(_observations(source.loc[dates]),type('Replay',(),{'frame':frame.loc[dates]})())
            rows.append({'reservoir_id':identifier,'arm':'historical_reference','role':'seed_balanced','seed':seed,'evaluation_id':selected['evaluation_id'],'assessment_feasible':True,'common_samples':len(dates),**{'assessment_'+k:v for k,v in metrics.items()}})
    table=pd.DataFrame(rows);table.to_csv(OUT/'comparison_scores.csv',index=False,float_format='%.17g')
    pooled=table[table.role=='pooled_selected'].copy();pooled.to_csv(OUT/'pooled_comparison.csv',index=False,float_format='%.17g')
    contrasts=[]
    for identifier in [s['reservoir_id'] for s in design['reservoirs']]:
        part=pooled[pooled.reservoir_id==identifier].set_index('arm')
        for name,before,after in [('year_selection','chronological_pooled','mixed_pooled'),('regime_weighting','mixed_pooled','mixed_balanced')]:
            for var in ('storage','release'):
                for metric in ('nse','macro_nse'):
                    col='assessment_'+(('regime_balanced_'+var+'_nse') if metric=='macro_nse' else var+'_nse')
                    contrasts.append({'reservoir_id':identifier,'contrast':name,'metric':var+'_'+metric,'change':float(part.loc[after,col]-part.loc[before,col])})
    pd.DataFrame(contrasts).to_csv(OUT/'paired_strategy_contrasts.csv',index=False,float_format='%.17g')
    write(OUT/'scientific_verification.json',{'status':'passed' if table.assessment_feasible.all() else 'checkpoint_and_balance_checks_passed_with_assessment_failures','checkpoint_jobs_verified':checks,'search_trials_verified':checks*512,'mixed_initial_population_parity':True,'selection_frozen_before_assessment':True,'assessment_rows_per_reservoir':{s['reservoir_id']:objective_for(s['reservoir_id'],'chronological_pooled')[3].roles.count('assessment') for s in design['reservoirs']},
          'calibration_replays':cal_calls,'assessment_forward_replays':assessment_calls,'assessment_from_cached_calibration_when_possible':True,'maximum_balance_residual_m3':max(r['maximum_balance_residual_m3'] for r in physical_checks),'assessment_failures':int((~table.assessment_feasible).sum()),'profiles_promoted':False})
    print(pooled[['reservoir_id','arm','assessment_storage_nse','assessment_release_nse','assessment_regime_balanced_storage_nse','assessment_regime_balanced_release_nse']].to_string(index=False))


if __name__=='__main__':main()
