"""Prepare forcing-stratified >=65% whole-year protocols for calibrated samples."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from simulate_calibrated_reservoirs import ROOT,read,write,sha
sys.path.insert(0,str(ROOT))
from forcing_batch import FrozenProfiles
from dztr.integrations.collection import load_collection,_verified_inputs
from dztr.integrations.world_reservoirs import _forcing
from strategies.representative_years import select_representative_years,fit_selected_years


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--reservoir-id',nargs='+')
    parser.add_argument('--minimum-fraction',type=float,default=.65)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--water-year-start-month',type=int,default=1)
    args=parser.parse_args()
    store=FrozenProfiles();ids=sorted(store.registry['profiles_by_reservoir_id']) if not args.reservoir_id else args.reservoir_id
    if len(set(ids))!=len(ids) or any(i not in store.registry['profiles_by_reservoir_id'] for i in ids):raise ValueError('Select unique calibrated reservoir IDs')
    root,manifest,manifest_path,_=load_collection(ROOT/'data/reservoirs/manifest.json')
    parent=ROOT/'outputs/policy_fleet_20261012';old=read(ROOT/'outputs/twenty_reservoir_workflow_20261012/protocol.json')
    assert sha(parent/'execution_verification.json')==old['parent_execution_sha256']
    assert sha(parent/'protocol.json')==read(parent/'execution_verification.json')['artifact_sha256']['protocol.json']
    hashes={p.replace('\\','/'):d for p,d in read(parent/'protocol.json')['inputs_sha256'].items()}
    assert sha(manifest_path)==hashes[manifest_path.relative_to(ROOT).as_posix()]
    entries={x['reservoir_id']:x for x in manifest['reservoirs']}
    output=Path(args.output).resolve()
    if output.exists() or not output.is_relative_to(ROOT/'outputs'):raise ValueError('Use a fresh repository outputs directory')
    output.mkdir(parents=True);results=[];consumed={manifest_path.relative_to(ROOT).as_posix():sha(manifest_path)}
    for identifier in ids:
        folder=output/identifier;folder.mkdir()
        try:
            profile,_,config=store.load({'reservoir_id':identifier})
            _,input_config,observations,input_hashes=_verified_inputs(root,entries[identifier])
            for path,digest in input_hashes.items():
                relative=path.relative_to(ROOT).as_posix()
                assert digest==hashes[relative];consumed[relative]=digest
            assert all(input_config[k]==config[k] for k in ('sampling','model','inflow_preprocessing'))
            observations=_forcing(observations,config['inflow_preprocessing'])
            from dztr.simulation.sampled import SamplingSettings
            sampling=SamplingSettings(**config['sampling'])
            selection,annual,warnings=select_representative_years(observations,sampling,minimum_fraction=args.minimum_fraction,seed=args.seed,water_year_start_month=args.water_year_start_month)
            selection.table().to_csv(folder/'row_selection.csv',index_label='date')
            annual.to_csv(folder/'year_selection.csv',float_format='%.17g')
            write(folder/'selection.json',selection.as_dict())
            mask=selection.mask(observations.frame.index,'calibration')
            inputs=observations.frame.copy();inputs.to_csv(folder/'observations_si.csv',index_label='date',float_format='%.17g')
            result={'reservoir_id':identifier,'GRAND_ID':profile['grand_id'],'status':'selection_prepared','total_rows':len(mask),'calibration_rows':int(mask.sum()),'calibration_fraction_total':float(mask.mean()),
                    'calibration_fraction_eligible':float(mask.sum()/sum(r in ('calibration','assessment') for r in selection.roles)),
                    'calibration_years':[int(y) for y in annual.index[annual.role=='calibration']], 'assessment_years':[int(y) for y in annual.index[annual.role=='assessment']],
                    'regime_counts':{role:{regime:int(((annual.role==role)&(annual.regime==regime)).sum()) for regime in ('dry','normal','wet')} for role in ('calibration','assessment')},
                    'warnings':warnings,'selection_sha256':selection.signature,'cost_mode':'regime_balanced_nse','heldout_interpretation':'retrospective representative-year assessment; not a blind future forecast test'}
            for variable,column in (('storage','storage_m3'),('release','release_m3s')):
                total=int(observations.frame[column].notna().sum());count=int(observations.frame.loc[mask,column].notna().sum())
                result['calibration_'+variable+'_pairs']=count;result['calibration_'+variable+'_fraction_total_pairs']=count/total if total else None
            write(folder/'model_config.json',{'physical_constants':profile['physical_constants'],'sampling':profile['sampling'],'root_policy':profile['root_policy'],'atol_storage_m3':config['atol_storage_m3'],'source_preprocessing_already_applied':True,'cost_mode':'regime_balanced_nse','quantile_method':'hazen'})
            try:
                template,bounds=fit_selected_years(observations,selection,profile['physical_constants'],allow_feasible_initialization=True)
                bounds.generalized_targets.to_csv(folder/'raw_generalized_targets_si.csv',float_format='%.17g')
                result['initialization_kind']=bounds.initialization_kind
                if bounds.initialization_kind!='empirical_generalized':
                    result['warnings'].append('Raw generalized targets fail physical ordering; preserved unchanged. Separate feasible bounds midpoint is an optimizer initialization, not a corrected generalized baseline.')
                template.monthly_targets.to_csv(folder/'calibration_targets_si.csv',float_format='%.17g')
                bounds.empirical.monthly_bounds.to_csv(folder/'calibration_bounds_si.csv',float_format='%.17g')
                # Observation variation check precedes any optimization job.
                for regime in sorted(set(np.asarray(selection.regimes)[mask])):
                    group=observations.frame.loc[mask & (np.asarray(selection.regimes)==regime)]
                    for column in ('storage_m3','release_m3s'):
                        values=group[column].dropna().to_numpy()
                        if len(values)<2 or not np.isfinite(values).all() or np.ptp(values)==0:raise ValueError(f'Undefined calibration NSE in {regime}/{column}')
                result['status']='fit_prepared_not_simulated'
            except (TypeError,ValueError) as error:
                result['status']='selection_prepared_fit_needs_review';result['fitting_issue']=str(error)
        except Exception as error:result={'reservoir_id':identifier,'status':'blocked','reason':str(error)}
        write(folder/'preparation.json',result);results.append(result);print(f"{identifier}: {result['status']}",flush=True)
    summary={'strategy':'representative_years_v1','minimum_fraction':args.minimum_fraction,'seed':args.seed,'jobs':results,'optimization_evaluations':0,'simulation_calls':0,'old_checkpoints_compatible':False,'prior_profiles_changed':False}
    write(output/'preparation_summary.json',summary)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in results]).to_csv(output/'fleet_selection_summary.csv',index=False,float_format='%.17g')
    consumed.update(store.hashes)
    write(output/'preparation_manifest.json',{'source_sha256':consumed,'strategy_sha256':sha(ROOT/'strategies/representative_years.py'),'runner_sha256':sha(__file__),'artifact_sha256':{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})


if __name__=='__main__':main()
