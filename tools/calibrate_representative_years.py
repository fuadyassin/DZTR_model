"""One fresh or identity-checked representative-year optimization job."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from simulate_calibrated_reservoirs import ROOT,read,write,sha
sys.path.insert(0,str(ROOT))
from strategies.representative_years import YearSelection,RepresentativeYearObjective,fit_selected_years
from dztr.calibration.sampled import _observations
from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.optimizer import NSGA2Settings,optimize_nsga2
from dztr.calibration.checkpoint import save_optimizer_checkpoint,load_optimizer_checkpoint
from dztr.simulation.sampled import SamplingSettings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('preparation',type=Path)
    parser.add_argument('--reservoir-id',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--evaluation-budget',required=True,type=int)
    parser.add_argument('--population',type=int,default=32)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    base=args.preparation.resolve(strict=True);folder=base/args.reservoir_id
    if not folder.resolve().is_relative_to(base):raise ValueError('Invalid reservoir path')
    manifest=read(base/'preparation_manifest.json')
    if sha(ROOT/'strategies/representative_years.py')!=manifest['strategy_sha256']:raise ValueError('Strategy code changed; prepare a new version')
    consumed={}
    for relative,digest in manifest['source_sha256'].items():
        if sha(ROOT/relative)!=digest:raise ValueError(f'Input or numeric source changed: {relative}')
    for name in ('selection.json','model_config.json','observations_si.csv','preparation.json'):
        path=folder/name;relative=path.relative_to(base).as_posix()
        if sha(path)!=manifest['artifact_sha256'][relative]:raise ValueError(f'Preparation changed: {relative}')
        consumed[name]=sha(path)
    if read(folder/'preparation.json')['status']!='fit_prepared_not_simulated':raise ValueError('Prepared fitting needs review; do not bypass physical gates')
    selection=YearSelection(**read(folder/'selection.json'));config=read(folder/'model_config.json')
    frame=pd.read_csv(folder/'observations_si.csv',index_col='date',parse_dates=True,float_precision='round_trip')
    observations=_observations(frame,str(folder/'observations_si.csv'))
    template,bounds=fit_selected_years(observations,selection,config['physical_constants'],allow_feasible_initialization=True)
    objective=RepresentativeYearObjective(observations,selection,template,sampling=SamplingSettings(**config['sampling']),bounds=bounds,
                policy=config['root_policy'],atol_storage_m3=config['atol_storage_m3'],cost_mode=config['cost_mode'])
    identity={'kind':'representative_years_v1_optimizer','objective':objective.identity,'prepared_files_sha256':consumed,
              'strategy_sha256':manifest['strategy_sha256'],'numeric_source_sha256':manifest['source_sha256'],
              'runner_sha256':sha(__file__),'sampling':config['sampling'],'physical_constants':config['physical_constants'],'initialization_kind':bounds.initialization_kind,
              'numpy_version':np.__version__,'pandas_version':pd.__version__,'python_version':__import__('platform').python_version()}
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'outputs'):raise ValueError('Output must be inside repository outputs')
    state=None
    if args.resume:
        if read(output/'checkpoint_identity.json')!=identity:raise ValueError('Checkpoint identity differs from selection, inputs, code or scoring')
        state=load_optimizer_checkpoint(output/'checkpoint.npz',identity)
    elif output.exists():raise ValueError('Use a fresh output directory unless resuming')
    else:output.mkdir(parents=True)
    write(output/'checkpoint_identity.json',identity)
    def checkpoint(current):
        save_optimizer_checkpoint(output/'checkpoint.npz',current,identity)
        write(output/'progress.json',{'evaluations':len(current.evaluated),'generation':current.generation,'selection_sha256':selection.signature})
    settings=NSGA2Settings(seed=args.seed,population_size=args.population,evaluation_budget=args.evaluation_budget)
    result=optimize_nsga2(objective,parameters_to_vector(template),settings,bounds=bounds,resume_state=state,checkpoint_callback=checkpoint)
    pd.DataFrame([{'evaluation_id':p.evaluation_id,'release_cost':p.objectives[0],'storage_cost':p.objectives[1],'feasible':p.feasible,'reason':p.reason} for p in result.evaluated]).to_csv(output/'trials.csv',index=False,float_format='%.17g')
    pd.DataFrame([p.vector for p in result.evaluated],index=[p.evaluation_id for p in result.evaluated]).to_csv(output/'vectors.csv',index_label='evaluation_id',float_format='%.17g')
    pd.DataFrame([{'evaluation_id':p.evaluation_id,'release_cost':p.objectives[0],'storage_cost':p.objectives[1]} for p in result.front]).to_csv(output/'pareto_front.csv',index=False,float_format='%.17g')
    pd.DataFrame([asdict(record) for record in result.history]).to_csv(output/'generation_history.csv',index=False,float_format='%.17g')
    write(output/'result.json',{'status':'completed','reservoir_id':args.reservoir_id,'evaluations':result.evaluations,'termination':result.termination,
          'settings':asdict(settings),'objective':objective.identity,'assessment_evaluations':0,'profile_promoted':False})
    # Assessment remains a separate explicitly chosen step after calibration selection.
    print(f'{args.reservoir_id}: completed {result.evaluations} calibration evaluations; assessment not used')


if __name__=='__main__':main()
