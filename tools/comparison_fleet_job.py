"""One resumable fixed strategy-comparison search; no assessment evaluation."""
from dataclasses import asdict
import sys
import pandas as pd
from simulate_calibrated_reservoirs import ROOT,read,write,sha
sys.path.insert(0,str(ROOT))
from strategies.representative_years import YearSelection,RepresentativeYearObjective,fit_selected_years
from dztr.calibration.sampled import _observations
from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.optimizer import NSGA2Settings,optimize_nsga2
from dztr.calibration.checkpoint import load_optimizer_checkpoint,save_optimizer_checkpoint
from dztr.simulation.sampled import SamplingSettings

OUT=ROOT/'outputs/strategy_comparison_fleet_20261012'


def objective_for(identifier,arm):
    folder=OUT/'prepared'/identifier/arm
    plan=YearSelection(**read(folder/'selection.json'));cfg=read(folder/'config.json')
    frame=pd.read_csv(folder/'observations_si.csv',index_col='date',parse_dates=True,float_precision='round_trip')
    obs=_observations(frame)
    template,bounds=fit_selected_years(obs,plan,cfg['physical_constants'],allow_feasible_initialization=True)
    objective=RepresentativeYearObjective(obs,plan,template,sampling=SamplingSettings(**cfg['sampling']),bounds=bounds,
               policy=cfg['root_policy'],atol_storage_m3=cfg['atol_storage_m3'],cost_mode=cfg['cost_mode'])
    return objective,template,bounds,plan,frame


def main():
    identifier,arm,seed=sys.argv[1],sys.argv[2],int(sys.argv[3])
    job=next(j for j in read(OUT/'design.json')['jobs'] if (j['reservoir_id'],j['arm'],j['seed'])==(identifier,arm,seed))
    folder=OUT/'jobs'/identifier/arm/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    protocol=read(OUT/'protocol.json')
    for relative,digest in {**protocol['source_sha256'],**protocol['code_sha256']}.items():assert sha(ROOT/relative)==digest,relative
    for relative,digest in protocol['prepared_sha256'].items():
        if relative.startswith(f'prepared/{identifier}/{arm}/'):assert sha(OUT/relative)==digest,relative
    objective,template,bounds,plan,_=objective_for(identifier,arm)
    identity={'kind':'paired_strategy_comparison_v1','protocol_sha256':sha(OUT/'protocol.json'),'job':job,'objective':objective.identity,'python_version':sys.version,'numpy_version':__import__('numpy').__version__,'pandas_version':pd.__version__}
    state=None
    if (folder/'checkpoint.npz').exists():
        if read(folder/'checkpoint_identity.json')!=identity:raise ValueError('Checkpoint identity changed')
        state=load_optimizer_checkpoint(folder/'checkpoint.npz',identity)
    write(folder/'checkpoint_identity.json',identity)
    def save(current):
        save_optimizer_checkpoint(folder/'checkpoint.npz',current,identity)
        write(folder/'progress.json',{'evaluations':len(current.evaluated),'generation':current.generation})
    settings=NSGA2Settings(seed=seed,population_size=job['population'],evaluation_budget=job['evaluation_budget'])
    result=optimize_nsga2(objective,parameters_to_vector(template),settings,bounds=bounds,resume_state=state,checkpoint_callback=save)
    pd.DataFrame([{'evaluation_id':p.evaluation_id,'generation':p.generation,'feasible':p.feasible,'release_cost':p.objectives[0],'storage_cost':p.objectives[1],'reason':p.reason} for p in result.evaluated]).to_csv(folder/'trials.csv',index=False,float_format='%.17g')
    pd.DataFrame([p.vector for p in result.evaluated],index=[p.evaluation_id for p in result.evaluated]).to_csv(folder/'vectors.csv',index_label='evaluation_id',float_format='%.17g')
    pd.DataFrame([{'evaluation_id':p.evaluation_id,'release_cost':p.objectives[0],'storage_cost':p.objectives[1]} for p in result.front]).to_csv(folder/'pareto_front.csv',index=False,float_format='%.17g')
    pd.DataFrame([asdict(g) for g in result.history]).to_csv(folder/'generation_history.csv',index=False,float_format='%.17g')
    write(folder/'result.json',{'status':'completed','evaluations':result.evaluations,'termination':result.termination,'job':job,'assessment_evaluations':0,'selection_sha256':plan.signature})
    write(folder/'completion_manifest.json',{'artifact_sha256':{p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name not in ('execution.log','completion_manifest.json')}})
    print('Completed checkpointed calibration search; no assessment scores used',flush=True)


if __name__=='__main__':main()
