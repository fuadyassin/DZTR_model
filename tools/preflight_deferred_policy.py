"""One fixed existing continuity policy for the already saved witness."""
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows import reservoir as w
from strategies.representative_years import YearSelection, fit_selected_years, RepresentativeYearObjective
from dztr.calibration.candidates import parameters_to_vector
from dztr.integrations.collection import _verified_inputs
from dztr.integrations.world_reservoirs import _forcing
from dztr.schemas import ReservoirParameters
from dztr.simulation.sampled import SamplingSettings


def main():
    review=ROOT/'outputs/deferred_readiness_20261013';prior=ROOT/'outputs/deferred_preflight_20261013';out=ROOT/'outputs/deferred_policy_preflight_20261013'
    if out.exists():raise ValueError('Use a fresh preflight version')
    proof=w.read(prior/'verification.json')
    for rel,digest in proof['artifact_sha256'].items():assert w.sha(prior/rel)==digest,rel
    previous=w.read(prior/'summary.json')
    witnesses=[r for r in previous['results'] if r['candidate']=='saved_sufficient_gap_witness' and 'Multiple implicit storage/release roots' in (r['reason'] or '')]
    out.mkdir();policy='nearest_previous_storage'
    w.write(out/'protocol.json',{'policy':policy,'rationale':'Same declared continuity heuristic used for eleven existing standard-workflow reservoirs; chooses among valid roots without adding roots to gaps.',
        'target_selection':'unchanged saved sufficient-gap witness, only when its strict failure reports multiple valid roots','scope':'warmup and calibration prefix only','assessment_used':False,
        'prior_verification_sha256':w.sha(prior/'verification.json'),'optimization_evaluations':0})
    records=[]
    entries={e['reservoir_id']:e for e in w.read(ROOT/'data/reservoirs/manifest.json')['reservoirs']}
    for old in witnesses:
        identifier=old['reservoir_id'];_,cfg,obs,_=_verified_inputs(ROOT,entries[identifier]);obs=_forcing(obs,cfg['inflow_preprocessing'])
        plan=YearSelection(**w.read(review/identifier/'proposed_year_selection.json'));template,bounds=fit_selected_years(obs,plan,cfg['model'],allow_feasible_initialization=True)
        targets=w.read(review/identifier/'mixed_bounds_audit.json')['sufficient_critical_gap_screen']['monthly_targets_si']
        parameters=ReservoirParameters(pd.DataFrame(targets,index=template.monthly_targets.index,columns=template.monthly_targets.columns),**cfg['model'])
        objective=RepresentativeYearObjective(obs,plan,template,sampling=SamplingSettings(**cfg['sampling']),bounds=bounds,policy=policy,atol_storage_m3=cfg['atol_storage_m3'],cost_mode='pooled_nse')
        evaluated=objective.evaluate(parameters_to_vector(parameters))
        record={'reservoir_id':identifier,'feasible':evaluated.feasible,'reason':evaluated.reason,'root_policy':policy,'default_initializer_compatible':False,'calibration_metrics':evaluated.metrics,'assessment_used':False}
        if evaluated.feasible:
            record['maximum_independent_balance_residual_m3']=w.independent_balance(evaluated.simulation.frame,cfg['model']['inflow_correction'])
            assert record['maximum_independent_balance_residual_m3']<=cfg['atol_storage_m3']
            evaluated.simulation.frame.to_csv(out/(identifier+'_calibration_simulation.csv'),index_label='date',float_format='%.17g')
            parameters.monthly_targets.to_csv(out/(identifier+'_proposed_initializer_si.csv'),float_format='%.17g')
        records.append(record)
    w.write(out/'summary.json',{'results':records,'model_evaluations':len(records),'optimization_evaluations':0,'held_out_evaluations':0,'standard_calibration_admitted':False,
        'next_step':'If feasible, support an explicit frozen initializer alongside the declared root policy before using the unified calibration runner. Existing profile registries and default estimates remain unchanged.'})
    w.write(out/'verification.json',{'status':'passed_recorded_policy_preflight','source_sha256':{'tools/preflight_deferred_policy.py':w.sha(__file__)},'artifact_sha256':{p.name:w.sha(p) for p in out.iterdir() if p.is_file() and p.name!='verification.json'}})
    print(records)


if __name__=='__main__':main()
