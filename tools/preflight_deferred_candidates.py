"""Predeclared default/witness calibration-only tests for readiness candidates."""
from pathlib import Path
import sys
import numpy as np
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

REVIEW=ROOT/'outputs/deferred_readiness_20261013'
OUT=ROOT/'outputs/deferred_preflight_20261013'


def main():
    if OUT.exists():raise ValueError('Use a fresh preflight directory')
    proof=w.read(REVIEW/'verification.json')
    for rel,digest in proof['code_sha256'].items():assert w.sha(ROOT/rel)==digest,rel
    for rel,digest in proof['artifact_sha256'].items():assert w.sha(REVIEW/rel)==digest,rel
    review=w.read(REVIEW/'readiness_review.json')
    candidates=[r for r in review['records'] if r['current_status']=='calibration_only_preflight_candidate']
    manifest=w.read(ROOT/'data/reservoirs/manifest.json');entries={e['reservoir_id']:e for e in manifest['reservoirs']}
    OUT.mkdir();protocol={'selection_uses_assessment':False,'optimization_evaluations':0,'candidates':[r['reservoir_id'] for r in candidates],
        'sequence':['default_generalized_or_feasible_midpoint','saved_sufficient_gap_witness_only_if_default_fails'],
        'policy':'reject','model_constants_changed':False,'evaluation_scope':'continuous warmup/calibration prefix; selected calibration years only; no held-out simulation or scores',
        'review_verification_sha256':w.sha(REVIEW/'verification.json')}
    w.write(OUT/'protocol.json',protocol);results=[];calls=0
    for record in candidates:
        identifier=record['reservoir_id'];folder=OUT/identifier;folder.mkdir()
        _,cfg,obs,_=_verified_inputs(ROOT,entries[identifier]);obs=_forcing(obs,cfg['inflow_preprocessing'])
        plan=YearSelection(**w.read(REVIEW/identifier/'proposed_year_selection.json'))
        template,bounds=fit_selected_years(obs,plan,cfg['model'],allow_feasible_initialization=True)
        objective=RepresentativeYearObjective(obs,plan,template,sampling=SamplingSettings(**cfg['sampling']),bounds=bounds,
            policy='reject',atol_storage_m3=cfg['atol_storage_m3'],cost_mode='pooled_nse')
        default=parameters_to_vector(template)
        gap=w.read(REVIEW/identifier/'mixed_bounds_audit.json')['sufficient_critical_gap_screen']['monthly_targets_si']
        witness=ReservoirParameters(pd.DataFrame(gap,index=template.monthly_targets.index,columns=template.monthly_targets.columns),**cfg['model'])
        for label,parameters in [('default_generalized_or_feasible_midpoint',template),('saved_sufficient_gap_witness',witness)]:
            evaluated=objective.evaluate(parameters_to_vector(parameters));calls+=1
            result={'reservoir_id':identifier,'candidate':label,'feasible':evaluated.feasible,'reason':evaluated.reason,
                'calibration_metrics':evaluated.metrics,'assessment_used':False,'root_policy':'reject','initialization_kind':bounds.initialization_kind}
            if evaluated.feasible:
                residual=w.independent_balance(evaluated.simulation.frame,cfg['model']['inflow_correction'])
                assert residual<=cfg['atol_storage_m3']
                result['maximum_independent_balance_residual_m3']=residual
                evaluated.simulation.frame.to_csv(folder/(label+'_calibration_simulation.csv'),index_label='date',float_format='%.17g')
            w.write(folder/(label+'.json'),result);results.append(result)
            if evaluated.feasible:break
    ready=[r['reservoir_id'] for r in results if r['feasible'] and r['candidate']=='default_generalized_or_feasible_midpoint']
    witness_only=[r['reservoir_id'] for r in results if r['feasible'] and r['candidate']=='saved_sufficient_gap_witness']
    summary={'status':'completed','model_evaluations':calls,'optimization_evaluations':0,'held_out_evaluations':0,
        'default_initialization_preflight_passed':ready,'witness_feasible_initializer_adapter_required':witness_only,'results':results}
    w.write(OUT/'summary.json',summary)
    lines=['# Deferred-candidate calibration-only preflight','',f'{calls} fixed initialization simulations; zero optimization or held-out evaluations. All source limits, bounds, clocks and strict root equations remain unchanged.','']
    for r in results:lines.append(f"- {r['reservoir_id']}, {r['candidate']}: {'completed' if r['feasible'] else 'failed'}; {r['reason'] or 'independent mass balance passed'}.")
    lines+=['','A successful default initialization supports calibration admission under the inherited assumptions; it does not certify source definitions or convergence. A successful alternate witness still needs an explicit initializer input before using the current unified runner. A failed witness is one failed candidate, not proof the entire box is impossible.','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    w.write(OUT/'verification.json',{'status':'passed_recorded_preflight_outcomes','source_sha256':{'tools/preflight_deferred_candidates.py':w.sha(__file__)},
        'review_verification_sha256':w.sha(REVIEW/'verification.json'),'artifact_sha256':{p.relative_to(OUT).as_posix():w.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='verification.json'}})
    print(summary)


if __name__=='__main__':main()
