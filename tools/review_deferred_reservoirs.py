"""Review deferred records using saved evidence and calibration-only preparation.

No optimization, simulation, source repairs or parameter-limit changes occur.
"""
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows import reservoir as w
from strategies.representative_years import select_representative_years, fit_selected_years
from strategies.fleet_comparison_v2 import comparison_selections
from dztr.integrations.collection import _verified_inputs
from dztr.integrations.world_reservoirs import _forcing
from dztr.simulation.sampled import SamplingSettings
from fleet_admission import ordered_candidate

OUT=ROOT/'outputs/deferred_readiness_20261013'
ADMISSION=ROOT/'outputs/fleet_admission_verified_20261012'


def special_action(identifier, status):
    if identifier=='ex35_Bull_Lake':return 'Reuse the signed computed-net interval results. Add a separate interval-policy route; do not relabel them as continuous independent-inflow calibration.'
    if identifier=='ex12_Trinity':return 'Implement the saved raw/provider conversion, version, quality mask, storage definition and interval timing decisions before re-estimating targets.'
    if identifier=='ex26_Ghost':return 'Resolve storage coordinate/curve, observation hour, capacity/floor and inflow/outflow boundary definitions before a source-profile import.'
    if identifier=='ex31_Brazeau':return 'Identify and pin the MATLAB external forcing series, dates, units and boundary interpretation; do not substitute packaged observations silently.'
    if identifier=='ex22_Mcgregor':return 'Review a short-record protocol: six complete years cannot pass the current six post-warmup-year policy. Four calibration years could meet 65%, but a one-year holdout and regime limitations must be declared before preflight.'
    if status=='empirical_box_incompatible':return 'Resolve capacity/storage coordinate and operating-floor definitions; then refit unchanged percentile rules from the approved calibration records.'
    if status=='critical_gap_constraint_incompatible':return 'Review monthly release/storage compatibility. The gap inequality is a sufficient continuity screen, not proof every candidate lacks roots; an unchanged-equation calibration-only preflight is the next numerical task.'
    return 'Resolve the named above-capacity/below-operating-floor records and storage coordinate. Preserve potential flood surcharge and separate physical dead storage from the DZTR floor; do not clip or enlarge capacity from model fit.'


def main():
    if OUT.exists():raise ValueError('Use a fresh review version')
    proof=w.read(ADMISSION/'final_verification.json')
    for rel,digest in proof['source_sha256'].items():
        if w.sha(ROOT/rel)!=digest:raise ValueError('Prior admission source changed')
    if w.sha(ADMISSION/'admission.json')!=proof['artifact_sha256']['admission.json']:raise ValueError('Admission evidence changed')
    admission=w.read(ADMISSION/'admission.json');registry=w.read(ROOT/'configs/calibrated_twenty.json')
    manifest=w.read(ROOT/'data/reservoirs/manifest.json')
    completed=set(registry['profiles_by_reservoir_id'])
    previous={r['reservoir_id']:r for r in admission['reservoirs']}
    entries=[e for e in manifest['reservoirs'] if e['reservoir_id'] not in completed]
    assert len(entries)==21 and len(completed)==20
    assert any(e['reservoir_id']=='ex43_TuyenQuang' for e in manifest['excluded_reservoirs'])
    consumed={'outputs/fleet_admission_verified_20261012/final_verification.json':w.sha(ADMISSION/'final_verification.json'),
              'outputs/fleet_admission_verified_20261012/admission.json':w.sha(ADMISSION/'admission.json'),
              'configs/calibrated_twenty.json':w.sha(ROOT/'configs/calibrated_twenty.json'),
              'data/reservoirs/manifest.json':w.sha(ROOT/'data/reservoirs/manifest.json')}
    OUT.mkdir();records=[]
    for entry in entries:
        identifier=entry['reservoir_id'];old=previous[identifier];folder=OUT/identifier;folder.mkdir()
        _,config,obs,hashes=_verified_inputs(ROOT,entry)
        consumed.update({p.relative_to(ROOT).as_posix():d for p,d in hashes.items()})
        frame=obs.frame;capacity=config['model']['capacity_m3'];floor=capacity*config['model']['dead_storage_fraction']
        full_flags={'above_configured_capacity':int(frame.storage_m3.gt(capacity).sum()),'below_configured_operating_floor':int(frame.storage_m3.lt(floor).sum()),
                    'negative_inflow':int(frame.inflow_m3s.lt(0).sum()),'missing_inflow':int(frame.inflow_m3s.isna().sum())}
        record={'reservoir_id':identifier,'name':entry['name'],'previous_status':old['status'],'standard_calibration_admitted':False,
            'current_status':old['status'],'new_optimization_evaluations':0,'new_simulations':0,'old_calibration_flags':old.get('calibration_flags'),
            'full_record_screen_audit_only':full_flags,'capacity_m3_inherited':capacity,'operating_floor_m3_inherited':floor,
            'storage_max_over_configured_capacity':float(frame.storage_m3.max()/capacity),'source_definitions_certified':False,
            'next_action':special_action(identifier,old['status']),'preparation_status':'not_attempted_semantics_gate','preparation_reason':None,
            'mixed_ordered_box_feasible':None,'mixed_gap_screen_feasible':None,'calibration_fraction':None,'common_assessment_years':None}
        if identifier not in {'ex12_Trinity','ex26_Ghost','ex31_Brazeau','ex35_Bull_Lake'}:
            try:
                adjusted=_forcing(obs,config['inflow_preprocessing']);sampling=SamplingSettings(**config['sampling'])
                _,annual,_=select_representative_years(adjusted,sampling)
                plans,held=comparison_selections(adjusted.frame,annual);plan=plans['mixed_pooled']
                record['calibration_fraction']=plan.roles.count('calibration')/len(plan.roles);record['common_assessment_years']=held
                record['missing_assessment_regimes']=sorted({'dry','normal','wet'}-set(annual.loc[held,'regime']))
                w.write(folder/'proposed_year_selection.json',plan.as_dict())
                # Bounds can be analyzed independently even if generalized targets are invalid.
                from dztr.parameterization import estimate_percentiles
                from dztr.calibration.sampled import target_observations, _observations
                fit=target_observations(_observations(adjusted.frame.loc[plan.mask(adjusted.frame.index,'calibration')]))
                quantiles=estimate_percentiles(fit,[5,35,75,95],quantile_method='hazen').frame.set_index(['variable','month','percentile'])
                lower=[];upper=[]
                for month in range(1,13):
                    lower.append([float(quantiles.loc[(var,month,p),'value']) for var in ('storage_m3','release_m3s') for p in (5,35,75)])
                    upper.append([float(quantiles.loc[(var,month,p),'value']) for var in ('storage_m3','release_m3s') for p in (35,75,95)])
                days=sampling.fixed_step_days or (1 if sampling.frequency=='daily' else 31)
                args=(lower,upper,capacity,config['model']['dead_storage_fraction'],days*86400)
                ordered=ordered_candidate(*args);gap=ordered_candidate(*args,close_critical_gap=True)
                w.write(folder/'mixed_bounds_audit.json',{'lower_si':lower,'upper_si':upper,'ordered_box':ordered,'sufficient_critical_gap_screen':gap,'timestep_days':days,'simulation_verified':False})
                record['mixed_ordered_box_feasible']=ordered['feasible'];record['mixed_gap_screen_feasible']=gap['feasible']
                try:
                    template,bounds=fit_selected_years(adjusted,plan,config['model'],allow_feasible_initialization=True)
                    template.monthly_targets.to_csv(folder/'proposed_initial_targets_si.csv',float_format='%.17g')
                    bounds.generalized_targets.to_csv(folder/'raw_generalized_targets_si.csv',float_format='%.17g')
                    bounds.empirical.monthly_bounds.to_csv(folder/'proposed_bounds_si.csv',float_format='%.17g')
                    record['preparation_status']='mathematical_preparation_passed_source_gates_unchanged';record['initialization_kind']=bounds.initialization_kind
                except ValueError as error:
                    record['preparation_status']='fitting_initialization_failed';record['preparation_reason']=str(error)
                if old['status']=='critical_gap_constraint_incompatible' and gap['feasible']:
                    record['current_status']='calibration_only_preflight_candidate';record['next_action']='The mixed-year box passes the sufficient gap screen. Test the unmodified ordered witness on warmup/calibration with strict roots before admission; never use held-out behavior to choose it.'
            except (ValueError,TypeError) as error:
                record['preparation_status']='year_selection_or_input_gate_failed';record['preparation_reason']=str(error)
                if identifier=='ex22_Mcgregor':record['current_status']='short_record_protocol_required'
        if identifier=='ex35_Bull_Lake':record['current_status']='calibrated_separate_interval_workflow'
        w.write(folder/'readiness.json',record);records.append(record)
        print(identifier,record['current_status'],record['preparation_status'],flush=True)
    for name in ['outputs/source_reconciliation_20261008/calibration_gates.json','outputs/bull_lake_interval_three_seed_20261009/README.md','outputs/bull_lake_interval_population_20261009/README.md']:
        consumed[name]=w.sha(ROOT/name)
    summary={'active_reservoirs':41,'standard_reference_reservoirs':20,'deferred_records_reviewed':21,'excluded_reservoirs':manifest['excluded_reservoirs'],
             'counts':dict(Counter(r['current_status'] for r in records)),'new_calibration_admissions':0,'new_simulations':0,'new_optimization_evaluations':0,'records':records}
    w.write(OUT/'readiness_review.json',summary)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in records]).to_csv(OUT/'reservoir_readiness.csv',index=False,float_format='%.17g')
    w.write(OUT/'work_queue.json',{'priority_order':['calibration_only_preflight_candidate','short_record_protocol_required','source_assumptions_review','empirical_box_incompatible','source_profile_required','forcing_adapter_required','critical_gap_constraint_incompatible','calibrated_separate_interval_workflow'],
        'tasks':[{'reservoir_id':r['reservoir_id'],'status':r['current_status'],'action':r['next_action'],'execution_started':False} for r in records]})
    lines=['# Deferred-reservoir readiness review','',
           'All 41 active reservoirs are accounted for: twenty standard-workflow reference reservoirs plus the twenty-one records reviewed here. Tuyen remains excluded. Bull Lake is already calibrated under a separate interval policy; it is not an uncalibrated failure.', '',
           'No source records, limits, equations, defaults, registries or checkpoints were changed. No simulations or optimization were run. New mathematical preparation uses the existing mixed-year selector and calibration-only Hazen bounds, with at least 65% support. Full-record flags are audit evidence only, never an assessment-fit selection criterion.', '',
           '| Review outcome | Reservoirs |','| --- | ---: |']
    for status,count in summary['counts'].items():lines.append(f'| {status} | {count} |')
    lines+=['','| Reservoir | Current outcome | Mixed ordered box | Mixed sufficient gap screen | Next action |','| --- | --- | --- | --- | --- |']
    for r in records:lines.append(f"| [{r['name']}]({r['reservoir_id']}/readiness.json) | {r['current_status']} | {r['mixed_ordered_box_feasible']} | {r['mixed_gap_screen_feasible']} | {r['next_action']} |")
    lines+=['','Passing a box or sufficient critical-gap screen is not proof of simulation feasibility or source correctness. Failing the sufficient gap screen is not proof no roots exist for every timestep. Neither case silently authorizes different capacities, floors, timesteps, clipping or solver policies.', '',
            'Mcgregor has six complete calendar years and five post-warmup years. The current shared selector requires at least six post-warmup years. This is an implementation/protocol restriction, not proof that 65% calibration is mathematically impossible: four calibration years could provide 66.7% support with one warmup and one held-out year. A named short-record protocol and regime limitations are needed before admission.', '',
            'Source-limit priorities: Canyon Ferry, Charvak and Toktogul have only a few previously flagged calibration records, so their documented capacity/storage-coordinate review is a bounded next task. Sparse flags do not establish harmlessness or justify ignoring a flood-surcharge event. Fort Peck, Falcon and Bennett need the capacity/coordinate issue resolved before empirical-box expansion; the observed maximum is not automatically the correct capacity.', '',
            'Bull Lake results retain signed computed net forcing, nominal daily-mean interval accounting, quarantine masks and observed-state restart anchors. They are segmented conditional hindcasts with shared forcing/observation information, not independent continuous-inflow validation. Reuse the existing results; add an explicit interval-policy route before unified admission.', '',
            '[Trinity/Ghost source requirements](../source_reconciliation_20261008/calibration_gates.json), [latest Bull Lake population review](../bull_lake_interval_population_20261009/README.md), [complete queue](work_queue.json), [readiness table](reservoir_readiness.csv).','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    w.write(OUT/'verification.json',{'status':'passed_readiness_review_not_calibration_admission','consumed_sha256':consumed,'code_sha256':w.source_hashes()|{'tools/review_deferred_reservoirs.py':w.sha(__file__),'tools/fleet_admission.py':w.sha(ROOT/'tools/fleet_admission.py')},
        'artifact_sha256':{p.relative_to(OUT).as_posix():w.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='verification.json'},'new_simulations':0,'new_optimization_evaluations':0})


if __name__=='__main__':main()
