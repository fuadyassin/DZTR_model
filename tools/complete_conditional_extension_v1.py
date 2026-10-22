"""Audit completed conditional jobs, preserve prior work and publish index v4."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows import conditional_cohort_v1 as w
from dztr.calibration.candidates import parameters_to_vector


def prior_checks():
    base=ROOT/'outputs/hungry_horse_completion_v1_20261021';proof=w.d.read(base/'completion_manifest.json');count=0
    for key,folder in [('source_sha256',ROOT),('files_sha256',ROOT),
        ('run_artifact_sha256',ROOT/'outputs/hungry_horse_daily_interval_v1_20261021'),('artifact_sha256',base)]:
        for name,expected in proof[key].items():
            if w.d.sha(folder/name)!=expected:raise ValueError('Prior verified work changed: '+name)
            count+=1
    _,imported=w.d.collection.verify(ROOT/'data/reservoirs_etienne_v1_20261021')
    return {'hungry_horse_and_prior_pins_verified':count,'original_extension_artifacts_verified':len(imported['artifact_sha256'])}


def audit(out,completion):
    out,proof=w.checked(out);w.native.verify_registry(out/'report/registry.json');cfg=proof['config']
    preserved=prior_checks();rows=[];seasonal=[];zones=[];summary=[];counts={'evaluations':0,'feasible_vectors':0,'proposal_rng_trials':0,'volume_intervals':0,'raw_target_values':0,'maximum_volume_residual_m3':0.}
    for case in cfg['cases']:
        rid=case['reservoir_id'];obj,frame,plan,support=w.objective(out,rid);mask=plan.mask(frame.index,'calibration')&support
        assert mask.mean()>=.65 and obj.observed.loc[~mask].isna().all().all()
        original=w.d.frame_read(ROOT/case['observations_csv']);expected=w.d.conditional_inflow(original).loc[case['start']:case['end'],w.d.COLUMNS]
        pd.testing.assert_frame_equal(frame,expected)
        raw=pd.read_csv(out/'prepared'/rid/'raw_generalized_targets_si.csv',index_col='month',float_precision='round_trip')
        for month in range(1,13):
            part=frame.loc[mask&(frame.index.month==month)]
            targets=[*np.percentile(part.storage_m3.dropna(),[10,45,85],method='hazen'),*np.percentile(part.release_m3s.dropna(),[10,45,85],method='hazen')]
            np.testing.assert_array_equal(raw.loc[month].to_numpy(),targets);counts['raw_target_values']+=6
        for seed in cfg['optimization']['seeds']:
            folder=out/'jobs'/rid/f'seed_{seed}';checkpoint=w.d.read(folder/'checkpoint.json');state=checkpoint['state'];receipt=w.d.read(folder/'completed.json')
            if receipt['checkpoint_sha256']!=w.d.sha(folder/'checkpoint.json') or receipt['evaluations']!=len(state['trials']):raise ValueError('Job receipt mismatch')
            position=0
            def recorded(vector):
                nonlocal position
                trial=state['trials'][position];np.testing.assert_array_equal(vector,trial['vector']);position+=1
                return SimpleNamespace(cost=float('inf') if trial['cost'] is None else trial['cost'],feasible=trial['feasible'],reason=trial['reason'])
            regenerated=w.dds_v1.search(recorded,parameters_to_vector(obj.template),obj.bounds.lower_vector,obj.bounds.upper_vector,
                seed=seed,horizon=cfg['optimization']['horizon'],budget=cfg['optimization']['evaluation_budget'],perturbation=cfg['optimization']['perturbation'])
            if regenerated!=state:raise ValueError('Proposal/RNG/cost history differs')
            counts['evaluations']+=len(state['trials']);counts['proposal_rng_trials']+=position
            for trial in state['trials']:
                if trial['feasible']:obj.bounds.validate(trial['vector'],obj.template);counts['feasible_vectors']+=1
        # Exported representative trajectories already passed canonical net-policy
        # parity in the report. Independently check their saved interval volumes.
        for path in sorted((out/'report'/rid).glob('simulation_*.csv')):
            sim=w.d.frame_read(path);complete=sim.interval_completed.to_numpy(bool)
            residual=sim.storage_m3.diff()-frame.inflow_m3s*86400+sim.interval_mean_total_outflow_m3s*86400
            maximum=float(residual[complete].abs().max());assert maximum<=cfg['balance_tolerance_m3']
            counts['volume_intervals']+=int(complete.sum());counts['maximum_volume_residual_m3']=max(counts['maximum_volume_residual_m3'],maximum)
            label=path.stem.replace('simulation_','');threshold=float(np.percentile(frame.release_m3s[mask],90,method='hazen'))
            for role in ['calibration','assessment']:
                select=plan.mask(frame.index,role)&support
                for variable,observed,predicted in [('storage','storage_m3','storage_m3'),('release','release_m3s','interval_mean_total_outflow_m3s')]:
                    errors=sim[predicted]-frame[observed]
                    for month in range(1,13):
                        chosen=select&(frame.index.month==month)
                        seasonal.append({'reservoir_id':rid,'profile':label,'role':role,'variable':variable,'month':month,'count':int(chosen.sum()),'mean_bias':float(errors[chosen].mean())})
                for zone in range(1,5):
                    chosen=select&sim.zone.eq(zone).to_numpy()
                    zones.append({'reservoir_id':rid,'profile':label,'role':role,'zone':zone,'count':int(chosen.sum()),
                        'storage_bias_m3':float((sim.storage_m3-frame.storage_m3)[chosen].mean()),
                        'release_bias_m3s':float((sim.interval_mean_total_outflow_m3s-frame.release_m3s)[chosen].mean())})
                peak=select&(frame.release_m3s.to_numpy()>=threshold)
                summary.append({'reservoir_id':rid,'profile':label,'role':role,'count':int(select.sum()),
                    'storage_bias_m3':float((sim.storage_m3-frame.storage_m3)[select].mean()),
                    'release_bias_m3s':float((sim.interval_mean_total_outflow_m3s-frame.release_m3s)[select].mean()),
                    'high_release_threshold_calibration_q90_m3s':threshold,'high_release_count':int(peak.sum()),
                    'high_release_bias_m3s':float((sim.interval_mean_total_outflow_m3s-frame.release_m3s)[peak].mean())})
        replay=w.replay(out,rid);assert replay['new_optimizer_evaluations']==0
        annual=plan.table().groupby(['role','regime']).size().reset_index(name='calendar_days');annual.insert(0,'reservoir_id',rid)
        annual.to_csv(completion/f'{rid}_selection_coverage.csv',index=False)
        rows.append({'reservoir_id':rid,'calendar_days':len(frame),'actual_fitting_days':int(mask.sum()),'actual_fitting_fraction':float(mask.mean()),
            'assessment_days':int((plan.mask(frame.index,'assessment')&support).sum()),'fitting_start':str(frame.index[mask][0].date()),'fitting_end':str(frame.index[mask][-1].date()),
            'observed_storage_above_assumed_capacity':int((frame.storage_m3>case['physical_constants']['capacity_m3']).sum()),
            'minimum_reconstructed_net_m3s':float(frame.inflow_m3s.min()),'assessment_responses_enter_forcing':True,
            'direct_assessment_initialization':False,'shared_registry_replay':replay['status']})
    pd.DataFrame(rows).to_csv(completion/'admission_and_selection.csv',index=False)
    pd.DataFrame(seasonal).to_csv(completion/'seasonal_bias.csv',index=False)
    pd.DataFrame(zones).to_csv(completion/'zone_diagnostics.csv',index=False)
    pd.DataFrame(summary).to_csv(completion/'bias_and_release_peaks.csv',index=False)
    checks={**counts,**preserved,'canonical_representative_trajectories':len(w.d.read(out/'report/scientific_checks.json')['representatives']),
        'new_optimizer_evaluations_in_audit':0,'selection_uses_assessment_metrics':False,'assessment_responses_enter_forcing':True,
        'independent_validation_established':False,'convergence_established':False,'all_original_source_records_preserved':True}
    w.write(completion/'scientific_checks.json',checks);return checks


def publish(out,completion):
    out,proof=w.checked(out);cfg=proof['config'];base=w.d.read(ROOT/cfg['base_catalog']);catalog=deepcopy(base)
    registry=w.native.verify_registry(out/'report/registry.json')
    for rid in registry['profiles_by_reservoir_id']:
        profile,parameters=w.native.load_profile(out/'report/registry.json',identifier=rid)
        row=next(r for r in catalog['reservoirs'] if r['reservoir_id']==rid)
        row.update(active_profile_available=True,active_profile_parameter_status='calibrated_conditional_exploration',primary_observation_collection_preserved=True)
        row['active_profile']={'active_profile_route':profile['route'],'active_profile_registry':str((out/'report/registry.json').relative_to(ROOT)),
            'profile_observations_file':str((out/'prepared'/rid/'observations_si.csv').relative_to(ROOT)),
            'profile_selection_file':str((out/'prepared'/rid/'selection.csv').relative_to(ROOT)),
            'parameters_ready_under_declared_scenario':True,'simulation_ready_under_declared_scenario':profile['assessment']['assessment_feasible'],
            'calibration_completed_under_declared_scenario':True,'source_gates_resolved':False,'physical_capacity_verified':False,
            'provider_clock_verified':False,'assessment_responses_enter_forcing':True,'independent_prediction_validated':False,
            'selection_strategy':profile['year_selection'],'seed':profile['seed'],'evaluation_budget':profile['evaluation_budget'],
            'unverified_GRAND_candidate':profile['source_decision']['unverified_GRAND_candidate'],
            'remaining_source_action':profile['source_decision']['source_reconciliation_pending']}
        assert row['GRAND_ID'] is None
    for old,new in zip(base['reservoirs'],catalog['reservoirs']):
        if old['reservoir_id'] not in registry['profiles_by_reservoir_id']:assert old==new
    assert len(catalog['reservoirs'])==62 and len({r['reservoir_id'] for r in catalog['reservoirs']})==62
    catalog['kind']='reservoir_collection_index_v4';catalog['previous_catalog']=cfg['base_catalog']
    catalog['conditional_extension']={'protocol':str((out/'protocol.json').relative_to(ROOT)),'protocol_sha256':w.d.sha(out/'protocol.json'),
        'registry':str((out/'report/registry.json').relative_to(ROOT)),'source_gates_resolved':False}
    target=ROOT/'data/catalogs/reservoir_collection_v4_20261021.json'
    if target.exists():assert w.d.read(target)==catalog
    else:w.write(target,catalog)
    table=ROOT/'data/catalogs/reservoir_inventory_v4_20261021.csv';data=(out/'report/reservoir_parameterization_all_62_by_GRAND_ID.csv').read_bytes()
    if table.exists():assert table.read_bytes()==data
    else:table.write_bytes(data)
    before=pd.read_csv(ROOT/cfg['base_inventory'],float_precision='round_trip').set_index('reservoir_id')
    after=pd.read_csv(table,float_precision='round_trip').set_index('reservoir_id')
    # Every pre-existing profile column/value, especially Hungry Horse, survives.
    untouched=[rid for rid in before.index if not rid.startswith('etienne_') or rid=='etienne_Hungry_Horse']
    pd.testing.assert_frame_equal(before.loc[untouched],after.loc[untouched,before.columns],check_dtype=False)
    receipt={'catalog':str(target.relative_to(ROOT)),'catalog_sha256':w.d.sha(target),'inventory':str(table.relative_to(ROOT)),
        'inventory_sha256':w.d.sha(table),'reservoirs':62,'new_conditional_profiles':2,'remaining_extension_source_gated':18,
        'prior_catalog_rows_unchanged':60,'GRAND_IDs_assigned':0,'Hungry_Horse_and_paper_profiles_preserved':True}
    w.write(completion/'publication_receipt.json',receipt);return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--completion',required=True)
    a=parser.parse_args();out=Path(a.output).resolve(strict=True);completion=Path(a.completion).resolve()
    assert completion.is_relative_to(ROOT/'outputs') and completion!=ROOT/'outputs'
    completion.mkdir(parents=True,exist_ok=True)
    checks=audit(out,completion);publication=publish(out,completion)
    scores=pd.read_csv(out/'report/scores_vs_baseline.csv');lines=['# Conditional extension completion','','Two further Etienne additions completed: Hugh Keenleyside and Revelstoke. Hungry Horse remains unchanged; three of the 21 additions now have completed exploratory profiles. All 62 reservoirs remain indexed. Eighteen additions still require source information.','',
        '**These results are conditional on observed responses.** Reconstructed net forcing uses storage changes and release observations in both fitting and assessment periods. Seed selection uses fitting costs only, but assessment is not independent validation. Capacity and dead-storage references remain assumptions.','',
        '[Plots and per-seed report](../'+out.name+'/report/README.md), [all 20 decisions](../'+out.name+'/report/remaining_20_status.csv), [all 62 parameterizations](../../'+publication['inventory']+'), [collection v4](../../'+publication['catalog']+').','',
        '| Reservoir | Profile | Fit storage NSE | Fit release NSE | Conditional assessment storage NSE | Conditional assessment release NSE |',
        '|---|---|---:|---:|---:|---:|']
    for _,row in scores.iterrows():
        lines.append('| '+' | '.join([row.reservoir_id,row['profile']+(' (selected)' if row.selected else ''),*[f'{row[k]:.4f}' for k in ['calibration_storage_nse','calibration_release_nse','assessment_storage_nse','assessment_release_nse']]])+' |')
    lines+=['','Hugh Keenleyside uses mixed complete years stratified by reconstructed net flux. Revelstoke uses a separate chronological short-record experiment; it cannot support claims of wet/dry-year transferability. Both retain at least 65% actual fitting support and all 12 fitting months.','',
        'The five other paired cases remain deferred: Aishihik has an unresolved multi-lake boundary/capacity; Minidoka has active/total storage and outlet conflicts; Nechako has a capacity/storage-coordinate mismatch; Seton has capacity/datum, outlet and gap issues; Winnebago lacks a verified compatible lake-system capacity. Ten storage-only and three flow-only additions still need suitable time-varying forcing and physical metadata; weaker-data parameter priors do not create that forcing.','',
        f"Scientific checks: {checks['evaluations']} recorded optimizer evaluations; {checks['proposal_rng_trials']} exact proposal/RNG replays using recorded costs; {checks['canonical_representative_trajectories']} canonical trajectories; {checks['volume_intervals']} independently checked interval volumes, maximum residual {checks['maximum_volume_residual_m3']:.6g} m3 against 0.0001 m3 tolerance. No new optimizer evaluations were used by the audit.",'',
        '[Selection coverage](admission_and_selection.csv), [scientific checks](scientific_checks.json), [seasonal bias](seasonal_bias.csv), [DZTR zone diagnostics](zone_diagnostics.csv), [release peaks](bias_and_release_peaks.csv).','',
        'The 512-evaluation result is a budget checkpoint. It does not establish convergence. GRAND candidates remain separate from verified IDs, so both new profiles are loaded by reservoir ID. See [workflow instructions](../../docs/conditional_extension_v1.md).','']
    (completion/'README.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps({'checks':checks,'publication':publication},indent=2))


if __name__=='__main__':main()
