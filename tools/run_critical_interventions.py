"""Two fixed calibration-only interventions; archived comparators are reused."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulate_calibrated_reservoirs import ROOT, ARCHIVE, read, write, sha
from forcing_batch import FrozenProfiles
from diagnose_storage_transfer import difference_budget
from dztr.calibration.sampled import _observations
from dztr.evaluation import evaluate_simulation
from dztr.schemas import ReservoirParameters
from dztr.simulation.sampled import SamplingSettings
from dztr.simulation.root_policy import simulate_root_policy_reservoir

SOURCE=ROOT/'outputs/critical_release_review_20261012'
OUT=ROOT/'outputs/critical_interventions_20261012'


def validate_intervention(targets, selected, baseline):
    others=[c for c in selected if c!='release_critical_m3s']
    pd.testing.assert_frame_equal(targets[others],selected[others],check_exact=True)
    np.testing.assert_array_equal(targets.release_critical_m3s,baseline.release_critical_m3s)


def independent_balance(frame, correction):
    s,q,i,dt=[frame[c].to_numpy() for c in ('storage_m3','release_m3s','inflow_m3s','step_seconds')]
    residual=np.diff(s)-dt[1:]/2*(correction*(i[1:]+i[:-1])-q[1:]-q[:-1])
    np.testing.assert_array_equal(residual,frame.balance_residual_m3.to_numpy()[1:])
    return float(np.abs(residual).max())


def main():
    if OUT.exists():raise ValueError('Use a fresh experiment directory')
    proof=read(SOURCE/'final_verification.json');consumed={}
    def prepared(relative):
        path=SOURCE/relative;digest=proof['artifact_sha256'][relative]
        if sha(path)!=digest:raise ValueError(f'Prepared source changed: {relative}')
        consumed[path.relative_to(ROOT).as_posix()]=digest
        return path
    jobs=read(prepared('prepared_intervention_profiles.json'))['jobs']
    if len(jobs)!=2 or {j['reservoir_id'] for j in jobs}!={'ex5_Garrison','ex6_Oahe'}:raise ValueError('Expected the two fixed interventions')
    prepared('proposed_calibration_test_protocol.json')
    store=FrozenProfiles()
    # Imported adapters must still match their prior release receipts.
    for release,helper in [('forcing_workflow_v1_1_20261012','tools/forcing_batch.py'),('garrison_oahe_diagnosis_20261012','tools/diagnose_storage_transfer.py')]:
        receipt=ROOT/'outputs'/release/'final_verification.json'
        if sha(ROOT/helper)!=read(receipt)['source_sha256'][helper]:raise ValueError(f'Frozen helper changed: {helper}')
        consumed[helper]=sha(ROOT/helper);consumed[receipt.relative_to(ROOT).as_posix()]=sha(receipt)
    OUT.mkdir();metrics_rows=[];summaries=[];calls=0
    for job in jobs:
        identifier=job['reservoir_id'];folder=OUT/identifier;folder.mkdir()
        write(folder/'request.json',job)
        try:
            profile,selected_params,config=store.load({'reservoir_id':identifier})
            if job['physical_constants']!=profile['physical_constants'] or job['sampling']!=profile['sampling'] or job['root_policy']!=profile['root_policy']:
                raise ValueError('Fixed assumptions differ from selected profile')
            if job['period']['end']!=profile['periods']['calibration']['end'] or job['score_period']!=profile['periods']['calibration'] or job['period']['start']!=profile['periods']['simulation_start']:
                raise ValueError('Intervention must use continuous warmup and calibration only')
            pilot=ARCHIVE/'pilots'/identifier/f"seed_{profile['seed']}"
            baseline_targets=pd.read_csv(store.checked(pilot/'calibration_targets_si.csv'),index_col='month',float_precision='round_trip')
            targets=pd.read_csv(prepared(job['targets']),index_col='month',float_precision='round_trip')
            validate_intervention(targets,selected_params.monthly_targets,baseline_targets)
            parameters=ReservoirParameters(targets,**job['physical_constants'])
            dates=job['period'];period=job['score_period']
            baseline=pd.read_csv(store.checked(pilot/'baseline_simulation.csv'),index_col='date',parse_dates=True,float_precision='round_trip').loc[dates['start']:dates['end']].copy()
            selected=pd.read_csv(store.checked(ARCHIVE/'selected'/identifier/'simulation.csv'),index_col='date',parse_dates=True,float_precision='round_trip').loc[dates['start']:dates['end']].copy()
            if not baseline.index.equals(selected.index) or baseline.index[-1]!=pd.Timestamp(dates['end']):raise ValueError('Exact calibration prefix required')
            for col in ('inflow_m3s','observed_storage_m3','observed_release_m3s'):
                np.testing.assert_array_equal(baseline[col],selected[col])
            inputs=baseline[['inflow_m3s','observed_storage_m3','observed_release_m3s']].rename(columns={'observed_storage_m3':'storage_m3','observed_release_m3s':'release_m3s'})
            observations=_observations(inputs,'Archived model-input calibration prefix; source preprocessing already applied')
            inputs.to_csv(folder/'model_inputs_si.csv',index_label='date',float_format='%.17g')
            targets.to_csv(folder/'monthly_targets_si.csv',float_format='%.17g')
            calls+=1
            result=simulate_root_policy_reservoir(observations,parameters,sampling=SamplingSettings(**job['sampling']),policy=job['root_policy'],atol_storage_m3=config['atol_storage_m3'])
            frame=result.frame.copy()
            maximum=independent_balance(frame,parameters.inflow_correction)
            if maximum>config['atol_storage_m3']:raise ValueError('Independent mass balance exceeds tolerance')
            for col in ('storage_m3','release_m3s'):frame['observed_'+col]=inputs[col]
            frame['period']=baseline.period
            frame.to_csv(folder/'intervention_simulation.csv',index_label='date',float_format='%.17g')
            budget=difference_budget(frame,selected)
            budget.to_csv(folder/'intervention_minus_selected_budget.csv',index_label='date',float_format='%.17g')
            group=[];zones=[]
            for name,data in [('generalized_baseline',baseline),('frozen_selected',selected),('qc_restored_intervention',frame)]:
                part=data.loc[period['start']:period['end']]
                measured=_observations(inputs.loc[part.index])
                values=evaluate_simulation(measured,SimpleNamespace(frame=part))
                if name=='frozen_selected':
                    for v in ('storage','release'):np.testing.assert_allclose(values[v+'_nse'],profile['calibration_'+v+'_nse'],rtol=0,atol=1e-12)
                row={'reservoir_id':identifier,'GRAND_ID':profile['grand_id'],'parameter_set':name,'samples':len(part),**{k:float(v) if np.isfinite(v) else None for k,v in values.items()},
                     'storage_mean_bias_m3':float((part.storage_m3-measured.frame.storage_m3).mean()),'release_mean_bias_m3s':float((part.release_m3s-measured.frame.release_m3s).mean()),'zone1_samples':int((part.zone==1).sum()),'zone1_mean_release_m3s':float(part.loc[part.zone==1,'release_m3s'].mean()) if (part.zone==1).any() else None}
                group.append(row);metrics_rows.append(row)
                # Selected-zone membership fixes the comparison dates across profiles.
                mask=selected.loc[part.index].zone==1
                fixed=part.loc[mask]
                zones.append({'parameter_set':name,'selected_zone1_comparison_samples':len(fixed),'same_date_mean_release_m3s':float(fixed.release_m3s.mean()),'same_date_mean_storage_error_m3':float((fixed.storage_m3-measured.frame.loc[fixed.index,'storage_m3']).mean())})
            pd.DataFrame(group).to_csv(folder/'calibration_metrics.csv',index=False,float_format='%.17g')
            pd.DataFrame(zones).to_csv(folder/'fixed_date_low_storage_comparison.csv',index=False,float_format='%.17g')
            original=group[1];changed=group[2]
            summary={'reservoir_id':identifier,'status':'passed','simulation_calls':1,'maximum_balance_residual_m3':maximum,'calibration_storage_nse_change_vs_selected':changed['storage_nse']-original['storage_nse'],'calibration_release_nse_change_vs_selected':changed['release_nse']-original['release_nse'],
                     'end_calibration_storage_difference_m3':float(budget.storage_difference_m3.iloc[-1]),'end_calibration_cumulative_extra_release_m3':float(budget.cumulative_extra_release_volume_m3.iloc[-1]),'difference_budget_error_m3':float(np.abs(budget.storage_difference_m3-budget.budget_reconstructed_difference_m3).max()),'metrics':group,'heldout_simulated':False,'profile_promoted':False}
            fig,axes=plt.subplots(3,1,figsize=(11,9),constrained_layout=True)
            axes[0].plot(inputs.index,inputs.storage_m3/1e9,label='Observed')
            axes[1].plot(inputs.index,inputs.release_m3s,label='Observed',alpha=.6)
            for name,data in [('Baseline',baseline),('Selected',selected),('Qc restored',frame)]:
                axes[0].plot(data.index,data.storage_m3/1e9,label=name);axes[1].plot(data.index,data.release_m3s,label=name,alpha=.7)
            axes[2].plot(budget.index,budget.storage_difference_m3/1e9,label='Intervention minus selected storage')
            axes[2].plot(budget.index,budget.budget_reconstructed_difference_m3/1e9,'--',label='Conservation reconstruction')
            for ax,title in zip(axes,['Storage (billion m³)','Native monthly release (m³/s)','Storage-difference budget (billion m³)']):
                ax.set_title(title);ax.axvspan(inputs.index[0],pd.Timestamp(period['start']),alpha=.1,color='grey');ax.grid(alpha=.2);ax.legend(fontsize=8)
            fig.suptitle(identifier+' calibration-only fixed intervention; shaded warmup')
            fig.savefig(folder/'calibration_intervention.png',dpi=180);fig.savefig(folder/'calibration_intervention.pdf');plt.close(fig)
        except Exception as error:
            summary={'reservoir_id':identifier,'status':'failed','reason':str(error),'heldout_simulated':False,'profile_promoted':False}
        write(folder/'result.json',summary);summaries.append(summary);print(summary,flush=True)
    pd.DataFrame(metrics_rows).to_csv(OUT/'calibration_comparison.csv',index=False,float_format='%.17g')
    write(OUT/'experiment_summary.json',{'status':'passed' if all(s['status']=='passed' for s in summaries) else 'failed','jobs':summaries,'forward_simulation_calls':calls,'optimization_evaluations':0,'heldout_simulated':False,'frozen_profiles_changed':False})
    consumed.update(store.hashes)
    write(OUT/'execution_manifest.json',{'prepared_receipt_sha256':sha(SOURCE/'final_verification.json'),'source_sha256':consumed,'runner_sha256':sha(__file__),'artifact_sha256':{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}})
    if not all(s['status']=='passed' for s in summaries):raise SystemExit('One or more fixed interventions failed; inspect result.json')


if __name__=='__main__':main()
