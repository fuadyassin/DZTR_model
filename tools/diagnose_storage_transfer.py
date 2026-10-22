"""Garrison/Oahe archived trajectory budgets and monthly-target review."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulate_calibrated_reservoirs import ROOT, ARCHIVE, read, write, sha

OUT=ROOT/"outputs/garrison_oahe_diagnosis_20261012"


def difference_budget(selected, baseline):
    if not selected.index.equals(baseline.index):
        raise ValueError("Trajectory dates differ")
    for column in ("inflow_m3s","step_seconds"):
        np.testing.assert_array_equal(selected[column],baseline[column])
    delta_s=(selected.storage_m3-baseline.storage_m3).to_numpy()
    delta_q=(selected.release_m3s-baseline.release_m3s).to_numpy()
    release_volume=np.r_[0.,np.cumsum(selected.step_seconds.to_numpy()[1:]/2*(delta_q[1:]+delta_q[:-1]))]
    correction=np.cumsum((selected.balance_residual_m3-baseline.balance_residual_m3).to_numpy())
    predicted=delta_s[0]-release_volume+correction
    np.testing.assert_allclose(predicted,delta_s,rtol=0,atol=1e-3)
    return pd.DataFrame({"storage_difference_m3":delta_s,"cumulative_extra_release_volume_m3":release_volume,
                         "residual_correction_m3":correction,"budget_reconstructed_difference_m3":predicted},index=selected.index)


def first_sustained(mask, count=6):
    """First start of count consecutive qualifying samples; no smoothing/filling."""
    values=np.asarray(mask,dtype=bool)
    for i in range(len(values)-count+1):
        if values[i:i+count].all(): return mask.index[i]
    return None


def main():
    if OUT.exists(): raise ValueError("Use a fresh diagnosis directory")
    proof=read(ARCHIVE/"final_verification.json");consumed={}
    def checked(path):
        if path.is_relative_to(ARCHIVE): digest=proof["artifact_sha256"][path.relative_to(ARCHIVE).as_posix()]
        else:
            # Packaged sample hashes remain pinned to the calibration inputs.
            protocol=read(ROOT/"outputs/policy_fleet_20261012/protocol.json")
            hashes={p.replace('\\','/'):d for p,d in protocol['inputs_sha256'].items()}
            digest=hashes[path.relative_to(ROOT).as_posix()]
        if sha(path)!=digest: raise ValueError(f"Source changed: {path}")
        consumed[str(path)]=digest
        return path
    OUT.mkdir(); summaries=[]
    for identifier in ('ex5_Garrison','ex6_Oahe'):
        folder=OUT/identifier;folder.mkdir()
        profile=read(checked(ARCHIVE/'selected'/identifier/'parameters.json'))
        pilot=ARCHIVE/'pilots'/identifier/f"seed_{profile['seed']}"
        assert read(checked(pilot/'initialization.json'))['selected_candidate']=='generalized'
        config=read(checked(pilot/'resolved_config.json'))
        provenance=read(checked(ROOT/'data/reservoirs'/identifier/'observation_provenance.json'))
        source=read(checked(ROOT/'data/reservoirs'/identifier/'matlab_source.json'))
        writer=Path(provenance['writer_file'])
        assert sha(writer)==provenance['source_sha256'][str(writer)]
        consumed[str(writer)]=sha(writer)
        code=writer.read_text(encoding='utf-8-sig')
        (folder/'source_writer_excerpt.txt').write_text('\n'.join(code.splitlines()[:23])+'\n',encoding='utf-8')
        selected=pd.read_csv(checked(ARCHIVE/'selected'/identifier/'simulation.csv'),index_col='date',parse_dates=True,float_precision='round_trip')
        baseline=pd.read_csv(checked(pilot/'baseline_simulation.csv'),index_col='date',parse_dates=True,float_precision='round_trip')
        budget=difference_budget(selected,baseline)
        aligned=selected[['observed_storage_m3','observed_release_m3s','inflow_m3s','period']].copy()
        for name,frame in (('baseline',baseline),('selected',selected)):
            for col in ('storage_m3','release_m3s','zone'): aligned[name+'_'+col]=frame[col]
        aligned=aligned.join(budget)
        aligned.to_csv(folder/'aligned_trajectories.csv',index_label='date',float_format='%.17g')
        targets=pd.read_csv(checked(pilot/'calibration_targets_si.csv'),index_col='month',float_precision='round_trip')
        optimized=pd.read_csv(checked(ARCHIVE/'selected'/identifier/'monthly_targets_si.csv'),index_col='month',float_precision='round_trip')
        changes=[]
        for month in range(1,13):
            for col in targets.columns:
                a,b=float(targets.loc[month,col]),float(optimized.loc[month,col])
                changes.append({'month':month,'target':col,'baseline':a,'selected':b,'change':b-a,'storage_change_percent_capacity':100*(b-a)/profile['physical_constants']['capacity_m3'] if col.startswith('storage') else None})
        pd.DataFrame(changes).to_csv(folder/'monthly_target_changes.csv',index=False,float_format='%.17g')
        yearly=aligned.groupby(aligned.index.year).agg(baseline_storage_m3=('baseline_storage_m3','mean'),selected_storage_m3=('selected_storage_m3','mean'),observed_storage_m3=('observed_storage_m3','mean'),baseline_release_m3s=('baseline_release_m3s','mean'),selected_release_m3s=('selected_release_m3s','mean'),observed_release_m3s=('observed_release_m3s','mean'))
        yearly.to_csv(folder/'annual_means.csv',index_label='year',float_format='%.17g')
        capacity=profile['physical_constants']['capacity_m3']
        periods=[]
        for label in ('calibration','validation'):
            dates=profile['periods'][label];part=aligned.loc[dates['start']:dates['end']]
            se=part.selected_storage_m3-part.observed_storage_m3;be=part.baseline_storage_m3-part.observed_storage_m3
            onset=first_sustained((se.abs()>be.abs())&(se.abs()>0.05*capacity))
            periods.append({'period':label,'baseline_mean_storage_error_m3':float(be.mean()),'selected_mean_storage_error_m3':float(se.mean()),
                            'selected_minus_baseline_mean_release_m3s':float((part.selected_release_m3s-part.baseline_release_m3s).mean()),
                            'storage_difference_first_m3':float(part.storage_difference_m3.iloc[0]),'storage_difference_last_m3':float(part.storage_difference_m3.iloc[-1]),
                            'first_sustained_worse_5pct_capacity_six_month_start':str(onset.date()) if onset is not None else None})
        worst=(aligned.selected_storage_m3-aligned.observed_storage_m3).abs().idxmax()
        summary={'reservoir_id':identifier,'GRAND_ID':profile['grand_id'],'periods':periods,'maximum_budget_reconstruction_error_m3':float(np.max(np.abs(budget.storage_difference_m3-budget.budget_reconstructed_difference_m3))),
                 'worst_selected_storage_error_date':str(worst.date()),'max_target_storage_change_percent_capacity':float(pd.DataFrame(changes).storage_change_percent_capacity.abs().max()),
                 'decision':'retain_frozen_profiles; review calibration-rule transfer and source timing before any parameter/forcing correction',
                 'demonstrated_unit_or_row_shift_error':False,'storage_snapshot_timing_confirmed':False,'monthly_flow_statistic_confirmed':False,'simulations':0,'optimization_evaluations':0}
        summaries.append(summary);write(folder/'diagnosis.json',summary)
        fig,axes=plt.subplots(4,1,figsize=(12,13),constrained_layout=True)
        for col,label in [('observed_storage_m3','Observed'),('baseline_storage_m3','Baseline'),('selected_storage_m3','Selected')]: axes[0].plot(aligned.index,aligned[col]/1e9,label=label)
        for col,label in [('observed_release_m3s','Observed'),('baseline_release_m3s','Baseline'),('selected_release_m3s','Selected')]: axes[1].plot(aligned.index,aligned[col],label=label,alpha=.7)
        axes[2].plot(budget.index,budget.storage_difference_m3/1e9,label='Selected minus baseline storage')
        axes[2].plot(budget.index,budget.budget_reconstructed_difference_m3/1e9,'--',label='Reconstructed from release difference + residual')
        for table,name,style in [(targets,'Baseline','--'),(optimized,'Selected','-')]:
            for col in ('storage_critical_m3','storage_normal_m3','storage_flood_m3'): axes[3].plot(table.index,table[col]/1e9,style,label=name+' '+col.split('_')[1])
        for ax in axes[:3]: ax.axvline(pd.Timestamp(profile['periods']['validation']['start']),color='black',linestyle=':',label='Held-out starts')
        for ax in axes: ax.legend(fontsize=8,ncol=3);ax.grid(alpha=.2)
        for ax,title in zip(axes,['Storage (billion m³)','Native monthly release (m³/s)','Storage-difference budget (billion m³)','Monthly storage targets (billion m³)']): ax.set_title(title)
        fig.suptitle(identifier+' — archived trajectories; no new simulation')
        fig.savefig(folder/'diagnosis.png',dpi=180);fig.savefig(folder/'diagnosis.pdf');plt.close(fig)
        lines=[f'# {identifier}: storage-transfer diagnosis','',f"GRAND_ID {profile['grand_id']}. Calibration 1971–1984; held-out 1985–1999. Same inflows, initial states, root policy and 30.5-day monthly clock are used by baseline and selected simulations.",'','![Archived comparison](diagnosis.png)','',
               'The source writer multiplies storage by one million and copies flow columns unchanged. Workbook/MAT matching is recorded in the packaged provenance. The code uses synthetic 30.5-day dates; normalized data uses workbook month labels. These establish processing conventions, not whether storage is month-end or whether flows are monthly means. No demonstrated unit conversion or row-shift error was found; do not invent a correction.','']
        for p in periods:
            lines.append(f"- {p['period']}: mean storage errors baseline {p['baseline_mean_storage_error_m3']/1e9:+.4g}, selected {p['selected_mean_storage_error_m3']/1e9:+.4g} billion m³; mean selected-minus-baseline release {p['selected_minus_baseline_mean_release_m3s']:+.4g} m³/s. Storage difference begins at {p['storage_difference_first_m3']/1e9:+.4g} and ends at {p['storage_difference_last_m3']/1e9:+.4g} billion m³. First six-month sustained selected-error excess above 5% capacity: {p['first_sustained_worse_5pct_capacity_six_month_start'] or 'not found'}.")
        lines+=['',f"The difference-budget identity reconstructs storage difference to within {summary['maximum_budget_reconstruction_error_m3']:.4g} m³. Identical forcing cancels, so accumulated release differences explain simulated storage separation up to recorded roundoff residuals. This is bookkeeping, not an independent demonstration of why the selected release rule is inappropriate.",'',
                'The held-out run carries the calibration-end state forward; it is not restarted at observed storage. Initial held-out separation can therefore be inherited. The six-month/5%-capacity screen is a declared descriptive locator, not a statistically estimated change point. Inspect annual means and the full trajectories before attributing a sudden operating change.','',
                f"Largest monthly storage-target change is {summary['max_target_storage_change_percent_capacity']:.4g}% of capacity. [All six target changes](monthly_target_changes.csv) retain units and signs; target changes jointly affect zone membership and release behavior, so this review cannot isolate one target's causal contribution.",'',
                '**Decision:** keep the archived profile unchanged, treat generalized baseline as an important comparator, and review whether calibration-period monthly rules transfer to the later record. Obtain source definitions for storage snapshot timing and inflow/release averaging intervals before altering clock, shifting observations or adjusting inflows. No targeted recalibration is yet justified by a proven input correction.','',
                '[Aligned trajectories and conservation budget](aligned_trajectories.csv), [annual means](annual_means.csv), [numeric diagnosis](diagnosis.json), [source writer excerpt](source_writer_excerpt.txt), [PDF](diagnosis.pdf).','']
        (folder/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    write(OUT/'summary.json',{'reservoirs':summaries,'simulations':0,'optimization_evaluations':0})
    (OUT/'README.md').write_text('# Garrison then Oahe: detailed diagnosis\n\n[ Garrison](ex5_Garrison/README.md) · [Oahe](ex6_Oahe/README.md)\n\nReuses archived trajectories and checks source processing, storage separation, monthly targets and conservation. No new simulations, optimization, parameter changes or assumed input corrections. Source snapshot/averaging definitions remain unresolved.\n',encoding='utf-8')
    write(OUT/'analysis_manifest.json',{'consumed_sha256':consumed,'runner_sha256':sha(__file__),'artifact_sha256':{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}})
    print(summaries)


if __name__=='__main__': main()
