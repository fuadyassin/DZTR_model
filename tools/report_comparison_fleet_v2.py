"""Summarize the complete twenty-reservoir comparison from stored results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulate_calibrated_reservoirs import ROOT, read, write, sha
from comparison_fleet_v2_job import OUT

PILOT = ROOT/'outputs/strategy_comparison_pilot_20261012'
ARMS = ['chronological_pooled', 'mixed_pooled', 'mixed_balanced']
LABELS = ['Chronological', 'Mixed years', 'Mixed years, equal-regime weighting']


def direction(storage, release, tolerance=1e-8):
    if not np.isfinite([storage, release]).all(): return 'Assessment unavailable'
    if abs(storage)<=tolerance and abs(release)<=tolerance: return 'No material difference'
    if storage>=-tolerance and release>=-tolerance: return 'Both improve or one improves with the other unchanged'
    if storage<=tolerance and release<=tolerance: return 'Both decline or one declines with the other unchanged'
    return 'Storage-release trade-off'


def number(value):
    return f'{value:.3f}' if np.isfinite(value) else 'unavailable'


def main():
    proof = read(PILOT/'final_verification.json')
    for name in ('comparison_scores.csv', 'pooled_comparison.csv', 'design.json'):
        assert sha(PILOT/name)==proof['artifact_sha256'][name]
    designs = read(PILOT/'design.json')['reservoirs'] + read(OUT/'design.json')['reservoirs']
    all_rows = pd.concat([pd.read_csv(p/'comparison_scores.csv', float_precision='round_trip').assign(experiment_directory=p.relative_to(ROOT).as_posix()) for p in (PILOT, OUT)], ignore_index=True)
    all_rows.to_csv(OUT/'all_twenty_scores.csv', index=False, float_format='%.17g')
    pooled = all_rows[all_rows.role=='pooled_selected'].copy()
    assert len(pooled)==60 and pooled.reservoir_id.nunique()==20
    pooled.to_csv(OUT/'all_twenty_pooled_comparison.csv', index=False, float_format='%.17g')
    registry = read(ROOT/'configs/calibrated_twenty.json')
    grand = {Path(path.replace('\\','/')).parent.name: int(gid) for gid,path in registry['profiles_by_GRAND_ID'].items()}
    summaries=[];regimes=[]; contrasts=[]
    for spec in sorted(designs, key=lambda s:s['reservoir_id']):
        identifier=spec['reservoir_id'];folder=OUT/'reservoir_reports'/identifier;folder.mkdir(parents=True,exist_ok=True)
        rows=pooled[pooled.reservoir_id==identifier].set_index('arm')
        summary={'reservoir_id':identifier,'GRAND_ID':grand.get(identifier),'assessment_years':','.join(map(str,spec['common_assessment_years'])),
                 'assessment_regimes_missing':','.join(sorted({'dry','normal','wet'}-set(spec['common_assessment_regimes'].values()))),
                 'calibration_fraction_min':min(a['calibration_fraction_total'] for a in spec['arms'].values()),
                 'chronological_and_mixed_years_identical':spec['arms']['chronological_pooled']['calibration_years']==spec['arms']['mixed_pooled']['calibration_years']}
        lines=[f'# {identifier}', '', 'These profiles were selected using calibration evidence only. Scores below use the same excluded assessment years for every strategy.', '',
               '| Strategy | Storage NSE | Release NSE | Equal-regime storage NSE | Equal-regime release NSE |', '| --- | ---: | ---: | ---: | ---: |']
        fig, axes=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
        observed=None
        for arm,label in zip(ARMS,LABELS):
            row=rows.loc[arm]
            for var in ('storage','release'):
                summary[arm+'_'+var+'_nse']=row.get('assessment_'+var+'_nse',np.nan)
            lines.append('| '+label+' | '+' | '.join(number(row.get('assessment_'+metric,np.nan)) for metric in ('storage_nse','release_nse','regime_balanced_storage_nse','regime_balanced_release_nse'))+' |')
            base=ROOT/row.experiment_directory
            target=base/'assessed'/identifier/arm/f'pooled_selected_seed{int(row.seed)}_eval{int(row.evaluation_id)}'
            if not row.assessment_feasible:
                lines+=['',f'Assessment failed for {label}: {row.assessment_reason}. No score is substituted.'];continue
            data=pd.read_csv(target/'simulation.csv',index_col='date',parse_dates=True,float_precision='round_trip')
            observed=data
            part=data[data.role=='assessment']
            for regime,group in part.groupby('regime'):
                se=group.storage_m3-group.observed_storage_m3;qe=group.release_m3s-group.observed_release_m3s
                regimes.append({'reservoir_id':identifier,'arm':arm,'regime':regime,'samples':len(group),
                    'storage_nse':row.get('assessment_'+regime+'_storage_nse',np.nan),'release_nse':row.get('assessment_'+regime+'_release_nse',np.nan),
                    'storage_rmse_m3':np.sqrt(np.nanmean(se**2)),'storage_mean_bias_m3':se.mean(),
                    'observed_storage_std_m3':group.observed_storage_m3.std(ddof=0),
                    'release_rmse_m3s':np.sqrt(np.nanmean(qe**2)),'release_mean_bias_m3s':qe.mean()})
            axes[0].plot(data.index,data.storage_m3/1e9,label=label)
            axes[1].plot(data.index,data.release_m3s,label=label,alpha=.7)
        for name,before,after in [('year_selection','chronological_pooled','mixed_pooled'),('regime_weighting','mixed_pooled','mixed_balanced')]:
            ds=summary[after+'_storage_nse']-summary[before+'_storage_nse'];dq=summary[after+'_release_nse']-summary[before+'_release_nse']
            summary[name+'_storage_change']=ds;summary[name+'_release_change']=dq;summary[name+'_outcome']=direction(ds,dq)
            contrasts.append({'reservoir_id':identifier,'contrast':name,'storage_nse_change':ds,'release_nse_change':dq,'outcome':direction(ds,dq)})
            lines+=['',f"{'Mixed years versus chronology' if name=='year_selection' else 'Equal-regime weighting versus mixed pooled'}: storage NSE changes by {number(ds)}; release NSE changes by {number(dq)}. {direction(ds,dq)}."]
        seed_rows=all_rows[(all_rows.reservoir_id==identifier)&(all_rows.role=='seed_balanced')&all_rows.arm.isin(ARMS)]
        lines+=['','| Strategy | Storage NSE across seeds (median; min to max) | Release NSE across seeds (median; min to max) |','| --- | --- | --- |']
        for arm,label in zip(ARMS,LABELS):
            part=seed_rows[seed_rows.arm==arm]
            values=[]
            for var in ('storage','release'):
                x=part['assessment_'+var+'_nse'].dropna()
                values.append(f'{number(x.median())}; {number(x.min())} to {number(x.max())}')
                summary[arm+'_seed_median_'+var+'_nse']=x.median()
            lines.append('| '+label+' | '+' | '.join(values)+' |')
        missing=summary['assessment_regimes_missing']
        lines+=['',f"Common assessment years: {summary['assessment_years']}. Minimum calibration coverage: {summary['calibration_fraction_min']:.1%}."]
        if missing:lines+=['',f'No common assessment year represents: {missing}. Performance in these regimes is not assessed.']
        if summary['chronological_and_mixed_years_identical']:lines+=['','Under this assessment design, chronological and mixed calibration years coincide; this reservoir does not independently test the year-selection contrast.']
        negative=[f'{arm}: {sum(r["storage_nse"]<0 for r in regimes if r["reservoir_id"]==identifier and r["arm"]==arm)}' for arm in ARMS]
        undefined=[f'{arm}: {sum(not np.isfinite(r["storage_nse"]) for r in regimes if r["reservoir_id"]==identifier and r["arm"]==arm)}' for arm in ARMS]
        lines+=['','Regimes with negative storage NSE: '+ '; '.join(negative)+'. Undefined regime storage NSE: '+ '; '.join(undefined)+'. Missing regimes are not counted as successful. See the fleet regime diagnostics for RMSE, bias and observed variability.', '',
                'Seed ranges describe optimizer variability, not confidence intervals. This is a retrospective assessment on previously inspected records. The 512-evaluation budget does not certify convergence. Historical profiles use different training periods and are descriptive references.', '',
                '[Comparison figure](comparison.png). Existing production/exploratory registry profiles were not replaced.','']
        if observed is not None:
            axes[0].plot(observed.index,observed.observed_storage_m3/1e9,color='black',label='Observed')
            axes[1].plot(observed.index,observed.observed_release_m3s,color='black',label='Observed',alpha=.8)
            for year in spec['common_assessment_years']:
                for ax in axes:ax.axvspan(pd.Timestamp(year,1,1),pd.Timestamp(year,12,31),color='grey',alpha=.13)
            for ax,title in zip(axes,('Storage (billion mÂ³)','Release (mÂ³/s)')):ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
            fig.suptitle(identifier+'; common assessment years shaded')
            fig.savefig(folder/'comparison.png',dpi=160);fig.savefig(folder/'comparison.pdf')
        plt.close(fig)
        (folder/'README.md').write_text('\n'.join(lines),encoding='utf-8');summaries.append(summary)
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'reservoir_outcomes_by_GRAND_ID.csv',index=False,float_format='%.17g')
    pd.DataFrame(regimes).to_csv(OUT/'all_twenty_regime_diagnostics.csv',index=False,float_format='%.17g')
    contrast=pd.DataFrame(contrasts);contrast.to_csv(OUT/'all_twenty_strategy_contrasts.csv',index=False,float_format='%.17g')
    seed=all_rows[(all_rows.role=='seed_balanced')&all_rows.arm.isin(ARMS)].groupby(['reservoir_id','arm'])[['assessment_storage_nse','assessment_release_nse']].agg(['median','min','max','count'])
    seed.columns=['_'.join(c) for c in seed.columns];seed.to_csv(OUT/'all_twenty_seed_agreement.csv',float_format='%.17g')
    lines=['# Twenty-reservoir calibration-strategy comparison','',
           'Eighteen new reservoirs (162 searches; 82,944 search evaluations) extend the verified Garrison/Oahe pilot. Across all twenty: 180 searches, population 32, seeds 42/43/44, 512 evaluations per search. The pilot outputs are reused without repeating its searches.', '',
           'Each reservoir compares chronological pooled NSE, mixed-year pooled NSE and the same mixed years with equal-regime NSE. Calibration uses at least 65% of recorded rows and available observation pairs. All arms use the same number of whole calibration years; daily leap-year row counts may differ slightly. Common assessment years are excluded from all arms before optimization. Continuous forcing and simulated storage carryover are retained, without observation resets.', '',
           'For shorter or regime-limited later records, reserve up to four later years covering available regimes where the record permits, with a later training opportunity retained; absent regimes are explicitly listed. The original pilot design is retained where applicable. This forces an honest common assessment rather than creating artificial wet/dry labels.', '']
    counts={name:part.outcome.value_counts().to_dict() for name,part in contrast.groupby('contrast')}
    for name,part in contrast.groupby('contrast'):
        lines += [f"{'Mixed-year selection versus chronology' if name=='year_selection' else 'Equal-regime weighting versus mixed pooled'}:",'']
        for outcome,count in part.outcome.value_counts().items():lines.append(f'- {outcome}: {count} reservoirs.')
        lines.append('')
    lines += ['| Reservoir | Chronological S / Q NSE | Mixed S / Q NSE | Equal-regime S / Q NSE | Mixed-year outcome |', '| --- | ---: | ---: | ---: | --- |']
    for _,r in summary.iterrows():
        cells=[number(r[a+'_storage_nse'])+' / '+number(r[a+'_release_nse']) for a in ARMS]
        lines.append(f"| [{r.reservoir_id}](reservoir_reports/{r.reservoir_id}/README.md) | "+' | '.join(cells)+f" | {r.year_selection_outcome} |")
    lines += ['', 'S = storage; Q = release. Rows show pooled calibration-selected compromises, not winners chosen from assessment results. Numerical changes count as improvement/decline; no statistical significance threshold is claimed.', '',
              f"Assessment failures among the sixty pooled-selected profiles: {int((~pooled.assessment_feasible).sum())}. Failed or undefined scores remain unavailable and are not replaced by successful seeds.", '',
              f"Missing assessment regimes affect {int(summary.assessment_regimes_missing.ne('').sum())} reservoirs. Identical chronological/mixed masks affect {int(summary.chronological_and_mixed_years_identical.sum())} reservoirs. Inspect their individual reports before interpreting the contrast.", '',
              'Existing parameter registry and GRAND_ID parameters remain unchanged. This comparison informs strategy choice; it does not promote the strategy with the largest assessment score. Fort Randall retains a null GRAND_ID pending verification.', '',
              '[All scores](all_twenty_scores.csv), [pooled profiles](all_twenty_pooled_comparison.csv), [GRAND_ID outcome table](reservoir_outcomes_by_GRAND_ID.csv), [regime diagnostics](all_twenty_regime_diagnostics.csv), [seed agreement](all_twenty_seed_agreement.csv), [contrasts](all_twenty_strategy_contrasts.csv), [scientific verification](scientific_verification.json), [test receipt](test_receipt.json).','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    write(OUT/'outcome_summary.json',{'counts':counts,'reservoirs':20,'new_reservoirs':18,'profiles_promoted':False,'assessment_failures':int((~pooled.assessment_feasible).sum())})
    write(OUT/'assessment_manifest.json',{'code_sha256':{p:sha(ROOT/p) for p in ['tools/assess_comparison_fleet_v2.py','tools/report_comparison_fleet_v2.py']},'artifact_sha256':{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('assessment_manifest.json','final_verification.json')}})
    print(summary[['reservoir_id','year_selection_outcome','regime_weighting_outcome']].to_string(index=False))


if __name__=='__main__': main()
