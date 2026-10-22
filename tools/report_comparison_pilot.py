"""Report completed paired comparisons using stored results only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulate_calibrated_reservoirs import ROOT,ARCHIVE,read,write,sha
from comparison_pilot_job import OUT
from dztr.calibration.sampled import _observations
from dztr.evaluation import evaluate_simulation


def main():
    table=pd.read_csv(OUT/'comparison_scores.csv',float_precision='round_trip')
    assert table.assessment_feasible.all()
    design=read(OUT/'design.json');reference=[];regime_rows=[]
    original=read(ARCHIVE/'final_verification.json')
    for spec in design['reservoirs']:
        identifier=spec['reservoir_id']
        path=ARCHIVE/'selected'/identifier/'simulation.csv'
        assert sha(path)==original['artifact_sha256'][path.relative_to(ARCHIVE).as_posix()]
        frame=pd.read_csv(path,index_col='date',parse_dates=True,float_precision='round_trip')
        chosen=frame.index.year.isin(spec['common_assessment_years'])
        obs=_observations(frame.loc[chosen,['observed_storage_m3','observed_release_m3s','inflow_m3s']].rename(columns={'observed_storage_m3':'storage_m3','observed_release_m3s':'release_m3s'}))
        metrics=evaluate_simulation(obs,type('Stored',(),{'frame':frame.loc[chosen]})())
        metadata=read(ARCHIVE/'selected'/identifier/'parameters.json')
        reference.append({'reservoir_id':identifier,'role':'historical_frozen_1024_pooled_selected','seed':metadata['seed'],'evaluation_id':metadata['evaluation_id'],
                          'budget_per_search':1024,'calibration_years':14,'common_samples':int(chosen.sum()),**{'assessment_'+k:v for k,v in metrics.items()}})
        fig,axes=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
        for arm in spec['arms']:
            row=table[(table.reservoir_id==identifier)&(table.arm==arm)&(table.role=='pooled_selected')].iloc[0]
            folder=OUT/'assessed'/identifier/arm/f"pooled_selected_seed{int(row.seed)}_eval{int(row.evaluation_id)}"
            data=pd.read_csv(folder/'simulation.csv',index_col='date',parse_dates=True,float_precision='round_trip')
            part=data[data.role=='assessment']
            assert len(part)==48
            for regime in ('dry','normal','wet'):
                group=part[part.regime==regime]
                se=group.storage_m3-group.observed_storage_m3;qe=group.release_m3s-group.observed_release_m3s
                regime_rows.append({'reservoir_id':identifier,'arm':arm,'regime':regime,'samples':len(group),'storage_mean_bias_m3':float(se.mean()),'storage_rmse_m3':float(np.sqrt(np.mean(se**2))),
                                    'observed_storage_std_m3':float(group.observed_storage_m3.std(ddof=0)),
                                    'release_mean_bias_m3s':float(qe.mean()),'release_rmse_m3s':float(np.sqrt(np.mean(qe**2))),
                                    'storage_nse':float(row['assessment_'+regime+'_storage_nse']),'release_nse':float(row['assessment_'+regime+'_release_nse'])})
            axes[0].plot(data.index,data.storage_m3/1e9,label=arm)
            axes[1].plot(data.index,data.release_m3s,label=arm,alpha=.7)
        axes[0].plot(data.index,data.observed_storage_m3/1e9,label='Observed',color='black',linewidth=1.2)
        axes[1].plot(data.index,data.observed_release_m3s,label='Observed',color='black',linewidth=1.2)
        for year in spec['common_assessment_years']:
            for ax in axes:ax.axvspan(pd.Timestamp(year,1,1),pd.Timestamp(year,12,31),alpha=.13,color='grey')
        for ax,title in zip(axes,['Storage (billion m³)','Native monthly release (m³/s)']):ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(identifier+' — pooled calibration-selected profiles; common assessment shaded')
        fig.savefig(OUT/f'{identifier}_comparison.png',dpi=180);fig.savefig(OUT/f'{identifier}_comparison.pdf');plt.close(fig)
    pd.DataFrame(reference).to_csv(OUT/'previous_frozen_profile_reference.csv',index=False,float_format='%.17g')
    pd.DataFrame(regime_rows).to_csv(OUT/'common_regime_diagnostics.csv',index=False,float_format='%.17g')
    seedtable=table[table.role=='seed_balanced']
    summary=seedtable.groupby(['reservoir_id','arm'])[['assessment_storage_nse','assessment_release_nse']].agg(['median','min','max'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(OUT/'seed_agreement.csv',float_format='%.17g')
    # A simple fixed-reference diagnostic is descriptive, not a stopping rule.
    convergence=[]
    def hv(points):
        p=np.asarray(points,dtype=float);p=p[np.isfinite(p).all(axis=1)&(p<1.).all(axis=1)]
        previous=1.;total=0.
        for x,y in p[np.argsort(p[:,0])] if len(p) else []:
            if y<previous:total+=(1.-x)*(previous-y);previous=y
        return float(total)
    for job in design['jobs']:
        folder=OUT/'jobs'/job['reservoir_id']/job['arm']/f"seed_{job['seed']}"
        trials=pd.read_csv(folder/'trials.csv',float_precision='round_trip')
        earlier=trials.iloc[:384];earlier=earlier[earlier.feasible]
        last=trials[trials.feasible]
        before=hv(earlier[['release_cost','storage_cost']]);after=hv(last[['release_cost','storage_cost']])
        convergence.append({**job,'feasible_trials':len(last),'hypervolume384':before,'hypervolume512':after,'last128_gain':after-before,'relative_last128_gain':(after-before)/after if after else np.nan,
                            'reference_excluded_feasible_points':int((last[['release_cost','storage_cost']]>=1).any(axis=1).sum()),'convergence_established':False})
    pd.DataFrame(convergence).to_csv(OUT/'budget_diagnostics.csv',index=False,float_format='%.17g')
    pooled=table[table.role=='pooled_selected']
    lines=['# Garrison/Oahe: evaluated calibration-strategy pilot','',
           'Eighteen completed searches compare chronological years with pooled NSE, mixed years with pooled NSE, and the same mixed years with equal-regime NSE. Each search uses population32, seeds42/43/44 and exactly512 evaluations:9,216 new search evaluations. All three arms use twenty calibration years (66.7% of all recorded rows). Four common assessment years per reservoir were excluded from every arm before searches began.','',
           '**Result:** mixed-year selection substantially improves Oahe storage on the common assessment set, but Garrison shows a storage–release trade-off. Equal-regime weighting is not uniformly better than mixed years with pooled NSE. No universal winner or production profile is declared.','',
           '| Reservoir | Strategy | Common storage NSE | Common release NSE | Equal-regime storage NSE | Equal-regime release NSE |','| --- | --- | --- | --- | --- | --- |']
    for _,r in pooled.iterrows():lines.append(f"| {r.reservoir_id} | {r.arm} | {r.assessment_storage_nse:.4f} | {r.assessment_release_nse:.4f} | {r.assessment_regime_balanced_storage_nse:.4f} | {r.assessment_regime_balanced_release_nse:.4f} |")
    lines+=['','Rows show one pooled compromise per arm, selected using calibration costs only. All three seed-balanced results and ranges are reported separately; no validation seed was selected.','',
            'Across three seed-balanced fits, Garrison chronological storage NSE is0.640–0.664 versus mixed pooled0.542–0.623; release is0.488–0.535 versus mixed pooled0.547–0.602. Oahe chronological storage is negative for all seeds (-4.246 to -1.576), while both mixed arms are positive for all seeds: mixed pooled0.437–0.445 and mixed balanced0.465–0.491. These are optimization-seed ranges, not confidence or predictive uncertainty intervals.','',
            '**Remaining weakness:** equal-regime common storage NSE is negative for all six pooled-selected profiles. Each individual common inflow regime also has negative storage NSE. Positive pooled storage NSE therefore does not establish adequate dry/normal/wet behavior. Regime sets are small (24 dry months,12 normal,12 wet); use the accompanying RMSE, bias and observed-variability table to interpret NSE denominators rather than claiming statistical significance.','',
            'For Garrison, mixed pooled is a clear release improvement paired with a storage loss relative to matched chronology. Equal-regime weighting partly recovers pooled storage and loses some release performance. For Oahe, the mixed design removes the large chronological storage failure; equal weighting adds a small pooled-storage gain and loses release performance, while its equal-regime storage score is actually lower than mixed pooled. These results support keeping both pooled and regime diagnostics rather than adopting equal weighting automatically.','',
            'The year-selection contrast includes refitted targets, bounds and initialization from the respective calibration years, not only a change of scoring dates. The weighting contrast holds calibration dates, targets/bounds and initial populations identical. The first32 trial vectors match between mixed arms for every seed. Model constants, solver policies, source forcing and initial states remain fixed. All model simulations retain continuous carryover; no observation reset occurs in assessment years.','',
            'Common assessment: Garrison1991/1992(dry),1995(normal),1997(wet); Oahe1992/1994(dry),1997(wet),1998(normal). Chronological calibration is1971–1990; mixed training years are listed in design.json. Other unscored years provide continuous forcing context. This is a retrospective comparison on a previously inspected record, not blind prospective forecasting validation.','',
            'The old1971–1984 seed-balanced512-evaluation fits and the latest frozen1024-evaluation pooled profiles are rescored from their archived trajectories on the same common years as descriptive references. Their smaller training period (and the latter different budget) prevents treating them as matched experimental arms. The original registry/GRAND_ID parameters were not overwritten.','',
            'Budget512 is a controlled pilot checkpoint, not certified convergence. Fixed-reference hypervolume diagnostics are provided within each objective definition; hypervolumes from different objective definitions must not be ranked against one another. Any extension should retain masks/settings and preserve the512-evaluation archive rather than tuning to assessment scores.','',
            'Decision: retain the frozen profiles; treat mixed selection as promising for Oahe and application-dependent for Garrison. Do not roll out a new universal objective or select parameters from common assessment performance. A broader matched-reservoir comparison and predeclared robustness assessment are needed before changing the fleet strategy.','',
            '[Full scores](comparison_scores.csv), [pooled results](pooled_comparison.csv), [paired strategy contrasts](paired_strategy_contrasts.csv), [seed agreement](seed_agreement.csv), [regime diagnostics](common_regime_diagnostics.csv), [old frozen profiles](previous_frozen_profile_reference.csv), [budget diagnostics](budget_diagnostics.csv).','',
            '[Garrison figure](ex5_Garrison_comparison.png) · [Oahe figure](ex6_Oahe_comparison.png). [Frozen design](design.json), [calibration-only selection](frozen_selection.json), [scientific verification](scientific_verification.json), [test receipt](test_receipt.json), [final verification](final_verification.json).','']
    content='\n'.join(lines)
    for old,new in [('population32','population 32'),('seeds42/43/44','seeds 42/43/44'),('exactly512','exactly 512'),('evaluations:9,216','evaluations: 9,216'),
                    ('is0.','is 0.'),('pooled0.','pooled 0.'),('balanced0.','balanced 0.'),('months,12','months, 12'),('normal,12','normal, 12'),
                    ('Garrison1991','Garrison 1991'),('Oahe1992','Oahe 1992'),('),1995','), 1995'),('),1997','), 1997'),('),1998','), 1998'),
                    ('is1971','is 1971'),('old1971','old 1971'),('balanced512','balanced 512'),('frozen1024','frozen 1024'),('Budget512','Budget 512'),('first32','first 32'),('the512','the 512')]:content=content.replace(old,new)
    (OUT/'README.md').write_text(content,encoding='utf-8')
    write(OUT/'assessment_manifest.json',{'code_sha256':{p:sha(ROOT/p) for p in ['tools/assess_comparison_pilot.py','tools/report_comparison_pilot.py','tools/run_critical_interventions.py']},'artifact_sha256':{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('assessment_manifest.json',)}})
    print('Complete paired pilot report generated from stored scores; historical references explicitly unmatched.')


if __name__=='__main__':main()
