"""Archived calibration-only critical-release bounds and Pareto-front review."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulate_calibrated_reservoirs import ROOT, ARCHIVE, read, write, sha

OUT=ROOT/'outputs/critical_release_review_20261012'


def critical_block(vectors):
    values=np.asarray(vectors,dtype=float)
    if values.ndim!=2 or values.shape[1]!=72:raise ValueError('Expected native 72-coordinate vectors')
    return values[:,36:48]


def nondominated_mask(costs):
    values=np.asarray(costs,dtype=float)
    if values.ndim!=2 or values.shape[1]!=2 or not np.isfinite(values).all():raise ValueError('Expected finite two-objective costs')
    return np.array([not np.any(np.all(values<=x,axis=1)&np.any(values<x,axis=1)) for x in values])


def main():
    if OUT.exists():raise ValueError('Use a fresh review directory')
    receipt=read(ARCHIVE/'final_verification.json');consumed={}
    def checked(relative):
        path=ARCHIVE/relative
        digest=receipt['artifact_sha256'][relative]
        if sha(path)!=digest:raise ValueError(f'Archived file changed: {relative}')
        consumed[relative]=digest
        return path
    OUT.mkdir();summaries=[]
    for identifier in ('ex5_Garrison','ex6_Oahe'):
        folder=OUT/identifier;folder.mkdir()
        profile=read(checked(f'selected/{identifier}/parameters.json'))
        pilot=f"pilots/{identifier}/seed_{profile['seed']}"
        baseline=pd.read_csv(checked(pilot+'/calibration_targets_si.csv'),index_col='month',float_precision='round_trip')
        bounds=pd.read_csv(checked(pilot+'/calibration_bounds_si.csv'),index_col='month',float_precision='round_trip')
        selected=pd.read_csv(checked(f'selected/{identifier}/monthly_targets_si.csv'),index_col='month',float_precision='round_trip')
        result=read(checked(pilot+'/result.json'))
        qc=baseline.release_critical_m3s.to_numpy()
        lo=bounds.lower_release_critical_m3s.to_numpy();hi=bounds.upper_release_critical_m3s.to_numpy()
        assert ((lo<=qc)&(qc<=hi)).all()
        audit=pd.DataFrame({'month':range(1,13),'lower_qcritical_m3s':lo,'baseline_qcritical_m3s':qc,'selected_qcritical_m3s':selected.release_critical_m3s.to_numpy(),'upper_qcritical_m3s':hi})
        audit['selected_position_in_bound']=(audit.selected_qcritical_m3s-lo)/(hi-lo)
        audit['baseline_position_in_bound']=(qc-lo)/(hi-lo)
        audit['selected_minus_baseline_m3s']=audit.selected_qcritical_m3s-qc
        assert audit.selected_position_in_bound.between(0,1).all()
        audit.to_csv(folder/'monthly_bound_audit.csv',index=False,float_format='%.17g')
        parts=[];trial_summaries=[]
        for seed in (42,43,44):
            base=f'pilots/{identifier}/seed_{seed}'
            front=pd.read_csv(checked(base+'/pareto_front.csv'),index_col='evaluation_id',float_precision='round_trip')
            vectors=pd.read_csv(checked(base+'/vectors.csv'),index_col='evaluation_id',float_precision='round_trip')
            trials=pd.read_csv(checked(base+'/trials.csv'),index_col='evaluation_id',float_precision='round_trip')
            assert front.feasible.all()
            q=critical_block(vectors.loc[front.index])
            front['seed']=seed;front['mean_monthly_qcritical_m3s']=q.mean(axis=1)
            front['all_months_qcritical_at_most_baseline']=(q<=qc).all(axis=1)
            parts.append(front.reset_index())
            feasible=trials.loc[trials.feasible]
            tq=critical_block(vectors.loc[feasible.index])
            trial_summaries.append({'seed':seed,'feasible_trials':len(feasible),'feasible_trials_mean_qcritical_at_most_baseline':int((tq.mean(axis=1)<=qc.mean()).sum()),'feasible_trials_all_months_at_most_baseline':int((tq<=qc).all(axis=1).sum()),
                                    'front_points':len(front),'front_mean_qcritical_min_m3s':float(q.mean(axis=1).min()),'front_mean_qcritical_max_m3s':float(q.mean(axis=1).max())})
        pooled=pd.concat(parts,ignore_index=True)
        pooled['pooled_nondominated']=nondominated_mask(pooled[['release_cost','storage_cost']])
        basecost=np.array([-result['baseline_calibration']['release_nse'],-result['baseline_calibration']['storage_nse']])
        costs=pooled[['release_cost','storage_cost']].to_numpy()
        pooled['dominates_baseline_on_calibration']=np.all(costs<=basecost,axis=1)&np.any(costs<basecost,axis=1)
        pooled.to_csv(folder/'archived_seed_front_candidates.csv',index=False,float_format='%.17g')
        pd.DataFrame(trial_summaries).to_csv(folder/'seed_archive_summary.csv',index=False,float_format='%.17g')
        summary={'reservoir_id':identifier,'GRAND_ID':profile['grand_id'],'baseline_mean_monthly_qcritical_m3s':float(qc.mean()),'selected_mean_monthly_qcritical_m3s':float(selected.release_critical_m3s.mean()),'lower_bound_mean_monthly_qcritical_m3s':float(lo.mean()),'upper_bound_mean_monthly_qcritical_m3s':float(hi.mean()),
                 'seed_front_candidates':len(pooled),'pooled_nondominated_candidates':int(pooled.pooled_nondominated.sum()),'seed_front_candidates_mean_at_most_baseline':int((pooled.mean_monthly_qcritical_m3s<=qc.mean()).sum()),'seed_front_candidates_all_months_at_most_baseline':int(pooled.all_months_qcritical_at_most_baseline.sum()),
                 'seed_front_candidates_dominating_baseline_calibration':int(pooled.dominates_baseline_on_calibration.sum()),'minimum_seed_front_mean_qcritical_m3s':float(pooled.mean_monthly_qcritical_m3s.min()),'selected_months_above_90pct_bound':int((audit.selected_position_in_bound>=.9).sum()),
                 'decision':'retain_existing_bounds; test calibration-only Qcritical intervention before changing objective or model', 'heldout_used_for_candidate_selection':False,'profiles_changed':False,'simulations':0,'optimization_evaluations':0}
        write(folder/'review.json',summary);summaries.append(summary)
        fig,ax=plt.subplots(figsize=(8,6),constrained_layout=True)
        scatter=ax.scatter(pooled.release_nse,pooled.storage_nse,c=pooled.mean_monthly_qcritical_m3s,cmap='viridis',s=25)
        ax.scatter(-basecost[0],-basecost[1],marker='s',color='red',label='Generalized baseline')
        ax.scatter(profile['calibration_release_nse'],profile['calibration_storage_nse'],marker='*',s=160,color='orange',edgecolors='black',label='Frozen selected')
        ax.set(xlabel='Calibration release NSE',ylabel='Calibration storage NSE',title=identifier+' archived seed fronts');ax.grid(alpha=.2);ax.legend()
        fig.colorbar(scatter,ax=ax,label='Unweighted mean monthly critical-release target (m³/s)')
        fig.savefig(folder/'calibration_front.png',dpi=180);fig.savefig(folder/'calibration_front.pdf');plt.close(fig)
    write(OUT/'review_summary.json',{'reservoirs':summaries,'simulations':0,'optimization_evaluations':0,'validation_used_for_selection':False})
    write(OUT/'proposed_calibration_test_protocol.json',{'status':'prepared_not_executed','purpose':'Isolate critical-release contribution on calibration evidence before any objective/bound changes','reservoirs':['ex5_Garrison','ex6_Oahe'],
          'period':'Existing continuous warmup and calibration only; exclude held-out dates','comparators':['Reuse archived generalized baseline','Reuse archived frozen selected profile','One diagnostic profile retaining selected storage/normal/flood targets and restoring twelve critical-release targets to generalized baseline values'],
          'fixed_assumptions':['Same source forcing and preprocessing','Same initial states','Same 30.5-day clock and operating-month convention','Same conservative overflow and root policy'],
          'checks':['Validate storage ordering and critical <= normal <= flood release ordering before any run','Independent mass balance','Calibration storage and release NSE versus both comparators','Low-storage zone occupancy and cumulative release/storage budget'],
          'decision_rule':'Describe calibration trade-off and mechanism; do not promote a profile, alter bounds or change equations automatically','new_optimization_evaluations':0,'holdout_policy':'Do not choose targets or seeds using held-out performance; any later retrospective held-out diagnostic must be explicitly labeled and separate'})
    lines=['# Critical-release bounds and calibration trade-off review','','This review uses only archived calibration costs/vectors to assess candidates. Held-out degradation motivated the question, but no held-out scores enter the candidate comparisons or a new selection. Frozen profiles are unchanged.','',
           '| Reservoir | Lower-bound mean Qc | Baseline mean Qc | Selected mean Qc | Minimum mean Qc on any seed front | Front points | Calibration front points dominating baseline |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for s in summaries:
        lines.append(f"| [{s['reservoir_id']}]({s['reservoir_id']}/review.json) | {s['lower_bound_mean_monthly_qcritical_m3s']:.3f} | {s['baseline_mean_monthly_qcritical_m3s']:.3f} | {s['selected_mean_monthly_qcritical_m3s']:.3f} | {s['minimum_seed_front_mean_qcritical_m3s']:.3f} | {s['seed_front_candidates']} | {s['seed_front_candidates_dominating_baseline_calibration']} |")
    lines+=['','Qc values are unweighted averages of twelve monthly targets (m³/s), not realized release or volume. Each original seed-front entry retains its identity; pooled nondominance is also recorded.','',
            'The existing calibration-percentile bounds allow lower critical release than baseline in every month. No final seed-front candidate has mean Qc at or below baseline, and none is at or below baseline in all twelve months. Thus an overly high lower bound is not demonstrated. Simply widening bounds downward is not yet justified. Calibration improvements can favor critical-release settings that transfer poorly; no archived final-front replacement satisfying that screening criterion can be identified. This does not prove that no feasible low-Qc solution exists.','',
            'Do not turn the baseline-Qc screening criterion into a physical safety rule: the generalized baseline is a comparator, not documented operational demand. Pareto association does not isolate the effect of critical release because all six target families vary jointly. The original equal-weight range-normalized pooled selection remains unchanged.','',
            '**Decision:** retain current bounds and equations. The next justified experiment is the prepared calibration-only intervention restoring selected Qc to baseline while holding the other five target families fixed. It would isolate a trade-off before considering a constrained recalibration or a different low-storage formulation. No such experiment or new optimization was run in this review.','',
            '[Prepared test protocol](proposed_calibration_test_protocol.json), [review summary](review_summary.json), [prior verified low-storage mechanism](../garrison_oahe_diagnosis_20261012/release_mechanism.md). Each reservoir folder includes twelve-month bound positions, all archived candidate identities, seed-level trial counts and a calibration-only Pareto figure.','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    write(OUT/'analysis_manifest.json',{'archive_receipt_sha256':sha(ARCHIVE/'final_verification.json'),'consumed_sha256':consumed,'runner_sha256':sha(__file__),'artifact_sha256':{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}})
    print(summaries)


if __name__=='__main__':main()
