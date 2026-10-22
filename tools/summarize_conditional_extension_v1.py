"""Explain completed results without choosing profiles from assessment scores."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows import conditional_cohort_v1 as w


def main():
    out=ROOT/'outputs/conditional_extension_cohort_v1_20261021';completion=ROOT/'outputs/conditional_extension_completion_v1_20261021'
    w.native.verify_registry(out/'report/registry.json')
    scores=pd.read_csv(out/'report/scores_vs_baseline.csv');curves=pd.read_csv(out/'report/convergence.csv')
    coverage=pd.read_csv(completion/'admission_and_selection.csv');seasonal=pd.read_csv(completion/'seasonal_bias.csv')
    bias=pd.read_csv(completion/'bias_and_release_peaks.csv');qualifications=[];convergence=[]
    lines=['# Interpretation of the new conditional profiles','','These diagnostics describe response-conditioned consistency. They do not remove unresolved source gates or establish independent prediction skill. All seed choices remain frozen from calibration costs.','']
    for rid,part in scores.groupby('reservoir_id',sort=False):
        selected=part.loc[part.selected].iloc[0];base=part.loc[part['profile']=='baseline'].iloc[0];seeds=part.loc[part.seed.notna()]
        rows=curves[curves.reservoir_id==rid].pivot(index='seed',columns='evaluations',values='best_cost')
        for seed,row in rows.iterrows():convergence.append({'reservoir_id':rid,'seed':int(seed),'gain_256_to_512':row[256]-row[512],
            'last_checkpoint_cost':row[512],'convergence_established':False})
        strong_storage=bool(selected.assessment_storage_nse>0)
        qualifications.append({'reservoir_id':rid,'selected_seed':int(selected.seed),
            'qualification':'conditional_comparison_only' if strong_storage else 'conditional_storage_assessment_failed',
            'independent_prediction_validated':False,'source_gates_resolved':False,'convergence_established':False})
        spec=coverage.loc[coverage.reservoir_id==rid].iloc[0]
        lines += [f'## {rid}','',f"Selected seed {int(selected.seed)}, using calibration cost only. Actual fitting support: {spec.actual_fitting_days} of {spec.calendar_days} days ({100*spec.actual_fitting_fraction:.2f}%); {spec.assessment_days} supported assessment days.",'',
            '| Metric | Baseline | Selected | Change |','|---|---:|---:|---:|']
        for key in ['calibration_storage_nse','calibration_release_nse','assessment_storage_nse','assessment_release_nse']:
            lines.append(f'| {key} | {base[key]:.4f} | {selected[key]:.4f} | {selected[key]-base[key]:+.4f} |')
        lines+=['',f"Across seeds, final calibration cost ranges from {seeds.calibration_cost.min():.6f} to {seeds.calibration_cost.max():.6f}. All three searches improved between 256 and 512 evaluations. Similar costs do not establish parameter uniqueness or convergence.",'']
        record=bias[(bias.reservoir_id==rid)&(bias['profile']==selected['profile'])&(bias.role=='assessment')].iloc[0]
        lines += [f"Selected conditional assessment mean storage bias is {record.storage_bias_m3/1e6:+.2f} million m3; mean release bias is {record.release_bias_m3s:+.2f} m3/s. At releases above calibration Q90 ({record.high_release_threshold_calibration_q90_m3s:.2f} m3/s), mean release bias is {record.high_release_bias_m3s:+.2f} m3/s across {int(record.high_release_count)} assessment days.",'']
        monthly=seasonal[(seasonal.reservoir_id==rid)&(seasonal['profile']==selected['profile'])&(seasonal.role=='assessment')&(seasonal.variable=='storage')].dropna()
        worst=monthly.assign(absolute=monthly.mean_bias.abs()).sort_values('absolute',ascending=False).head(3)
        lines+=['Largest conditional assessment monthly storage biases: '+', '.join(f"month {int(r.month)}: {r.mean_bias/1e6:+.2f} million m3" for _,r in worst.iterrows())+'.','']
        if rid.endswith('Revelstoke'):
            lines+=['All three seeds have negative storage NSE in the short assessment period (approximately -3.30 to -3.76), despite improved release scores. This is not an accepted storage-prediction profile. Its assessment covers July-December 2019 only. The narrow observed storage range makes NSE sensitive to storage drift, and the plots show a late-2019 positive storage departure. Small release errors can accumulate into substantial storage errors through the water balance. This observation is consistent with the model diagnostics; it does not identify a unique cause. Obtain a longer storage record and reconcile physical metadata/clock before claiming transferability.','']
        else:
            lines += ['All seeds improve calibration storage and release; seed 42 slightly reduces assessment storage NSE while improving release. The selected seed improves both conditional assessment metrics, but it was selected from calibration evidence. The 101 observed storage values above the assumed 11.3 billion m3 ceiling remain unmodified and explicitly flagged. Capacity, live/total storage and outlet/clock definitions therefore still require reconciliation.','']
    pd.DataFrame(qualifications).to_csv(completion/'profile_use_qualification.csv',index=False)
    pd.DataFrame(convergence).to_csv(completion/'convergence_review.csv',index=False)
    lines+=['## Collection decision','','The collection remains 62 reservoirs: the original 41 plus 21 Etienne additions. Hungry Horse and these two additions provide three completed exploratory scenarios among the 21 additions. The other 18 remain visible and source-gated: five paired records, ten storage-only records and three release-only records. No new GRAND ID has been certified. Existing paper and Hungry Horse profiles remain unchanged.','',
        'Do not use these conditional scenarios to claim ungauged performance. The next substantive evidence would be independent time-varying forcing and compatible storage/outlet/clock definitions, plus longer storage support for Revelstoke. Additional search evaluations alone cannot resolve those limitations.','']
    (completion/'diagnostic_findings.md').write_text('\n'.join(lines),encoding='utf-8')
    readme=completion/'README.md';content=readme.read_text(encoding='utf-8')
    content+='\n[Interpretation and remaining limitations](diagnostic_findings.md), [profile-use qualifications](profile_use_qualification.csv), [convergence review](convergence_review.csv). Revelstoke retains a failed conditional storage assessment; calibration completion is not performance acceptance.\n'
    readme.write_text(content,encoding='utf-8');print(completion/'diagnostic_findings.md')


if __name__=='__main__':main()
