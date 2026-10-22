"""Seal completed scientific results and annotate quality without changing selections."""
import argparse
from pathlib import Path
import re
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from workflows import reservoir_v2 as w

def seal(output):
    out=Path(output).resolve(strict=True);dest=out/'report'
    if not out.is_relative_to(ROOT/'outputs'):raise ValueError('Study must be inside outputs')
    if (dest/'manifest.json').exists():raise ValueError('Preserve sealed results')
    proof=w.read(dest/'verification.json')
    for rel,digest in proof['source_sha256'].items():
        if w.sha(ROOT/rel)!=digest:raise ValueError('Verified source changed')
    invariant=w.read(out/'treatment_invariants.json')
    assert invariant['status']=='passed' and len(invariant['reservoirs'])==8
    summary=pd.read_csv(dest/'new_reservoir_scores.csv',float_precision='round_trip')
    both_positive=summary.assessment_storage_nse.gt(0)&summary.assessment_release_nse.gt(0)
    assert int(both_positive.sum())==3
    assert summary.assessment_feasible.all()
    programme=pd.read_csv(dest/'prioritized_programme.csv')
    programme=programme.rename(columns={'exploratory_treatment_status':'preflight_treatment_outcome'})
    programme['new_profile_quality_note']='not reassessed in this batch'
    for _,row in summary.iterrows():
        good=row.assessment_storage_nse>0 and row.assessment_release_nse>0
        programme.loc[programme.reservoir_id==row.reservoir_id,'new_profile_quality_note']=(
            'Both held-out NSE values positive; source assumptions and convergence remain unresolved' if good else
            'At least one held-out NSE nonpositive; review forcing/timing/model before application')
    assert len(programme)==42 and int(programme.included.sum())==41
    assert int((programme.in_published_paper&programme.included).sum())==37
    assert int(programme.loc[programme.reservoir_id!='ex35_Bull_Lake','programme_status'].str.startswith('calibrated').sum())==30
    assert int(programme.loc[programme.in_published_paper&programme.reservoir_id.ne('ex35_Bull_Lake'),'programme_status'].str.startswith('calibrated').sum())==29
    programme.to_csv(dest/'prioritized_programme.csv',index=False)
    for path in (dest/'README.md',out/'README.md',ROOT/'docs/paper_exploration.md'):
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if not link.startswith(('http://','https://','#')):assert (path.parent/link).exists(),(path,link)
    for relative in proof['profile_registries']:w.verify_registry(ROOT/relative)
    proof.update({'new_profiles_with_both_assessment_NSE_positive':3,'new_profiles_with_nonpositive_assessment_NSE':5,
        'calibration_both_objectives_improved':int((summary.calibration_storage_nse_gain.gt(0)&summary.calibration_release_nse_gain.gt(0)).sum()),
        'reported_sources_and_links_verified':True,'input_invariants_verified':True,
        'seal_source_sha256':w.sha(Path(__file__)), 'new_fixed_model_preflight_calls':sum(
            a['stage']=='fixed_calibration_preflight' for r in w.read(out/'review.json')['reservoirs'] for a in r.get('attempts',[]))})
    w.write(dest/'verification.json',proof)
    extra=[ROOT/'data/paper_cohort_v1/hess-23-3735-2019-t01.xlsx',ROOT/'data/paper_cohort_v1/download.json',
           ROOT/'docs/paper_exploration.md',Path(__file__),ROOT/'tools/report_paper_exploration.py']
    w.write(dest/'manifest.json',{'kind':'paper_first_exploratory_study_seal_v1',
        'artifact_sha256':{p.relative_to(out).as_posix():w.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=dest/'manifest.json'},
        'related_artifact_sha256':{p.relative_to(ROOT).as_posix():w.sha(p) for p in extra}})
    print('Sealed: 37 paper cases, 29 standard paper profiles + Bull Lake + 7 pending; 30 standard profiles across 41 active cases.')
    return dest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(seal(parser.parse_args().output))
