"""Scientific distinctions and archived evidence integrity for diagnostic triage."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"tools"))
from performance_assessment import recommendation, largest_bias, OUT
from simulate_calibrated_reservoirs import ROOT, ARCHIVE, read, sha


def evidence(cal=(.5,.5),val=(.5,.5),delta=(.1,.1)):
    return {f"{prefix}_{period}_{v}_nse":values[i]
            for prefix,period,values in (("selected","calibration",cal),("selected","validation",val),("delta","validation",delta))
            for i,v in enumerate(("storage","release"))}


def test_improvement_does_not_replace_absolute_skill_or_calibration_review():
    assert recommendation(evidence(val=(-.1,.6),delta=(.8,.3)))[0]=='review_forcing_and_timing'
    assert recommendation(evidence(cal=(-.1,.6),val=(-.1,.6)))[0]=='review_model_assumptions'
    assert recommendation(evidence(val=(0,.6)))[0]=='review_forcing_and_timing'


def test_positive_skill_does_not_replace_baseline_comparison():
    assert recommendation(evidence(val=(.9,.9),delta=(-.001,.1)))[0]=='review_calibration_tradeoff'
    assert recommendation(evidence(delta=(0,.1)))[0]=='retain_exploratory'
    assert recommendation(evidence(val=(np.nan,.8)))[0]=='review_missing_evidence'


def test_signed_seasonal_bias_uses_absolute_magnitude_and_skips_missing():
    data=pd.DataFrame({'month':[1,2,3,4],'bias':[2.,-3.,np.nan,3.]})
    chosen=largest_bias(data,'bias')
    assert chosen.month==2 and chosen.bias==-3
    assert largest_bias(data.iloc[[2]],'bias') is None


def test_assessment_matches_frozen_scores_without_changing_profiles():
    if not OUT.exists():
        pytest.skip('Assessment artifact is generated independently')
    table=pd.read_csv(OUT/'reservoir_performance_and_actions.csv',float_precision='round_trip').set_index('reservoir_id')
    source=pd.read_csv(ARCHIVE/'reservoir_parameterization_by_GRAND_ID.csv',float_precision='round_trip').set_index('reservoir_id')
    assert len(table)==20 and table.index.is_unique and set(table.index)==set(source.index)
    score_columns=[c for c in table if c.startswith(('baseline_','selected_','delta_')) and c.endswith('_nse')]
    pd.testing.assert_frame_equal(table[score_columns].sort_index(),source[score_columns].sort_index(),check_exact=True)
    assert not table.selection_changed_by_assessment.any()
    assert table.GRAND_ID.notna().sum()==19 and pd.isna(table.loc['ex4_Fort_randall','GRAND_ID'])
    manifest=read(OUT/'analysis_manifest.json')
    for relative,digest in manifest['consumed_sha256'].items(): assert sha(ROOT/relative)==digest
    for relative,digest in manifest['artifact_sha256'].items(): assert sha(OUT/relative)==digest
    assert manifest['runner_sha256']==sha(ROOT/'tools/performance_assessment.py')
    assert read(OUT/'assessment_summary.json')['optimization_evaluations']==0
    assert read(OUT/'assessment_summary.json')['simulations']==0
