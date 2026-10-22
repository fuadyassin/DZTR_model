"""Budget continuation and calibration-only pooled selection invariants."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from dztr.calibration.optimizer import NSGA2Settings, optimize_nsga2
from twenty_analysis import pooled_selection, pair_metrics


def test_continuation_preserves_prefix_and_matches_uninterrupted_search():
    baseline = np.full(72, .5)
    objective = SimpleNamespace(bounds=SimpleNamespace(lower_vector=np.zeros(72), upper_vector=np.ones(72)), evaluate=lambda v: SimpleNamespace(feasible=True, objectives=(float(np.sum(v*v)), float(np.sum((v-1)**2))), reason=None))
    settings = NSGA2Settings(seed=42, population_size=4, evaluation_budget=8)
    saved = []
    first = optimize_nsga2(objective, baseline, settings, checkpoint_callback=saved.append)
    continued = optimize_nsga2(objective, baseline, replace(settings, evaluation_budget=16), resume_state=saved[-1])
    uninterrupted = optimize_nsga2(objective, baseline, replace(settings, evaluation_budget=16))
    np.testing.assert_array_equal([p.vector for p in first.evaluated], [p.vector for p in continued.evaluated[:8]])
    np.testing.assert_array_equal([p.vector for p in uninterrupted.evaluated], [p.vector for p in continued.evaluated])
    np.testing.assert_array_equal([p.objectives for p in uninterrupted.evaluated], [p.objectives for p in continued.evaluated])


def test_pooled_selection_ignores_validation_and_input_order():
    points=pd.DataFrame({"seed":[44,42,43,42],"evaluation_id":[4,4,4,8],"release_cost":[-.5,-.9,-.7,-.7],"storage_cost":[-.9,-.5,-.7,-.7],"validation_nse":[100,-100,0,1000]})
    first,_=pooled_selection(points)
    points["validation_nse"]=[-1000,999,999,0]
    second,_=pooled_selection(points.sample(frac=1,random_state=3))
    assert (int(first.seed),int(first.evaluation_id))==(42,8)
    assert (int(second.seed),int(second.evaluation_id))==(42,8)


def test_sparse_zone_scores_remain_explicitly_undefined():
    frame=pd.DataFrame({"storage_m3":[2.],"observed_storage_m3":[1.],"release_m3s":[3.],"observed_release_m3s":[2.]},index=pd.to_datetime(["2000-01-01"]))
    result=pair_metrics(frame,{"start":"2000-01-01","end":"2000-01-01"})
    assert result["storage_nse"] is None and result["release_nse"] is None
    assert result["storage_mean_bias"]==1 and result["release_rmse"]==1
