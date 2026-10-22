"""Historical sampled adapter preserves settings, selection, and validation."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("cohort_export", Path(__file__).parents[1] / "tools/standardize_cohort_archives.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root, frequency="monthly"):
    folder = root / "pilots/daily/seed_42"
    folder.mkdir(parents=True)
    config = {"reservoir": {"id": "daily"}, "units": {"storage": "m3", "flows": "m3/s"},
              "sampling": {"frequency": frequency, "fixed_step_days": 30.5 if frequency == "monthly" else 1},
              "periods": {"calibration": {"start": "2001-01-01", "end": "2002-12-31"}},
              "model": {"capacity_m3": 1000, "dead_storage_fraction": .1}}
    config_file = root / "configs/daily.json"
    put(config_file, config)
    settings = {"seed": 42, "population_size": 16, "evaluation_budget": 128}
    identity = {"settings": settings, "split": config["periods"], "sampling": config["sampling"],
                "input_sha256": {"configs/daily.json": module.base.sha(config_file)}}
    roles = {role: index for role, index in zip(("best_release", "best_storage", "balanced"), (1, 2, 3))}
    put(folder / "checkpoint_identity.json", identity)
    put(folder / "resolved_config.json", config)
    put(folder / "selection.json", {"representatives": roles, "validation_used": False, "written_before_validation": True})
    result = {"reservoir_id": "daily", "seed": 42, "settings": settings, "status": "pilot_completed", "termination": "budget_reached",
              "search_evaluations": 128, "identity": identity, "selection_uses_validation": False, "feasible_search_evaluations": 128,
              "representatives": [{"role": role, "evaluation_id": i, "calibration_release_nse": .5, "calibration_storage_nse": .6,
                                   "validation_feasible": True, "validation_reason": None, "validation_release_nse": .4,
                                   "validation_storage_nse": -.8} for role, i in roles.items()],
              "baseline_calibration": {"release_nse": .5, "storage_nse": .6},
              "baseline_validation": {"feasible": True, "reason": None, "metrics": {"release_nse": .4, "storage_nse": .48}}}
    put(folder / "result.json", result)
    vectors = np.tile(np.repeat([.2, .4, .8, 1, 2, 3], 12), (128, 1))
    pd.DataFrame(vectors, index=pd.Index(range(128), name="evaluation_id")).to_csv(folder / "vectors.csv")
    costs = np.tile([-.5, -.6], (128, 1))
    pd.DataFrame({"evaluation_id": range(128), "feasible": True, "release_cost": costs[:, 0], "storage_cost": costs[:, 1]}).to_csv(folder / "trials.csv", index=False)
    np.savez(folder / "checkpoint.npz", vectors=vectors, objectives=costs)
    proof = {"artifact_sha256": {p.relative_to(root).as_posix(): module.base.sha(p) for p in folder.iterdir()}}
    entry = {"reservoir_id": "daily", "seed": 42, "population_size": 16, "evaluation_budget": 128}
    protocol = {"pilot": {"population_size": 16, "evaluation_budget_per_job": 128}}
    review = {"eligible_for_exploratory_calibration": True, "pilot_selected": True}
    return folder, entry, proof, protocol, review


@pytest.mark.parametrize("frequency", ["daily", "monthly"])
def test_native_settings_and_negative_historical_validation_preserved(tmp_path, frequency):
    folder, entry, proof, protocol, review = fixture(tmp_path, frequency)
    result, config, roles, targets = module.load_pilot(tmp_path, folder, entry, proof, protocol, review, {})
    assert result["settings"]["population_size"] == 16 and result["search_evaluations"] == 128
    assert config["sampling"]["fixed_step_days"] == (1 if frequency == "daily" else 30.5)
    assert roles["balanced"]["validation_storage_nse"] == -.8
    assert roles["generalized_baseline"]["validation_storage_nse"] == .48
    assert roles["balanced"]["evaluation_id"] == 3
    np.testing.assert_array_equal(targets["balanced"].values, np.tile([200, 400, 800, 1, 2, 3], (12, 1)))


def test_historical_population_cannot_be_relabelled(tmp_path):
    folder, entry, proof, protocol, review = fixture(tmp_path)
    entry["population_size"] = 32
    with pytest.raises(ValueError, match="identity or settings"):
        module.load_pilot(tmp_path, folder, entry, proof, protocol, review, {})


def test_changed_artifact_rejected(tmp_path):
    folder, entry, proof, protocol, review = fixture(tmp_path)
    (folder / "vectors.csv").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        module.load_pilot(tmp_path, folder, entry, proof, protocol, review, {})


@pytest.mark.parametrize("flag,value", [("validation_used", True), ("written_before_validation", False)])
def test_validation_based_selection_rejected(tmp_path, flag, value):
    folder, entry, proof, protocol, review = fixture(tmp_path)
    path = folder / "selection.json"
    selection = module.base.read(path)
    selection[flag] = value
    put(path, selection)
    proof["artifact_sha256"][path.relative_to(tmp_path).as_posix()] = module.base.sha(path)
    with pytest.raises(ValueError, match="frozen before validation"):
        module.load_pilot(tmp_path, folder, entry, proof, protocol, review, {})


def test_checkpoint_vector_disagreement_rejected(tmp_path):
    folder, entry, proof, protocol, review = fixture(tmp_path)
    path = folder / "checkpoint.npz"
    with np.load(path) as saved:
        vectors, costs = saved["vectors"].copy(), saved["objectives"].copy()
    vectors[0, 0] += .01
    np.savez(path, vectors=vectors, objectives=costs)
    proof["artifact_sha256"][path.relative_to(tmp_path).as_posix()] = module.base.sha(path)
    with pytest.raises(AssertionError):
        module.load_pilot(tmp_path, folder, entry, proof, protocol, review, {})
