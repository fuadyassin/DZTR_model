"""Portable policy comparisons, calibration isolation and review integrity."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dztr.calibration.objectives import CandidateEvaluation
from dztr.calibration.sampled import _observations
from dztr.integrations import collection
from dztr.integrations import root_policy_review as module
from dztr.simulation.sampled import simulate_sampled_reservoir
from test_collection_runner import _json, collection_fixture


PHYSICAL_COLUMNS = ["storage_m3", "release_m3s", "inflow_m3s", "effective_inflow_m3s",
    "zone", "overflow_volume_m3", "balance_residual_m3", "release_rule_residual_m3s",
    "step_seconds", "rule_month"]


def csv(path):
    return pd.read_csv(path, index_col="date", parse_dates=["date"], float_precision="round_trip")


def pin_config(root, path, manifest, entry, config):
    config_path = root / entry["config_file"]
    _json(config_path, config)
    entry["config_sha256"] = module.sha256(config_path)
    entry["files_sha256"][entry["config_file"]] = entry["config_sha256"]
    _json(path, manifest)


def pin_frame(root, path, manifest, entry, frame):
    observations_path = root / entry["observations_file"]
    frame.to_csv(observations_path, float_format="%.17g")
    entry["observations_sha256"] = module.sha256(observations_path)
    entry["files_sha256"][entry["observations_file"]] = entry["observations_sha256"]
    config = json.loads((root / entry["config_file"]).read_text())
    config["observations_sha256"] = entry["observations_sha256"]
    pin_config(root, path, manifest, entry, config)


def run(path, output, **kwargs):
    return module.run_root_policy_review(path, output, all_reservoirs=True, **kwargs)


def policies(record):
    return {value["policy"]: value for value in record["policies"]}


def no_root_fixture(root):
    path, manifest = collection_fixture(root)
    entry = manifest["reservoirs"][0]
    frame = csv(root / entry["observations_file"])
    frame.loc[frame.index[:2], "inflow_m3s"] = 2.15
    pin_frame(root, path, manifest, entry, frame)
    config = json.loads((root / entry["config_file"]).read_text())
    config["model"]["dead_storage_fraction"] = 0.3025
    pin_config(root, path, manifest, entry, config)
    return path, manifest, frame, config


def test_unique_root_control_matches_existing_baseline_exactly(tmp_path):
    path, _ = collection_fixture(tmp_path / "repository")
    old_output, output = tmp_path / "existing", tmp_path / "review"
    original = collection.run_collection(path, old_output, all_reservoirs=True)
    assert original["counts"] == {"completed": 1}
    result = run(path, output)
    assert result["status"] == "review_completed"
    assert result["default_policy_unchanged"] == "reject" and result["policy_selected"] is None
    assert not result["selection_uses_validation"] and result["search_evaluations"] == 0
    record = result["reservoirs"][0]
    assert record["first_calibration_failure"] is None and record["policy_selected"] is None
    original_frame = csv(old_output / "ex1_Example/experiment/baseline_simulation.csv")
    for policy in module.POLICIES:
        comparison = policies(record)[policy]
        assert comparison["status"] == "completed"
        simulated = csv(output / "ex1_Example" / policy / "simulation.csv")
        pd.testing.assert_frame_equal(simulated[PHYSICAL_COLUMNS], original_frame[PHYSICAL_COLUMNS], check_exact=True)
        assert not simulated.root_policy_applied.any() and not simulated.root_policy_tie_break.any()
        assert comparison["trajectory_quality"]["multiple_root_steps"] == 0
        metadata = ["selected_root_storage_m3", "selected_root_release_m3s", "root_admissibility_tolerance_m3"]
        assert simulated.loc[simulated.index[0], metadata].isna().all()
        assert simulated.root_admissibility_tolerance_m3.iloc[1:].ge(1e-4).all()
        assert not simulated.overflow_volume_m3.any()
        np.testing.assert_array_equal(simulated.selected_root_storage_m3.iloc[1:], simulated.storage_m3.iloc[1:])
        np.testing.assert_array_equal(simulated.selected_root_release_m3s.iloc[1:], simulated.release_m3s.iloc[1:])
    for filename in ("calibration_targets_si.csv", "calibration_bounds_si.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(output / "ex1_Example" / filename, float_precision="round_trip"),
            pd.read_csv(old_output / "ex1_Example/experiment" / filename, float_precision="round_trip"), check_exact=True)


def test_no_root_remains_a_gap_and_reference_is_numerically_invalid(tmp_path):
    path, _, frame, config = no_root_fixture(tmp_path / "repository")
    fitting = module.target_observations(_observations(frame.loc["2001-01-01":"2003-12-01"]))
    estimates = module.estimate_generalized_parameters(fitting, config["target_percentiles"],
        quantile_method=config["quantile_method"], channel_capacity_m3s=config["model"]["channel_capacity_m3s"])
    parameters = module.ReservoirParameters(estimates.monthly_targets, **config["model"])
    sampling = module.SamplingSettings(**config["sampling"])
    observed = _observations(frame)
    with pytest.raises(ValueError, match="No feasible implicit") as before:
        simulate_sampled_reservoir(observed, parameters, sampling=sampling)
    result = run(path, tmp_path / "review")
    record = result["reservoirs"][0]
    failure = record["first_calibration_failure"]
    assert failure["failure_code"] == "implicit_no_root" and failure["distinct_root_count"] == 0
    assert failure["matlab_reference"]["tail_period"] == 2
    assert not failure["matlab_reference"]["converged"] and not failure["matlab_reference"]["balance_valid"]
    for policy in module.POLICIES:
        comparison = policies(record)[policy]
        assert comparison["calibration_status"] == "blocked" and comparison["validation_status"] == "not_run"
        assert "No feasible implicit" in comparison["reason"]
    reference = policies(record)[module.REFERENCE]
    assert reference["status"] == "numerically_invalid_reference" and reference["reference_only"]
    assert not reference["calibration_quality"]["numerically_valid"]
    assert record["calibration_pareto_policies"] == [] and result["policy_selected"] is None
    with pytest.raises(ValueError, match="No feasible implicit") as after:
        simulate_sampled_reservoir(observed, parameters, sampling=sampling)
    assert str(before.value) == str(after.value)


def test_held_out_changes_do_not_change_fit_search_or_calibration_selection(tmp_path):
    root = tmp_path / "repository"
    path, manifest = collection_fixture(root)
    before_output, after_output = tmp_path / "before", tmp_path / "after"
    settings = {"policies": ("reject", "maximum_storage"), "include_matlab_reference": False,
                "pilot_policy": "maximum_storage", "optimization_budget": 2, "seeds": (42, 43)}
    before = run(path, before_output, **settings)
    entry = manifest["reservoirs"][0]
    frame = csv(root / entry["observations_file"])
    held_out = frame.index >= "2004-01-01"
    frame.loc[held_out, "storage_m3"] *= 1.7
    frame.loc[held_out, "release_m3s"] = frame.loc[held_out, "release_m3s"] * 1.5 + 1
    frame.loc[held_out, "inflow_m3s"] *= 3
    pin_frame(root, path, manifest, entry, frame)
    after = run(path, after_output, **settings)
    first, second = before["reservoirs"][0], after["reservoirs"][0]
    assert first["calibration_pareto_policies"] == second["calibration_pareto_policies"]
    for policy in settings["policies"]:
        assert policies(first)[policy]["metrics"]["calibration"] == policies(second)[policy]["metrics"]["calibration"]
        pd.testing.assert_frame_equal(csv(before_output / "ex1_Example" / policy / "calibration_simulation.csv"),
                                     csv(after_output / "ex1_Example" / policy / "calibration_simulation.csv"), check_exact=True)
    assert policies(first)["reject"]["metrics"]["validation"] != policies(second)["reject"]["metrics"]["validation"]
    for filename in ("calibration_targets_si.csv", "calibration_bounds_si.csv"):
        assert (before_output / "ex1_Example" / filename).read_bytes() == (after_output / "ex1_Example" / filename).read_bytes()
    assert before["search_evaluations"] == after["search_evaluations"] == 4
    for old, new in zip(first["pilots"], second["pilots"]):
        assert old["settings"] == new["settings"] and old["calibration_representatives"] == new["calibration_representatives"]
        assert not old["selection_uses_validation"] and not new["selection_uses_validation"]
        for filename in ("trials.csv", "vectors.csv", "history.csv"):
            pd.testing.assert_frame_equal(pd.read_csv(before_output / "ex1_Example/pilots" / f"seed_{old['seed']}" / filename,
                float_precision="round_trip"), pd.read_csv(after_output / "ex1_Example/pilots" / f"seed_{new['seed']}" / filename,
                float_precision="round_trip"), check_exact=True)
    assert before["policy_selected"] is after["policy_selected"] is None


def test_validation_forcing_failure_preserves_calibration_and_finished_search(tmp_path):
    root = tmp_path / "repository"
    path, manifest = collection_fixture(root)
    entry = manifest["reservoirs"][0]
    frame = csv(root / entry["observations_file"])
    frame.loc[frame.index >= "2004-01-01", "inflow_m3s"] = -1
    pin_frame(root, path, manifest, entry, frame)
    result = run(path, tmp_path / "review", policies=("maximum_storage",),
                 include_matlab_reference=False, pilot_policy="maximum_storage", optimization_budget=2, seeds=(42,))
    record = result["reservoirs"][0]
    comparison = policies(record)["maximum_storage"]
    assert comparison["calibration_status"] == "completed" and comparison["validation_status"] == "blocked"
    assert record["calibration_pareto_policies"] == ["maximum_storage"]
    pilot = record["pilots"][0]
    assert pilot["status"] == "completed" and pilot["evaluations"] == 2
    assert pilot["validation"] and all(not point["feasible"] for point in pilot["validation"])
    assert not result["selection_uses_validation"] and result["policy_selected"] is None


def test_zero_feasible_search_preserves_every_trial_and_vector(tmp_path, monkeypatch):
    path, _ = collection_fixture(tmp_path / "repository")
    def infeasible(objective, vector):
        return CandidateEvaluation(False, (np.inf, np.inf), {}, {"release": 36, "storage": 36},
            "Controlled infeasible candidate", None, None, objective.calibration_period)
    def validation_must_not_run(*args, **kwargs):
        pytest.fail("A zero-feasible front must not evaluate held-out representatives.")
    monkeypatch.setattr(module.RootPolicyCalibrationObjective, "evaluate", infeasible)
    monkeypatch.setattr(module.RootPolicyCalibrationObjective, "evaluate_validation", validation_must_not_run)
    output = tmp_path / "review"
    result = run(path, output, policies=("minimum_storage",), include_matlab_reference=False,
                 pilot_policy="minimum_storage", optimization_budget=2, seeds=(42, 43))
    assert result["search_evaluations"] == 4
    for pilot in result["reservoirs"][0]["pilots"]:
        assert pilot["status"] == "failed" and pilot["evaluations"] == 2 and pilot["feasible_evaluations"] == 0
        folder = output / "ex1_Example/pilots" / f"seed_{pilot['seed']}"
        trials = pd.read_csv(folder / "trials.csv")
        vectors = pd.read_csv(folder / "vectors.csv")
        assert len(trials) == len(vectors) == 2 and vectors.shape[1] == 72
        assert not trials.feasible.any() and trials.reason.eq("Controlled infeasible candidate").all()
        assert np.isposinf(trials.release_cost).all() and np.isposinf(trials.storage_cost).all()


@pytest.mark.parametrize("damage", ["csv_hash", "null_hash_metadata"])
def test_invalid_member_does_not_prevent_independent_review(tmp_path, damage):
    root = tmp_path / "repository"
    path, manifest = collection_fixture(root, count=2)
    entry = manifest["reservoirs"][0]
    if damage == "csv_hash":
        target = root / entry["observations_file"]
        target.write_text(target.read_text() + "tamper\n")
    else:
        entry["files_sha256"] = None
        _json(path, manifest)
    output = tmp_path / "review"
    result = run(path, output, policies=("reject",), include_matlab_reference=False)
    assert result["reservoir_counts"] == {"input_error": 1, "review_completed": 1}
    assert not (output / "ex1_Example").exists() and (output / "ex2_Example/review.json").is_file()


def test_output_protection_and_nonempty_output_rejection(tmp_path):
    root = tmp_path / "repository"
    path, _ = collection_fixture(root, count=2)
    with pytest.raises(ValueError, match="protected source inputs"):
        module.run_root_policy_review(path, root / "data/reservoirs/ex2_Example", reservoirs="Example 1")
    output = tmp_path / "review"
    run(path, output, policies=("reject",), include_matlab_reference=False)
    existing = (output / "root_policy_review.json").read_bytes()
    with pytest.raises(ValueError, match="new or empty output"):
        run(path, output)
    assert (output / "root_policy_review.json").read_bytes() == existing


@pytest.mark.parametrize("mutation", ["manifest", "input", "code"])
def test_end_guards_invalidate_concurrent_changes_without_editing_production(tmp_path, monkeypatch, mutation):
    root = tmp_path / "repository"
    path, manifest = collection_fixture(root)
    if mutation == "code":
        original = module._code_hashes
        calls = [0]
        def changed_hashes():
            calls[0] += 1
            value = original()
            return value if calls[0] == 1 else {**value, "controlled_source_change": "changed"}
        monkeypatch.setattr(module, "_code_hashes", changed_hashes)
    else:
        original = module._baseline
        def changed_baseline(*args, **kwargs):
            value = original(*args, **kwargs)
            target = path if mutation == "manifest" else root / manifest["reservoirs"][0]["observations_file"]
            target.write_text(target.read_text() + " ")
            return value
        monkeypatch.setattr(module, "_baseline", changed_baseline)
    result = run(path, tmp_path / "review", policies=("reject",), include_matlab_reference=False)
    assert result["status"] == "invalid_integrity"
    field = {"manifest": "manifest_file_unchanged", "input": "input_files_unchanged", "code": "code_files_unchanged"}[mutation]
    assert not result[field]
    if mutation == "input":
        assert result["reservoirs"][0]["status"] == "input_error"


@pytest.mark.parametrize("kwargs", [
    {"optimization_budget": 2}, {"pilot_policy": "minimum_storage"},
    {"pilot_policy": "minimum_storage", "optimization_budget": 1},
    {"pilot_policy": "minimum_storage", "optimization_budget": True},
    {"policies": ("reject",), "pilot_policy": "minimum_storage", "optimization_budget": 2},
    {"seeds": (42, 42)}, {"seeds": (True,)}, {"seeds": ()}, {"policies": ("unknown",)},
])
def test_invalid_or_incomplete_search_settings_do_not_create_outputs(tmp_path, kwargs):
    path, _ = collection_fixture(tmp_path / "repository")
    output = tmp_path / "review"
    with pytest.raises((ValueError, TypeError)):
        run(path, output, **kwargs)
    assert not output.exists()
