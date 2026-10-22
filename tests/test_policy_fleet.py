"""Policy-aware orchestration, initialization and checkpoint identity guards."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import policy_fleet as controller
import policy_fleet_job as runner
from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.checkpoint import load_optimizer_checkpoint, save_optimizer_checkpoint
from dztr.calibration.optimizer import NSGA2Settings, optimize_nsga2
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters


def fixture_parameters():
    targets = pd.DataFrame(np.tile([20., 40., 70., 1., 2., 3.], (12, 1)), index=pd.Index(range(1, 13), name="month"), columns=MONTHLY_TARGET_COLUMNS)
    return ReservoirParameters(targets, 100, 50, 2, 20, .1)


def test_failed_generalized_initialization_can_use_declared_witness():
    template = fixture_parameters()
    calls = []
    def evaluate(vector):
        calls.append(vector)
        return SimpleNamespace(feasible=len(calls) == 2, reason="no root" if len(calls) == 1 else None)
    bounds = SimpleNamespace(validate=lambda vector, template: None)
    vector, result, ledger = runner.initialize(SimpleNamespace(evaluate=evaluate), [("generalized", template.monthly_targets), ("critical_gap_witness", template.monthly_targets)], bounds, template)
    assert result.feasible and len(calls) == 2
    assert [r["feasible"] for r in ledger] == [False, True]
    np.testing.assert_array_equal(vector, parameters_to_vector(template))


def test_invalid_targets_do_not_call_model_or_modify_estimates():
    template = fixture_parameters()
    invalid = template.monthly_targets.copy()
    invalid.iloc[0, 0] = 0
    before = invalid.copy()
    def forbidden(vector):
        pytest.fail("Invalid physical candidate reached model")
    vector, result, ledger = runner.initialize(SimpleNamespace(evaluate=forbidden), [("generalized", invalid)], SimpleNamespace(validate=forbidden), template)
    assert vector is None and result is None and not ledger[0]["model_called"]
    pd.testing.assert_frame_equal(invalid, before)


def test_validation_support_failure_is_reported_separately():
    def reject(vector):
        raise ValueError("Signed held-out forcing needs interpretation")
    result, report = runner.safe_validation(SimpleNamespace(evaluate_validation=reject), np.zeros(72))
    assert result is None and not report["feasible"] and "Signed" in report["reason"]


def test_policy_changed_checkpoint_is_rejected(tmp_path):
    # Algebraic optimizer fixture: no hydrological model or reservoir experiment.
    baseline = parameters_to_vector(fixture_parameters())
    settings = NSGA2Settings(seed=42, population_size=4, evaluation_budget=8)
    identity = {"root_policy": "nearest_previous_storage", "seed": 42}
    path = tmp_path / "checkpoint.npz"
    objective = SimpleNamespace(bounds=SimpleNamespace(lower_vector=baseline - .1, upper_vector=baseline + .1), evaluate=lambda vector: SimpleNamespace(feasible=True, objectives=(np.sum(vector ** 2), np.sum((vector - 1) ** 2)), reason=None))
    result = optimize_nsga2(objective, baseline, settings, checkpoint_callback=lambda state: save_optimizer_checkpoint(path, state, identity))
    state = load_optimizer_checkpoint(path, identity)
    resumed = optimize_nsga2(objective, baseline, settings, resume_state=state)
    np.testing.assert_array_equal([p.vector for p in result.evaluated], [p.vector for p in resumed.evaluated])
    with pytest.raises(ValueError):
        load_optimizer_checkpoint(path, {**identity, "root_policy": "reject"})


def test_identity_records_policy_and_initialization_protocol():
    config = {"sampling": {"frequency": "daily"}, "periods": {"calibration": {"start": "2000-01-01", "end": "2001-01-01"}}}
    job = {"reservoir_id": "sample", "seed": 42, "population_size": 32, "evaluation_budget": 512, "root_policy": "nearest_previous_storage"}
    identity = runner.identity_for({"code_sha256": {}}, job, config, {})
    assert identity["root_policy"] == job["root_policy"]
    assert "critical_gap_witness" in identity["initialization_protocol"]


def test_existing_incomplete_job_is_not_modified(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "checked", lambda path: {})
    runner.write(tmp_path / "job_plan.json", {"jobs": [{"reservoir_id": "sample", "seed": 42}]})
    folder = tmp_path / "pilots/sample/seed_42"
    folder.mkdir(parents=True)
    (folder / "checkpoint.npz").write_bytes(b"retained")
    with pytest.raises(FileExistsError):
        controller.run_job(tmp_path, "sample", 42)
    assert list(p.name for p in folder.iterdir()) == ["checkpoint.npz"]


def test_undeclared_reservoir_cannot_start(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "checked", lambda path: {})
    runner.write(tmp_path / "job_plan.json", {"jobs": []})
    with pytest.raises(ValueError, match="outside"):
        controller.run_job(tmp_path, "blocked", 42)
    assert not (tmp_path / "pilots").exists()


def test_failed_process_does_not_cancel_other_jobs_or_reporting(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "checked", lambda path: {})
    jobs = [{"reservoir_id": name, "seed": 42} for name in ("good", "fails")]
    runner.write(tmp_path / "job_plan.json", {"jobs": jobs})
    reports = []
    def execute(command, **kwargs):
        if "--completed-fleet" in command:
            reports.append(command)
            return SimpleNamespace(returncode=0)
        kwargs["stdout"].write("synthetic job log\n")
        return SimpleNamespace(returncode=1 if "fails" in command else 0)
    monkeypatch.setattr(controller.subprocess, "run", execute)
    with pytest.raises(SystemExit, match="successful jobs"):
        controller.run_all(tmp_path)
    summary = runner.read(tmp_path / "execution_summary.json")
    assert len(summary["jobs"]) == 2 and len(reports) == 1
    assert {r["status"] for r in summary["jobs"]} == {"completed", "failed"}


def test_job_keeps_calibration_output_when_held_out_forcing_fails(tmp_path, monkeypatch):
    # Real parameter/split/bounds construction, with explicit synthetic model
    # and search stand-ins. This performs no reservoir simulation experiment.
    import fleet_admission
    from dztr.calibration.sampled import _observations
    monkeypatch.setattr(controller, "checked", lambda path: {"code_sha256": {}})
    dates = pd.date_range("2000-01-01", "2003-12-01", freq="MS", name="date")
    observed = _observations(pd.DataFrame({"storage_m3": 40. + (dates.year - 2000) * 2 + dates.month * .1, "release_m3s": 1. + (dates.year - 2000) * .1, "inflow_m3s": 2.}, index=dates))
    model = {k: getattr(fixture_parameters(), k) for k in ("capacity_m3", "initial_storage_m3", "initial_release_m3s", "channel_capacity_m3s", "dead_storage_fraction", "inflow_correction", "flood_mode")}
    config = {"reservoir": {"id": "sample", "name": "Synthetic"}, "model": model, "sampling": {"frequency": "monthly", "fixed_step_days": 30.5}, "periods": {"simulation_start": "2000-01-01", "calibration": {"start": "2001-01-01", "end": "2002-12-01"}, "validation": {"start": "2003-01-01", "end": "2003-12-01"}, "warmup_rows": 12}, "quantile_method": "hazen", "target_percentiles": [10, 45, 85], "inflow_preprocessing": {"factor": 1, "offset_m3s": 0}, "atol_storage_m3": 1e-4}
    audit = fleet_admission.bounds_audit(observed.frame, config)
    runner.write(tmp_path / "admission.json", {"reservoirs": [{"reservoir_id": "sample", "bounds_audit": audit}]})
    monkeypatch.setattr(runner, "load_collection", lambda path: (runner.ROOT, {"reservoirs": [{"reservoir_id": "sample"}]}, None, None))
    monkeypatch.setattr(runner, "_verified_inputs", lambda root, entry: (None, config, observed, {}))
    folder = tmp_path / "pilots/sample/seed_42"
    class Objective:
        def __init__(self, *args, **kwargs):
            assert kwargs["policy"] == "nearest_previous_storage"
        def evaluate(self, vector):
            frame = observed.frame.loc[:"2002-12-01"].copy()
            frame["balance_residual_m3"] = 0.
            return SimpleNamespace(feasible=True, reason=None, objectives=(-.5, -.6), metrics={"release_nse": .5, "storage_nse": .6}, simulation=SimpleNamespace(frame=frame))
        def evaluate_validation(self, vector):
            assert runner.read(folder / "selection.json")["written_before_validation"]
            raise ValueError("Unavailable held-out forcing")
    monkeypatch.setattr(runner, "RootPolicyCalibrationObjective", Objective)
    def search(objective, vector, settings, **kwargs):
        point = SimpleNamespace(evaluation_id=0, generation=0, feasible=True, objectives=(-.5, -.6), reason=None, vector=vector)
        return SimpleNamespace(evaluated=[point], front=[point], history=[])
    monkeypatch.setattr(runner, "optimize_nsga2", search)
    job = {"reservoir_id": "sample", "seed": 42, "root_policy": "nearest_previous_storage", "population_size": 32, "evaluation_budget": 512}
    result = runner.run(tmp_path, job)
    assert result["status"] == "completed" and not result["baseline_validation"]["feasible"]
    assert all(not r["validation_feasible"] for r in result["representatives"])
    assert (folder / "fleet_completion.json").exists()
    saved = pd.read_csv(folder / "selected_0_simulation.csv")
    assert "validation" not in set(saved.period)
    assert runner.read(folder / "checkpoint_identity.json")["root_policy"] == job["root_policy"]
