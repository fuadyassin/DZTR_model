"""Fleet control refuses drift/undeclared jobs and isolates process failures."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("fleet_control", Path(__file__).parents[1] / "tools/fleet_workflow.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def put(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_undeclared_job_cannot_create_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "checked", lambda path: {})
    put(tmp_path / "job_plan.json", {"jobs": [{"reservoir_id": "ready", "seed": 42}]})
    with pytest.raises(ValueError, match="outside the reviewed plan"):
        module.run_job(tmp_path, "blocked", 42)
    assert not (tmp_path / "pilots").exists()


def test_existing_job_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "checked", lambda path: {})
    put(tmp_path / "job_plan.json", {"jobs": [{"reservoir_id": "ready", "seed": 42}]})
    saved = tmp_path / "pilots/ready/seed_42"
    saved.mkdir(parents=True)
    checkpoint = saved / "checkpoint.npz"
    checkpoint.write_bytes(b"saved")
    with pytest.raises(FileExistsError):
        module.run_job(tmp_path, "ready", 42)
    assert checkpoint.read_bytes() == b"saved"


def test_process_failure_does_not_cancel_other_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "checked", lambda path: {})
    jobs = [{"reservoir_id": name, "seed": 42} for name in ("first", "fails", "last")]
    put(tmp_path / "job_plan.json", {"jobs": jobs})
    def fake_run(command, **kwargs):
        kwargs["stdout"].write("test process log\n")
        return SimpleNamespace(returncode=1 if "fails" in command else 0)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.run_all(tmp_path)
    summary = module.read(tmp_path / "execution_summary.json")
    assert summary["status"] == "completed_with_failures"
    assert len(summary["jobs"]) == 3
    assert sorted(r["exit_code"] for r in summary["jobs"]) == [0, 0, 1]
    assert len(module.read(tmp_path / "job_status.json")) == 3


def test_numerical_drift_refused_before_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "code_hashes", lambda: {"model": "changed"})
    put(tmp_path / "protocol.json", {"code_sha256": {"model": "original"}, "fleet_driver_sha256": module.sha(Path(module.__file__))})
    with pytest.raises(ValueError, match="numerical code or driver changed"):
        module.checked(tmp_path)


def test_pipeline_reports_successful_jobs_after_isolated_failure(tmp_path, monkeypatch):
    pipeline_spec = importlib.util.spec_from_file_location("fleet_pipeline_test", Path(__file__).parents[1] / "tools/fleet_pipeline.py")
    pipeline = importlib.util.module_from_spec(pipeline_spec)
    pipeline_spec.loader.exec_module(pipeline)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline.sys, "argv", ["fleet_pipeline", "--output", str(tmp_path), "--run-authorized"])
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            put(tmp_path / "execution_summary.json", {"status": "completed_with_failures"})
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    with pytest.raises(SystemExit, match="isolated failures"):
        pipeline.main()
    assert len(calls) == 2
    assert calls[1][2].endswith("fleet_reports.py")
