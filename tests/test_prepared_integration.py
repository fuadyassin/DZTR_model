"""Local integration barriers, immutable plans and interrupted-write recovery.

Publication verification is replaced with a controlled saved-artifact fixture;
the separately tested population verifier owns scientific archive validation.
These tests exercise real files and the installer's atomic replacement helper.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


HELPER_PATH = Path(__file__).resolve().parents[1] / "examples/integrate_prepared_simulation.py"
BALANCE = "src/dztr/model/balance.py"
RUNNER = "src/dztr/simulation/runner.py"
RECEIPT = "outputs/simulation_performance/integration_receipt.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def files_under(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def installation(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("prepared_integration_test_helper", HELPER_PATH)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    source, target = tmp_path / "staged", tmp_path / "main"
    source.mkdir()
    target.mkdir()
    old_code = {
        "model/balance.py": b"original public balance implementation\n",
        "simulation/runner.py": b"original uncached simulation\n",
        "model/release.py": b"unchanged release law\n",
    }
    new_code = dict(old_code)
    new_code["model/balance.py"] = b"prepared public balance and arithmetic helpers\n"
    new_code["simulation/runner.py"] = b"prepared local monthly simulation cache\n"
    for name in old_code:
        write(source / "src/dztr" / name, new_code[name])
        write(target / "src/dztr" / name, old_code[name])
    additions = (
        "tests/test_prepared_simulation.py",
        "tests/test_prepared_integration.py",
        "tests/reference_cases/uncached_simulation/release.py",
        "tests/reference_cases/uncached_simulation/balance.py",
        "tests/reference_cases/uncached_simulation/runner.py",
        "tests/reference_cases/uncached_simulation/provenance.json",
        "examples/benchmark_prepared_simulation.py",
        "docs/simulation_performance.md",
    )
    entries = []
    for target_name in (BALANCE, RUNNER, *additions):
        staged = source / target_name
        if not staged.exists():
            write(staged, f"Reviewed additive fixture: {target_name}\n".encode("utf-8"))
        existing = target / target_name
        entries.append({"source": target_name, "target": target_name,
                        "sha256": digest(staged.read_bytes()),
                        "original_sha256": digest(existing.read_bytes()) if existing.exists() else None})
    benchmark = source / "outputs/prepared_simulation_benchmark"
    write(benchmark / "summary.json", json_bytes({"status": "complete", "candidate_pairs_verified": 16}))
    write(benchmark / "benchmark_manifest.json", json_bytes({
        "schema_version": 1, "artifact_sha256": {"summary.json": digest((benchmark / "summary.json").read_bytes())},
    }))
    folder = target / "outputs/trinity_population_study"
    write(folder / "population_identity.json", json_bytes({"experiment": "controlled completed population study"}))
    write(folder / "summary.json", json_bytes({"failed_job_count": 0}))
    write(folder / "study_manifest.json", json_bytes({"schema_version": 1, "status": "complete"}))
    manifest_digest = digest((folder / "study_manifest.json").read_bytes())
    write(folder / "population_results.md", f"Verified publication. Study manifest SHA256: `{manifest_digest}`.\n".encode("utf-8"))
    write(folder / "population_comparison.png", b"\x89PNG\r\n\x1a\ncontrolled published figure bytes")
    archive = target / "outputs/historical_seed_42"
    for name, data in {
        "optimizer_checkpoint.npz": b"historical immutable checkpoint bytes",
        "run_identity.json": json_bytes({"search_code_sha256": {name: digest(data) for name, data in old_code.items()}}),
        "adaptive_manifest.json": b"historical adaptive manifest bytes",
        "evaluated_vectors.csv": b"evaluation_id,parameter\n0,1.25\n",
    }.items():
        write(archive / name, data)
    archived_bytes = files_under(archive)
    settings = {"population_sizes": [16, 32, 64], "seeds": [42, 43, 44],
                "evaluation_checkpoints": [512, 1024, 2048]}
    verified = {"settings": settings, "jobs": [{"population_size": population, "seed": seed,
        "search_evaluations": 2048, "source_proof": {"directory": str(archive),
        "identity": {"search_code_sha256": {name: digest(data) for name, data in old_code.items()}}}}
        for population in settings["population_sizes"] for seed in settings["seeds"]]}
    verification_calls = []

    def verify(requested):
        assert Path(requested) == folder
        verification_calls.append(Path(requested))
        if files_under(archive) != archived_bytes:
            raise ValueError("Historical archive changed.")
        return json.loads(json.dumps(verified))

    monkeypatch.setattr(helper, "verify_population_study", verify)
    plan = {
        "schema_version": 1, "workspace_root": str(tmp_path), "source_root": str(source),
        "target_root": str(target), "helper_sha256": digest(HELPER_PATH.read_bytes()),
        "files": entries, "original_code_sha256": {name: digest(data) for name, data in old_code.items()},
        "prepared_code_sha256": {name: digest(data) for name, data in new_code.items()},
        "proof_file_sha256": {path.relative_to(source).as_posix(): digest(path.read_bytes())
                              for path in benchmark.iterdir()},
        "study_directory": folder.relative_to(target).as_posix(),
        "study_identity_sha256": digest((folder / "population_identity.json").read_bytes()),
        "study_settings": settings, "receipt": RECEIPT,
    }
    plan_path = source / "prepared_plan.json"
    write(plan_path, json_bytes(plan))
    return SimpleNamespace(helper=helper, source=source, target=target, folder=folder,
        archive=archive, plan=plan, plan_path=plan_path, plan_sha=digest(plan_path.read_bytes()),
        receipt=target / RECEIPT, verification_calls=verification_calls,
        archived_bytes=archived_bytes, verified=verified)


def repin(installation):
    write(installation.plan_path, json_bytes(installation.plan))
    installation.plan_sha = digest(installation.plan_path.read_bytes())


def invoke(installation, **kwargs):
    return installation.helper.integrate(installation.plan_path, installation.plan_sha, **kwargs)


def inject_runner_failure(installation, monkeypatch):
    original_replace = installation.helper._replace_checkpoint

    def fail_runner(temporary, destination):
        if Path(destination) == installation.target / RUNNER:
            raise PermissionError("Injected replacement failure after the balance module.")
        return original_replace(temporary, destination)

    monkeypatch.setattr(installation.helper, "_replace_checkpoint", fail_runner)
    return original_replace


@pytest.mark.parametrize("missing", ["study_manifest.json", "population_results.md", "population_comparison.png"])
def test_pending_or_unpublished_study_never_changes_source_files(installation, missing):
    (installation.folder / missing).unlink()
    before = files_under(installation.target)
    with pytest.raises(RuntimeError, match="still running"):
        invoke(installation)
    assert files_under(installation.target) == before
    assert installation.verification_calls == []


def test_failed_study_is_not_integrated(installation):
    write(installation.folder / "summary.json", json_bytes({"failed_job_count": 1}))
    before = files_under(installation.target)
    with pytest.raises(RuntimeError, match="jobs failed"):
        invoke(installation)
    assert files_under(installation.target) == before
    assert installation.verification_calls == []


def test_wait_timeout_keeps_all_install_targets_unchanged(installation, monkeypatch):
    (installation.folder / "population_results.md").unlink()
    moments = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(installation.helper.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(installation.helper.time, "sleep", lambda _: pytest.fail("No sleep is needed after deadline."))
    before = files_under(installation.target)
    with pytest.raises(TimeoutError, match="Timed out"):
        invoke(installation, wait=True, timeout_hours=1 / 3600)
    assert files_under(installation.target) == before


@pytest.mark.parametrize("kind", ["original", "unchanged_source", "prepared", "addition", "occupied_addition", "proof"])
def test_source_user_or_proof_edits_reject_before_any_installation(installation, kind):
    paths = {
        "original": installation.target / BALANCE,
        "unchanged_source": installation.target / "src/dztr/model/release.py",
        "prepared": installation.source / RUNNER,
        "addition": installation.source / "docs/simulation_performance.md",
        "occupied_addition": installation.target / "docs/simulation_performance.md",
        "proof": installation.source / "outputs/prepared_simulation_benchmark/summary.json",
    }
    write(paths[kind], b"Intervening user edit must survive.\n")
    before = files_under(installation.target)
    with pytest.raises(ValueError, match="changed"):
        invoke(installation)
    assert files_under(installation.target) == before
    assert installation.verification_calls == []


@pytest.mark.parametrize("change", ["plan_bytes", "helper_pin"])
def test_plan_and_helper_hashes_are_immutable(installation, change):
    if change == "plan_bytes":
        installation.plan_path.write_bytes(installation.plan_path.read_bytes() + b" ")
    else:
        installation.plan["helper_sha256"] = "0" * 64
        repin(installation)
    before = files_under(installation.target)
    with pytest.raises(ValueError, match="changed"):
        invoke(installation)
    assert files_under(installation.target) == before


@pytest.mark.parametrize("relative", ["../outside.py", "/outside.py", "C:/outside.py",
                                     "src\\dztr\\model\\release.py", "./src/dztr/model/balance.py"])
def test_noncanonical_or_escaping_source_paths_are_rejected(installation, relative):
    installation.plan["files"][0]["source"] = relative
    repin(installation)
    before = files_under(installation.target)
    with pytest.raises(ValueError, match="Unsafe|escapes"):
        invoke(installation)
    assert files_under(installation.target) == before


@pytest.mark.parametrize("target", ["src/dztr/model/release.py", "src\\dztr\\model\\release.py",
                                   "configs/user_settings.json", "outputs/historical_seed_42/optimizer_checkpoint.npz"])
def test_third_production_module_or_unreviewed_target_is_rejected(installation, target):
    installation.plan["files"].append({"source": "docs/simulation_performance.md", "target": target,
        "sha256": digest((installation.source / "docs/simulation_performance.md").read_bytes()),
        "original_sha256": None})
    repin(installation)
    before = files_under(installation.target)
    with pytest.raises(ValueError, match="two reviewed|allowlist|Unsafe"):
        invoke(installation)
    assert files_under(installation.target) == before


@pytest.mark.parametrize("change", ["overlap", "outside_workspace", "reversed_production", "receipt_collision"])
def test_root_dependency_order_and_receipt_guards(installation, change):
    if change == "overlap":
        installation.plan["source_root"] = str(installation.target / "nested_source")
    elif change == "outside_workspace":
        installation.plan["target_root"] = str(installation.target.parent.parent / "outside_workspace")
    elif change == "reversed_production":
        installation.plan["files"][:2] = reversed(installation.plan["files"][:2])
    else:
        installation.plan["receipt"] = "outputs/historical_seed_42/run_identity.json"
    repin(installation)
    before = files_under(installation.target)
    with pytest.raises(ValueError):
        invoke(installation)
    assert files_under(installation.target) == before


def test_note_must_name_the_current_verified_manifest(installation):
    write(installation.folder / "population_results.md", f"Wrong manifest: `{'0' * 64}`.\n".encode("utf-8"))
    before = files_under(installation.target)
    with pytest.raises(ValueError, match="note does not identify"):
        invoke(installation)
    assert files_under(installation.target) == before


def test_complete_publication_installs_exact_allowlist_and_preserves_archived_bytes(installation):
    before = files_under(installation.target)
    source_before = files_under(installation.source)
    receipt = invoke(installation)
    after = files_under(installation.target)
    expected_changes = {entry["target"] for entry in installation.plan["files"]} | {RECEIPT}
    assert {name for name in set(before) | set(after) if before.get(name) != after.get(name)} == expected_changes
    for entry in installation.plan["files"]:
        assert after[entry["target"]] == source_before[entry["source"]]
    assert files_under(installation.source) == source_before
    assert files_under(installation.archive) == installation.archived_bytes
    assert receipt["status"] == "complete"
    assert receipt["historical_checkpoint_identities_rewritten"] is False
    assert receipt["installed_files"] == [entry["target"] for entry in installation.plan["files"]]
    assert json.loads(installation.receipt.read_text(encoding="utf-8")) == receipt
    assert not installation.receipt.with_suffix(".lock").exists()
    assert not list(installation.target.rglob(".prepared_install_*.tmp"))


def test_partial_atomic_failure_has_a_durable_receipt_and_exact_recovery(installation, monkeypatch):
    before = files_under(installation.target)
    original_replace = inject_runner_failure(installation, monkeypatch)
    with pytest.raises(PermissionError, match="Injected replacement"):
        invoke(installation)
    failed = json.loads(installation.receipt.read_text(encoding="utf-8"))
    assert failed["status"] == "partial_failure"
    assert failed["installed_files"] == [BALANCE]
    assert failed["pending_file"] == RUNNER
    assert "Injected replacement" in failed["error"]
    assert (installation.target / BALANCE).read_bytes() == (installation.source / BALANCE).read_bytes()
    assert (installation.target / RUNNER).read_bytes() == before[RUNNER]
    assert files_under(installation.archive) == installation.archived_bytes
    assert not installation.receipt.with_suffix(".lock").exists()
    assert not list(installation.target.rglob(".prepared_install_*.tmp"))
    balance_stat = (installation.target / BALANCE).stat()
    monkeypatch.setattr(installation.helper, "_replace_checkpoint", original_replace)
    recovered = invoke(installation)
    assert recovered["status"] == "complete"
    assert recovered["installed_files"] == [entry["target"] for entry in installation.plan["files"]]
    assert (installation.target / BALANCE).stat().st_mtime_ns == balance_stat.st_mtime_ns
    assert files_under(installation.archive) == installation.archived_bytes


@pytest.mark.parametrize("edited_target", [BALANCE, RUNNER])
def test_user_edits_after_partial_installation_are_never_overwritten(installation, monkeypatch, edited_target):
    original_replace = inject_runner_failure(installation, monkeypatch)
    with pytest.raises(PermissionError):
        invoke(installation)
    monkeypatch.setattr(installation.helper, "_replace_checkpoint", original_replace)
    write(installation.target / edited_target, b"User changes after interrupted installation.\n")
    before = files_under(installation.target)
    with pytest.raises(ValueError, match="Target source changed"):
        invoke(installation)
    assert files_under(installation.target) == before


def test_changed_publication_cannot_resume_an_interrupted_installation(installation, monkeypatch):
    original_replace = inject_runner_failure(installation, monkeypatch)
    with pytest.raises(PermissionError):
        invoke(installation)
    monkeypatch.setattr(installation.helper, "_replace_checkpoint", original_replace)
    note = installation.folder / "population_results.md"
    note.write_bytes(note.read_bytes() + b"An edited results interpretation.\n")
    before = files_under(installation.target)
    with pytest.raises(ValueError, match="publication changed"):
        invoke(installation)
    assert files_under(installation.target) == before


def test_user_edit_after_pending_file_journal_is_caught_before_replacement(installation, monkeypatch):
    original_journal = installation.helper._journal
    intervening_bytes = b"User edit arriving immediately before runner replacement.\n"
    changed = False

    def edit_after_journal(path, state):
        nonlocal changed
        original_journal(path, state)
        if state.get("pending_file") == RUNNER and state["status"] == "installing" and not changed:
            changed = True
            write(installation.target / RUNNER, intervening_bytes)

    monkeypatch.setattr(installation.helper, "_journal", edit_after_journal)
    with pytest.raises(ValueError, match="changed immediately before installation"):
        invoke(installation)
    assert changed
    assert (installation.target / RUNNER).read_bytes() == intervening_bytes
    assert (installation.target / BALANCE).read_bytes() == (installation.source / BALANCE).read_bytes()
    state = json.loads(installation.receipt.read_text(encoding="utf-8"))
    assert state["status"] == "partial_failure"
    assert state["installed_files"] == [BALANCE]
    assert state["pending_file"] == RUNNER
    assert files_under(installation.archive) == installation.archived_bytes


def test_single_installer_lock_does_not_disturb_another_attempt(installation):
    lock = installation.receipt.with_suffix(".lock")
    write(lock, b"Another installer owns this lock.\n")
    before = files_under(installation.target)
    with pytest.raises(RuntimeError, match="lock already exists"):
        invoke(installation)
    assert files_under(installation.target) == before
    assert lock.read_bytes() == b"Another installer owns this lock.\n"


@pytest.mark.parametrize("options", [{"wait": 1}, {"timeout_hours": True}, {"timeout_hours": 0},
                                    {"timeout_hours": float("nan")}, {"timeout_hours": float("inf")},
                                    {"timeout_hours": 169}])
def test_invalid_wait_or_timeout_is_rejected_without_filesystem_mutation(installation, options):
    before = files_under(installation.target)
    with pytest.raises((ValueError, TypeError)):
        invoke(installation, **options)
    assert files_under(installation.target) == before
