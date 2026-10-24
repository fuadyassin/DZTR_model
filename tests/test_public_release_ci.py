"""Public release boundaries; self-contained and usable on GitHub runners."""
import pytest
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile
from tools.run_public_ci_v1 import audit_data, validate_metadata, ROOT
from tools.build_researcher_bundle_v1 import selected_files
from tools import build_researcher_bundle_v2 as bundle
from dztr._release import __version__
from dztr.cli import main


def test_public_data_audit_accepts_only_declared_reservoirs():
    result = audit_data(['data/reservoirs/ex14_Gardiner/observations_si.csv', 'src/dztr/cli.py'])
    assert result['tracked_paths_checked'] == 2


@pytest.mark.parametrize('path', ['data/reservoirs/private_case/observations_si.csv',
    'data/private_download/station.csv', 'data/reservoirs/ex14_Gardiner/raw_stations.csv',
    'outputs/experiment/simulation.csv', '.cache/credentials', 'presentations/deck.pptx'])
def test_public_data_audit_rejects_unapproved_records_and_local_artifacts(path):
    with pytest.raises(ValueError):
        audit_data([path])


def test_release_citation_version_and_license_agree():
    assert validate_metadata(ROOT)['version'] == __version__


def test_stable_command_reports_the_distribution_release(capsys):
    with pytest.raises(SystemExit) as result:
        main(['--version'])
    assert result.value.code == 0
    assert 'DZTR ' + __version__ in capsys.readouterr().out


@pytest.mark.parametrize('path', [
    '.ipynb_checkpoints/README-checkpoint.md',
    'dztr_param/.ipynb_checkpoints/csv_creator-checkpoint.py',
    'dztr_param/__pycache__/columns.cpython-310.pyc',
    'src/dztr/old_module.pyc',
    r'dztr_param\__pycache__\columns.cpython-310.pyc',
    'tests/.pytest_cache/v/cache/nodeids',
])
def test_tracked_generated_files_are_rejected_even_when_already_in_git(path):
    with pytest.raises(ValueError, match='Generated development file'):
        audit_data([path])


def test_bundle_selection_excludes_retained_local_checkpoints_and_agent_instructions(tmp_path):
    files = ['dztr_param/module.py', 'AGENTS.md', 'docs/AGENTS.md',
        'dztr_param/.ipynb_checkpoints/module-checkpoint.py',
        'docs/.ipynb_checkpoints/guide-checkpoint.md',
        'examples/portable/example/__pycache__/module.pyc']
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('local file', encoding='utf-8')
    selected = {path.relative_to(tmp_path).as_posix() for path in selected_files(tmp_path)}
    assert selected == {'dztr_param/module.py'}
    assert all((tmp_path / name).is_file() for name in files)


def test_versioned_zip_filters_generated_files_from_extra_directories(tmp_path, monkeypatch):
    files = ['pyproject.toml', 'README.md', 'LICENSE', 'CITATION.cff',
        'THIRD_PARTY_NOTICES.md', 'mkdocs.yml', 'AGENTS.md',
        'src/dztr/example.py', 'tests/test_public_release_ci.py',
        'ci/.ipynb_checkpoints/requirements-checkpoint.txt',
        '.github/workflows/.ipynb_checkpoints/workflow-checkpoint.yml',
        'docs/javascripts/__pycache__/module.pyc']
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture', encoding='utf-8')
    config = tmp_path / 'configs/release/v0.3.0.json'
    config.parent.mkdir(parents=True)
    config.write_text('{"software_doi": "10.5281/zenodo.23606593"}', encoding='utf-8')
    monkeypatch.setattr(bundle, 'ROOT', tmp_path)
    monkeypatch.setattr(bundle.subprocess, 'run', lambda command, **kwargs:
        SimpleNamespace(stdout='fixture-commit\n' if command[1] == 'rev-parse' else ''))
    archive = bundle.build(tmp_path / 'outputs/bundle')
    with ZipFile(archive) as source:
        names = source.namelist()
        assert 'DZTR_researcher_application_v1/src/dztr/example.py' in names
        assert all(not any(part in Path(name).parts for part in
            ('AGENTS.md', '.ipynb_checkpoints', '__pycache__')) for name in names)
    assert all((tmp_path / name).is_file() for name in files)
