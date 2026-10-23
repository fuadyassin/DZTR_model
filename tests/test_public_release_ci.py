"""Public release boundaries; self-contained and usable on GitHub runners."""
import pytest
from tools.run_public_ci_v1 import audit_data, validate_metadata, ROOT
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
