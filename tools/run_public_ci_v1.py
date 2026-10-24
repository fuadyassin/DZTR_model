"""Public-only installation, parameterization and frozen-application checks."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from dztr._release import __version__
from dztr.portable import file_digest, read_json, write_json
from tools.build_researcher_bundle_v2 import build
from tools.verify_researcher_bundle_v1 import verify
from tools.release_hygiene_v1 import is_generated_file


def audit_data(paths):
    """Refuse tracked observations outside the explicit public reservoir set."""
    allowed = {'ex14_Gardiner', 'ex12_Trinity', 'ex6_Oahe', 'ex5_Garrison'}
    names = {'observations_si.csv', 'observation_provenance.json', 'matlab_source.json',
             'legacy_targets_si.csv', 'legacy_bounds_si.csv'}
    other_data = {'data/README.md', 'data/reservoirs/README.md',
        'data/catalogs/grand_v1_1_reference.csv', 'data/catalogs/grand_v1_1_reference.provenance.json',
        'data/public_examples/README.md', 'data/public_examples/manifest.json',
        'data/gardiner_levels_v1_20261017/README.md',
        'data/geometry_profiles_v1_20261017/253/curve_si.csv',
        'data/geometry_profiles_v1_20261017/253/source_regeom.csv'}
    other_data.update('data/gardiner_levels_v1_20261017/acquisition/' + name for name in
        ('levels_si.csv', 'observations_si.csv', 'station_components_si.csv', 'provenance.json'))
    for name in paths:
        path = Path(name)
        if is_generated_file(name):
            raise ValueError('Generated development file entered the tracked release: ' + name)
        if path.parts and path.parts[0] == 'data':
            reservoir_file = (len(path.parts) == 4 and path.parts[1] == 'reservoirs'
                              and path.parts[2] in allowed and path.parts[3] in names)
            if not reservoir_file and path.as_posix() not in other_data:
                raise ValueError('Unapproved data file: ' + name)
        if path.parts and path.parts[0] in ('outputs', '.cache', 'presentations'):
            raise ValueError('Local artifacts entered the tracked release: ' + name)
    return {'approved_observation_reservoirs': sorted(allowed), 'tracked_paths_checked': len(paths)}


def validate_metadata(root):
    import yaml
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    metadata = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    citation = yaml.safe_load((root / 'CITATION.cff').read_text(encoding='utf-8'))
    if metadata['version'] != __version__ or citation['version'] != __version__ or citation['license'] != 'MIT':
        raise ValueError('Distribution version, citation and license disagree')
    if citation.get('doi') == '10.5194/hess-23-3735-2019':
        raise ValueError('The paper DOI cannot identify the software release')
    if not any(row.get('doi') == '10.5194/hess-23-3735-2019' for row in citation['references']):
        raise ValueError('Missing separate model-paper citation')
    if 'MIT License' not in (root / 'LICENSE').read_text(encoding='utf-8'):
        raise ValueError('Missing MIT license')
    return {'version': __version__, 'license': 'MIT', 'software_doi': citation.get('doi'),
            'model_paper_doi': '10.5194/hess-23-3735-2019'}


def run(output):
    out = Path(output).resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Use a fresh CI directory inside repository outputs')
    out.mkdir(parents=True)
    paths = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True, check=True).stdout.decode().split('\0')
    audit = audit_data([name for name in paths if name])
    metadata = validate_metadata(ROOT)
    write_json(out / 'input_audit.json', {'data': audit, 'metadata': metadata})
    temporary = ROOT / 'outputs' / ('t_' + out.name)
    if temporary.exists():
        raise ValueError('Pytest base must be fresh')
    command = [sys.executable, '-B', '-m', 'pytest', '-q', 'tests/test_portable_application.py',
        'tests/test_public_release_ci.py', '--basetemp', str(temporary), '--junitxml', str(out / 'public_pytest.xml')]
    tested = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (out / 'public_pytest.log').write_text(tested.stdout + tested.stderr, encoding='utf-8')
    if tested.returncode:
        raise RuntimeError('Public tests failed: ' + str(out / 'public_pytest.log'))
    archive = build(out / 'bundle')
    verify(archive, out / 'portability')
    verified = read_json(out / 'portability/verification.json')
    versions = read_json(out / 'portability/environment_versions.json')
    if versions['dztr_param'] != __version__:
        raise ValueError('The installed wheel has the wrong release version')
    short = Path(verified['short_source_workspace'])
    campaign = short / 'outputs/public_parameterization/stages/targets/campaign'
    summary = read_json(campaign / 'report/summary.json')
    if any(summary['methods'][method]['completed_baselines'] != 1 for method in ('observed_monthly', 'flow_only')):
        raise ValueError('A public fixed baseline failed')
    import pandas as pd
    inventory = pd.read_csv(campaign / 'reservoir_inventory.csv')
    fraction = float(inventory.calibration_fraction.iloc[0])
    if fraction < .65:
        raise ValueError('Public calibration coverage is below 65%')
    # Only this curated directory is uploaded by Actions.
    artifacts = ROOT / 'outputs/ci_artifacts'
    if artifacts.exists():
        raise ValueError('CI artifact directory must be fresh')
    artifacts.mkdir()
    for path in [archive, out / 'public_pytest.xml', out / 'public_pytest.log', out / 'input_audit.json',
                 out / 'portability/verification.json', out / 'portability/environment_versions.json']:
        shutil.copyfile(path, artifacts / path.name)
    for path in (out / 'portability/wheels').glob('*.whl'):
        shutil.copyfile(path, artifacts / path.name)
    for path in (out / 'portability').glob('*.log'):
        shutil.copyfile(path, artifacts / path.name)
    result = {'status': 'passed', 'platform': sys.platform, 'version': __version__,
        'github_commit': os.environ.get('GITHUB_SHA'), 'calibration_fraction': fraction,
        'public_fixed_baselines_completed': 2, 'private_archives_required': False,
        'new_calibration_searches': 0, 'installed_wheel_verified': True,
        'historical_and_restart_examples_verified': 3, 'metadata': metadata,
        'scientific_qualifications': {'source_certified': False, 'convergence_established': False,
                                     'independent_predictive_skill_established': False}}
    write_json(artifacts / 'release_verification.json', result)
    write_json(artifacts / 'SHA256SUMS.json', {p.name: file_digest(p) for p in sorted(artifacts.iterdir()) if p.is_file()})
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a', encoding='utf-8') as stream:
            stream.write(f'## DZTR {__version__}: {sys.platform}\n\nPublic installation, three frozen profiles, '
                f'restarts, resume and two parameterization baselines passed. Calibration support: {fraction:.2%}. '
                'No private archives or new calibration searches. Source certification, forecast skill and '
                'optimizer convergence remain unproven.\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
