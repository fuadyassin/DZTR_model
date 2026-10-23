"""Install and exercise the copied researcher bundle without private archives.

Uses local installed dependencies through a system-site-packages environment;
this proves source isolation, not offline dependency distribution.
"""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import os
import shutil
import subprocess
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dztr.portable import file_digest, read_json, write_json


def verify(bundle, output):
    bundle, out = Path(bundle).resolve(strict=True), Path(output).resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Use a fresh verification directory inside repository outputs')
    out.mkdir(parents=True)
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    results = []
    def run(name, command, cwd):
        completed = subprocess.run([str(v) for v in command], cwd=cwd, env=env,
                                   capture_output=True, text=True)
        (out / (name + '.log')).write_text(completed.stdout + completed.stderr, encoding='utf-8')
        results.append({'name': name, 'exit_code': completed.returncode})
        write_json(out / 'progress.json', {'stages': results})
        if completed.returncode:
            raise ValueError(name + ' failed; inspect ' + str(out / (name + '.log')))
        return completed.stdout
    # A short source root avoids Windows MAX_PATH in nested prepared profiles.
    # Never delete/reuse a computed directory; require a fresh checked target.
    source = ROOT / 'outputs' / ('p_' + sha256(str(out).encode()).hexdigest()[:8])
    source = source.resolve()
    if source.exists() or not source.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Short verification source directory must be fresh and inside project outputs')
    source.mkdir()
    with ZipFile(bundle) as archive:
        for name in archive.namelist():
            parts = Path(name).parts
            if not parts or parts[0] != 'DZTR_researcher_application_v1':
                raise ValueError('Unknown archive root')
            target = source.joinpath(*parts[1:]).resolve()
            if not target.is_relative_to(source):
                raise ValueError('Bundle path escapes extraction directory')
            if name.endswith('/'):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as content, target.open('wb') as destination:
                    shutil.copyfileobj(content, destination)
    manifest = read_json(source / 'BUNDLE_MANIFEST.json')
    for name, expected in manifest['artifact_sha256'].items():
        if file_digest(source / name) != expected:
            raise ValueError('Bundle file changed: ' + name)
    assert not (source / 'outputs').exists() and not (source / '.git').exists()
    run('build_wheel', [sys.executable, '-B', '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                       '--wheel-dir', out / 'wheels', '.'], source)
    environment = out / 'environment'
    run('create_environment', [sys.executable, '-B', '-m', 'venv', '--system-site-packages', environment], out)
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    wheels = list((out / 'wheels').glob('*.whl')); assert len(wheels) == 1
    run('install_wheel', [python, '-B', '-m', 'pip', 'install', '--no-deps', '--force-reinstall', wheels[0]], out)
    isolated = out / 'isolated_application'; isolated.mkdir()
    origin = run('import_origin', [python, '-I', '-B', '-c',
        'import dztr,json; from pathlib import Path; '
        'print(json.dumps({"module":str(Path(dztr.__file__).resolve())}))'], isolated)
    installed = Path(json.loads(origin)['module'])
    assert installed.is_relative_to(environment) and not installed.is_relative_to(source)
    command = [python, '-I', '-B', '-m', 'dztr']
    run('module_help', [*command, '--help'], isolated)
    console = environment / ('Scripts/dztr.exe' if os.name == 'nt' else 'bin/dztr')
    run('console_help', [console, '--help'], isolated)
    import pandas as pd
    import numpy as np
    applications = []
    for name in ('gardiner', 'oahe', 'garrison'):
        local = isolated / name
        shutil.copytree(source / 'examples/portable' / name, local)
        profile = local / 'profile.json'; forcing = local / 'forcing_si.csv'; result = local / 'result'
        run(name + '_inspect', [*command, 'inspect', '--profile', profile], isolated)
        simulate = [*command, 'simulate', '--profile', profile, '--forcing', forcing,
                    '--forcing-stage', 'prepared', '--output', result, '--plot']
        run(name + '_apply', simulate, isolated)
        expected = read_json(local / 'expected_checks.json')
        canonical = sha256((result / 'simulation.csv').read_text(encoding='utf-8').encode()).hexdigest()
        assert canonical == expected['historical_state_canonical_lf_sha256']
        prior_mtime = (result / 'simulation.csv').stat().st_mtime_ns
        run(name + '_resume', [*simulate, '--resume'], isolated)
        assert (result / 'simulation.csv').stat().st_mtime_ns == prior_mtime
        restart = local / 'restart'
        run(name + '_restart', [*command, 'simulate', '--profile', profile,
            '--forcing', local / 'forcing_restart_si.csv', '--forcing-stage', 'prepared',
            '--initial-state', local / 'initial_state_restart.json', '--output', restart], isolated)
        full = pd.read_csv(result / 'simulation.csv', index_col='date', float_precision='round_trip')
        subset = pd.read_csv(restart / 'simulation.csv', index_col='date', float_precision='round_trip')
        np.testing.assert_array_equal(full.loc[subset.index, ['storage_m3', 'release_m3s']],
                                      subset[['storage_m3', 'release_m3s']])
        applications.append({'name': name, 'historical_rows': len(full), 'restart_rows': len(subset),
                             'exact_historical_replay': True, 'exact_restart': True, 'resume_reused': True})
    # Plans and a fresh preparation/baseline run are executed only in the copied source tree.
    for name in ('gardiner_workflow', 'gardiner_calibration', 'gardiner_scalar_calibration'):
        run(name + '_plan', [*command, 'plan', '--config', source / 'configs/portable' / (name + '.json'),
                             '--project-root', source], source)
    workflow_output = source / 'outputs/public_parameterization'
    workflow = [*command, 'run', '--config', source / 'configs/portable/gardiner_workflow.json',
                '--project-root', source, '--output', workflow_output]
    run('public_workflow', workflow, source)
    run('public_workflow_resume', [*workflow, '--resume'], source)
    run('public_workflow_status', [*command, 'status', '--output', workflow_output], source)
    # Read dependency versions from the tested interpreter, not from the parent source.
    versions = run('dependency_versions', [python, '-I', '-B', '-c',
        'import importlib.metadata as m,json,platform; '
        'print(json.dumps({"python":platform.python_version(),**{p:m.version(p) for p in '
        '["dztr_param","numpy","pandas","requests","openpyxl","matplotlib","pytest"]}},indent=2))'], isolated)
    write_json(out / 'environment_versions.json', json.loads(versions))
    result = {'kind': 'dztr_clean_bundle_verification_v1', 'bundle_sha256': file_digest(bundle),
        'installed_module': str(installed), 'wheel_sha256': file_digest(wheels[0]),
        'short_source_workspace': str(source),
        'no_original_repository_imports_for_application': True, 'private_archives_present': False,
        'system_site_dependencies_reused': True, 'applications': applications,
        'public_preparation_baseline_run_and_resume': True, 'optional_calibration_configs_planned_only': True,
        'new_optimizer_evaluations': 0, 'stages': results}
    write_json(out / 'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    verify(args.bundle, args.output)
