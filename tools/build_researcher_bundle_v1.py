"""Build an explicit allowlist source/application ZIP without private observations."""
from pathlib import Path
import argparse
import json
import shutil
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dztr.portable import file_digest, write_json

PUBLIC_RESERVOIRS = ('ex14_Gardiner', 'ex12_Trinity', 'ex6_Oahe', 'ex5_Garrison')


def selected_files(root):
    paths = [root / name for name in ('pyproject.toml', 'README.md', 'AGENTS.md', '.gitignore', '.gitattributes')]
    for folder in ('src/dztr', 'dztr_param', 'workflows', 'strategies', 'tools'):
        paths.extend((root / folder).rglob('*.py'))
    paths.append(root / 'tools/Set-ProjectCache.ps1')
    paths.extend((root / 'docs').rglob('*.md'))
    paths.extend((root / 'docs/assets/researcher_v1').glob('*'))
    paths.extend((root / 'examples').glob('*.py'))
    paths.extend((root / 'examples/portable').rglob('*'))
    paths.extend((root / 'configs/portable').glob('*.json'))
    paths.append(root / 'configs/run_manager/gardiner_model_v1.json')
    paths.append(root / 'tests/test_portable_application.py')
    for name in ('README.md', 'manifest.json'):
        paths.append(root / 'data/public_examples' / name)
    for rid in PUBLIC_RESERVOIRS:
        paths.extend((root / 'data/reservoirs' / rid).glob('*'))
        paths.append(root / 'configs/reservoirs' / (rid + '.json'))
    for name in ('README.md', 'acquisition/levels_si.csv', 'acquisition/observations_si.csv',
                 'acquisition/station_components_si.csv', 'acquisition/provenance.json'):
        paths.append(root / 'data/gardiner_levels_v1_20261017' / name)
    for name in ('curve_si.csv', 'source_regeom.csv'):
        paths.append(root / 'data/geometry_profiles_v1_20261017/253' / name)
    for name in ('LICENSE', 'LICENSE.md', 'CITATION.cff'):
        if (root / name).is_file():
            paths.append(root / name)
    # Public examples are intentionally small and explicit. No wildcard data tree.
    data_names = {'observations_si.csv', 'observation_provenance.json', 'matlab_source.json',
                  'legacy_targets_si.csv', 'legacy_bounds_si.csv'}
    result = []
    for path in sorted(set(paths)):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        rel = path.relative_to(root)
        if rel.parts[0] == 'data' and 'reservoirs' in rel.parts and path.name not in data_names:
            raise ValueError('Unreviewed observation file: ' + str(rel))
        if path.suffix == '.pyc' or any(part in ('outputs', '.git', '.cache', 'presentations') for part in rel.parts):
            raise ValueError('Private/generated source selected: ' + str(rel))
        result.append(path)
    return result


def build(output):
    out = Path(output).resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Use a fresh bundle directory inside repository outputs')
    source = out / 'DZTR_researcher_application_v1'; source.mkdir(parents=True)
    for path in selected_files(ROOT):
        destination = source / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    manifest = {'kind': 'dztr_researcher_application_candidate_v1', 'published': False, 'software_doi': None,
        'license_decision_pending': not any((source / name).exists() for name in ('LICENSE', 'LICENSE.md')),
        'public_observation_reservoirs': list(PUBLIC_RESERVOIRS),
        'private_runtime_archives_included': False,
        'artifact_sha256': {p.relative_to(source).as_posix(): file_digest(p)
            for p in sorted(source.rglob('*')) if p.is_file()}}
    write_json(source / 'BUNDLE_MANIFEST.json', manifest)
    archive = out / 'DZTR_researcher_application_v1.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED) as zipped:
        for path in sorted(source.rglob('*')):
            if path.is_file():
                zipped.write(path, path.relative_to(out).as_posix())
    write_json(out / 'build_receipt.json', {'source_files': len(manifest['artifact_sha256']),
        'zip_sha256': file_digest(archive), 'zip_bytes': archive.stat().st_size,
        'source_directory': source.name, 'published': False, 'software_doi': None})
    print(json.dumps({'output': str(out), 'source_files': len(manifest['artifact_sha256']),
                      'zip_bytes': archive.stat().st_size}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    build(parser.parse_args().output)
