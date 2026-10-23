"""Seal a versioned, allowlisted release with citation and CI metadata."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from dztr._release import __version__
from dztr.portable import file_digest, read_json, write_json
from tools.build_researcher_bundle_v1 import selected_files, PUBLIC_RESERVOIRS


def build(output):
    out = Path(output).resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Use a fresh release directory inside project outputs')
    source = out / 'DZTR_researcher_application_v1'; source.mkdir(parents=True)
    paths = selected_files(ROOT)
    paths += [ROOT / name for name in ('LICENSE', 'CITATION.cff', 'THIRD_PARTY_NOTICES.md', 'mkdocs.yml')]
    for folder in ('ci', '.github/workflows', 'docs/javascripts', 'docs/stylesheets', 'configs/release'):
        paths += [p for p in (ROOT / folder).rglob('*') if p.is_file()]
    paths += [ROOT / 'tests/test_public_release_ci.py']
    for path in sorted(set(paths)):
        if not path.is_file():
            raise ValueError('Missing release input: ' + str(path.relative_to(ROOT)))
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip())
    write_json(source / 'SOURCE_IDENTITY.json', {'commit': commit, 'working_tree_dirty': dirty,
        'version': __version__, 'source_date': '2026-10-23'})
    manifest = {'kind': 'dztr_researcher_release_v2', 'version': __version__, 'license': 'MIT',
        'software_doi': read_json(ROOT / 'configs/release/v0.3.0.json')['software_doi'],
        'source_commit': commit, 'working_tree_dirty': dirty,
        'public_observation_reservoirs': list(PUBLIC_RESERVOIRS), 'private_runtime_archives_included': False,
        'artifact_sha256': {p.relative_to(source).as_posix(): file_digest(p)
            for p in sorted(source.rglob('*')) if p.is_file()}}
    write_json(source / 'BUNDLE_MANIFEST.json', manifest)
    archive = out / f'DZTR-v{__version__}-source.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED) as zipped:
        for path in sorted(source.rglob('*')):
            if path.is_file():
                zipped.write(path, path.relative_to(out).as_posix())
    write_json(out / 'build_receipt.json', {'version': __version__, 'source_files': len(manifest['artifact_sha256']),
        'zip_sha256': file_digest(archive), 'zip_bytes': archive.stat().st_size,
        'source_commit': commit, 'working_tree_dirty': dirty})
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    print(build(parser.parse_args().output))
