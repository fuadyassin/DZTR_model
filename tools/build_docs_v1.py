"""Build public documentation while retaining private archive links as labels."""
from pathlib import Path
import argparse
import json
import re
import shutil
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dztr.portable import file_digest, write_json

LINK = re.compile(r'(!?)\[([^\]]+)\]\(([^\n)]+)\)')


def render_links(text, source):
    """Keep available docs/media; send public code links to their repository."""
    def link(match):
        image, label, raw = match.groups()
        target = raw.strip().strip('<>')
        if target.startswith(('#', 'https://', 'http://', 'mailto:')):
            return match.group(0)
        clean = unquote(target.split('#', 1)[0])
        if not clean:
            return match.group(0)
        absolute = (source.parent / clean.replace('\\', '/')).resolve()
        if 'outputs' in absolute.parts or re.match(r'^[A-Za-z]:', target):
            return label + ' (local study archive)'
        if absolute.is_relative_to(ROOT / 'docs') and absolute.is_file():
            return match.group(0)
        if absolute.is_relative_to(ROOT) and absolute.is_file():
            relative = absolute.relative_to(ROOT).as_posix()
            # Publishing a link does not publish ignored private inputs.
            if relative.startswith('data/') and not relative.startswith(('data/public_examples/', 'data/reservoirs/README')):
                return label + ' (local input reference)'
            return f'[{label}](https://github.com/fuadyassin/DZTR_model/blob/main/{relative})'
        return label + ' (archive reference)'
    # Commands/examples retain their literal syntax.
    pieces = re.split(r'(```[\s\S]*?```)', text)
    return ''.join(piece if piece.startswith('```') else LINK.sub(link, piece) for piece in pieces)


def build(output):
    from mkdocs.config import load_config
    from mkdocs.commands.build import build as compile_site
    out = Path(output).resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Use a fresh documentation site inside project outputs')
    source = ROOT / '.cache' / ('docs_' + out.name)
    if source.exists():
        raise ValueError('Documentation preparation must be fresh')
    source.mkdir(parents=True)
    for path in sorted((ROOT / 'docs').glob('*.md')):
        (source / path.name).write_text(render_links(path.read_text(encoding='utf-8'), path), encoding='utf-8')
    for name in ('assets/researcher_v1', 'javascripts', 'stylesheets'):
        shutil.copytree(ROOT / 'docs' / name, source / name)
    config = load_config(config_file=str(ROOT / 'mkdocs.yml'), docs_dir=str(source), site_dir=str(out), strict=True)
    compile_site(config)
    if not (out / 'index.html').exists() or not (out / 'scientific_assessment_v1/index.html').exists():
        raise ValueError('Missing core public documentation page')
    receipt = {'kind': 'dztr_public_documentation_v1', 'site': str(out), 'strict_build': True,
        'private_archive_files_copied': False, 'source_sha256': {p.relative_to(ROOT).as_posix(): file_digest(p)
            for p in sorted((ROOT / 'docs').rglob('*')) if p.is_file()}}
    write_json(out.parent / (out.name + '_receipt.json'), receipt)
    print(json.dumps({'site': str(out), 'strict_build': True}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    build(parser.parse_args().output)
