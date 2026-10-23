"""Stable public entry point for frozen application and versioned workflows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from dztr.portable import load_profile, read_json, simulate_csv
from dztr._release import __version__


def project_root(explicit=None):
    candidates = [Path(explicit)] if explicit else [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        root = candidate.resolve()
        if (root / 'tools/dztr_run_v10.py').is_file() and (root / 'pyproject.toml').is_file():
            return root
    raise ValueError('Workflow commands require the source/application bundle; use --project-root PATH. '
                     'Installed-wheel frozen-profile simulation works without a checkout.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version='DZTR ' + __version__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('plan', 'run'):
        command = sub.add_parser(name, help='Delegate to the original versioned workflow manager')
        command.add_argument('--config', required=True, type=Path)
        command.add_argument('--project-root', type=Path)
        if name == 'run':
            command.add_argument('--output', required=True, type=Path)
            command.add_argument('--resume', action='store_true')
            command.add_argument('--until')
    command = sub.add_parser('status', help='Read a saved manager/application status without running a model')
    command.add_argument('--output', required=True, type=Path)
    command = sub.add_parser('inspect', help='Verify a standalone frozen profile')
    command.add_argument('--profile', required=True, type=Path)
    command = sub.add_parser('export-profile', help='Export a verified native registry profile without observations')
    command.add_argument('--registry', required=True, type=Path)
    select = command.add_mutually_exclusive_group(required=True)
    select.add_argument('--reservoir-id'); select.add_argument('--grand-id', type=int)
    command.add_argument('--preprocessing-json', required=True, type=Path)
    command.add_argument('--method', required=True)
    command.add_argument('--output', required=True, type=Path)
    command.add_argument('--project-root', type=Path)
    command = sub.add_parser('simulate', help='Apply a standalone profile to inflow-only forcing')
    command.add_argument('--profile', required=True, type=Path)
    command.add_argument('--forcing', required=True, type=Path)
    command.add_argument('--forcing-stage', required=True, choices=('canonical', 'prepared'))
    command.add_argument('--initial-state', type=Path)
    command.add_argument('--output', required=True, type=Path)
    command.add_argument('--resume', action='store_true')
    command.add_argument('--plot', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'export-profile':
            root = project_root(args.project_root)
            command = [sys.executable, '-B', str(root / 'tools/export_portable_profile_v1.py'),
                '--registry', str(args.registry.resolve()), '--output', str(args.output.resolve()),
                '--preprocessing-json', str(args.preprocessing_json.resolve()), '--method', args.method]
            command += ['--reservoir-id', args.reservoir_id] if args.reservoir_id else ['--grand-id', str(args.grand_id)]
            return subprocess.call(command, cwd=root)
        if args.command in ('plan', 'run'):
            root = project_root(args.project_root)
            command = [sys.executable, '-B', str(root / 'tools/dztr_run_v10.py'), args.command,
                       '--config', str(args.config.resolve())]
            if args.command == 'run':
                command += ['--output', str(args.output.resolve())]
                if args.resume:
                    command.append('--resume')
                if args.until:
                    command += ['--until', args.until]
            return subprocess.call(command, cwd=root)
        if args.command == 'status':
            target = args.output / 'status.json'
            if not target.exists():
                target = args.output / 'simulation_receipt.json'
            result = read_json(target)
        elif args.command == 'inspect':
            profile, _ = load_profile(args.profile)
            result = {k: profile[k] for k in ('reservoir_id', 'GRAND_ID', 'method', 'sampling', 'numerics', 'provenance')}
        else:
            initial = read_json(args.initial_state) if args.initial_state else None
            result = simulate_csv(args.profile, args.forcing, args.output, forcing_stage=args.forcing_stage,
                                  initial_state=initial, resume=args.resume, plot=args.plot)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError) as error:
        print('DZTR: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
