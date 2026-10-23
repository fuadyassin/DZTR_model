"""Self-contained frozen profiles using the established DZTR solvers unchanged.

Application requires inflow and an initial state, never response observations.
Checksums establish reproducibility, not physical source certification.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import csv
import json
from numbers import Real
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from dztr.schemas import DataQuality, MONTHLY_TARGET_COLUMNS, ReservoirObservations, ReservoirParameters
from dztr.simulation.root_policy import POLICIES, simulate_root_policy_reservoir
from dztr.simulation.sampled import SamplingSettings, validate_sampling_index
from dztr.simulation.signed_net import SignedNetPolicy, simulate_signed_net_reservoir

KIND = 'dztr_portable_profile_v1'
PHYSICAL = {'capacity_m3', 'initial_storage_m3', 'initial_release_m3s', 'channel_capacity_m3s',
            'dead_storage_fraction', 'inflow_correction', 'flood_mode'}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON field: ' + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=pairs, parse_constant=constant)


def source_hashes():
    """Canonical LF text hashes survive Windows/Unix checkout line endings."""
    root = Path(__file__).resolve().parent
    files = [root / 'schemas.py', root / 'portable.py']
    files += list((root / 'model').glob('*.py')) + list((root / 'simulation').glob('*.py'))
    return {p.relative_to(root).as_posix(): sha256(p.read_text(encoding='utf-8').encode()).hexdigest()
            for p in sorted(files)}


def _finite(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or positive and value <= 0:
        raise ValueError(label + ' must be finite' + (' and positive' if positive else ''))


def seal_profile(payload):
    payload = dict(payload)
    if 'payload_sha256' in payload:
        raise ValueError('Seal a payload without a previous checksum')
    return {**payload, 'payload_sha256': digest(payload)}


def load_profile(path):
    profile = read_json(path)
    required = {'kind', 'schema_version', 'reservoir_id', 'GRAND_ID', 'method', 'physical_constants',
                'sampling', 'numerics', 'inflow_preprocessing', 'initial_date', 'monthly_targets',
                'provenance', 'source_sha256', 'payload_sha256'}
    if set(profile) != required or (profile['kind'], profile['schema_version']) != (KIND, 1):
        raise ValueError('Unknown portable profile schema or fields')
    body = {k: v for k, v in profile.items() if k != 'payload_sha256'}
    if digest(body) != profile['payload_sha256']:
        raise ValueError('Portable profile checksum changed')
    if profile['source_sha256'] != source_hashes():
        raise ValueError('Portable numerical/application source changed; export a new version')
    if not isinstance(profile['reservoir_id'], str) or not profile['reservoir_id'].strip():
        raise ValueError('Declare a reservoir ID')
    gid = profile['GRAND_ID']
    if gid is not None and (type(gid) is not int or gid <= 0):
        raise ValueError('GRAND_ID must be a verified positive integer or null')
    provenance = profile['provenance']
    if (not isinstance(provenance, dict) or provenance.get('selection_uses_assessment') is not False
            or provenance.get('source_certified') is not False or provenance.get('source_gates')):
        raise ValueError('Declare calibration-only selection, uncertified sources and no unresolved admission gate')
    physical = profile['physical_constants']
    if set(physical) != PHYSICAL:
        raise ValueError('Declare all physical constants without unknown fields')
    for key in PHYSICAL - {'flood_mode'}:
        _finite(physical[key], key)
    rows = profile['monthly_targets']
    if not isinstance(rows, list) or any(set(row) != {'month', *MONTHLY_TARGET_COLUMNS} for row in rows):
        raise ValueError('Monthly rows require month and exactly six SI target fields')
    if any(type(row['month']) is not int for row in rows):
        raise ValueError('Months must be integers')
    for row in rows:
        for key in MONTHLY_TARGET_COLUMNS:
            _finite(row[key], key)
    targets = pd.DataFrame(rows).set_index('month')
    parameters = ReservoirParameters(targets, **physical)
    SamplingSettings(**profile['sampling'])
    date = pd.Timestamp(profile['initial_date'])
    if pd.isna(date) or date.tz is not None or date != date.normalize():
        raise ValueError('Declare the original initial date at midnight without a timezone')
    numerical = profile['numerics']
    if set(numerical) != {'route', 'root_policy', 'atol_storage_m3', 'forcing_policy'}:
        raise ValueError('Declare the complete frozen numerical policy')
    if numerical['root_policy'] not in POLICIES:
        raise ValueError('Unknown root policy')
    _finite(numerical['atol_storage_m3'], 'Storage tolerance', positive=True)
    if numerical['route'] not in ('standard_native_v1', 'signed_net_native_v1'):
        raise ValueError('Portable endpoint application does not support this numerical route')
    if numerical['route'] == 'signed_net_native_v1':
        SignedNetPolicy(**numerical['forcing_policy'])
    elif numerical['forcing_policy'] is not None or profile['sampling'].get('allow_signed_inflow', False):
        raise ValueError('Standard forcing cannot declare a signed policy')
    preprocessing = profile['inflow_preprocessing']
    if set(preprocessing) != {'factor', 'offset_m3s'}:
        raise ValueError('Declare factor and offset_m3s separately from physical inflow correction')
    _finite(preprocessing['factor'], 'Preprocessing factor', positive=True)
    _finite(preprocessing['offset_m3s'], 'Preprocessing offset')
    return profile, parameters


def read_forcing(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    if len(header) != 2 or set(header) != {'date', 'inflow_m3s'}:
        raise ValueError('Application CSV must contain exactly date,inflow_m3s; keep responses separate')
    frame = pd.read_csv(path, float_precision='round_trip')
    index = pd.DatetimeIndex(pd.to_datetime(frame.pop('date'), errors='raise'), name='date')
    frame.index = index
    frame['inflow_m3s'] = pd.to_numeric(frame.inflow_m3s, errors='raise')
    if not np.isfinite(frame.inflow_m3s.to_numpy()).all():
        raise ValueError('Forcing must be finite; missing values are not filled')
    return frame


def run_profile(profile_path, forcing, *, forcing_stage, initial_state=None):
    """Run a fixed profile on endpoint-clock forcing without fitting or resets."""
    profile, parameters = load_profile(profile_path)
    if list(forcing.columns) != ['inflow_m3s']:
        raise ValueError('Pass an inflow-only frame')
    sampling = SamplingSettings(**profile['sampling'])
    validate_sampling_index(forcing.index, sampling.frequency)
    if forcing_stage not in ('canonical', 'prepared'):
        raise ValueError('Declare canonical or prepared forcing explicitly')
    if initial_state is None:
        if forcing.index[0] != pd.Timestamp(profile['initial_date']):
            raise ValueError('A different forcing start requires explicit storage_m3 and release_m3s')
    else:
        if not isinstance(initial_state, dict) or set(initial_state) != {'storage_m3', 'release_m3s'}:
            raise ValueError('Initial state requires exactly storage_m3 and release_m3s')
        for key, value in initial_state.items():
            _finite(value, key)
        parameters = replace(parameters, initial_storage_m3=initial_state['storage_m3'],
                             initial_release_m3s=initial_state['release_m3s'])
    if sampling.month_clock == 'legacy' and forcing.index[0] != pd.Timestamp(profile['initial_date']):
        raise ValueError('A changed legacy target clock requires a separate profile')
    frame = forcing.copy(deep=True).astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Forcing must be finite')
    prep = profile['inflow_preprocessing']
    if forcing_stage == 'canonical':
        frame['inflow_m3s'] = frame.inflow_m3s * prep['factor'] + prep['offset_m3s']
    for name in ('storage_m3', 'release_m3s'):
        frame[name] = np.nan
    observations = ReservoirObservations(frame, DataQuality('available', len(frame),
        {'inflow_m3s': 0, 'storage_m3': len(frame), 'release_m3s': len(frame)}, 0, ()))
    numerical = profile['numerics']
    options = {'sampling': sampling, 'policy': numerical['root_policy'],
               'atol_storage_m3': numerical['atol_storage_m3']}
    if numerical['route'] == 'signed_net_native_v1':
        result = simulate_signed_net_reservoir(observations, parameters,
            forcing_policy=SignedNetPolicy(**numerical['forcing_policy']), **options)
    else:
        result = simulate_root_policy_reservoir(observations, parameters, **options)
    output = result.frame
    s, q, inflow, seconds = (output[c].to_numpy() for c in
        ('storage_m3', 'release_m3s', 'inflow_m3s', 'step_seconds'))
    residual = np.diff(s) - seconds[1:] / 2 * (
        parameters.inflow_correction * (inflow[1:] + inflow[:-1]) - q[1:] - q[:-1])
    maximum = float(np.abs(residual).max())
    if (maximum > numerical['atol_storage_m3']
            or not np.allclose(residual, output.balance_residual_m3.to_numpy()[1:], rtol=0, atol=1e-9)
            or not output.storage_m3.between(0, parameters.capacity_m3).all()
            or (output.release_m3s < 0).any()):
        raise ValueError('Independent balance or physical-state check failed')
    return result, {'maximum_independent_balance_residual_m3': maximum,
        'multiple_root_steps': int(output.root_count.gt(1).sum()),
        'root_policy_applied_steps': int(output.root_policy_applied.sum()),
        'overflow_steps': int(output.overflow_volume_m3.gt(0).sum())}


def plot_application(frame, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(frame.index, frame.storage_m3 / 1e6)
    axes[0].set_ylabel('Storage (million m3)')
    axes[1].plot(frame.index, frame.inflow_m3s, label='Input forcing', alpha=.65)
    axes[1].plot(frame.index, frame.release_m3s, label='Total release')
    axes[1].set_ylabel('Flow (m3/s)'); axes[1].legend()
    axes[2].plot(frame.index[1:], frame.balance_residual_m3.iloc[1:])
    axes[2].set_ylabel('Balance residual (m3)')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Frozen-profile application; no calibration or response assimilation')
    fig.savefig(Path(output) / 'simulation.png', dpi=150)
    plt.close(fig)


def simulate_csv(profile_path, forcing_path, output, *, forcing_stage, initial_state=None,
                 resume=False, plot=False):
    """Persist a fixed application; resume verifies and reuses its sealed output."""
    profile, _ = load_profile(profile_path)
    out = Path(output).resolve()
    if out == Path(profile_path).resolve().parent or out == Path(forcing_path).resolve().parent:
        raise ValueError('Keep application outputs separate from inputs')
    identity = {'profile_sha256': file_digest(profile_path), 'forcing_sha256': file_digest(forcing_path),
                'forcing_stage': forcing_stage, 'initial_state': initial_state,
                'source_sha256': source_hashes(), 'plot': plot}
    if out.exists():
        if not resume:
            raise ValueError('Use a fresh output directory or --resume')
        manifest = read_json(out / 'manifest.json')
        if manifest['identity'] != identity:
            raise ValueError('Application inputs or decisions changed; use a new output directory')
        for name, expected in manifest['artifact_sha256'].items():
            path = (out / name).resolve()
            if not path.is_relative_to(out) or file_digest(path) != expected:
                raise ValueError('Completed application artifact changed: ' + name)
        return read_json(out / 'simulation_receipt.json')
    forcing = read_forcing(forcing_path)
    result, checks = run_profile(profile_path, forcing, forcing_stage=forcing_stage, initial_state=initial_state)
    out.mkdir(parents=True)
    result.frame.to_csv(out / 'simulation.csv', index_label='date', float_format='%.17g')
    result.parameters.monthly_targets.to_csv(out / 'monthly_targets_si.csv', float_format='%.17g')
    write_json(out / 'profile.json', profile)
    receipt = {'kind': 'dztr_portable_application_v1', 'status': 'passed',
        'reservoir_id': profile['reservoir_id'], 'GRAND_ID': profile['GRAND_ID'], 'method': profile['method'],
        'rows': len(result.frame), 'first_date': str(result.frame.index[0].date()),
        'last_date': str(result.frame.index[-1].date()), 'sampling': profile['sampling'],
        'numerics': profile['numerics'], 'forcing_stage': forcing_stage,
        'source_transform_applied': forcing_stage == 'canonical',
        'physical_inflow_correction': result.parameters.inflow_correction,
        'initial_state_override': initial_state, 'responses_read': False,
        'new_optimizer_evaluations': 0, 'source_certified': False,
        'independent_predictive_skill_established': False,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
        **checks}
    if plot:
        plot_application(result.frame, out)
    write_json(out / 'simulation_receipt.json', receipt)
    write_json(out / 'manifest.json', {'identity': identity, 'artifact_sha256':
        {p.name: file_digest(p) for p in sorted(out.iterdir()) if p.is_file()}})
    return receipt
