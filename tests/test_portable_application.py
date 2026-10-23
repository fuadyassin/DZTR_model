"""Scientific application and reproducibility boundaries of the public gateway."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dztr import cli, portable
from dztr.schemas import MONTHLY_TARGET_COLUMNS
from dztr.simulation.sampled import SamplingSettings
from dztr.simulation.signed_net import SignedNetPolicy


@pytest.fixture
def profile(tmp_path):
    targets = pd.DataFrame(np.tile([2e8, 5e8, 8e8, 1., 2., 3.], (12, 1)),
                          columns=MONTHLY_TARGET_COLUMNS, index=pd.Index(range(1, 13), name='month'))
    payload = {'kind': portable.KIND, 'schema_version': 1, 'reservoir_id': 'synthetic', 'GRAND_ID': None,
        'method': 'observed_monthly', 'initial_date': '2000-01-01',
        'physical_constants': {'capacity_m3': 1e9, 'initial_storage_m3': 4e8, 'initial_release_m3s': 1.,
            'channel_capacity_m3s': 20., 'dead_storage_fraction': .1, 'inflow_correction': 1., 'flood_mode': 'storage'},
        'sampling': asdict(SamplingSettings()),
        'numerics': {'route': 'standard_native_v1', 'root_policy': 'nearest_previous_storage',
                     'atol_storage_m3': 1e-4, 'forcing_policy': None},
        'inflow_preprocessing': {'factor': 1.45, 'offset_m3s': 8.},
        'monthly_targets': targets.reset_index().to_dict('records'),
        'provenance': {'selection_uses_assessment': False, 'source_certified': False, 'source_gates': []},
        'source_sha256': portable.source_hashes()}
    path = tmp_path / 'profile.json'
    portable.write_json(path, portable.seal_profile(payload))
    return path


def frame(values=(1., 2., 3.)):
    return pd.DataFrame({'inflow_m3s': values}, index=pd.date_range('2000-01-01', periods=len(values), name='date'))


def change(profile, mutate):
    payload = portable.read_json(profile)
    payload.pop('payload_sha256')
    mutate(payload)
    portable.write_json(profile, portable.seal_profile(payload))


def test_frozen_application_does_not_mutate_inputs_and_applies_transform_once(profile):
    raw = frame(); copy = raw.copy(deep=True)
    canonical, checks = portable.run_profile(profile, raw, forcing_stage='canonical')
    prepared = raw * 1.45 + 8.
    second, _ = portable.run_profile(profile, prepared, forcing_stage='prepared')
    pd.testing.assert_frame_equal(canonical.frame, second.frame)
    pd.testing.assert_frame_equal(raw, copy)
    assert checks['maximum_independent_balance_residual_m3'] <= 1e-4


def test_new_start_requires_explicit_initial_state_without_response_assimilation(profile):
    source = frame(); source.index += pd.Timedelta(days=20)
    with pytest.raises(ValueError, match='explicit'):
        portable.run_profile(profile, source, forcing_stage='prepared')
    result, _ = portable.run_profile(profile, source, forcing_stage='prepared',
                                    initial_state={'storage_m3': 3e8, 'release_m3s': 2.})
    assert result.frame.storage_m3.iloc[0] == 3e8
    assert result.frame.release_m3s.iloc[0] == 2.
    source['storage_m3'] = 999.
    with pytest.raises(ValueError, match='inflow-only'):
        portable.run_profile(profile, source, forcing_stage='prepared')


@pytest.mark.parametrize('bad', [{'storage_m3': True, 'release_m3s': 2.},
    {'storage_m3': 2e9, 'release_m3s': 2.}, {'storage_m3': 3e8}, {'storage_m3': 3e8, 'release_m3s': -1.}])
def test_invalid_initial_state_is_rejected(profile, bad):
    with pytest.raises((ValueError, TypeError)):
        portable.run_profile(profile, frame(), forcing_stage='prepared', initial_state=bad)


def test_signed_route_must_be_explicit_and_retains_net_losses(profile):
    forcing = frame((1., -2., 3.))
    with pytest.raises(ValueError, match='nonnegative'):
        portable.run_profile(profile, forcing, forcing_stage='prepared')
    change(profile, lambda p: p['numerics'].update(route='signed_net_native_v1', forcing_policy=asdict(SignedNetPolicy())))
    result, _ = portable.run_profile(profile, forcing, forcing_stage='prepared')
    np.testing.assert_array_equal(result.frame.inflow_m3s, forcing.inflow_m3s)


def test_overflow_uses_existing_conservative_release_and_independent_balance(profile):
    result, checks = portable.run_profile(profile, frame((1e6, 1e6)), forcing_stage='prepared')
    assert result.frame.storage_m3.iloc[-1] == result.parameters.capacity_m3
    assert result.frame.overflow_volume_m3.iloc[-1] > 0
    assert checks['overflow_steps'] == 1
    assert checks['maximum_independent_balance_residual_m3'] <= 1e-4


@pytest.mark.parametrize('mutation,match', [
    (lambda p: p['source_sha256'].update({'schemas.py': '0' * 64}), 'source changed'),
    (lambda p: p['provenance'].update(selection_uses_assessment=True), 'selection'),
    (lambda p: p['provenance'].update(source_gates=['unknown storage definition']), 'admission'),
    (lambda p: p['physical_constants'].update(capacity_m3=True), 'finite'),
    (lambda p: p['monthly_targets'][0].update(storage_critical_m3=9e8), 'Storage targets'),
    (lambda p: p['numerics'].update(route='unreviewed_interval_adapter'), 'numerical route'),
])
def test_incompatible_profiles_cannot_be_promoted(profile, mutation, match):
    change(profile, mutation)
    with pytest.raises(ValueError, match=match):
        portable.load_profile(profile)


def test_profile_tampering_and_duplicate_json_are_detected(profile):
    payload = portable.read_json(profile); payload['physical_constants']['capacity_m3'] *= 2
    portable.write_json(profile, payload)
    with pytest.raises(ValueError, match='checksum'):
        portable.load_profile(profile)
    profile.write_text('{"kind": 1, "kind": 2}')
    with pytest.raises(ValueError, match='Duplicate'):
        portable.read_json(profile)


def test_missing_forcing_dates_and_values_are_not_filled(profile):
    with pytest.raises(ValueError, match='consecutive'):
        portable.run_profile(profile, frame().iloc[[0, 2]], forcing_stage='prepared')
    with pytest.raises(ValueError, match='finite'):
        portable.run_profile(profile, frame((1., np.nan, 2.)), forcing_stage='prepared')
    with pytest.raises(ValueError, match='Declare'):
        portable.run_profile(profile, frame(), forcing_stage='automatic')


def test_application_csv_rejects_response_columns_and_duplicate_headers(tmp_path):
    path = tmp_path / 'input.csv'
    for text in ('date,inflow_m3s,storage_m3\n2000-01-01,1,2\n', 'date,inflow_m3s,inflow_m3s\n2000-01-01,1,2\n'):
        path.write_text(text)
        with pytest.raises(ValueError, match='exactly'):
            portable.read_forcing(path)


def test_sealed_resume_reuses_results_and_rejects_changed_input_or_artifact(profile, tmp_path, monkeypatch):
    forcing = tmp_path / 'forcing.csv'; frame().to_csv(forcing)
    output = tmp_path / 'application'
    receipt = portable.simulate_csv(profile, forcing, output, forcing_stage='prepared')
    assert receipt['responses_read'] is False and receipt['new_optimizer_evaluations'] == 0
    def unexpected(*args, **kwargs):
        raise AssertionError('Completed application must not simulate again')
    monkeypatch.setattr(portable, 'run_profile', unexpected)
    assert portable.simulate_csv(profile, forcing, output, forcing_stage='prepared', resume=True) == receipt
    with pytest.raises(ValueError, match='decisions changed'):
        portable.simulate_csv(profile, forcing, output, forcing_stage='canonical', resume=True)
    (output / 'simulation.csv').write_text('changed')
    with pytest.raises(ValueError, match='artifact changed'):
        portable.simulate_csv(profile, forcing, output, forcing_stage='prepared', resume=True)


def test_stable_gateway_preserves_workflow_arguments_and_requires_source_bundle(tmp_path, monkeypatch):
    root = tmp_path / 'bundle'; (root / 'tools').mkdir(parents=True)
    (root / 'tools/dztr_run_v10.py').write_text('')
    (root / 'pyproject.toml').write_text('')
    calls = []
    monkeypatch.setattr(cli.subprocess, 'call', lambda command, cwd: calls.append((command, cwd)) or 0)
    assert cli.main(['run', '--config', 'case.json', '--output', 'result', '--resume', '--until', 'baseline',
                     '--project-root', str(root)]) == 0
    command, cwd = calls[0]
    assert cwd == root.resolve() and '--resume' in command and command[-2:] == ['--until', 'baseline']
    with pytest.raises(ValueError, match='source/application bundle'):
        cli.project_root(tmp_path / 'absent')


def test_verified_registry_export_embeds_targets_without_observations(profile, tmp_path):
    from tools.export_portable_profile_v1 import export
    payload, parameters = portable.load_profile(profile)
    release = tmp_path / 'native'; release.mkdir()
    parameters.monthly_targets.to_csv(release / 'monthly_targets_si.csv', float_format='%.17g')
    metadata = {'reservoir_id': 'synthetic', 'grand_id': None, 'selection_uses_assessment': False,
        'physical_constants': payload['physical_constants'], 'sampling': payload['sampling'],
        'initial_date': payload['initial_date'], 'root_policy': 'nearest_previous_storage',
        'atol_storage_m3': 1e-4, 'method': 'observed_monthly',
        'monthly_targets_file': 'monthly_targets_si.csv',
        'monthly_targets_sha256': portable.file_digest(release / 'monthly_targets_si.csv')}
    portable.write_json(release / 'parameters.json', metadata)
    portable.write_json(release / 'registry.json', {'kind': 'unified_dztr_profiles',
        'selection_uses_assessment': False, 'profiles_by_reservoir_id': {'synthetic': 'parameters.json'},
        'profiles_by_GRAND_ID': {}})
    portable.write_json(release / 'manifest.json', {'code_sha256': {},
        'artifact_sha256': {p.name: portable.file_digest(p) for p in release.iterdir()}})
    prep = tmp_path / 'preprocessing.json'; portable.write_json(prep, {'factor': 1., 'offset_m3s': 0.})
    result = export(release / 'registry.json', tmp_path / 'export.json', prep,
                    reservoir_id='synthetic', method='observed_monthly')
    assert result['observations_copied'] is False
    _, actual = portable.load_profile(tmp_path / 'export.json')
    pd.testing.assert_frame_equal(parameters.monthly_targets, actual.monthly_targets)
    with pytest.raises(ValueError, match='disagrees'):
        export(release / 'registry.json', tmp_path / 'wrong_method.json', prep,
               reservoir_id='synthetic', method='flow_only')
    (release / 'monthly_targets_si.csv').write_text('modified')
    with pytest.raises(ValueError, match='artifact changed'):
        export(release / 'registry.json', tmp_path / 'tampered.json', prep,
               reservoir_id='synthetic', method='observed_monthly')
