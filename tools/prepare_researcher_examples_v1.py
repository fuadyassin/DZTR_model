"""Export three public frozen profiles and replay them without private runtime inputs."""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from dztr.portable import KIND, file_digest, load_profile, seal_profile, simulate_csv, source_hashes, write_json
from workflows import profile_application_v1 as archived

EXAMPLES = {'ex14_Gardiner': 'gardiner', 'ex6_Oahe': 'oahe', 'ex5_Garrison': 'garrison'}


def export_examples(output):
    out = Path(output).resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Use a fresh output directory inside repository outputs')
    destination = ROOT / 'examples/portable'
    if destination.exists():
        raise ValueError('Public example release already exists; use a new version rather than overwrite it')
    catalog_path = ROOT / 'outputs/calibrated_paper_cohort_v1_20261017_final/catalog.json'
    catalog = archived.read_catalog(catalog_path)
    out.mkdir(parents=True); destination.mkdir(parents=True)
    receipts = []
    for rid, name in EXAMPLES.items():
        entry, metadata, parameters = archived.load_profile(catalog_path, reservoir_id=rid)
        folder = destination / name; folder.mkdir()
        registry = ROOT / entry['registry']
        original_profile = registry.parent / archived.w.read(registry)['profiles_by_reservoir_id'][rid]
        preparation = ROOT / entry['prepared_forcing']
        public = ROOT / 'data/reservoirs' / rid / 'observations_si.csv'
        forcing = pd.read_csv(preparation, index_col='date', parse_dates=True, float_precision='round_trip')[['inflow_m3s']]
        shared = pd.read_csv(public, index_col='date', parse_dates=True, float_precision='round_trip')[['inflow_m3s']]
        pd.testing.assert_index_equal(forcing.index, shared.index)
        expected = shared.inflow_m3s * entry['inflow_preprocessing']['factor'] + entry['inflow_preprocessing']['offset_m3s']
        np.testing.assert_allclose(forcing.inflow_m3s, expected, rtol=0, atol=1e-9)
        payload = {'kind': KIND, 'schema_version': 1, 'reservoir_id': rid, 'GRAND_ID': entry['GRAND_ID'],
            'method': 'observed_monthly', 'physical_constants': metadata['physical_constants'],
            'sampling': metadata['sampling'], 'initial_date': metadata['initial_date'],
            'numerics': {'route': entry['route'], 'root_policy': metadata['root_policy'],
                'atol_storage_m3': metadata['atol_storage_m3'], 'forcing_policy': metadata.get('forcing_policy')},
            'inflow_preprocessing': entry['inflow_preprocessing'],
            'monthly_targets': parameters.monthly_targets.reset_index().to_dict(orient='records'),
            'provenance': {'source_certified': False, 'source_gates': [], 'convergence_established': False,
                'selection_uses_assessment': False, 'year_selection': metadata['year_selection'],
                'scoring': metadata['scoring'], 'seed': metadata['seed'], 'evaluation_id': metadata['evaluation_id'],
                'evaluation_budget_per_seed': metadata['evaluation_budget'],
                'selection_sha256': metadata['selection_sha256'], 'quantile_method': 'hazen',
                'source_profile_sha256': file_digest(original_profile),
                'source_targets_sha256': metadata['monthly_targets_sha256'],
                'source_catalog_sha256': file_digest(catalog_path),
                'source_registry_sha256': file_digest(registry),
                'public_observations_sha256': file_digest(public),
                'calibration': metadata['calibration'], 'assessment': metadata['assessment'],
                'assumptions': ['Retrospective conditional calibration; forcing independence is unverified.',
                    'Monthly 30.5-day steps are retained where declared; they are not calendar-month means.',
                    'Targets, capacity and root policy were not changed during export.']},
            'source_sha256': source_hashes()}
        profile = folder / 'profile.json'; write_json(profile, seal_profile(payload))
        portable, reconstructed = load_profile(profile)
        np.testing.assert_array_equal(reconstructed.monthly_targets, parameters.monthly_targets)
        forcing.to_csv(folder / 'forcing_si.csv', index_label='date', float_format='%.17g')
        receipt = simulate_csv(profile, folder / 'forcing_si.csv', out / name,
            forcing_stage='prepared', plot=True)
        cache = ROOT / 'outputs/cohort_diagnostic_cache_v1_20261017' / rid / 'selected'
        cached_receipt = archived.w.read(cache / 'receipt.json')
        if file_digest(cache / 'simulation.csv') != cached_receipt['simulation_sha256']:
            raise ValueError('Archived reference trajectory changed')
        reference = pd.read_csv(cache / 'simulation.csv', index_col='date', parse_dates=True, float_precision='round_trip')
        actual = pd.read_csv(out / name / 'simulation.csv', index_col='date', parse_dates=True, float_precision='round_trip')
        pd.testing.assert_index_equal(actual.index, reference.index)
        for column in ('storage_m3', 'release_m3s', 'inflow_m3s'):
            np.testing.assert_array_equal(actual[column], reference[column])
        # A later-start forcing needs a model-carried initial state, never a response observation.
        start = 365 if metadata['sampling']['frequency'] == 'daily' else 12
        end = start + (366 if metadata['sampling']['frequency'] == 'daily' else 25)
        subset = forcing.iloc[start:end]
        subset.to_csv(folder / 'forcing_restart_si.csv', index_label='date', float_format='%.17g')
        initial = {'storage_m3': float(actual.storage_m3.iloc[start]), 'release_m3s': float(actual.release_m3s.iloc[start])}
        write_json(folder / 'initial_state_restart.json', initial)
        restart = simulate_csv(profile, folder / 'forcing_restart_si.csv', out / (name + '_restart'),
            forcing_stage='prepared', initial_state=initial)
        restarted = pd.read_csv(out / (name + '_restart') / 'simulation.csv', index_col='date',
                                parse_dates=True, float_precision='round_trip')
        np.testing.assert_array_equal(restarted[['storage_m3', 'release_m3s']], actual.loc[subset.index, ['storage_m3', 'release_m3s']])
        write_json(folder / 'expected_checks.json', {'reservoir_id': rid, 'GRAND_ID': entry['GRAND_ID'],
            'historical_rows': len(actual), 'historical_storage_release_exactly_reproduced': True,
            'restart_rows': len(subset), 'restart_initial_state_source': 'frozen_model_trajectory_not_observations',
            'restart_storage_release_exactly_reproduced': True,
            'historical_state_sha256': file_digest(out / name / 'simulation.csv'),
            'historical_state_canonical_lf_sha256': sha256((out / name / 'simulation.csv').read_text(encoding='utf-8').encode()).hexdigest(),
            'maximum_balance_residual_m3': receipt['maximum_independent_balance_residual_m3'],
            'independent_predictive_validation': False})
        receipts.append({'name': name, 'historical': receipt, 'restart': restart,
                         'archived_replay_exact': True, 'restart_replay_exact': True})
    write_json(destination / 'manifest.json', {'kind': 'dztr_public_portable_examples_v1',
        'reservoirs': list(EXAMPLES), 'private_runtime_dependencies': False,
        'artifact_sha256': {p.relative_to(destination).as_posix(): file_digest(p)
            for p in sorted(destination.rglob('*')) if p.is_file()}})
    write_json(out / 'verification.json', {'examples': receipts, 'new_optimizer_evaluations': 0,
        'existing_profiles_modified': False, 'unshared_observations_copied': False,
        'catalog_sha256': file_digest(catalog_path)})
    print(json.dumps({'output': str(out), 'examples': len(receipts), 'exact_historical_replays': 3,
                      'exact_later_start_replays': 3, 'new_optimizer_evaluations': 0}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    export_examples(parser.parse_args().output)
