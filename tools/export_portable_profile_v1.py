"""Export one verified native registry profile without copying observations."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from dztr.portable import KIND, file_digest, load_profile, read_json, seal_profile, source_hashes, write_json
from workflows import reservoir_v2 as native


def export(registry_path, output, preprocessing_path, *, reservoir_id=None, grand_id=None, method):
    registry_path = Path(registry_path).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new portable profile filename')
    registry = native.verify_registry(registry_path)
    metadata, parameters = native.load_profile(registry_path, identifier=reservoir_id, grand_id=grand_id)
    if metadata.get('selection_uses_assessment') is not False or metadata.get('source_gates'):
        raise ValueError('Export requires calibration-only selection and no unresolved admission gate')
    if metadata.get('method', method) != method:
        raise ValueError('Requested method disagrees with frozen profile')
    identifier = metadata['reservoir_id']
    original = registry_path.parent / registry['profiles_by_reservoir_id'][identifier]
    payload = {'kind': KIND, 'schema_version': 1, 'reservoir_id': identifier,
        'GRAND_ID': metadata.get('grand_id'), 'method': method,
        'physical_constants': metadata['physical_constants'], 'sampling': metadata['sampling'],
        'initial_date': metadata['initial_date'],
        'numerics': {'route': metadata.get('route', 'standard_native_v1'), 'root_policy': metadata['root_policy'],
                     'atol_storage_m3': metadata['atol_storage_m3'], 'forcing_policy': metadata.get('forcing_policy')},
        'inflow_preprocessing': read_json(preprocessing_path),
        'monthly_targets': parameters.monthly_targets.reset_index().to_dict('records'),
        'provenance': {'source_certified': False, 'source_gates': [], 'selection_uses_assessment': False,
            'convergence_established': metadata.get('convergence_established', False),
            'source_registry_sha256': file_digest(registry_path), 'source_profile_sha256': file_digest(original),
            'source_targets_sha256': metadata['monthly_targets_sha256'],
            'preprocessing_declaration_sha256': file_digest(preprocessing_path),
            'frozen_selection': {k: metadata[k] for k in ('seed', 'evaluation_id', 'year_selection', 'scoring',
                'evaluation_budget', 'selection_sha256') if k in metadata},
            'assumptions': ['Source interpretation and numerical settings remain those of the original profile.',
                            'Export does not establish source certification or independent forecast skill.']},
        'source_sha256': source_hashes()}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate in a sibling temporary file before exposing the completed profile.
    temporary = output.with_name(output.name + '.pending')
    if temporary.exists():
        raise ValueError('A previous partial export exists; retain it and choose a fresh filename')
    write_json(temporary, seal_profile(payload))
    load_profile(temporary)
    temporary.replace(output)
    return {'profile': str(output), 'reservoir_id': identifier, 'profile_sha256': file_digest(output),
            'observations_copied': False, 'new_optimizer_evaluations': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', required=True)
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument('--reservoir-id'); select.add_argument('--grand-id', type=int)
    parser.add_argument('--preprocessing-json', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = export(args.registry, args.output, args.preprocessing_json,
                    reservoir_id=args.reservoir_id, grand_id=args.grand_id, method=args.method)
    print(json.dumps(result, indent=2))
