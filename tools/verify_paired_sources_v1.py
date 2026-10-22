"""Independent checks of prepared boundaries; no optimization or integration."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from workflows import reservoir_v2 as io, station_inputs_v1 as station
from tools import review_paired_sources_v1 as review
from tools.complete_minidoka_v1 import preserved


def main():
    out = ROOT / 'outputs/paired_source_review_v1_20261022'
    validation = ROOT / 'outputs/paired_source_review_validation_v1_20261022'
    validation.mkdir(exist_ok=True)
    pin = io.read(out / 'completion_manifest.json')
    before = {p: (io.sha(ROOT / p), (ROOT / p).stat().st_mtime_ns) for p in pin['sha256']}
    checks = review.verify(out)
    checks.update(preserved())
    receipts = io.read(out / 'stations/receipts.json')
    reparsed = 0
    for sid, wrapper in receipts.items():
        proof = wrapper['receipt']; query = proof['query']
        cache = out / 'stations/raw' / station.digest({'endpoint': station.ENDPOINT, 'query': query})
        pages = [cache / name for name in proof['page_sha256']]
        for name, expected in proof['page_sha256'].items():
            assert io.sha(cache / name) == expected
        start, end = query['datetime'].split('/')
        index = pd.date_range(start, end, name='date')
        frame = review.read_frame(out / ('stations/' + sid + '.csv'))
        for prop, col in [('LEVEL', 'level_m'), ('DISCHARGE', 'discharge_m3s')]:
            actual = station._series(pages, sid, prop, index)
            np.testing.assert_array_equal(actual, frame[col]); reparsed += 1
    cfg = io.read(review.CONFIG); negative = []; expressions = 0
    for case in cfg['cases']:
        rid = case['reservoir_id']
        candidate = review.read_frame(out / 'prepared' / rid / 'source_candidates_si.csv')
        original = review.read_frame(ROOT / 'data/reservoirs_etienne_v1_20261021' / rid / 'observations_si.csv')
        np.testing.assert_array_equal(candidate.storage_archive_m3, original.storage_m3)
        np.testing.assert_array_equal(candidate.release_archive_m3s, original.release_m3s)
        assert candidate.inflow_m3s.isna().all()
        if not case['release_terms']:
            assert candidate.candidate_release_m3s.isna().all(); continue
        # Check the expression directly instead of calling the producer's composition helper.
        values = []; all_available = np.ones(len(candidate), dtype=bool)
        for term in case['release_terms']:
            s = review.read_frame(out / ('stations/' + term['station'] + '.csv')).discharge_m3s.reindex(candidate.index)
            np.testing.assert_array_equal(candidate['gauge_' + term['station'] + '_discharge_m3s'], s)
            values.append(s.to_numpy() * term['coefficient']); all_available &= s.notna().to_numpy()
        expected = sum(values); expected[~all_available] = np.nan
        np.testing.assert_array_equal(candidate.candidate_release_m3s, expected)
        old_net = original.storage_m3.diff() / 86400 + original.release_m3s
        new_net = original.storage_m3.diff() / 86400 + expected
        np.testing.assert_array_equal(candidate.diagnostic_archive_net_flux_m3s, old_net)
        np.testing.assert_array_equal(candidate.diagnostic_candidate_net_flux_m3s, new_net)
        expressions += 1
        for date in candidate.index[candidate.candidate_release_m3s < 0]:
            negative.append({'reservoir_id': rid, 'date': str(date.date()),
                             'candidate_release_m3s': float(candidate.loc[date, 'candidate_release_m3s']),
                             'storage_available': bool(pd.notna(original.loc[date, 'storage_m3'])),
                             'adopted_as_physical_release': False})
    a, b = io.read(ROOT / cfg['base_catalog']), io.read(ROOT / 'data/catalogs/reservoir_collection_v6_20261022.json')
    case_ids = {c['reservoir_id'] for c in cfg['cases']}
    allowed_additions = {'source_review_file', 'source_admission_gates', 'source_candidate_file'}
    assert len(a['reservoirs']) == len(b['reservoirs']) == 62
    for old, new in zip(a['reservoirs'], b['reservoirs']):
        if old['reservoir_id'] in case_ids:
            assert {k: v for k, v in new.items() if k not in allowed_additions} == old
            assert not new['calibration_ready']
        else:
            assert old == new
    old_inv = pd.read_csv(ROOT / cfg['base_inventory'], float_precision='round_trip')
    new_inv = pd.read_csv(ROOT / 'data/catalogs/reservoir_inventory_v6_20261022.csv', float_precision='round_trip')
    for column in old_inv:
        if column != 'next_action':
            pd.testing.assert_series_equal(old_inv[column], new_inv[column])
        else:
            keep = ~old_inv.reservoir_id.isin(case_ids)
            pd.testing.assert_series_equal(old_inv.loc[keep, column], new_inv.loc[keep, column])
    review.verify(out)
    for name, signature in before.items():
        assert (io.sha(ROOT / name), (ROOT / name).stat().st_mtime_ns) == signature
    checks.update(station_series_reparsed_from_raw_pages=reparsed,
                  boundary_expressions_checked_independently=expressions,
                  all_62_existing_target_and_profile_fields_preserved=True,
                  unchanged_preparation_files_and_mtimes=len(before),
                  negative_candidate_release_records=negative,
                  hydrological_simulations_rerun=0)
    io.write(validation / 'independent_source_checks.json', checks)
    io.write(validation / 'proof.json', {'source_review_manifest_sha256': io.sha(out / 'completion_manifest.json'),
        'verification_tool_sha256': io.sha(Path(__file__)),
        'independent_source_checks_sha256': io.sha(validation / 'independent_source_checks.json')})
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
