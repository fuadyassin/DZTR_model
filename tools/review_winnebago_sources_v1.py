"""Winnebago lake-boundary sensitivity with explicit datum and filter choices.

Reuse cached source evidence and existing estimators; never replace an active
profile or silently promote a relative volume to surveyed total storage.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import re
import sys
from html import unescape

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from dztr.schemas import MONTHLY_TARGET_COLUMNS
from strategies.data_availability_v1 import empirical_targets
from workflows import lake_level_proxy_v1 as lake
from workflows import boundary_source_v1 as boundary
from workflows import reservoir_v2 as io

CONFIG = ROOT / 'configs/winnebago_source_reconciliation_v1_20261022.json'


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_frame(path):
    return pd.read_csv(path, index_col=0, parse_dates=True, float_precision='round_trip')


def verify_pins(folder):
    proof = io.read(folder / 'completion_manifest.json')
    for name, expected in proof['sha256'].items():
        if io.sha(ROOT / name) != expected:
            raise ValueError('Source review pin changed: ' + name)
    return len(proof['sha256'])


def verify(out):
    proof = io.read(out / 'completion_manifest.json')
    return {'pinned_files_verified': verify_pins(out),
        'paired_review_pins_verified': verify_pins(ROOT / proof['previous_review']),
        'aishihik_review_pins_verified': verify_pins(ROOT / proof['aishihik_review']),
        'new_optimizer_evaluations': 0, 'new_calibration_admissions': 0}


def datum_sample(path, station, expected_offset):
    """Forensic check of simultaneous instantaneous samples, not daily data."""
    data = io.read(path); streams = {}
    for series in data['value']['timeSeries']:
        assert [v['value'] for v in series['sourceInfo']['siteCode'] if v.get('agencyCode') == 'USGS'] == [station]
        code = series['variable']['variableCode'][0]['value']
        assert code in ('00065', '62615') and code not in streams
        assert series['variable']['unit']['unitCode'] == 'ft' and len(series['values']) == 1
        missing = float(series['variable']['noDataValue']); points = {}
        for row in series['values'][0]['value']:
            value = float(row['value']); time = row['dateTime']
            assert time not in points and np.isfinite(value)
            if value != missing: points[time] = value
        streams[code] = points
    assert set(streams) == {'00065', '62615'}
    times = sorted(streams['00065'].keys() & streams['62615'].keys()); assert times
    diff = np.array([streams['62615'][time] - streams['00065'][time] for time in times])
    assert abs(diff - expected_offset).max() <= .010000001
    return {'station': station, 'paired_instantaneous_timestamps': len(times),
        'first_timestamp': times[0], 'last_timestamp': times[-1],
        'published_offset_navd88_ft': expected_offset,
        'minimum_elevation_minus_height_ft': float(diff.min()),
        'maximum_elevation_minus_height_ft': float(diff.max()),
        'absolute_tolerance_ft': .010000001, 'full_historical_datum_verified': False,
        'sample_was_aggregated_to_daily_elevation': False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(); cfg = io.read(CONFIG); out = ROOT / cfg['output']
    if args.verify or (out / 'completion_manifest.json').exists():
        print(json.dumps(verify(out), indent=2)); return
    assert cfg['kind'] == 'winnebago_source_reconciliation_v1' and not cfg['calibration_admitted']
    old = ROOT / cfg['previous_review']; old_pins = verify_pins(old)
    aishihik_pins = verify_pins(ROOT / cfg['aishihik_review'])
    evidence = out / 'evidence'; receipts = io.read(evidence / 'receipts.json')
    for name, receipt in receipts.items():
        if 'error' not in receipt:
            assert io.sha(evidence / name) == receipt['sha256'], name
    # Official station descriptions establish current datum/area values only.
    for filename, required in {
        'usgs_oshkosh_metadata.html': ['743.91 NAVD88', '215 square miles', '5,880'],
        'usgs_stockbridge_metadata.html': ['744.07 NAVD88', '215 square miles', '5,880'],
        'usgs_appleton_metadata.html': ['5,950']}.items():
        text = re.sub(r'\s+', ' ', unescape(re.sub(r'<[^>]+>', ' ', (evidence / filename).read_text())))
        assert all(value in text for value in required), filename
    samples = [datum_sample(evidence / ('usgs_' + name + '_instantaneous_2020_sample.json'), station, offset)
        for name, station, offset in [('oshkosh', '04082500', 743.91), ('stockbridge', '04084255', 744.07)]]
    io.write(out / 'datum_sample_checks.json', {'samples': samples,
        'full_period_datum_history_verified': False, 'daily_elevation_parameter_available': False})
    for name in ('oshkosh', 'stockbridge'):
        assert not io.read(evidence / ('usgs_' + name + '_elevation_2019_2020.json'))['value']['timeSeries']
    source = old / 'prepared/etienne_Winnebago/source_candidates_si.csv'; frame = read_frame(source)
    heights = frame[['gauge_04082500_level_ft', 'gauge_04084255_level_ft']].copy()
    heights.columns = ['04082500', '04084255']
    area = cfg['surface_area_square_miles'] * lake.MI_TO_M ** 2
    relative_frame, reference_proof = lake.relative_storage_candidate(heights, cfg['gauge_datums_navd88_ft'],
        reference_elevation_ft=cfg['relative_volume_reference_navd88_ft'], surface_area_m2=area,
        datum_history_assumption=cfg['historical_datum_assumption'], allow_exploratory=cfg['allow_exploratory_relative_storage'])
    relative_frame.to_csv(out / 'datum_aligned_relative_storage_si.csv')
    adapter = cfg['release_adapter']
    q = boundary.read_usgs_daily(old / 'stations/usgs_04084445.json', '04084445', '00060', method_id=adapter['usgs_method_id']) * .028316846592
    q = q.reindex(frame.index)
    pd.testing.assert_series_equal(q, frame.gauge_04084445_appleton_discharge_m3s, check_names=False)
    ratio = adapter['lake_outlet_drainage_area_square_miles'] / adapter['appleton_drainage_area_square_miles']
    proxy_q = q * ratio
    comparison = boundary.compare(frame.release_archive_m3s, proxy_q, .006)
    candidate = frame.copy(); candidate['release_area_prorated_proxy_m3s'] = proxy_q
    profiles = {}; filter_proofs = {}; target_files = {}
    fitting_mask = pd.Series(True, index=frame.index)
    for days in cfg['filter_candidates_days']:
        tag = f'matched_trailing_{days}d'
        profile, proof = lake.matched_budget_candidates(relative_frame.relative_storage_candidate_m3, proxy_q, days=days)
        assert proof['maximum_filter_balance_difference_m3'] <= 1e-4
        profile.to_csv(out / f'relative_volume_profile_{tag}_si.csv')
        profiles[tag] = profile; filter_proofs[tag] = proof
        source_fit_mask = lake.eligible_source_windows(fitting_mask, filter_days=days)
        candidate['relative_storage_candidate_' + tag + '_m3'] = profile.relative_storage_candidate_m3
        candidate['release_candidate_' + tag + '_m3s'] = profile.release_candidate_m3s
        targets = pd.DataFrame(index=pd.Index(range(1, 13), name='month'), columns=MONTHLY_TARGET_COLUMNS, dtype=float)
        counts = pd.DataFrame(index=targets.index)
        for variable, values, columns in [('storage', profile.relative_storage_candidate_m3, MONTHLY_TARGET_COLUMNS[:3]),
                                         ('release', profile.release_candidate_m3s, MONTHLY_TARGET_COLUMNS[3:])]:
            estimates, n = empirical_targets(values, source_fit_mask, monthly=True,
                percentiles=tuple(cfg['descriptive_percentiles']), method=cfg['quantile_method'])
            targets.loc[:, list(columns)] = estimates; counts[variable + '_samples'] = n
        assert targets.notna().all().all()
        target_path = out / f'descriptive_targets_{tag}_si.csv'; targets.to_csv(target_path, float_format='%.17g')
        counts.to_csv(out / f'monthly_sample_counts_{tag}.csv'); target_files[tag] = relative(target_path)
    raw = profiles['matched_trailing_1d']; smoothed = profiles['matched_trailing_7d']
    # The shifted-Q and unmatched-filter alternatives are diagnostics only.
    # A downstream reading assigned one day earlier uses future Q explicitly.
    budgets = pd.DataFrame({'archive_storage_archive_Q': frame.storage_archive_m3.diff()/86400 + frame.release_archive_m3s,
        'lake_relative_daily_prorated_Q': raw.diagnostic_net_flux_m3s,
        'lake_relative_daily_unscaled_Appleton_Q': raw.relative_storage_candidate_m3.diff()/86400 + q,
        'lake_relative_daily_next_day_prorated_Q_diagnostic': raw.relative_storage_candidate_m3.diff()/86400 + proxy_q.shift(-1),
        'lake_relative_matched_7d': smoothed.diagnostic_net_flux_m3s,
        'lake_relative_7d_storage_unmatched_raw_Q_diagnostic': smoothed.relative_storage_candidate_m3.diff()/86400 + proxy_q}, index=frame.index)
    common = budgets.notna().all(axis=1); summary_rows = []
    for name in budgets:
        sample = budgets.loc[common, name]
        summary_rows.append({'scenario': name, 'common_intervals': int(common.sum()),
            'negative_net_flux_intervals': int((sample < 0).sum()), 'minimum_net_flux_m3s': float(sample.min()),
            'mean_net_flux_m3s': float(sample.mean()), 'uses_future_Q_for_historical_diagnostic': 'next_day' in name,
            'automatically_selected': False, 'independent_inflow': False})
    summary = pd.DataFrame(summary_rows); summary.to_csv(out / 'budget_sensitivity_summary.csv', index=False)
    budgets['common_supported_interval'] = common; budgets.to_csv(out / 'signed_budget_diagnostics.csv')
    candidate['mean_surface_elevation_navd88_candidate_m'] = relative_frame.mean_surface_elevation_navd88_m
    candidate['calibration_ready'] = False; candidate.to_csv(out / 'source_candidates_si.csv')
    # Forensic slope only: never adopt an area by fitting the archived storage.
    slope = boundary.affine_source_diagnostic(frame.storage_archive_m3,
        frame[['unverified_archive_stage_proxy_m']])
    slope['fitted_slope_used_as_area'] = False
    decision = {'reservoir_id': cfg['reservoir_id'], 'status': 'exploratory_lake_only_relative_storage_prepared',
        'calibration_ready': False, 'boundary': cfg['exploratory_boundary'], 'surface_area_m2': area,
        'reference_proof': reference_proof, 'datum_samples': samples, 'filter_proofs': filter_proofs,
        'archive_outlet_comparison': comparison, 'drainage_area_proration_factor': ratio,
        'drainage_area_ratio_rationale_identified': True, 'routed_outlet_proxy_certified': False,
        'archive_storage_forensic_slope': slope, 'archive_3_7112_offset_used_as_bed': False,
        'capacity_adopted_m3': None, 'dead_storage_adopted_m3': None,
        'descriptive_target_files': target_files, 'target_fitting_scope': 'all_available_records_descriptive_only',
        'historical_centered_filter_imported': False,
        'future_calibration_filter_support_rule': 'Every source day of a filtered target must be a calibration day; budget differences additionally require the preceding storage window. Recheck actual >=65% support after exclusions.',
        'source_gates': cfg['source_gates'],
        'next_action': ('Retain the explicitly lake-only 215 mi2 constant-area relative-volume scenario; '
            'confirm historical datum continuity and adopt capacity/dead storage on its declared reference plane. '
            'Freeze the Appleton drainage-area/routing proxy and matched-filter clock; use independent inflow '
            'or a separately admitted conditional signed-net experiment. Refit targets on >=65% frozen mixed calibration years.')}
    io.write(out / 'source_decision.json', decision)
    base = pd.read_csv(ROOT / cfg['base_inventory'], float_precision='round_trip'); inventory = base.copy(deep=True)
    select = inventory.reservoir_id == cfg['reservoir_id']; assert select.sum() == 1
    inventory.loc[select, 'next_action'] = decision['next_action']
    inventory.loc[select, 'source_review_file'] = relative(out / 'source_decision.json')
    inventory.loc[select, 'source_candidate_file'] = relative(out / 'source_candidates_si.csv')
    target_cols = [c for c in base if c.startswith(tuple(f'm{i:02d}_' for i in range(1, 13)))]
    assert len(target_cols) == 72 and len(base) == 62
    pd.testing.assert_frame_equal(base[target_cols], inventory[target_cols])
    pd.testing.assert_frame_equal(base.loc[~select], inventory.loc[~select])
    revised = deepcopy(io.read(ROOT / cfg['base_catalog'])); revised['kind'] = 'reservoir_collection_index_v9'
    revised['previous_catalog'] = cfg['base_catalog']
    row = next(r for r in revised['reservoirs'] if r['reservoir_id'] == cfg['reservoir_id'])
    row.update(source_review_file=relative(out / 'source_decision.json'), source_admission_gates=cfg['source_gates'],
        source_candidate_file=relative(out / 'source_candidates_si.csv'), exploratory_descriptive_targets=target_files,
        exploratory_boundary=cfg['exploratory_boundary'], full_period_datum_history_verified=False)
    catalog_path = ROOT / 'data/catalogs/reservoir_collection_v9_20261022.json'
    inventory_path = ROOT / 'data/catalogs/reservoir_inventory_v9_20261022.csv'
    assert not catalog_path.exists() and not inventory_path.exists()
    io.write(catalog_path, revised); inventory.to_csv(inventory_path, index=False, float_format='%.17g')
    reread = pd.read_csv(inventory_path, float_precision='round_trip')
    pd.testing.assert_frame_equal(base[target_cols], reread[target_cols])
    pd.testing.assert_frame_equal(base.loc[~select], reread.loc[~select])
    checks = {'paired_review_pins_verified': old_pins, 'aishihik_review_pins_verified': aishihik_pins,
        'preserved_monthly_target_cells': 4464, 'unchanged_other_inventory_rows': 61,
        'inventory_roundtrip_verified': True, 'new_optimizer_evaluations': 0, 'new_calibration_admissions': 0,
        'source_observations_overwritten': False, 'hydrological_equations_changed': False,
        'historical_centered_filter_future_levels_used': False, 'same_dates_for_budget_comparison': True,
        'matched_filter_balance_tolerance_m3': 1e-4, 'storage_reference_is_not_inferred_bed': True,
        'phase_pure_source_window_guard_available': True}
    io.write(out / 'scientific_checks.json', checks)
    plot(relative_frame, profiles, summary, out)
    write_report(out, decision, summary)
    files = [CONFIG, source, ROOT / cfg['base_inventory'], ROOT / cfg['base_catalog'],
        ROOT / 'workflows/lake_level_proxy_v1.py', ROOT / 'tests/test_lake_level_proxy_v1.py',
        ROOT / 'strategies/data_availability_v1.py', ROOT / 'workflows/boundary_source_v1.py', Path(__file__),
        ROOT / 'tools/fetch_winnebago_reference_v1.py', ROOT / 'docs/winnebago_source_reconciliation_v1.md',
        old / 'stations/usgs_04084445.json', old / 'evidence/winnebago_usgs_routing_1981.pdf',
        inventory_path, catalog_path] + [p for p in out.rglob('*') if p.is_file()]
    io.write(out / 'completion_manifest.json', {'kind': cfg['kind'], 'previous_review': cfg['previous_review'],
        'aishihik_review': cfg['aishihik_review'], 'sha256': {relative(p): io.sha(p) for p in files},
        'scientific_checks': checks})
    print(json.dumps(verify(out), indent=2))


def plot(relative_frame, profiles, summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    for name, profile in profiles.items():
        axes[0].plot(profile.index, profile.relative_storage_candidate_m3 / 1e6, label=name, lw=.75)
    axes[0].set(title='Lake Winnebago-only constant-area sensitivity; relative volume above declared plane',
                ylabel='Relative volume candidate (million m3)')
    axes[0].legend()
    labels = ['Archive storage / archived Q', 'Lake daily / prorated Q', 'Lake daily / unscaled Appleton Q',
              'Lake daily / next-day Q (historical diagnostic)', 'Lake matched trailing 7-day S and Q',
              'Lake 7-day S / unmatched raw Q (diagnostic)']
    axes[1].barh(labels, summary.negative_net_flux_intervals)
    axes[1].set(title='Signed-budget comparison on identical supported dates; no automatic selection',
                xlabel='Negative response-derived net-flux intervals')
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('Winnebago: datum-aligned storage anomalies, no lake-bed inference or calibration admission')
    fig.savefig(out / 'winnebago_source_diagnostics.png', dpi=160)
    fig.savefig(out / 'winnebago_source_diagnostics.svg'); plt.close(fig)


def write_report(out, decision, summary):
    proof = decision['reference_proof']; common = int(summary.common_intervals.iloc[0])
    filtered = decision['filter_proofs']['matched_trailing_7d']
    text = f'''# Winnebago datum, storage boundary and outlet preparation

**Prepared a lake-only relative-volume sensitivity and two descriptive monthly DZTR target profiles. No new calibration is admitted.** The archived storage and every existing monthly parameter remain unchanged.

[Plots](winnebago_source_diagnostics.png), [source decision](source_decision.json), [datum checks](datum_sample_checks.json), [daily targets](descriptive_targets_matched_trailing_1d_si.csv), [matched seven-day targets](descriptive_targets_matched_trailing_7d_si.csv), [budget comparison](budget_sensitivity_summary.csv), [updated 62-row inventory](../../data/catalogs/reservoir_inventory_v9_20261022.csv).

## Boundary decision

The [USGS pool-management study](https://pubs.usgs.gov/wri/1980/0107/report.pdf), printed pages 2 and 5-6, distinguishes the four-lake managed pool from Lake Winnebago alone and identifies wind/seiche contamination of level-derived budgets. Its stage-storage table was unpublished USACE material; this PDF does not supply a usable numeric curve. The exploratory candidate here is explicitly **Lake Winnebago alone**, using **215 mi2 = {decision['surface_area_m2']/1e6:.6f} km2** from [USGS lake station descriptions](https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=04082500). This area is not substituted for the four-lake pool area.

## Gauge references and relative volume

Current USGS descriptions give **743.91 ft NAVD88 for Oshkosh (04082500)** and **744.07 ft NAVD88 for Stockbridge (04084255)**. Each is added to its own native gauge height before averaging. The two June 2020 instantaneous samples provide **288 paired height/elevation timestamps each**, consistent with those offsets within 0.01 ft. This checks those samples, not the full 2007-2020 datum history. Daily NAVD88 requests returned no series; no instantaneous samples are relabelled as daily means. [Oshkosh](https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=04082500), [Stockbridge](https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=04084255).

The relative storage candidate is `S_rel = area * (mean elevation - declared reference elevation)`. Its reference is **743.91 ft NAVD88**, a computational plane, **not a lake bed**. No total lake volume, physical capacity or dead-storage volume is inferred. Both gauges are required on a day; gaps are preserved. Current datum constants are extended to earlier records only under an explicit, unverified historical assumption.

The archive's `3.7112 + 0.1524*Oshkosh_ft + 0.1524*Stockbridge_ft` remains a forensic construction, not a true-depth formula. It is never used as the new reference. A datum translation changes absolute anomalies but, under a constant reference, does not repair storage-change budget errors. The fitted archived storage/extra-field slope is recorded only as a diagnostic and is not adopted as surface area. Historical Oshkosh regulatory stages are not mixed with current NAVD88 metadata to invent capacity.

## Outlet proxy and clocks

The archive's factor **5880/5950 = {decision['drainage_area_proration_factor']:.12f}** matches the published drainage areas at the lake outlet and Appleton. This identifies a plausible area-proration rationale; it does not certify uniform runoff, total-outlet routing, or matching clocks. The candidate uses Appleton primary stream-flow method **226683**, with exact ft3/s-to-m3/s conversion, as a documented proxy. The two revised source dates remain separate from the original release archive. [Appleton description](https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=04084445).

Two profiles are prepared: unfiltered daily means and a **trailing seven-day triangular filter** with weights 1,2,3,4,3,2,1. The latter has a **three-day center-of-weight delay** recorded in provenance. Both storage and release receive the same filter. No future level is used, and no missing value is filled. The historical study's centered filter and future-neighbor editing are not imported as an operational preprocessing default.

On supported dates, `dF(S)/dt + F(Q) = F(dS/dt + Q)`. The maximum processing difference is **{filtered['maximum_filter_balance_difference_m3']:.3g} m3**, within the 1e-4 m3 audit tolerance. This is an algebraic filter check, not independent validation of reservoir forcing or source timing.

Six diagnostic combinations use the same **{common} dates**. The next-day-Q combination explicitly uses a future downstream observation for historical lag sensitivity only; it is not used in either target profile. The unmatched raw-Q combination demonstrates a different clock assumption. No option is selected by its negative count:

| Source/clock scenario | Common intervals | Negative net-flux intervals |
|---|---:|---:|
'''
    for _, row in summary.iterrows():
        text += f"| {row.scenario} | {int(row.common_intervals)} | {int(row.negative_net_flux_intervals)} |\n"
    text += '''
## Workflow status

Both profiles contain all 72 monthly storage/release targets from the shared P10/P45/P85 Hazen estimator. They use all available supported records for description and are neither calibrated nor independently assessed. The filtered profile describes filtered relative storage and filtered release, not instantaneous storage or unfiltered daily release. Refit on frozen mixed calibration years with at least 65% usable fitting data before any optimization.

The new `eligible_source_windows` guard requires every source day in a filtered target's window to belong to the fitting phase. Budget differences additionally require the preceding storage window. This prevents a January calibration target from incorporating December held-out responses. Recheck actual 65% support after these exclusions, rather than merely assigning 65% of calendar years. The same support rule can guard the earlier three-day Aishihik candidate when calibration is admitted; no earlier profile or source receipt is modified here.

Admission still requires explicit historical datum treatment, capacity/dead storage in the same relative coordinate, acceptance of the downstream discharge/routing proxy and observation clock, and independent inflow or a separately labelled conditional signed-net experiment. Response-derived net fluxes, including held-out responses if reconstructed across the record, cannot support independent predictive validation claims.

The v9 catalog/inventory changes only Winnebago's source-review links and next action. All 4464 existing monthly parameter cells and every other 61 inventory row are preserved. Observations and analysis remain locally ignored by Git. Reuse the frozen review with `tools/Set-ProjectCache.ps1` then `python -B tools/review_winnebago_sources_v1.py --verify`; initial evidence acquisition is separate in `tools/fetch_winnebago_reference_v1.py`.
'''
    (out / 'README.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
