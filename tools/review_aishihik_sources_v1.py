"""Prepare explicit Aishihik storage hypotheses and descriptive shared targets.

Existing observations, profiles and parameter cells are immutable. No optimizer
is run, and the inferred curve interpretation is never silently certified.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from dztr.schemas import MONTHLY_TARGET_COLUMNS
from strategies.data_availability_v1 import empirical_targets
from workflows import aishihik_storage_candidate_v1 as curve
from workflows import reservoir_v2 as io

CONFIG = ROOT / 'configs/aishihik_source_reconciliation_v1_20261022.json'


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_frame(path):
    return pd.read_csv(path, index_col=0, parse_dates=True, float_precision='round_trip')


def verify_pins(folder):
    proof = io.read(folder / 'completion_manifest.json')
    for name, expected in proof['sha256'].items():
        if io.sha(ROOT / name) != expected:
            raise ValueError('Pinned source/artifact changed: ' + name)
    return len(proof['sha256'])


def verify(out):
    proof = io.read(out / 'completion_manifest.json')
    return {'pinned_files_verified': verify_pins(out),
            'previous_paired_review_pins_verified': verify_pins(ROOT / proof['previous_review']),
            'nechako_review_pins_verified': verify_pins(ROOT / proof['nechako_review']),
            'new_optimizer_evaluations': 0, 'new_calibration_admissions': 0}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(); cfg = io.read(CONFIG); out = ROOT / cfg['output']
    if args.verify or (out / 'completion_manifest.json').exists():
        print(json.dumps(verify(out), indent=2)); return
    assert cfg['kind'] == 'aishihik_source_reconciliation_v1'
    old = ROOT / cfg['previous_review']; previous_pins = verify_pins(old)
    nechako_pins = verify_pins(ROOT / cfg['nechako_review'])
    assert io.sha(ROOT / cfg['operator_pdf']) == cfg['operator_pdf_sha256']
    assert cfg['gauge_elevation_offset_m'] == curve.GAUGE_ELEVATION_OFFSET_M
    assert cfg['curve_interpretation'] == 'inferred_x_equals_elevation_minus_912_no_3600'
    assert not cfg['calibration_admitted'] and not cfg['certified_storage_conversion']
    out.mkdir(parents=True, exist_ok=True)
    source = old / 'prepared/etienne_Aishihik/source_candidates_si.csv'; frame = read_frame(source)
    receipts = io.read(old / 'stations/receipts.json')
    station_paths = []
    for station in ['08AA005', '08AA012', '08AA010', '08AA009']:
        path = old / 'stations' / (station + '.csv'); station_paths.append(path)
        assert io.sha(path) == receipts[station]['csv_sha256']
    q = frame.gauge_08AA010_discharge_m3s - frame.gauge_08AA009_discharge_m3s
    pd.testing.assert_series_equal(q, frame.candidate_release_m3s, check_names=False)
    q_for_targets = q.mask(q < 0)  # explicit exclusion of one unsupported physical outflow
    south = frame.gauge_08AA005_level_m; north = frame.gauge_08AA012_level_m
    difference = north - south
    gauge_annual = pd.DataFrame({'paired_days': difference.groupby(difference.index.year).count(),
        'median_north_minus_south_m': difference.groupby(difference.index.year).median(),
        'minimum_difference_m': difference.groupby(difference.index.year).min(),
        'maximum_difference_m': difference.groupby(difference.index.year).max()})
    gauge_annual.index.name = 'year'; gauge_annual.to_csv(out / 'gauge_comparison_by_year.csv')
    conversions = {}; conversion_proofs = {}; target_files = {}
    candidate = frame.copy(); candidate['negative_boundary_release_flag'] = q < 0
    candidate['release_candidate_for_descriptive_targets_m3s'] = q_for_targets
    monthly_mask = pd.Series(True, index=frame.index)
    for days in cfg['smoothing_candidates_days']:
        tag = f'trailing_{days}d'
        converted, proof = curve.incremental_storage_candidate(south, station_number=cfg['storage_gauge'],
            smoothing_days=days, allow_exploratory=cfg['allow_exploratory_curve_candidate'])
        conversions[tag] = converted; conversion_proofs[tag] = proof
        converted.to_csv(out / f'level_storage_candidate_{tag}.csv')
        candidate['incremental_storage_candidate_' + tag + '_m3'] = converted.incremental_storage_candidate_m3
        targets = pd.DataFrame(index=pd.Index(range(1, 13), name='month'), columns=MONTHLY_TARGET_COLUMNS, dtype=float)
        counts = pd.DataFrame(index=targets.index)
        for name, values, columns in [('storage', converted.incremental_storage_candidate_m3, MONTHLY_TARGET_COLUMNS[:3]),
                                     ('release', q_for_targets, MONTHLY_TARGET_COLUMNS[3:])]:
            estimates, n = empirical_targets(values, monthly_mask, monthly=True,
                percentiles=tuple(cfg['target_percentiles']), method='hazen')
            targets.loc[:, list(columns)] = estimates; counts[name + '_samples'] = n
        assert targets.notna().all().all() and (np.diff(targets.iloc[:, :3], axis=1) > 0).all()
        target_path = out / f'descriptive_targets_{tag}_si.csv'; targets.to_csv(target_path, float_format='%.17g')
        counts.to_csv(out / f'monthly_sample_counts_{tag}.csv')
        target_files[tag] = relative(target_path)
    # Every source/timing comparison uses the identical intersection of dates.
    # Negative outlet subtraction is retained in diagnostics, never clipped.
    budgets = pd.DataFrame(index=frame.index)
    storage_versions = {'archive': frame.storage_archive_m3,
        **{tag: converted.incremental_storage_candidate_m3 for tag, converted in conversions.items()}}
    for label, s in storage_versions.items():
        budgets[label + '__same_day_Q'] = s.diff() / 86400. + q
        budgets[label + '__adjacent_mean_Q'] = s.diff() / 86400. + (q + q.shift(1)) / 2
    common = budgets.notna().all(axis=1); summaries = []
    for name in budgets:
        sample = budgets.loc[common, name]
        summaries.append({'source_clock_scenario': name, 'common_intervals': int(common.sum()),
            'negative_net_flux_intervals': int((sample < 0).sum()), 'minimum_net_flux_m3s': float(sample.min()),
            'mean_net_flux_m3s': float(sample.mean()), 'negative_net_volume_m3': float(-sample[sample < 0].sum() * 86400),
            'selected_automatically': False, 'independent_inflow': False})
    summary = pd.DataFrame(summaries); summary.to_csv(out / 'budget_sensitivity_summary.csv', index=False)
    budgets['common_supported_interval'] = common; budgets.to_csv(out / 'signed_budget_diagnostics.csv')
    candidate['calibration_ready'] = False; candidate.to_csv(out / 'source_candidates_si.csv')
    # Show the candidate curve on its documented fitting domain, including the
    # small independently fitted branch jump. These are not surveyed points.
    elevation = np.linspace(912., 916., 401)
    grid = pd.Series(elevation - curve.GAUGE_ELEVATION_OFFSET_M, index=pd.date_range('2000-01-01', periods=len(elevation)))
    curve_table, _ = curve.incremental_storage_candidate(grid, station_number='08AA005', allow_exploratory=True)
    curve_table[['elevation_candidate_m', 'incremental_storage_candidate_m3', 'area_derivative_candidate_m2', 'branch']].to_csv(out / 'inferred_curve_candidate_si.csv', index=False)
    offset_years = gauge_annual.dropna()
    decision = {'reservoir_id': cfg['reservoir_id'], 'status': 'exploratory_storage_candidate_and_descriptive_targets_prepared',
        'calibration_ready': False, 'datum_offset_from_published_report_m': curve.GAUGE_ELEVATION_OFFSET_M,
        'gauge_datum_history_verified': False, 'curve_interpretation_verified': False,
        'author_clarification': 'Lower branch has a linear term and no intercept; reviewer-added 7657633.28 intercept is not adopted.',
        'polynomial_coordinate_inference': 'x = gauge_08AA005_level_m + 911.565 - 912; branch chosen by absolute elevation 915 m.',
        'branch_diagnostic': curve.branch_diagnostic(), 'factor_3600_applied': False,
        'volume_is_total_lake_storage': False, 'capacity_adopted_m3': None, 'dead_storage_adopted_m3': None,
        'south_gauge_supported_days': int(south.notna().sum()), 'archive_storage_days': int(frame.storage_archive_m3.notna().sum()),
        'negative_boundary_release_days': int((q < 0).sum()), 'negative_release_clipped': False,
        'gauge_annual_median_difference_first_m': float(offset_years.median_north_minus_south_m.iloc[0]),
        'gauge_annual_median_difference_last_m': float(offset_years.median_north_minus_south_m.iloc[-1]),
        'north_gauge_fitted_offset_used': False, 'conversion_proofs': conversion_proofs,
        'descriptive_target_files': target_files, 'target_fitting_scope': cfg['fitting_scope'],
        'existing_targets_replaced': False, 'source_gates': cfg['unresolved_gates'],
        'independent_inflow_available': False, 'next_action': (
            'Confirm the inferred reduced coordinate x=H-912 and m3 units without the printed 3600 factor; '
            'adopt capacity/dead storage on the same incremental coordinate. Keep 08AA005 and 08AA012 separate; '
            'freeze daily-mean timing, review the single signed outlet subtraction, then admit an explicitly '
            'conditional signed-net experiment or supply independent inflow. Refit targets on >=65% frozen mixed calibration years.')}
    io.write(out / 'source_decision.json', decision)
    # A new inventory version records the candidate without changing targets.
    base = pd.read_csv(ROOT / cfg['base_inventory'], float_precision='round_trip'); inventory = base.copy(deep=True)
    select = inventory.reservoir_id == cfg['reservoir_id']; assert select.sum() == 1
    inventory.loc[select, 'next_action'] = decision['next_action']
    inventory.loc[select, 'source_review_file'] = relative(out / 'source_decision.json')
    inventory.loc[select, 'source_candidate_file'] = relative(out / 'source_candidates_si.csv')
    pd.testing.assert_frame_equal(base.loc[~select], inventory.loc[~select])
    target_cols = [c for c in base if c.startswith(tuple(f'm{i:02d}_' for i in range(1, 13)))]
    assert len(target_cols) == 72 and len(base) == 62
    pd.testing.assert_frame_equal(base[target_cols], inventory[target_cols])
    revised = deepcopy(io.read(ROOT / cfg['base_catalog'])); revised['kind'] = 'reservoir_collection_index_v8'
    revised['previous_catalog'] = cfg['base_catalog']
    row = next(r for r in revised['reservoirs'] if r['reservoir_id'] == cfg['reservoir_id'])
    row.update(source_review_file=relative(out / 'source_decision.json'), source_admission_gates=cfg['unresolved_gates'],
        source_candidate_file=relative(out / 'source_candidates_si.csv'),
        exploratory_descriptive_targets=target_files, candidate_storage_source_gauge='08AA005',
        candidate_curve_interpretation_verified=False)
    catalog_path = ROOT / 'data/catalogs/reservoir_collection_v8_20261022.json'
    inventory_path = ROOT / 'data/catalogs/reservoir_inventory_v8_20261022.csv'
    assert not catalog_path.exists() and not inventory_path.exists()
    io.write(catalog_path, revised); inventory.to_csv(inventory_path, index=False, float_format='%.17g')
    reread = pd.read_csv(inventory_path, float_precision='round_trip')
    pd.testing.assert_frame_equal(base[target_cols], reread[target_cols])
    pd.testing.assert_frame_equal(base.loc[~select], reread.loc[~select])
    checks = {'previous_paired_review_pins_verified': previous_pins, 'nechako_review_pins_verified': nechako_pins,
        'new_optimizer_evaluations': 0, 'new_calibration_admissions': 0,
        'preserved_monthly_target_cells': 4464, 'unchanged_other_inventory_rows': 61,
        'inventory_roundtrip_verified': True, 'source_observations_overwritten': False,
        'hydrological_equations_changed': False, 'source_interpretation_explicitly_exploratory': True,
        'signed_budget_values_retained': True, 'same_dates_for_budget_comparison': True,
        'future_levels_used_in_trailing_average': False}
    io.write(out / 'scientific_checks.json', checks)
    plot(frame, conversions, gauge_annual, summary, out)
    write_report(out, decision, summary)
    files = [CONFIG, source, ROOT / cfg['operator_pdf'], ROOT / cfg['base_inventory'], ROOT / cfg['base_catalog'],
        old / 'evidence/receipts.json', old / 'evidence/aishihik_levels.html', old / 'stations/receipts.json',
        ROOT / 'workflows/aishihik_storage_candidate_v1.py', ROOT / 'tests/test_aishihik_storage_candidate_v1.py',
        ROOT / 'strategies/data_availability_v1.py', ROOT / 'docs/aishihik_source_reconciliation_v1.md',
        Path(__file__), inventory_path, catalog_path] + station_paths
    files += [p for p in out.rglob('*') if p.is_file()]
    io.write(out / 'completion_manifest.json', {'kind': cfg['kind'], 'previous_review': cfg['previous_review'],
        'nechako_review': cfg['nechako_review'], 'sha256': {relative(p): io.sha(p) for p in files},
        'scientific_checks': checks})
    print(json.dumps(verify(out), indent=2))


def plot(frame, conversions, gauge_annual, summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax = axes[0, 0]
    for station in ('08AA005', '08AA012'):
        sample = frame['gauge_' + station + '_level_m']; ax.plot(sample.index, sample.rolling(30, min_periods=25).mean(), label=station, lw=.8)
    ax.set(title='Separate lake gauges: native reference heights', ylabel='Gauge height (m), 30-day means'); ax.legend()
    ax = axes[0, 1]; paired = gauge_annual.dropna(); ax.plot(paired.index, paired.median_north_minus_south_m, marker='.')
    ax.set(title='Annual median north-minus-south difference', ylabel='Difference (m)', xlabel='Year')
    ax = axes[1, 0]
    for tag, conversion in conversions.items():
        s = conversion.incremental_storage_candidate_m3
        ax.plot(s.index, s / 1e6, label=tag, lw=.7)
    ax.set(title='Exploratory incremental volume above 912 m', ylabel='Candidate volume (million m3)'); ax.legend()
    ax = axes[1, 1]; ax.barh(summary.source_clock_scenario, summary.negative_net_flux_intervals)
    ax.set(title='Budget sensitivity on identical supported dates', xlabel='Negative response-derived net-flux intervals')
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle('Aishihik source candidates: no certified storage conversion or calibration admission')
    fig.savefig(out / 'aishihik_source_diagnostics.png', dpi=160)
    fig.savefig(out / 'aishihik_source_diagnostics.svg'); plt.close(fig)


def write_report(out, decision, summary):
    branch = decision['branch_diagnostic']; proofs = decision['conversion_proofs']; common = int(summary.common_intervals.iloc[0])
    text = f'''# Aishihik storage, outlet and timing preparation

**Prepared a guarded exploratory level-to-incremental-storage converter and two sets of descriptive monthly DZTR targets through the shared estimator. Calibration remains source-gated.** Existing storage, completed parameters, and Nechako's preceding review remain preserved.

[Plots](aishihik_source_diagnostics.png), [candidate curve](inferred_curve_candidate_si.csv), [source decision](source_decision.json), [budget comparison](budget_sensitivity_summary.csv), [one-day descriptive targets](descriptive_targets_trailing_1d_si.csv), [three-day descriptive targets](descriptive_targets_trailing_3d_si.csv), [updated inventory](../../data/catalogs/reservoir_inventory_v8_20261022.csv).

## What the source now clarifies

The combined [operator-hosted PDF](https://yukonenergy.ca/media/site_documents/AtkinsRealis_Hydrotechnical_Study_2024.pdf) contains an embedded INRS hydrological report. Printed page 44 ties southern lake gauge **08AA005** to elevation using **911.565 m**. The technical-review appendix, PDF pages 371-372, confirms the lower polynomial's missing linear term and specifies **no intercept**. The reviewer's proposed 7,657,633.28 m3 intercept is not adopted. The authors describe the fitted quantities as **incremental volume**, and report approximately 0.011% difference between independent branches at elevation 915 m.

Evaluating the polynomials using **x = elevation - 912 m** reproduces a **{branch['relative_difference_percent']:.6f}%** difference: {branch['lower_limit_m3']/1e6:.6f} versus {branch['upper_value_m3']/1e6:.6f} million m3. The small {branch['upward_jump_m3']:.2f} m3 upward jump remains unchanged. **This coordinate is an inference from numerical consistency, not an explicit operator-approved correction.** The main text's additional factor of 3600 remains ambiguous. The candidate omits it and records that choice explicitly. A time multiplier is not applied to a quantity interpreted as stored volume.

The converter therefore requires an explicit exploratory opt-in, accepts only 08AA005, and returns incremental volume candidates only over the documented **912-916 m fitting range**. It does not infer total lake volume, bed elevation, capacity, or the gauge datum history. Out-of-domain levels and gaps remain missing; there is no extrapolation or observed-maximum alignment.

## Why the two gauges remain separate

The archive contains **{decision['archive_storage_days']}** storage days. The southern gauge has **{decision['south_gauge_supported_days']}** valid daily levels on the review calendar. The northern gauge 08AA012 has its own reference. Its annual median height difference from the southern gauge changes from about **{decision['gauge_annual_median_difference_first_m']:.3f} m** in the first paired year to **{decision['gauge_annual_median_difference_last_m']:.3f} m** in the last. This may contain datum, instrument, spatial water-surface or other effects; the difference alone does not identify their cause. A single fitted gauge offset is not used to splice or convert those records.

The archived storage near 4.2-4.5 billion m3 and the new incremental-volume candidates use different coordinates. Their absolute values are not interchangeable. Adding or subtracting a constant cannot reconcile differing storage-curve slopes or fix daily budget errors.

## Timing and release treatment

Lake-outflow candidate = **08AA010 downstream river minus 08AA009 Giltana Creek**, as stated in the embedded report. Both components are required. The single negative subtraction on **2012-06-09** is retained in source/budget outputs. It is explicitly missing in the separate nonnegative release view used to estimate descriptive targets; it is neither clipped to zero nor silently repaired.

Two storage candidates are exported: conversion of daily mean level, and conversion of the **trailing three-day mean level** described in the source. The latter requires all three days and uses no future levels. Neither is certified endpoint storage. The conversions support **{proofs['trailing_1d']['converted_days']}** and **{proofs['trailing_3d']['converted_days']}** days respectively.

Six storage/clock combinations compare the archive and both new candidates, with current-day Q or adjacent-day mean Q. Every comparison uses the same **{common} intervals**. Signed net flux `dS/86400 + Q` remains a response-derived diagnostic, not independent inflow. No candidate is chosen by minimizing its negative count:

| Storage / Q timing | Common intervals | Negative net-flux intervals |
|---|---:|---:|
'''
    for _, row in summary.iterrows():
        text += f"| {row.source_clock_scenario} | {int(row.common_intervals)} | {int(row.negative_net_flux_intervals)} |\n"
    text += '''
## Parameterization and admission

Both candidate profiles now have twelve months of storage and release targets (P10/P45/P85, Hazen), generated with the existing shared `empirical_targets` estimator. They are **descriptive all-record estimates**, not calibration-fitted or validated parameter sets. They remain separate files and replace none of the active registry or inventory's monthly parameters. Before optimization, refit on frozen mixed calibration years with at least 65% usable fitting data.

Remaining decisions are specific: confirm the reduced polynomial coordinate and volume units; adopt capacity and dead storage on that same incremental coordinate; freeze the daily-mean/trailing-mean observation clock and the signed outlet-day treatment; and supply independent inflow or declare a conditional signed-net experiment. Daily net flux reconstructed from responses, including held-out responses, cannot provide independent predictive validation.

The v8 collection/inventory preserves all 4464 existing monthly target cells and every other 61 reservoir row. Nechako remains pending its elevation tie; Seton and Winnebago retain their existing gates. This stage runs zero optimizer evaluations.

Reproduce with `tools/Set-ProjectCache.ps1` and `python -B tools/review_aishihik_sources_v1.py`; use `--verify` to reuse completed artifacts. The source PDF and daily station responses were already cached and hash-verified. All observations, candidates, plots and inventories stay locally ignored by Git.
'''
    (out / 'README.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
