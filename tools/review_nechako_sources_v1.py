"""Reconcile Nechako references and clocks using pinned local primary evidence.

No observation overwrite, new parameter profile or optimizer execution. The
previous review is immutable. Repeated runs verify completed artifacts only.
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
from workflows import reservoir_v2 as io
from workflows import source_clock_review_v1 as clock
from workflows.geometry_v1 import read_regeom

CONFIG = ROOT / 'configs/nechako_source_reconciliation_v1_20261022.json'


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_frame(path):
    return pd.read_csv(path, index_col=0, parse_dates=True, float_precision='round_trip')


def verify_previous(folder):
    manifest = io.read(folder / 'completion_manifest.json')
    for name, expected in manifest['sha256'].items():
        if io.sha(ROOT / name) != expected:
            raise ValueError('Previous source review changed: ' + name)
    return len(manifest['sha256'])


def verify(folder):
    proof = io.read(folder / 'completion_manifest.json')
    for name, expected in proof['sha256'].items():
        if io.sha(ROOT / name) != expected:
            raise ValueError('Nechako source review changed: ' + name)
    for name, expected in proof['external_sha256'].items():
        if io.sha(Path(name)) != expected:
            raise ValueError('External geometry source changed: ' + name)
    return {'pinned_files_verified': len(proof['sha256']),
            'external_files_verified': len(proof['external_sha256']),
            'previous_review_pins_verified': verify_previous(ROOT / proof['previous_review']),
            'new_optimizer_evaluations': 0, 'new_calibration_admissions': 0}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(); cfg = io.read(CONFIG); out = ROOT / cfg['output']
    if args.verify or (out / 'completion_manifest.json').exists():
        print(json.dumps(verify(out), indent=2)); return
    assert cfg['kind'] == 'nechako_source_reconciliation_v1'
    old = ROOT / cfg['previous_review']; previous_pins = verify_previous(old)
    evidence = out / 'evidence'; receipts = io.read(evidence / 'receipts.json')
    for name, receipt in receipts.items():
        if 'error' not in receipt and io.sha(evidence / name) != receipt['sha256']:
            raise ValueError('Cached primary evidence changed: ' + name)
    source = old / 'prepared/etienne_Nechako/source_candidates_si.csv'
    frame = read_frame(source); s = frame.storage_archive_m3
    components = frame[[f'gauge_{station}_discharge_m3s' for station in cfg['outlet_stations']]].copy()
    components.columns = cfg['outlet_stations']
    q = components.sum(axis=1, min_count=2)
    pd.testing.assert_series_equal(q, frame.candidate_release_m3s, check_names=False)
    meta = io.read(evidence / 'wsc_level_station.json')['features']
    assert len(meta) == 1 and meta[0]['properties']['STATION_NUMBER'] == cfg['level_station']
    datum = meta[0]['properties']['VERTICAL_DATUM']
    reference = cfg['operator_reference']
    assert reference['gauge_to_operator_elevation_offset_m'] is None
    audit = clock.capacity_diagnostic(s, reference['storage_at_full_supply_m3'])
    curve = read_regeom(cfg['regeom_curve_comparison_only'], expected_grand_id=15)
    geometry_audit = clock.capacity_diagnostic(s, curve.storage_m3[-1])
    paired = frame[['gauge_08JA023_level_m', 'storage_archive_m3']].dropna()
    diagnostic_fit = np.polynomial.Polynomial.fit(paired.iloc[:, 0], paired.iloc[:, 1], 1)
    fit_residual = paired.iloc[:, 1] - diagnostic_fit(paired.iloc[:, 0])
    scenarios = [dict(name='same_date_daily_mean', component_shifts_days={'08JA013': 0, '08FE002': 0}, quadrature='current_day'),
                 dict(name='adjacent_mean_to_mean_proxy', component_shifts_days={'08JA013': 0, '08FE002': 0}, quadrature='adjacent_day_average')]
    for spill_shift in cfg['timing_review']['component_shifts_days']:
        for power_shift in cfg['timing_review']['component_shifts_days']:
            if spill_shift == power_shift == 0:
                continue
            scenarios.append(dict(name=f'spill_shift_{spill_shift:+d}_power_shift_{power_shift:+d}',
                component_shifts_days={'08JA013': spill_shift, '08FE002': power_shift}, quadrature='current_day'))
    intervals, sensitivity = clock.budget_clock_sensitivity(s, components, scenarios)
    intervals.to_csv(out / 'clock_sensitivity_intervals.csv')
    sensitivity.to_csv(out / 'clock_sensitivity_summary.csv', index=False)
    io.write(out / 'clock_scenarios.json', scenarios)
    net = s.diff() / 86400. + q
    negatives = frame.loc[net < 0].copy(); negatives['net_flux_m3s'] = net[net < 0]
    negatives.to_csv(out / 'negative_budget_intervals.csv')
    anomaly = cfg['outlet_anomaly_review']; power = components[anomaly['station']]
    local_median = power.rolling(anomaly['rolling_days'], min_periods=anomaly['minimum_supported_days'], center=True).median()
    flagged = (power > anomaly['multiple_of_local_median'] * local_median) & (local_median > 0)
    flags = pd.DataFrame({'kemano_provider_m3s': power, 'rolling_local_median_m3s': local_median,
                          'ratio_to_local_median': power / local_median, 'review_flag': flagged})
    flags.loc[flagged].to_csv(out / 'outlet_review_flags.csv')
    # Candidates deliberately keep the archive coordinate and do not expose a
    # certified model observations schema. Unknown storage conversion is blank.
    prepared = frame.copy(); prepared['storage_operator_reference_candidate_m3'] = np.nan
    prepared['kemano_review_flag'] = flagged
    prepared['release_candidate_excluding_flagged_m3s'] = q.mask(flagged)
    prepared['candidate_ready_for_calibration'] = False
    prepared.to_csv(out / 'source_candidates_si.csv')
    annual = pd.DataFrame(index=sorted(set(frame.index.year)))
    for name, values in {'level_days': frame.gauge_08JA023_level_m.notna(), 'storage_days': s.notna(),
                         'two_outlet_days': q.notna(), 'supported_budget_intervals': net.notna(),
                         'negative_budget_intervals': net < 0, 'kemano_review_flags': flagged}.items():
        annual[name] = values.groupby(frame.index.year).sum()
    annual.index.name = 'year'; annual.to_csv(out / 'annual_support.csv')
    gates = ['historical_08JA023_to_operator_elevation_reference', 'numeric_operator_stage_storage_curve_on_verified_reference',
             'daily_mean_storage_and_two_outlet_clock_proxy_decision', 'kemano_2016_12_14_to_18_provider_value_review',
             'independent_inflow_or_explicit_conditional_signed_net_forcing']
    next_action = ('Obtain the historical 08JA023 gauge-zero/elevation tie and numeric Rio Tinto stage-storage curve; '
                   'replace the archived storage conversion in a new profile, then reconcile daily-mean timing and '
                   'Kemano 2016-12-14 to 2016-12-18 flags. Capacity at full supply is documented as 23.85e9 m3; '
                   'do not inflate capacity or offset storage to admit the archive. Keep two-outlet release and conditional-forcing labels.')
    decision = {'reservoir_id': cfg['reservoir_id'], 'status': 'source_gated_reference_identified',
        'calibration_ready': False, 'existing_profile_replaced': False,
        'documented_operator_reference': reference, 'wsc_vertical_datum': datum,
        'archive_vs_operator_capacity': audit, 'archive_vs_regeom_curve_top': geometry_audit,
        'regeom_curve_sha256': curve.source_sha256, 'archive_level_forensics_only': {
            'paired_days': len(paired), 'slope_m3_per_m': float(diagnostic_fit.convert().coef[1]),
            'maximum_affine_residual_m3': float(abs(fit_residual).max()),
            'fit_used_as_datum_or_parameter': False},
        'daily_mean_level_is_endpoint_storage': False, 'clock_selected_from_diagnostics': False,
        'raw_budget_intervals': int(net.notna().sum()), 'raw_negative_budget_intervals': int((net < 0).sum()),
        'outlet_review_flag_days': int(flagged.sum()), 'outlet_flags_certify_provider_error': False,
        'flagged_values_rescaled_or_clipped': False, 'inflow_independent': False,
        'source_gates': gates, 'next_action': next_action}
    io.write(out / 'source_decision.json', decision)
    # Version the queue while preserving every existing parameter cell and all
    # other reservoirs. Documented capacity is separate from adopted capacity.
    base = pd.read_csv(ROOT / cfg['base_inventory'], float_precision='round_trip')
    inventory = base.copy(deep=True); select = inventory.reservoir_id == cfg['reservoir_id']; assert select.sum() == 1
    inventory.loc[select, 'next_action'] = next_action
    inventory.loc[select, 'source_review_file'] = relative(out / 'source_decision.json')
    inventory.loc[select, 'source_candidate_file'] = relative(out / 'source_candidates_si.csv')
    pd.testing.assert_frame_equal(base.loc[~select], inventory.loc[~select])
    parameter_columns = [c for c in base if c.startswith(tuple(f'm{i:02d}_' for i in range(1, 13)))]
    assert len(parameter_columns) == 72
    pd.testing.assert_frame_equal(base[parameter_columns], inventory[parameter_columns])
    catalog = io.read(ROOT / cfg['base_catalog']); revised = deepcopy(catalog)
    revised['kind'] = 'reservoir_collection_index_v7'; revised['previous_catalog'] = cfg['base_catalog']
    row = next(r for r in revised['reservoirs'] if r['reservoir_id'] == cfg['reservoir_id'])
    row.update(source_review_file=relative(out / 'source_decision.json'), source_admission_gates=gates,
               source_candidate_file=relative(out / 'source_candidates_si.csv'),
               documented_operator_capacity_m3=reference['storage_at_full_supply_m3'],
               operator_capacity_adopted_for_archived_storage=False)
    inventory_path = ROOT / 'data/catalogs/reservoir_inventory_v7_20261022.csv'
    catalog_path = ROOT / 'data/catalogs/reservoir_collection_v7_20261022.json'
    assert not inventory_path.exists() and not catalog_path.exists()
    inventory.to_csv(inventory_path, index=False, float_format='%.17g'); io.write(catalog_path, revised)
    checks = {'previous_review_pins_verified': previous_pins, 'collection_rows': len(inventory),
              'unchanged_other_inventory_rows': int((~select).sum()), 'preserved_monthly_target_cells': len(base) * 72,
              'new_optimizer_evaluations': 0, 'new_calibration_admissions': 0,
              'source_observations_overwritten': False, 'hydrological_equations_changed': False,
              'signed_residuals_retained': True, 'identical_clock_comparison_support': True,
              'observed_maximum_used_to_set_capacity_or_datum': False}
    io.write(out / 'scientific_checks.json', checks)
    plot(frame, sensitivity, flags, reference, out)
    write_report(out, decision, sensitivity, flags.loc[flagged])
    files = [CONFIG, source, ROOT / cfg['base_catalog'], ROOT / cfg['base_inventory'], inventory_path, catalog_path,
             ROOT / 'workflows/source_clock_review_v1.py', ROOT / 'tests/test_source_clock_review_v1.py',
             Path(__file__), ROOT / 'tools/fetch_nechako_reference_v1.py', ROOT / 'docs/nechako_source_reconciliation_v1.md']
    files += [p for p in out.rglob('*') if p.is_file()]
    io.write(out / 'completion_manifest.json', {'kind': cfg['kind'], 'previous_review': cfg['previous_review'],
        'sha256': {relative(p): io.sha(p) for p in files},
        'external_sha256': {cfg['regeom_curve_comparison_only']: curve.source_sha256}, 'scientific_checks': checks})
    print(json.dumps(verify(out), indent=2))


def plot(frame, sensitivity, flags, reference, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    axes[0].plot(frame.index, frame.storage_archive_m3 / 1e9, lw=.8, label='Archived estimated storage (unresolved coordinate)')
    axes[0].axhline(reference['storage_at_full_supply_m3'] / 1e9, color='tab:red', ls='--', label='Operator full-supply volume, separate coordinate')
    axes[0].set(ylabel='Storage (billion m3)', title='Nechako: archived storage cannot be paired with operator capacity')
    axes[0].legend(fontsize=9)
    part = flags.loc['2016-12-01':'2016-12-31']; axes[1].plot(part.index, part.kemano_provider_m3s, marker='.', label='Kemano provider values, unchanged')
    marked = part[part.review_flag]; axes[1].scatter(marked.index, marked.kemano_provider_m3s, color='tab:red', label='Local-median review flag')
    axes[1].set(ylabel='Discharge (m3/s)', title='Five December 2016 provider records require review; no divide-by-ten correction')
    axes[1].legend(fontsize=9)
    axes[2].barh(sensitivity.scenario, sensitivity.negative_intervals)
    axes[2].set(xlabel='Negative response-derived net-flux intervals on identical dates', title='Clock sensitivity does not select or certify a physical clock')
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(out / 'nechako_source_diagnostics.png', dpi=160)
    fig.savefig(out / 'nechako_source_diagnostics.svg'); plt.close(fig)


def write_report(out, decision, sensitivity, flagged):
    a = decision['archive_vs_operator_capacity']; common = int(sensitivity.common_intervals.iloc[0])
    text = f'''# Nechako storage-reference and timing reconciliation

**Decision: preserve the two-outlet boundary candidate, replace the archived storage conversion only after its gauge reference is established, and keep calibration gated.** No new optimizer evaluations or parameter profiles were produced.

[Plots](nechako_source_diagnostics.png), [source decision](source_decision.json), [clock comparison](clock_sensitivity_summary.csv), [signed budget intervals](negative_budget_intervals.csv), [Kemano flags](outlet_review_flags.csv), [annual support](annual_support.csv), [new 62-row inventory](../../data/catalogs/reservoir_inventory_v7_20261022.csv).

## What is now established

The [Rio Tinto reference](https://www.getinvolvednechako.ca/wp-content/uploads/2021/11/Nechako-Reservoir-Licensed-Minimum-Water-Level-20_October_2020-WEI-M15-9_December_2020.pdf), dated 20 October 2020, distinguishes **23.85 billion m3 at 2800 ft (853.44 m)** from **7.10 billion m3 of licensed live storage**. The licensed minimum is about 2769.5 ft, corresponding to **16.75 billion m3**. The licensed minimum is not automatically the model's physical dead-storage threshold. The two-page document includes an operator stage-storage curve. Its endpoints establish reference quantities; they are not a numeric interpolation table.

The archived storage spans **{a['minimum_m3']/1e9:.3f}–{a['maximum_m3']/1e9:.3f} billion m3**; **all {a['valid_days']} valid storage days exceed the operator's full-supply volume**. The local ReGeom candidate curve for GRanD 15 also ends near 23.66 billion m3. Thus capacity inflation or replacing 23.8 with 32 billion m3 is unsupported. A constant storage offset also leaves every storage difference and reconstructed flux unchanged. No offset, clipping or rescaling is applied.

[Official WSC station metadata](https://api.weather.gc.ca/collections/hydrometric-stations/items?STATION_NUMBER=08JA023&f=json) identifies the level gauge as **{decision['wsc_vertical_datum']}**. Its approximately 9–13 m readings are heights above a local reference, not absolute elevations or reservoir depths. The required conversion is `operator_elevation_m = gauge_level_m + documented_offset_m`, followed by the operator's volume curve. The offset remains blank: neither maximum observed level, catalog dam height nor an affine fit to archived storage supplies a surveyed elevation tie. A current metadata label also does not prove a datum remained unchanged during 1993–2021.

## Outlet and clock review

The candidate total outlet remains **Skins Lake Spillway 08JA013 + Kemano Powerhouse 08FE002**, requiring both components. An [operator-commissioned naturalization study](https://www.getinvolvednechako.ca/wp-content/uploads/2022/08/Ecofish-Nechako-Flow-Naturalization-20221223-cert.pdf), sections 2 and 3.2, confirms these two outlet pathways and explains attenuation and travel through the Cheslatta system. Its downstream routing approximation is not a clock correction to apply to the reservoir boundary; its naturalized Vanderhoof hydrograph is not an independent reservoir-inflow record. The report mentions Rio Tinto daily reservoir inflow as an upstream source, but that numeric record is not supplied here.

The unchanged archival-storage/current-day outlet budget has **{decision['raw_negative_budget_intervals']} negative intervals out of {decision['raw_budget_intervals']} supported intervals**. The earlier 1830-to-15 comparison used 4735 dates shared with the old downstream release; this count uses all available corrected-boundary intervals. These are response-derived net fluxes, never independent inflow.

Ten predefined timing scenarios compare same-date daily Q, an adjacent-day mean-to-mean proxy, and separate outlet shifts of -1/0/+1 days. **All scenarios use the same {common} dates**. Negative counts range from **{int(sensitivity.negative_intervals.min())} to {int(sensitivity.negative_intervals.max())}**. No scenario is selected. Differences of daily mean level-derived storage do not equal observed midnight-to-midnight storage changes; the adjacent-day proxy is an assumption, not a certification. Every negative value and gap remains available in the interval output.

Kemano has **{len(flagged)} local-median review flags**. These are forensic quality flags, not proof of provider error or an engineering discharge limit:

| Date | Provider discharge (m3/s) | Nearby 31-day median (m3/s) |
|---|---:|---:|
'''
    for day, row in flagged.iterrows():
        text += f'| {day.date()} | {row.kemano_provider_m3s:g} | {row.rolling_local_median_m3s:g} |\n'
    text += '''
The records are retained unchanged. A separate candidate excludes flagged dates as missing; it is not adopted for calibration, and no divide-by-ten repair is inferred.

## Admission requirements and workflow connection

1. Obtain the historical gauge-zero/elevation tie for 08JA023, including any datum changes, and the operator's numeric stage-storage table. Regenerate storage in a new, explicitly labelled observation profile.
2. Decide and freeze a daily observation/interval clock and treatment of the five Kemano flags using provider evidence. Recompute signed net flux after changing the storage curve; the present 15-negative count need not survive a slope change.
3. Supply independent reservoir inflow, or explicitly admit a conditional signed-net-flux experiment. Such an experiment uses storage and release responses in forcing, including assessment data, and cannot demonstrate independent predictive validation.
4. Only after admission, refit monthly targets and bounds on frozen mixed calibration years (at least 65% usable fitting data), run the established checkpointed calibration and export a distinct profile. Preserve current descriptive and completed calibrated profiles.

The versioned v7 catalog/inventory changes only Nechako's review links and next action; all 62 × 72 target cells and all other 61 inventory rows remain unchanged. No GRanD mapping is promoted solely by a name/curve match. Aishihik, Seton and Winnebago retain their previous source gates and priorities.

Reproduce with `tools/Set-ProjectCache.ps1` then `python -B tools/review_nechako_sources_v1.py`; after completion use `--verify` to reuse the frozen artifacts without downloads or optimization. Evidence acquisition is separate in `tools/fetch_nechako_reference_v1.py`. All raw records, plots, receipts and inventories remain locally ignored by Git.
'''
    (out / 'README.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
