"""Prepare auditable boundary candidates, update the queue, and verify reuse.

No optimizer, model integration, parameter fitting or source overwrite occurs.
All affine fits are forensic diagnostics, excluded from model inputs.
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
from workflows import boundary_source_v1 as boundary
from workflows import reservoir_v2 as io

CONFIG = ROOT / 'configs/paired_source_review_v1_20261022.json'


def read_frame(path):
    return pd.read_csv(path, index_col=0, parse_dates=True, float_precision='round_trip')


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace('\\', '/')


def old_proofs():
    base = ROOT / 'outputs/minidoka_completion_v1_20261021'
    proof = io.read(base / 'completion_manifest.json')
    checked = 0
    for key, folder in [('source_sha256', ROOT), ('files_sha256', ROOT),
                        ('run_artifact_sha256', ROOT / 'outputs/minidoka_provider_interval_v1_20261021'),
                        ('artifact_sha256', base)]:
        for name, expected in proof[key].items():
            if io.sha(folder / name) != expected:
                raise ValueError('Previous completion changed: ' + name)
            checked += 1
    return checked


def verify(out):
    proof = io.read(out / 'completion_manifest.json')
    for name, expected in proof['sha256'].items():
        if io.sha(ROOT / name) != expected:
            raise ValueError('Source review changed: ' + name)
    return {'pinned_files_verified': len(proof['sha256']),
            'previous_completion_pins_verified': old_proofs(),
            'new_optimizer_evaluations': 0}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    cfg = io.read(CONFIG); out = ROOT / cfg['output']
    if args.verify or (out / 'completion_manifest.json').exists():
        print(json.dumps(verify(out), indent=2)); return
    assert cfg['kind'] == 'paired_extension_source_review_v1'
    stations = out / 'stations'; receipts = io.read(stations / 'receipts.json')
    input_paths = [CONFIG, ROOT / cfg['base_catalog'], ROOT / cfg['base_inventory']]
    for site, receipt in receipts.items():
        assert 'error' not in receipt, (site, receipt)
        assert io.sha(stations / (site + '.csv')) == receipt['csv_sha256']
        proof = receipt['receipt']
        for name, expected in proof['page_sha256'].items():
            candidates = list((stations / 'raw').glob('*/' + name))
            assert any(io.sha(path) == expected for path in candidates)
    checks = {'previous_completion_pins_verified': old_proofs(), 'new_optimizer_evaluations': 0,
              'source_observations_overwritten': False, 'hydrological_equations_changed': False,
              'new_calibration_admissions': 0, 'independent_validation_claimed': False,
              'forensic_fits_used_as_datum_or_parameters': False}
    comparisons = []; diagnostics = {}; prepared = {}; decisions = []
    levels = {s: boundary.read_usgs_daily(stations / ('usgs_' + s + '.json'), s, '00065')
              for s in ['04082500', '04084255']}
    appleton = boundary.read_usgs_daily(stations / 'usgs_04084445.json', '04084445', '00060',
                                       method_id=cfg['usgs_appleton_method_id'])
    for case in cfg['cases']:
        rid = case['reservoir_id']; name = rid.removeprefix('etienne_')
        source = ROOT / 'data/reservoirs_etienne_v1_20261021' / rid / 'observations_si.csv'
        input_paths.append(source); archive = read_frame(source)
        raw = pd.DataFrame(index=archive.index)
        candidate = pd.DataFrame({'storage_archive_m3': archive.storage_m3,
                                  'release_archive_m3s': archive.release_m3s,
                                  'inflow_m3s': np.nan,
                                  'candidate_release_m3s': np.nan}, index=archive.index)
        # Every gauge remains separate: lake levels with different datums are not added.
        for station in case['level_stations']:
            candidate['gauge_' + station + '_level_m'] = read_frame(stations / (station + '.csv')).level_m
        def comparison(label, left, right, tolerance):
            result = boundary.compare(left, right, tolerance)
            comparisons.append(dict(reservoir_id=rid, comparison=label, **result))
            return result
        if case['release_terms']:
            for term in case['release_terms']:
                s = term['station']; raw[s] = read_frame(stations / (s + '.csv')).discharge_m3s
                candidate['gauge_' + s + '_discharge_m3s'] = raw[s]
            corrected, expression = boundary.compose_discharge(raw, case['release_terms'])
            candidate['candidate_release_m3s'] = corrected
            diagnostics[rid] = {'candidate_expression': expression,
                                'alignment': cfg['candidate_flow_alignment'],
                                'candidate_adopted_for_calibration': False}
            # This is only a source-budget diagnostic, never an independent input.
            old_net = archive.storage_m3.diff() / 86400 + archive.release_m3s
            new_net = archive.storage_m3.diff() / 86400 + corrected
            common = old_net.notna() & new_net.notna()
            candidate['diagnostic_archive_net_flux_m3s'] = old_net
            candidate['diagnostic_candidate_net_flux_m3s'] = new_net
            diagnostics[rid]['common_interval_budget'] = {
                'common_intervals': int(common.sum()),
                'archive_negative_intervals': int((old_net[common] < 0).sum()),
                'candidate_negative_intervals': int((new_net[common] < 0).sum()),
                'source_storage_reference_unresolved': True,
                'timing_proxy_unverified': True, 'responses_enter_flux': True,
                'negative_net_flux_retained': True,
                'fewer_negative_intervals_certifies_boundary': False}
            comparison('archive_vs_candidate_boundary', archive.release_m3s, corrected, .0001)
        else:
            diagnostics[rid] = {'candidate_adopted_for_calibration': False,
                                'candidate_total_release_unavailable': True}
        if name == 'Aishihik':
            comparison('archive_vs_downstream_08AA010', archive.release_m3s, raw['08AA010'], .0001)
        elif name == 'Nechako':
            q = read_frame(stations / '08JA017.csv').discharge_m3s
            candidate['gauge_08JA017_downstream_discharge_m3s'] = q
            comparison('archive_vs_0.9_times_downstream_08JA017', archive.release_m3s, .9 * q, .0001)
            diagnostics[rid]['archive_0.9_scaling_cause_verified'] = False
        elif name == 'Seton':
            q = read_frame(stations / '08ME003.csv').discharge_m3s
            candidate['gauge_08ME003_river_only_discharge_m3s'] = q
            comparison('archive_vs_river_only_08ME003', archive.release_m3s, q, .0001)
            diagnostics[rid]['later_operator_storage_metadata'] = {
                'total_volume_m3': 2100e6, 'live_volume_before_1996_1999_m3': 22.6e6,
                'applies_directly_to_1958_1978_archive': False,
                'archive_is_established_total_or_live_storage': False}
        else:
            for station, level in levels.items():
                candidate['gauge_' + station + '_level_ft'] = level
            candidate['gauge_04084445_appleton_discharge_m3s'] = appleton * .028316846592
            source_path = ROOT / 'data/reservoirs_etienne_v1_20261021' / rid / 'observation_provenance.json'
            provenance = io.read(source_path); input_paths.append(source_path)
            original = Path(provenance['storage']['source_file'])
            assert io.sha(original) == provenance['storage']['source_sha256']
            # Preserve the overlooked field in the repo; analysis does not need the external drive again.
            copy = out / 'evidence/Winnebago_Storage_original.txt'
            if not copy.exists(): copy.write_bytes(original.read_bytes())
            raw_storage = pd.read_csv(copy, sep=r'\s+', header=None)
            raw_storage.index = pd.to_datetime(raw_storage[0].astype(str))
            extra = raw_storage.iloc[:, -1]
            candidate['unverified_archive_stage_proxy_m'] = extra
            x = pd.concat(levels, axis=1)
            diagnostics[rid]['archive_extra_field_relationship'] = boundary.affine_source_diagnostic(extra, x)
            comparison('archive_vs_0.9882352941176471_times_Appleton_SI', archive.release_m3s,
                       appleton * (5880 / 5950) * .028316846592, .006)
            diagnostics[rid]['archive_scaling'] = {
                'numerical_factor': 5880 / 5950,
                'physical_justification_and_historical_routing_verified': False,
                'selected_usgs_method_id': cfg['usgs_appleton_method_id'],
                'method_description': 'Primary Stream Flow; UpStream Location excluded explicitly'}
        case_dir = out / 'prepared' / rid; case_dir.mkdir(parents=True, exist_ok=True)
        candidate.to_csv(case_dir / 'source_candidates_si.csv', float_format='%.17g')
        assert candidate.inflow_m3s.isna().all()
        if not case['release_terms']: assert candidate.candidate_release_m3s.isna().all()
        pd.testing.assert_series_equal(candidate.storage_archive_m3, archive.storage_m3, check_names=False)
        pd.testing.assert_series_equal(candidate.release_archive_m3s, archive.release_m3s, check_names=False)
        prepared[rid] = candidate
        decisions.append(dict(case, calibration_ready=False, source_certified=False,
                              candidate_file=relative(case_dir / 'source_candidates_si.csv'),
                              storage_reference_estimated_from_extrema=False,
                              optimizer_evaluations=0))
    table = pd.DataFrame(comparisons); table.to_csv(out / 'source_comparisons.csv', index=False)
    io.write(out / 'source_diagnostics.json', diagnostics)
    io.write(out / 'source_decisions.json', {'kind': cfg['kind'], 'cases': decisions})

    # Update source knowledge in a new collection version without replacing a profile or target.
    catalog = io.read(ROOT / cfg['base_catalog']); revised = deepcopy(catalog)
    revised['kind'] = 'reservoir_collection_index_v6'
    revised['previous_catalog'] = cfg['base_catalog']
    revised['paired_source_review'] = relative(out / 'source_decisions.json')
    base_inventory = pd.read_csv(ROOT / cfg['base_inventory'], float_precision='round_trip')
    inventory = base_inventory.copy(deep=True)
    for case in decisions:
        rid = case['reservoir_id']; row = next(r for r in revised['reservoirs'] if r['reservoir_id'] == rid)
        assert row['calibration_ready'] is False
        row.update(source_review_file=relative(out / 'source_decisions.json'),
                   source_admission_gates=case['source_gates'], source_candidate_file=case['candidate_file'])
        select = inventory.reservoir_id == rid; assert select.sum() == 1
        inventory.loc[select, 'next_action'] = case['next_action']
        inventory.loc[select, 'source_review_file'] = relative(out / 'source_decisions.json')
        inventory.loc[select, 'source_candidate_file'] = case['candidate_file']
    unchanged = ~base_inventory.reservoir_id.isin(prepared)
    pd.testing.assert_frame_equal(base_inventory.loc[unchanged], inventory.loc[unchanged, base_inventory.columns])
    target_cols = [c for c in base_inventory if c.startswith('m0') or c.startswith('m1')]
    assert len(target_cols) == 72
    pd.testing.assert_frame_equal(base_inventory[target_cols], inventory[target_cols])
    assert len(inventory) == len(revised['reservoirs']) == 62
    checks.update(collection_rows=62, unchanged_other_inventory_rows=int(unchanged.sum()),
                  preserved_monthly_target_cells=62 * 72,
                  completed_extensions=4, source_gated_extensions=17)
    catalog_path = ROOT / 'data/catalogs/reservoir_collection_v6_20261022.json'
    inventory_path = ROOT / 'data/catalogs/reservoir_inventory_v6_20261022.csv'
    io.write(catalog_path, revised); inventory.to_csv(inventory_path, index=False, float_format='%.17g')
    plot(prepared, out)
    write_report(out, diagnostics, table)
    io.write(out / 'scientific_checks.json', checks)
    queue = pd.read_csv(ROOT / 'outputs/extension_source_intake_v1_20261021/remaining_17_source_queue.csv')
    for case in decisions:
        queue.loc[queue.reservoir_id == case['reservoir_id'], 'next_action'] = case['next_action']
    assert len(queue) == 17
    queue.to_csv(out / 'remaining_17_source_queue.csv', index=False)
    files = input_paths + [catalog_path, inventory_path,
        ROOT / 'workflows/boundary_source_v1.py', ROOT / 'tests/test_boundary_source_v1.py', Path(__file__),
        ROOT / 'tools/fetch_paired_stations_20261022.py', ROOT / 'tools/fetch_paired_source_evidence_20261022.py']
    files += [p for p in out.rglob('*') if p.is_file() and p.name != 'completion_manifest.json']
    io.write(out / 'completion_manifest.json', {'kind': 'paired_source_review_completion_v1',
             'sha256': {relative(p): io.sha(p) for p in files}, 'scientific_checks': checks})
    print(json.dumps(verify(out), indent=2))


def plot(prepared, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for ax, (rid, frame) in zip(axes.flat, prepared.items()):
        q = frame.release_archive_m3s; ax.plot(q.index, q.rolling(30, min_periods=25).mean(),
                                             label='Archived release (30-day mean)', linewidth=1)
        if frame.candidate_release_m3s.notna().any():
            q = frame.candidate_release_m3s
            ax.plot(q.index, q.rolling(30, min_periods=25).mean(), label='Candidate corrected boundary', linewidth=1)
        else:
            ax.text(.03, .93, 'Total lake outflow remains unresolved', transform=ax.transAxes, va='top', fontsize=9)
        ax.set(title=rid.removeprefix('etienne_'), ylabel='Discharge (m3/s)')
        ax.grid(alpha=.2); ax.legend(fontsize=8, loc='upper right'); ax.tick_params(axis='x', rotation=25)
    fig.suptitle('Source-boundary review: candidate expressions; no calibration admission')
    fig.savefig(out / 'boundary_comparison.png', dpi=160); fig.savefig(out / 'boundary_comparison.svg')
    plt.close(fig)


def write_report(out, diagnostic, table):
    get = lambda rid, label: table[(table.reservoir_id == rid) & (table.comparison == label)].iloc[0]
    a = get('etienne_Aishihik', 'archive_vs_downstream_08AA010')
    n = get('etienne_Nechako', 'archive_vs_0.9_times_downstream_08JA017')
    s = get('etienne_Seton', 'archive_vs_river_only_08ME003')
    w = get('etienne_Winnebago', 'archive_vs_0.9882352941176471_times_Appleton_SI')
    net = diagnostic['etienne_Nechako']['common_interval_budget']
    extra = diagnostic['etienne_Winnebago']['archive_extra_field_relationship']
    text = f'''# Four paired-reservoir source review

The next source-preparation stage is complete for Aishihik, Nechako, Seton and Winnebago. Official daily station data, raw response pages, hashes, separate gauge components and candidate boundary corrections are preserved. **No new calibration has been admitted:** the collection remains 62 reservoirs, with 4 completed exploratory Etienne extensions and 17 source-gated additions. This work used zero optimizer evaluations.

[Boundary comparison figure](boundary_comparison.png), [SVG](boundary_comparison.svg), [source comparisons](source_comparisons.csv), [decisions and gates](source_decisions.json), [budget/source diagnostics](source_diagnostics.json), [remaining 17 queue](remaining_17_source_queue.csv), [updated 62-row inventory](../../data/catalogs/reservoir_inventory_v6_20261022.csv), [collection v6](../../data/catalogs/reservoir_collection_v6_20261022.json).

| Reservoir | What the public records establish | Prepared action | Still required |
|---|---|---|---|
| Aishihik | Archived Q matches downstream gauge 08AA010 on {int(a.matching_days)} of {int(a.paired_days)} paired dates, within 0.0001 m3/s. | Candidate lake outflow = 08AA010 minus Giltana Creek 08AA009. Both lake-level gauges are retained separately. | A compatible storage curve/capacity, north/south gauge datum relationship and historical timing. |
| Nechako | Archived Q matches 0.9 times downstream Cheslatta Falls gauge 08JA017 on {int(n.matching_days)} of {int(n.paired_days)} paired dates, within 0.0001 m3/s. This numerical scaling does not establish its rationale. | Candidate total outflow = Skins Lake spillway 08JA013 plus Kemano powerhouse 08FE002. | Correct stage-storage coordinate and capacity, two-outlet timing and historical outlet completeness. |
| Seton | Archived Q matches river gauge 08ME003 on {int(s.matching_days)} of {int(s.paired_days)} paired dates, within 0.0001 m3/s. | Preserve river-only Q and lake gauge 08ME017; total-release candidate stays missing. | Canal/powerhouse discharge, storage reference/capacity and timing. |
| Winnebago | Archive matches a 0.9882352941176471 scaling of Appleton primary daily flow on {int(w.matching_days)} of {int(w.paired_days)} paired dates, within 0.006 m3/s. Two paired dates differ; current revisions are retained separately. | Preserve both level gauges and Appleton flow with explicit USGS method selection. | Lake versus connected-pool boundary, surveyed historical datums, a matching stage-storage curve and downstream routing/scaling. |

## Decisions that matter for the water balance

**Aishihik.** Yukon Energy's study explicitly subtracts the intervening creek from the downstream river gauge to estimate lake outflow. This is a source-boundary correction, not an optimizer choice. The study uses the Whitehorse lake gauge, while the archive also has support from the northern lake gauge; the two are not automatically interchangeable. Its printed stage-volume polynomial, absolute-stage branch thresholds and stated extra factor of 3600 need clarification before implementation. No polynomial is adopted here. [Operator study, printed pages 38 and 43-45](https://yukonenergy.ca/media/site_documents/AtkinsRealis_Hydrotechnical_Study_2024.pdf).

**Nechako.** A scaled downstream river record omits the westward diversion through Kemano. Federal assessment metadata distinguish diverted Kemano powerhouse flow from the natural river above the tailrace. On the same {net['common_intervals']} supported daily intervals, substituting the candidate two-outlet expression reduces negative response-derived budget intervals from {net['archive_negative_intervals']} to {net['candidate_negative_intervals']}. Every signed value is retained. This is a diagnostic of boundary sensitivity; it does not certify the archived storage curve, clocks or net forcing. The storage record around 30-32 billion m3 still cannot be paired with a 23.8 billion m3 catalog capacity without resolving their different coordinates. [ECCC station topology](https://wateroffice.ec.gc.ca/station_metadata/reference_index_e.html?stnIndex=2580), [federal assessment Table C-1](https://iaac-aeic.gc.ca/050/documents_staticpost/cearref_21799/2430/Marine_Physical_Environment.pdf).

**Seton.** The river and the power canal are different outlets. The archived river record cannot be treated as their sum. A later BC Hydro report distinguishes total lake volume of 2100 million m3 from live storage around 22-24 million m3, whereas the archive is about 132-155 million m3. Neither later figure establishes a replacement coordinate for the 1958-1978 storage archive. Changing only capacity would conceal this mismatch. [BC Hydro hydrology table 5](https://www.bchydro.com/content/dam/BCHydro/customer-portal/documents/corporate/environment-sustainability/water-use-planning/lower-mainland/brgmon-6-yr1-2015-03-31.pdf), [canal and river boundary description](https://www.bchydro.com/content/dam/BCHydro/customer-portal/documents/corporate/environment-sustainability/water-use-planning/lower-mainland/brgmon-13-yr8-2022-02-22.pdf).

**Winnebago.** On {extra['paired_days']} paired dates, the extra archive field is numerically equal to `3.7112 + 0.1524 * Oshkosh_gauge_ft + 0.1524 * Stockbridge_gauge_ft`; maximum affine residual is {extra['max_absolute_residual']:.3g} m. This identifies its construction, not a surveyed bottom elevation or true water depth. One missing gauge date remains unsupported. The fitted offset is never exported as a datum or used for parameterization. USGS returns both an upstream-location method and primary-stream-flow method for Appleton; method 226683 was selected explicitly, and conflicting location records were not merged. Historical USGS pool modeling also requires downstream routing; its one-day Rapide Croche approximation is evidence that timing matters, not a certified Appleton correction for this archive. [USGS daily source](https://waterservices.usgs.gov/nwis/dv/?format=json&sites=04084445&startDT=2007-10-01&endDT=2020-12-31&parameterCd=00060&statCd=00003), [USGS pool-routing study](https://pubs.usgs.gov/wri/1980/0107/report.pdf).

## How these outputs connect to the workflow

`prepared/<reservoir_id>/source_candidates_si.csv` keeps archived storage/release alongside individual station fields and explicitly named candidates. **Inflow remains missing.** A reconstructed budget diagnostic is response-dependent and is never relabeled as independent inflow. No component gaps are filled, no negative result is clipped, and missing canal discharge is never interpreted as zero.

The new collection and inventory reference these files and their source gates. All 62 x 72 existing target cells and all 58 other inventory rows are preserved. These source-candidate files deliberately distinguish `storage_archive_m3` and `candidate_release_m3s` from adopted model observations; they cannot silently become calibration inputs. Previous completed runs and their parameter registries remain unchanged.

**Next admission priority: Nechako, then Aishihik.** Their boundary expressions can now be prepared from public records. Resolve the stage-storage reference and matching capacity before fitting targets; then declare either independently sourced forcing or a conditional signed net-flux experiment. Seton additionally needs canal discharge; Winnebago needs the lake/pool and datum/routing reconciliation first. No mean-channel-flow series, arbitrary capacity inflation or extrema-based level reference is introduced.

Reproduce preparation after downloading the cached records with `python -B tools/review_paired_sources_v1.py`. Reuse/verify without downloads or model runs with `python -B tools/review_paired_sources_v1.py --verify`. Run `tools/Set-ProjectCache.ps1` first in the same PowerShell invocation. Source acquisition scripts and receipts are included in the completion manifest. [Scientific checks](scientific_checks.json).
'''
    (out / 'README.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
