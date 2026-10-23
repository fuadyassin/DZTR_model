"""Reproduce a curated aggregate assessment from verified cached results only."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from dztr.portable import file_digest, write_json

METHODS = ('observed_monthly', 'flow_only', 'storage_only', 'ungauged_mean_flow')


def summarize(source, output):
    source, out = Path(source).resolve(), Path(output).resolve()
    if out.exists():
        raise ValueError('Use a new assessment output directory')
    manifest = json.loads((source / 'manifest.json').read_text())
    for key in ('code_sha256', 'artifact_sha256'):
        for name, expected in manifest.get(key, {}).items():
            path = (ROOT if key == 'code_sha256' else source) / name
            if file_digest(path) != expected:
                raise ValueError('Archived assessment changed: ' + name)
    scores = pd.read_csv(source / 'all_route_phase_scores.csv', float_precision='round_trip')
    archived = pd.read_csv(source / 'summary.csv', float_precision='round_trip')
    held = scores.loc[(scores.phase == 'assessment') & scores.requested_benchmark_reservoir]
    success = held.loc[(held.status == 'completed') & held.storage_nse.notna() & held.release_nse.notna()]
    sets = [set(success.loc[success.method == method, 'reservoir_id']) for method in METHODS]
    common = set.intersection(*sets)
    rows = []
    for scope in ('available_cases', 'common_successful_cases'):
        for method in METHODS:
            group = success.loc[success.method == method]
            if scope == 'common_successful_cases':
                group = group.loc[group.reservoir_id.isin(common)]
            rows.append({'scope': scope, 'method': method,
                'requested': 32 if scope == 'available_cases' else len(common),
                'completed': len(group), 'storage_scored': len(group), 'release_scored': len(group),
                'median_storage_nse': group.storage_nse.median(),
                'median_release_nse': group.release_nse.median(),
                'both_nse_positive': int(((group.storage_nse > 0) & (group.release_nse > 0)).sum())})
    summary = pd.DataFrame(rows)
    pd.testing.assert_frame_equal(summary, archived, check_dtype=False, atol=1e-14, rtol=0)
    out.mkdir(parents=True)
    summary.to_csv(out / 'route_summary.csv', index=False, float_format='%.17g')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matched = summary.loc[summary.scope == 'common_successful_cases']
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    labels = ['Storage + release', 'Release only', 'Storage only', 'Neither response']
    for ax, column, title in zip(axes, ('median_storage_nse', 'median_release_nse'), ('Storage', 'Release')):
        values = matched[column].to_numpy()
        ax.barh(labels[::-1], values[::-1], color=['#526777', '#ad7444', '#4e9087', '#326ba8'][::-1])
        ax.axvline(0, color='black', lw=.8)
        ax.set(title=title, xlabel='Median held-out NSE; same 27 reservoirs')
        for y, value in enumerate(values[::-1]):
            ax.text(value, y, f' {value:.3f} ', va='center', ha='left' if value >= 0 else 'right')
        ax.margins(x=.3); ax.grid(axis='x', alpha=.2)
    fig.suptitle('Data availability matters: conditional retrospective comparison')
    fig.savefig(out / 'route_comparison.png', dpi=160); plt.close(fig)
    write_json(out / 'assessment_receipt.json', {'kind': 'dztr_curated_assessment_v1',
        'source_manifest_sha256': file_digest(source / 'manifest.json'),
        'source_scores_sha256': file_digest(source / 'all_route_phase_scores.csv'),
        'archived_summary_exactly_reproduced': True, 'common_reservoir_count': len(common),
        'aggregate_only_no_observation_series': True, 'new_optimizer_evaluations': 0,
        'new_model_simulations': 0, 'summary_sha256': file_digest(out / 'route_summary.csv')})
    print(json.dumps({'common_cases': len(common), 'summary_reproduced': True, 'output': str(out)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='outputs/managed_availability_paper_v1_20261020_final/stages/comparison/campaign')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    summarize(args.source, args.output)
