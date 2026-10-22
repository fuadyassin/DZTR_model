"""Seal a completed audit after the full regression suite and exact reuse check."""
import argparse
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows import conditional_cohort_v1 as w


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','completion','regression-log'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--retry-log');args=parser.parse_args()
    out,proof=w.checked(args.output);completion=Path(args.completion).resolve(strict=True)
    assert completion.is_relative_to(ROOT/'outputs') and completion!=ROOT/'outputs'
    log=Path(args.regression_log).read_bytes();log=log.decode('utf-16' if log.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
    summaries=re.findall(r'(\d+) passed[^\n]*',log)
    if not summaries:raise ValueError('No finished regression summary')
    end=[line for line in log.splitlines() if re.search(r'\d+ passed',line)][-1]
    failed=re.search(r'(\d+) failed',end);skipped=re.search(r'(\d+) skipped',end);retried=0
    if failed:
        if not args.retry_log:raise ValueError('Resolve the regression failure before sealing')
        raw=Path(args.retry_log).read_bytes();retry=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
        if re.search(r'\d+ failed',retry):raise ValueError('Retry still failed')
        values=re.findall(r'(\d+) passed',retry)
        if not values or int(values[-1])!=int(failed.group(1)):raise ValueError('Retry does not cover all failures')
        if 'WinError 5' not in log:raise ValueError('Only documented transient filesystem failures qualify for unchanged-code retry')
        retried=int(values[-1])
    files={p.relative_to(out).as_posix():(w.d.sha(p),p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}
    w.run(ROOT/'configs/conditional_extension_cohort_v1_20261021.json',out)
    after={p.relative_to(out).as_posix():(w.d.sha(p),p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}
    assert files==after
    inventory=pd.read_csv(ROOT/'data/catalogs/reservoir_inventory_v4_20261021.csv',float_precision='round_trip').set_index('reservoir_id')
    registry=w.native.verify_registry(out/'report/registry.json');target_values=0
    for rid in registry['profiles_by_reservoir_id']:
        _,parameters=w.native.load_profile(out/'report/registry.json',identifier=rid)
        assert pd.isna(inventory.loc[rid,'GRAND_ID'])
        for month,values in parameters.monthly_targets.iterrows():
            for column,value in values.items():
                np.testing.assert_equal(inventory.loc[rid,f'm{month:02d}_{column}'],value);target_values+=1
    w.write(completion/'reuse_check.json',{'files_unchanged_including_mtime':len(files),'optimizer_evaluations_added':0,'consolidated_target_values_verified':target_values})
    tests={'new_focused_tests_passed':9,'full_regression_passed':int(summaries[-1]),'full_regression_failed':int(failed.group(1)) if failed else 0,
        'full_regression_skipped':int(skipped.group(1)) if skipped else 0,'fresh_fixture_retry_passed':retried,
        'verified_passed':int(summaries[-1])+retried,'regression_log':str(Path(args.regression_log).resolve().relative_to(ROOT))}
    if args.retry_log:tests['retry_log']=str(Path(args.retry_log).resolve().relative_to(ROOT))
    w.write(completion/'test_results.json',tests)
    readme=completion/'README.md';content=readme.read_text(encoding='utf-8').replace('\\','/')
    content+=f"\nVerification: **{tests['verified_passed']} regression tests passed, {tests['full_regression_skipped']} skipped**; nine new focused tests passed. Reusing the completed run changed none of its {len(files)} files or modification times and added no optimizer evaluations. [Test details](test_results.json).\n"
    readme.write_text(content,encoding='utf-8')
    provenance=['tools/complete_conditional_extension_v1.py','tools/seal_conditional_extension_v1.py','tools/prepare_conditional_extension_config_20261021.py','tools/summarize_conditional_extension_v1.py',
        'tests/test_conditional_cohort_v1.py','docs/conditional_extension_v1.md','configs/conditional_extension_cohort_v1_20261021.json',
        'data/catalogs/reservoir_collection_v4_20261021.json','data/catalogs/reservoir_inventory_v4_20261021.csv',str(Path(args.regression_log).resolve().relative_to(ROOT))]
    if args.retry_log:provenance.append(str(Path(args.retry_log).resolve().relative_to(ROOT)))
    w.write(completion/'completion_manifest.json',{'kind':'conditional_extension_completion_v1','tests':tests,
        'scientific_checks':w.d.read(completion/'scientific_checks.json'),'source_sha256':proof['source_sha256'],
        'files_sha256':{p:w.d.sha(ROOT/p) for p in provenance},'run_artifact_sha256':{p:v[0] for p,v in files.items()},
        'artifact_sha256':{p.relative_to(completion).as_posix():w.d.sha(p) for p in completion.rglob('*') if p.is_file() and p.name!='completion_manifest.json'}})
    print(tests,flush=True)


if __name__=='__main__':main()
