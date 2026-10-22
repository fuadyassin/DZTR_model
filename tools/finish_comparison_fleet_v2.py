"""Finish the independent fleet job after the checkpointed controller exits.

Run with --wait to adopt an already-running comparison_fleet.py execute job.
Without --wait, resume searches first. Assessment, reporting and regression
verification follow automatically; failures are persisted, never silently skipped.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from uuid import uuid4
from simulate_calibrated_reservoirs import ROOT, read, write, sha
from comparison_fleet_v2_job import OUT


def state(stage, **extra):
    write(OUT/'workflow_status.json', {'stage':stage,'pid':os.getpid(),'updated_utc':datetime.now(timezone.utc).isoformat(),**extra})
    print(stage, flush=True)


def command(script, *args):
    with (OUT/(Path(script).stem+'.log')).open('a',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'-B',str(ROOT/script),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:
        raise RuntimeError(f'{script} failed with exit {result.returncode}; see persistent log')


def integrity():
    protocol=read(OUT/'protocol.json')
    for rel,digest in {**protocol['source_sha256'],**protocol['code_sha256']}.items():
        if sha(ROOT/rel)!=digest:raise ValueError(f'Protocol source changed: {rel}')
    for rel,digest in protocol['prepared_sha256'].items():
        if sha(OUT/rel)!=digest:raise ValueError(f'Prepared inputs changed: {rel}')
    assert sha(OUT/'design.json')==protocol['design_sha256']
    previous=ROOT/'outputs/strategy_comparison_pilot_20261012'
    old=read(previous/'final_verification.json')
    for rel,digest in old['artifact_sha256'].items():
        assert sha(previous/rel)==digest,rel
    registry=read(ROOT/'outputs/twenty_reservoir_workflow_20261012/final_verification.json')
    for rel,digest in registry['source_sha256'].items():assert sha(ROOT/rel)==digest,rel


def regression():
    temp=ROOT/'outputs'/('pytest_comparison_fleet_full_'+uuid4().hex)
    assert temp.is_relative_to(ROOT/'outputs') and not temp.exists()
    args=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','--tb=short','--basetemp='+str(temp)]
    log=OUT/'full_regression.log'
    with log.open('w',encoding='utf-8') as stream:
        result=subprocess.run(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    text=log.read_text(encoding='utf-8')
    receipt={'exit_code':result.returncode,'summary':text.splitlines()[-1] if text else '', 'log':'full_regression.log','sha256':sha(log)}
    if result.returncode:
        write(OUT/'test_receipt.json',{'status':'failed','full_regression':receipt})
        raise RuntimeError('Full regression failed; preserve log and investigate before marking complete')
    receipt['passed']=int(re.search(r'(\d+) passed',text).group(1))
    match=re.search(r'(\d+) skipped',text);receipt['skipped']=int(match.group(1)) if match else 0
    write(OUT/'test_receipt.json',{'status':'passed','full_regression':receipt,'initial_focused':{'passed':5,'skipped':1,'duration_seconds':5.24}})
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    lock=OUT/'workflow.lock'
    # Exclusive creation prevents two finishers from launching the same stages.
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(os.getpid()))
    try:
        integrity()
        if args.wait:
            state('searches_running',new_jobs=162,new_search_evaluations=82944,check_interval_seconds=300)
            while not (OUT/'execution_summary.json').exists():time.sleep(300)
        else:
            state('searches_running');command('tools/comparison_fleet_v2.py','execute')
        if not read(OUT/'execution_summary.json')['all_completed']:
            raise RuntimeError('Some calibration jobs failed; inspect job logs and resume explicitly')
        integrity()
        if not (OUT/'comparison_scores.csv').exists():
            state('assessing_common_years');command('tools/assess_comparison_fleet_v2.py')
        state('generating_twenty_reservoir_report');command('tools/report_comparison_fleet_v2.py')
        state('running_regression_suite');receipt=regression()
        integrity()
        manifest=read(OUT/'assessment_manifest.json')
        for rel,digest in manifest['code_sha256'].items():assert sha(ROOT/rel)==digest,rel
        for rel,digest in manifest['artifact_sha256'].items():
            # Live workflow files are not frozen scientific experiment artifacts.
            if rel not in ('workflow_status.json','workflow.lock') and not rel.endswith('.log'):
                assert sha(OUT/rel)==digest,rel
        state('completed',profiles_promoted=False,tests=receipt)
        sources=['strategies/fleet_comparison_v2.py','tools/comparison_fleet_v2.py','tools/comparison_fleet_v2_job.py','tools/assess_comparison_fleet_v2.py','tools/report_comparison_fleet_v2.py','tools/finish_comparison_fleet_v2.py','tools/reuse_comparison_checkpoints.py','tests/test_fleet_comparison.py']
        write(OUT/'final_verification.json',{'status':'passed','scientific_checks':read(OUT/'scientific_verification.json'),'outcomes':read(OUT/'outcome_summary.json'),'tests':read(OUT/'test_receipt.json'),
              'prior_pilot_preserved':True,'prior_registry_sources_unchanged':True,'source_sha256':{p:sha(ROOT/p) for p in sources},
              'artifact_sha256':{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('final_verification.json','workflow.lock','finish_stdout.log','finish_stderr.log')}})
    except Exception as error:
        state('failed',reason=str(error));raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
