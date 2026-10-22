"""Paper-first exploratory preparation and sequential reservoir calibration jobs."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from workflows import paper_exploration as exploration
from workflows import reservoir_v2 as w

def run(output):
    out=Path(output).resolve(strict=True)
    if not out.is_relative_to(ROOT/'outputs'):raise ValueError('Output must be inside repository outputs')
    review=w.read(out/'review.json')
    for rel,digest in review['source_sha256'].items():
        if w.sha(ROOT/rel)!=digest:raise ValueError('Exploratory controller changed; use a new version')
    status_path=out/'execution.json'
    completed=[]
    lock=out/'study.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(__import__('os').getpid()))
    try:
        for job in review['ready_jobs']:
            folder=ROOT/job['output'];status=folder/'status.json'
            if status.exists() and w.read(status).get('stage')=='completed':
                registry=folder/'releases/budget_512/registry.json';w.verify_registry(registry)
                completed.append({'reservoir_id':job['reservoir_id'],'exit_code':0,'reused':True})
            else:
                with (folder.parent/'execution.log').open('a',encoding='utf-8') as stream:
                    result=subprocess.run([sys.executable,'-B',str(ROOT/'tools/reservoir_workflow_v2.py'),'run',
                        '--config',str(ROOT/job['config']),'--output',str(folder)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                completed.append({'reservoir_id':job['reservoir_id'],'exit_code':result.returncode,'reused':False})
            w.write(status_path,{'jobs':completed,'total_reservoirs':len(review['ready_jobs']),
                                'completed':sum(j['exit_code']==0 for j in completed),'stage':'running'})
            print(completed[-1],flush=True)
        w.write(status_path,{'jobs':completed,'total_reservoirs':len(review['ready_jobs']),
                            'completed':sum(j['exit_code']==0 for j in completed),
                            'stage':'completed' if all(j['exit_code']==0 for j in completed) else 'failures_logged'})
    finally:lock.unlink(missing_ok=True)
    return status_path

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run']);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(exploration.prepare(args.output) if args.stage=='prepare' else run(args.output))
