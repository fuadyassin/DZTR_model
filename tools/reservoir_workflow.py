"""One configurable entry point for DZTR calibration, export and simulation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from workflows import reservoir as workflow


def run(config_path, output, budget=None):
    path,cfg=workflow.config(config_path);out=Path(output).resolve()
    if out.exists():
        _,_,prepared=workflow.check_protocol(out)
        if prepared!=cfg:raise ValueError('Configuration differs from prepared workflow')
    else:out=workflow.prepare(path,out)
    budget=cfg['optimization']['evaluation_budget'] if budget is None else budget
    jobs=[(e['reservoir_id'],seed) for e in cfg['reservoirs'] for seed in cfg['optimization']['seeds']]
    def execute(job):
        identifier,seed=job;folder=out/'jobs'/identifier/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
        args=[sys.executable,'-B',str(Path(__file__).resolve()),'calibrate','--output',str(out),'--reservoir-id',identifier,'--seed',str(seed),'--evaluation-budget',str(budget)]
        with (folder/'execution.log').open('a',encoding='utf-8') as stream:
            result=subprocess.run(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return {'reservoir_id':identifier,'seed':seed,'exit_code':result.returncode}
    lock=out/'run.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(__import__('os').getpid()))
    try:
        workflow.write(out/'status.json',{'stage':'calibrating','budget':budget,'jobs':len(jobs)})
        with ThreadPoolExecutor(max_workers=cfg['optimization']['workers']) as pool:
            results=[f.result() for f in as_completed([pool.submit(execute,job) for job in jobs])]
        workflow.write(out/f'execution_budget_{budget}.json',{'jobs':results})
        if any(j['exit_code'] for j in results):raise ValueError('Calibration jobs failed; inspect persistent logs and resume')
        workflow.write(out/'status.json',{'stage':'selecting_assessing_exporting','budget':budget})
        registry=workflow.finish(out,budget)
        workflow.write(out/'status.json',{'stage':'completed','budget':budget,'registry':str(registry)})
        return registry
    except Exception as error:
        workflow.write(out/'status.json',{'stage':'failed','budget':budget,'reason':str(error)});raise
    finally:lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    for name in ('prepare','run','adopt'):
        p=sub.add_parser(name);p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
        if name=='run':p.add_argument('--evaluation-budget',type=int)
    p=sub.add_parser('calibrate');p.add_argument('--output',required=True,type=Path);p.add_argument('--reservoir-id',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--evaluation-budget',type=int)
    p=sub.add_parser('finish');p.add_argument('--output',required=True,type=Path);p.add_argument('--evaluation-budget',type=int)
    p=sub.add_parser('verify');p.add_argument('--registry',required=True,type=Path)
    p=sub.add_parser('simulate');p.add_argument('--registry',required=True,type=Path);p.add_argument('--forcing-csv',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    choice=p.add_mutually_exclusive_group(required=True);choice.add_argument('--reservoir-id');choice.add_argument('--grand-id',type=int)
    p.add_argument('--initial-state',type=Path,help='JSON containing storage_m3 and release_m3s for a different forcing start')
    args=parser.parse_args()
    if args.stage=='prepare':result=workflow.prepare(args.config,args.output)
    elif args.stage=='run':result=run(args.config,args.output,args.evaluation_budget)
    elif args.stage=='adopt':result=workflow.adopt(args.config,args.output)
    elif args.stage=='calibrate':result=workflow.calibrate(args.output,args.reservoir_id,args.seed,args.evaluation_budget)
    elif args.stage=='finish':result=workflow.finish(args.output,args.evaluation_budget)
    elif args.stage=='verify':workflow.verify_registry(args.registry);result='Registry verified'
    else:result=workflow.simulate(args.registry,args.forcing_csv,args.output,identifier=args.reservoir_id,grand_id=args.grand_id,initial=workflow.read(args.initial_state) if args.initial_state else None)
    print(result)


if __name__=='__main__':main()
