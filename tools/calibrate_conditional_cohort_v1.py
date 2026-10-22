"""Run source-declared conditional reservoir experiments without altering old runs."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows import conditional_cohort_v1 as w

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ('prepare','run'):
        p=sub.add_parser(command);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('worker');p.add_argument('--output',required=True);p.add_argument('--reservoir-id',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--budget',type=int)
    p=sub.add_parser('replay');p.add_argument('--output',required=True);p.add_argument('--reservoir-id',required=True)
    for command in ('report','verify'):
        p=sub.add_parser(command);p.add_argument('--output',required=True)
    a=parser.parse_args()
    if a.command=='prepare':r=w.prepare(a.config,a.output)
    elif a.command=='run':r=w.run(a.config,a.output)
    elif a.command=='worker':r=w.worker(a.output,a.reservoir_id,a.seed,a.budget)
    elif a.command=='report':r=w.report(a.output)
    elif a.command=='replay':r=w.replay(a.output,a.reservoir_id)
    else:
        out,_=w.checked(a.output);r=w.native.verify_registry(out/'report/registry.json')['kind']
    print(r,flush=True)
