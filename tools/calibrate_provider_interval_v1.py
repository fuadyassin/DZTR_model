"""Prepare, run, resume or replay the provider-net storage-objective scenario."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows import provider_calibration_v1 as w
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','run'):
        p=sub.add_parser(name);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('worker');p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--budget',type=int)
    for name in ('report','replay','verify'):
        p=sub.add_parser(name);p.add_argument('--output',required=True)
    a=parser.parse_args()
    if a.command in ('prepare','run'):result=getattr(w,a.command)(a.config,a.output)
    elif a.command=='worker':result=w.worker(a.output,a.seed,a.budget)
    elif a.command in ('report','replay'):result=getattr(w,a.command)(a.output)
    else:
        out,_=w.checked(a.output);result=w.native.verify_registry(out/'report/registry.json')['kind']
    print(result,flush=True)
