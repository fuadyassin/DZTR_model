"""Prepare, resume, assess or replay a frozen mean-flow DDS calibration."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows import daily_interval_calibration_v1 as w

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ('prepare','run'):
        p=sub.add_parser(command);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('worker');p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--budget',type=int)
    for command in ('report','replay','verify'):
        p=sub.add_parser(command);p.add_argument('--output',required=True)
    a=parser.parse_args()
    if a.command=='prepare':result=w.prepare(a.config,a.output)
    elif a.command=='run':result=w.run(a.config,a.output)
    elif a.command=='worker':result=w.calibrate(a.output,a.seed,a.budget)
    elif a.command=='report':result=w.report(a.output)
    elif a.command=='replay':result=w.replay(a.output)
    else:
        out,_=w.checked(a.output);result=w.native.verify_registry(out/'report/registry.json')['kind']
    print(result,flush=True)
