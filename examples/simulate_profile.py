"""Run a frozen native or signed-net profile by verified GRAND/reservoir ID.

Default forcing is the exact archived prepared input. External forcing requires
an explicit input stage; a different start requires explicit initial states.
"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from workflows.profile_application_v1 import simulate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog",required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--grand-id",type=int);group.add_argument("--reservoir-id")
    parser.add_argument("--forcing");parser.add_argument("--forcing-stage",choices=["canonical","prepared"])
    parser.add_argument("--initial-storage-m3",type=float);parser.add_argument("--initial-release-m3s",type=float)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    if (args.initial_storage_m3 is None)!=(args.initial_release_m3s is None):
        parser.error("Declare both initial storage and release")
    initial=None if args.initial_storage_m3 is None else {"storage_m3":args.initial_storage_m3,"release_m3s":args.initial_release_m3s}
    print(simulate(args.catalog,args.output,reservoir_id=args.reservoir_id,grand_id=args.grand_id,
        forcing_csv=args.forcing,forcing_stage=args.forcing_stage,initial=initial))


if __name__=="__main__":main()
