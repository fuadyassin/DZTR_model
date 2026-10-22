"""Build the consolidated 32-paper-reservoir assessment without new searches."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from workflows.cohort_assessment_v1 import assess

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--diagnostic-cache",required=True)
    args=parser.parse_args()
    print(assess(args.output,args.diagnostic_cache))
