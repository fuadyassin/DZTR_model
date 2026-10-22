"""One command for the frozen continuation and its complete reporting workflow."""
from pathlib import Path
import subprocess
import sys

from twenty_workflow import OUT,ROOT,read


def command(script,*args):
    subprocess.run([sys.executable,"-B",str(ROOT/'tools'/script),*args],check=True)


def main():
    if not OUT.exists():command('twenty_workflow.py','--prepare')
    if not (OUT/'execution_summary.json').exists():command('twenty_workflow.py','--run-authorized')
    if read(OUT/'execution_summary.json')["status"]!='completed':
        raise SystemExit("Resolve isolated failures using retained checkpoints before final selection.")
    if not (OUT/'pooled_selection.json').exists():command('twenty_analysis.py','select')
    if not (OUT/'pooled_validation_receipt.json').exists():command('twenty_analysis.py','assess')
    if not (OUT/'workflow_summary.json').exists():command('twenty_analysis.py','finish')
    if not (OUT/'scientific_verification.json').exists():command('verify_twenty_workflow.py')
    if not (OUT/'reports/manifest.json').exists():command('twenty_reports.py')
    if not (OUT/'finalization_receipt.json').exists():command('twenty_finalize.py')


if __name__=="__main__":main()
