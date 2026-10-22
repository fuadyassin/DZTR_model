"""Migrate only byte-identical prepared objectives from the parked draft batch."""
import shutil
import sys
import numpy as np
import pandas as pd
from simulate_calibrated_reservoirs import ROOT, read, write, sha
from comparison_fleet_v2_job import OUT, objective_for
from dztr.calibration.checkpoint import load_optimizer_checkpoint, save_optimizer_checkpoint


def main():
    old=ROOT/'outputs/strategy_comparison_fleet_20261012'
    prior=read(old/'protocol.json');current=read(OUT/'protocol.json')
    for rel,digest in {**prior['source_sha256'],**prior['code_sha256']}.items():assert sha(ROOT/rel)==digest,rel
    for rel,digest in {**current['source_sha256'],**current['code_sha256']}.items():assert sha(ROOT/rel)==digest,rel
    # Only the output location differs in the actual optimizer worker.
    a=(ROOT/'tools/comparison_fleet_job.py').read_text()
    b=(ROOT/'tools/comparison_fleet_v2_job.py').read_text()
    assert a.replace('strategy_comparison_fleet_20261012','strategy_comparison_fleet_v2_20261012')==b
    records=[]
    for job in read(OUT/'design.json')['jobs']:
        identifier,arm,seed=job['reservoir_id'],job['arm'],job['seed']
        relative=f'jobs/{identifier}/{arm}/seed_{seed}'
        source=old/relative;target=OUT/relative
        if not (source/'checkpoint.npz').exists():continue
        prepared=f'prepared/{identifier}/{arm}/'
        files=[r for r in current['prepared_sha256'] if r.startswith(prepared)]
        if any(current['prepared_sha256'][r]!=prior['prepared_sha256'].get(r) for r in files):continue
        for rel in files:assert sha(old/rel)==sha(OUT/rel)==current['prepared_sha256'][rel]
        identity=read(source/'checkpoint_identity.json')
        assert identity['protocol_sha256']==sha(old/'protocol.json') and identity['job']==job
        objective=objective_for(identifier,arm)[0]
        assert identity['objective']==objective.identity
        assert identity['python_version']==sys.version and identity['numpy_version']==np.__version__ and identity['pandas_version']==pd.__version__
        previous=load_optimizer_checkpoint(source/'checkpoint.npz',identity)
        new_identity=dict(identity,protocol_sha256=sha(OUT/'protocol.json'))
        target.mkdir(parents=True)
        if (source/'result.json').exists():
            for name,digest in read(source/'completion_manifest.json')['artifact_sha256'].items():assert sha(source/name)==digest
            for path in source.iterdir():
                if path.is_file() and path.name not in ('checkpoint.npz','checkpoint_identity.json','completion_manifest.json','execution.log'):
                    shutil.copyfile(path,target/path.name)
        write(target/'checkpoint_identity.json',new_identity)
        save_optimizer_checkpoint(target/'checkpoint.npz',previous,new_identity)
        replay=load_optimizer_checkpoint(target/'checkpoint.npz',new_identity)
        assert len(previous.evaluated)==len(replay.evaluated)
        np.testing.assert_array_equal([p.vector for p in previous.evaluated],[p.vector for p in replay.evaluated])
        np.testing.assert_array_equal([p.objectives for p in previous.evaluated],[p.objectives for p in replay.evaluated])
        write(target/'progress.json',{'evaluations':len(replay.evaluated),'generation':replay.generation})
        if (target/'result.json').exists():write(target/'completion_manifest.json',{'artifact_sha256':{p.name:sha(p) for p in target.iterdir() if p.is_file() and p.name not in ('execution.log','completion_manifest.json')}})
        records.append({'job':job,'prefix_evaluations_preserved':len(previous.evaluated),'completed':(target/'result.json').exists(),'previous_checkpoint_sha256':sha(source/'checkpoint.npz'),'new_checkpoint_sha256':sha(target/'checkpoint.npz')})
    write(OUT/'checkpoint_reuse.json',{'status':'verified','records':records,'reused_evaluations':sum(r['prefix_evaluations_preserved'] for r in records),'prepared_files_byte_identical':True,'worker_equivalent_except_output_path':True,'source_sha256':sha(__file__)})
    write(old/'parked_design.json',{'status':'superseded_before_completion','reason':'Retain a later mixed-training opportunity in short records','replacement':OUT.relative_to(ROOT).as_posix(),'all_checkpoints_and_logs_preserved':True})
    print(f'Reused {len(records)} verified checkpoints; {sum(r["prefix_evaluations_preserved"] for r in records)} evaluations preserved.')


if __name__=='__main__':main()
