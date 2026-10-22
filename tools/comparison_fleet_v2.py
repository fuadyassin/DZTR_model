"""Prepare and execute remaining fleet checkpointed paired searches."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import subprocess
import sys
import pandas as pd
from simulate_calibrated_reservoirs import ROOT,read,write,sha
sys.path.insert(0,str(ROOT))
from strategies.fleet_comparison_v2 import comparison_selections
from strategies.representative_years import fit_selected_years
from dztr.calibration.sampled import _observations

OUT=ROOT/'outputs/strategy_comparison_fleet_v2_20261012'
SOURCE=ROOT/'outputs/representative_years_v1_20261012'


def prepare():
    if OUT.exists():raise ValueError('Use a fresh pilot directory')
    proof=read(SOURCE/'final_verification.json');consumed={}
    input_manifest=SOURCE/'preparation_manifest.json'
    assert sha(input_manifest)==proof['artifact_sha256']['preparation_manifest.json']
    for relative,digest in read(input_manifest)['source_sha256'].items():
        if relative.startswith('src/'):
            assert sha(ROOT/relative)==digest,relative
            consumed[relative]=digest
    for relative,digest in proof['source_sha256'].items():
        if sha(ROOT/relative)!=digest:raise ValueError(f'Strategy source changed: {relative}')
        consumed[relative]=digest
    OUT.mkdir();jobs=[];designs=[]
    for identifier in sorted(set(read(ROOT/'configs/calibrated_twenty.json')['profiles_by_reservoir_id'])-{'ex5_Garrison','ex6_Oahe'}):
        source=SOURCE/identifier
        for name in ('observations_si.csv','year_selection.csv','model_config.json'):
            path=source/name
            assert sha(path)==proof['artifact_sha256'][path.relative_to(SOURCE).as_posix()]
            consumed[path.relative_to(ROOT).as_posix()]=sha(path)
        frame=pd.read_csv(source/'observations_si.csv',index_col='date',parse_dates=True,float_precision='round_trip')
        annual=pd.read_csv(source/'year_selection.csv',index_col='water_year',float_precision='round_trip')
        config=read(source/'model_config.json');plans,held=comparison_selections(frame,annual)
        record={'reservoir_id':identifier,'common_assessment_years':held,'common_assessment_regimes':{str(y):str(annual.loc[y,'regime']) for y in held},'assessment_regimes_missing':sorted({'dry','normal','wet'}-set(annual.loc[held,'regime'])),'sampling_frequency':config['sampling']['frequency'],'arms':{}}
        for arm,plan in plans.items():
            folder=OUT/'prepared'/identifier/arm;folder.mkdir(parents=True)
            write(folder/'selection.json',plan.as_dict());plan.table().to_csv(folder/'row_selection.csv',index_label='date')
            cfg=dict(config,cost_mode='regime_balanced_nse' if arm=='mixed_balanced' else 'pooled_nse')
            write(folder/'config.json',cfg)
            frame.to_csv(folder/'observations_si.csv',index_label='date',float_format='%.17g')
            template,bounds=fit_selected_years(_observations(frame),plan,cfg['physical_constants'],allow_feasible_initialization=True)
            template.monthly_targets.to_csv(folder/'initial_targets_si.csv',float_format='%.17g')
            bounds.empirical.monthly_bounds.to_csv(folder/'bounds_si.csv',float_format='%.17g')
            selected=sorted(set(y for y,r in zip(plan.years,plan.roles) if r=='calibration'))
            record['arms'][arm]={'calibration_years':selected,'calibration_fraction_total':plan.roles.count('calibration')/len(plan.roles),'selection_sha256':plan.signature,
                                'calibration_regime_counts':{r:sum(annual.loc[y,'regime']==r for y in selected) for r in ('dry','normal','wet')},'cost_mode':cfg['cost_mode']}
            for seed in (42,43,44):jobs.append({'reservoir_id':identifier,'arm':arm,'seed':seed,'population':32,'evaluation_budget':512})
        designs.append(record)
    write(OUT/'design.json',{'kind':'paired_strategy_comparison_v1','reservoirs':designs,'jobs':jobs,'workers':3,'new_search_evaluations':sum(j['evaluation_budget'] for j in jobs),
          'assessment_kind':'retrospective shared-year assessment, previously inspected record','selection_rule':'Pooled calibration nondominated front; equal-weight range-normalized distance to ideal; release/storage cost then seed/evaluation ID ties',
          'historical_reference':'Old calibration periods are descriptive references, not matched-training-size comparison arms','no_profile_promotion':True})
    files=['strategies/strategy_comparison.py','strategies/fleet_comparison_v2.py','tools/comparison_fleet_v2.py','tools/comparison_fleet_v2_job.py']
    write(OUT/'protocol.json',{'source_sha256':consumed,'code_sha256':{p:sha(ROOT/p) for p in files},'prepared_sha256':{p.relative_to(OUT).as_posix():sha(p) for p in sorted((OUT/'prepared').rglob('*')) if p.is_file()},'design_sha256':sha(OUT/'design.json'),
                              'python_version':sys.version,'numpy_version':__import__('numpy').__version__,'pandas_version':pd.__version__})
    print(f'Prepared {len(jobs)} searches, 512 evaluations/population32/seeds42-44; common assessment frozen before execution.')


def execute():
    protocol=read(OUT/'protocol.json')
    for relative,digest in {**protocol['source_sha256'],**protocol['code_sha256']}.items():assert sha(ROOT/relative)==digest,relative
    assert sha(OUT/'design.json')==protocol['design_sha256']
    for relative,digest in protocol['prepared_sha256'].items():assert sha(OUT/relative)==digest,relative
    def run(job):
        folder=OUT/'jobs'/job['reservoir_id']/job['arm']/f"seed_{job['seed']}";folder.mkdir(parents=True,exist_ok=True)
        if (folder/'result.json').exists() and read(folder/'result.json')['status']=='completed':return {'job':job,'status':'reused_completed'}
        command=[sys.executable,'-B',str(ROOT/'tools/comparison_fleet_v2_job.py'),job['reservoir_id'],job['arm'],str(job['seed'])]
        with (folder/'execution.log').open('a',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        print(f"{job['reservoir_id']} {job['arm']} seed{job['seed']}: exit {result.returncode}",flush=True)
        return {'job':job,'status':'completed' if result.returncode==0 else 'failed','exit_code':result.returncode}
    jobs=read(OUT/'design.json')['jobs']
    with ThreadPoolExecutor(max_workers=3) as pool:results=[f.result() for f in as_completed([pool.submit(run,j) for j in jobs])]
    write(OUT/'execution_summary.json',{'jobs':results,'all_completed':all(r['status'] in ('completed','reused_completed') for r in results)})
    if not all(r['status'] in ('completed','reused_completed') for r in results):raise SystemExit('Some pilot jobs failed; inspect persistent logs')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','execute']);args=parser.parse_args()
    {'prepare':prepare,'execute':execute}[args.stage]()
