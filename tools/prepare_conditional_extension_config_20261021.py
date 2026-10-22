"""Freeze the already reviewed two admissions and all remaining source gates."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
old=json.loads((root/'configs/reservoir_extension_assessment_v1_20261021.json').read_text())
catalog=json.loads((root/'data/catalogs/reservoir_collection_v3_20261021.json').read_text())
decisions={k:v['next_action'] for k,v in old['decisions'].items() if k!='etienne_Hungry_Horse'}
cases=[]
for rid,start,end,capacity,split,candidate in [
    ('etienne_Hugh_Keenleyside','1973-01-01','2019-12-31',11.3e9,'mixed_whole_years',282),
    ('etienne_Revelstoke','2018-05-03','2019-12-31',5.18e9,'short_record_chronological',257)]:
    row=next(r for r in catalog['reservoirs'] if r['reservoir_id']==rid)
    cases.append({'reservoir_id':rid,'GRAND_ID':None,'unverified_GRAND_candidate':candidate,
        'observations_csv':row['observations_file'],'start':start,'end':end,'selection':split,
        'restart_warmup_days':30,'allow_feasible_initialization':True,
        'physical_constants':{'capacity_m3':capacity,'dead_storage_fraction':0.,'initial_storage_m3':capacity*.5,
            'initial_release_m3s':0.,'inflow_correction':1.,'flood_mode':'storage'},
        'source_decision':'Conditional sensitivity only: original storage coordinate retained, catalog capacity unverified, zero dead-storage reference assumed. Net inflow reconstructed from storage changes and observed release, including assessment responses. No independent validation claim.',
        'source_reconciliation_pending':decisions[rid],
        'channel_capacity':'Calibration-only Hazen Q99 benchmark; not outlet design capacity.',
        'timing':'Nominal 86400-second interval ending at storage row, with current-day Q assumed interval mean; provider clock not certified.',
        'capacity_evidence':old['decisions'][rid].get('capacity_reference_evidence','Prior catalog screening reference only; not verified same-coordinate capacity.'),
        'no_record_clipping_or_offsets':True})
    decisions[rid]='Completed conditional sensitivity profile; retain original source gate: '+decisions[rid]
cfg={'kind':'conditional_extension_cohort_v1','base_catalog':'data/catalogs/reservoir_collection_v3_20261021.json',
    'base_inventory':'outputs/hungry_horse_completion_v1_20261021/reservoir_parameterization_all_62_by_GRAND_ID.csv',
    'cases':cases,'decisions':decisions,'balance_tolerance_m3':1e-4,'objective_weights':{'storage':.5,'release':.5},
    'optimization':{'seeds':[42,43,44],'horizon':512,'evaluation_budget':512,'workers':3,'perturbation':.2}}
path=root/'configs/conditional_extension_cohort_v1_20261021.json'
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
print(path)
