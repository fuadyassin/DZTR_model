"""Preserve official records and freeze a documented exploratory scenario."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows import minidoka_source_v1 as s
from workflows import extension_assessment_v1 as d


def main():
    out=ROOT/'outputs/minidoka_source_reconciliation_v1_20261021';evidence=out/'evidence'
    frame=s.read_provider(evidence/'daily_min_1984_2018.html');frame.to_csv(out/'provider_native_units.csv')
    normalized=s.normalize(frame);normalized.to_csv(out/'observations_gross_storage_si.csv')
    previous=pd.read_csv(ROOT/'outputs/reservoir_extension_assessment_v1_20261021/supplementary_sources/horizon_usbrpn_min/raw.csv',index_col='date',parse_dates=True,float_precision='round_trip')
    pairs=frame.join(previous).dropna(subset=['min_af','s_af'])
    comparison=[]
    for a,b in [('min_af','s_af'),('min_id','i_cfs'),('min_qt','r_cfs')]:
        finite=pairs[[a,b]].dropna();diff=finite[a]-finite[b]
        comparison.append({'provider_field':a,'archive_field':b,'paired':len(finite),'different':int((diff.abs()>1e-8).sum()),'maximum_absolute_difference':float(diff.abs().max())})
    pd.DataFrame(comparison).to_csv(out/'provider_archive_comparison.csv',index=False)
    budget=pd.DataFrame({'storage_change_m3':normalized.storage_m3.diff(),'inflow_m3s':normalized.inflow_m3s,
        'qt_endpoint_m3s':normalized.release_m3s,'qt_trapezoid_proxy_m3s':(normalized.release_m3s+normalized.release_m3s.shift())/2})
    budget['endpoint_as_mean_budget_residual_m3']=budget.storage_change_m3-86400*(budget.inflow_m3s-budget.qt_endpoint_m3s)
    budget['endpoint_trapezoid_budget_residual_m3']=budget.storage_change_m3-86400*(budget.inflow_m3s-budget.qt_trapezoid_proxy_m3s)
    budget.to_csv(out/'source_clock_budget_comparison.csv')
    summary={'provider_fields':{'AF':'instant storage above historical reference','ID':'provider-computed daily average inflow','QT':'instant total release; score modeled endpoint, not interval mean'},
        'offset_af':115000.,'historical_operating_capacity_above_reference_af':107240.,'gross_capacity_scenario_af':222240.,
        'capacity_scenario_verified_for_1985_2017':False,'absolute_datum_reference_verified_for_all_years':False,
        'dead_storage_fraction_scenario':0.,'dead_storage_note':'Historical power-penstock reference is not certified all-outlet minimum storage; zero DZTR floor is an exploratory assumption.',
        'negative_provider_storage_days':int((frame.min_af<0).sum()),'negative_provider_computed_inflow_days':int((frame.min_id<0).sum()),
        'provider_inflow_negative_99_99_days':int(frame.min_id.eq(-99.99).sum()),'negative_gross_storage_days':int((normalized.storage_m3<0).sum()),
        'above_historical_capacity_days':int((frame.min_af>107240).sum()),'provider_records_sha256':d.sha(evidence/'daily_min_1984_2018.html'),
        'observation_independence_verified':False,'flow_clock_verified':False,'no_clipping_filling_or_reconstructed_inflow':True}
    d.write(out/'source_decisions.json',summary)
    # Evidence is frozen before model fitting. Full calendar years are selected
    # solely from source support; no performance-based window selection.
    cfg={'kind':'provider_endpoint_interval_calibration_v1','reservoir_id':'etienne_Minidoka','GRAND_ID':None,
        'observations_csv':str((out/'observations_gross_storage_si.csv').relative_to(ROOT)),
        'source_review':str((out/'source_decisions.json').relative_to(ROOT)),
        'base_inventory':'data/catalogs/reservoir_inventory_v4_20261021.csv','base_catalog':'data/catalogs/reservoir_collection_v4_20261021.json',
        'start':'1985-01-01','end':'2017-12-31','balance_tolerance_m3':1e-4,'restart_warmup_days':30,
        'physical_constants':{'capacity_m3':222240*d.ACRE_FOOT_M3,'dead_storage_fraction':0.,'initial_storage_m3':170000*d.ACRE_FOOT_M3,'initial_release_m3s':0.,'inflow_correction':1.,'flood_mode':'storage'},
        'objective_weights':{'storage':1.,'release':0.},'optimization':{'seeds':[42,43,44],'horizon':512,'evaluation_budget':512,'workers':3,'perturbation':.2}}
    path=ROOT/'configs/minidoka_provider_interval_v1_20261021.json'
    if path.exists():assert d.read(path)==cfg
    else:d.write(path,cfg)
    print(summary);print(pd.DataFrame(comparison).to_string(index=False))


if __name__=='__main__':main()
