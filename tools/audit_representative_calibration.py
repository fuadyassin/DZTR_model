"""Independently audit selected-year coverage and selected-only empirical fits."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulate_calibrated_reservoirs import ROOT,read,write,sha
sys.path.insert(0,str(ROOT))
from strategies.representative_years import YearSelection
from dztr.schemas import ReservoirParameters

OUT=ROOT/'outputs/representative_years_v1_20261012'


def main():
    receipt=read(OUT/'preparation_manifest.json')
    for relative,digest in receipt['artifact_sha256'].items():assert sha(OUT/relative)==digest,relative
    for relative,digest in receipt['source_sha256'].items():assert sha(ROOT/relative)==digest,relative
    assert receipt['strategy_sha256']==sha(ROOT/'strategies/representative_years.py')
    jobs=read(OUT/'preparation_summary.json')['jobs'];rows=[]
    for job in jobs:
        assert job['status']=='fit_prepared_not_simulated',job
        folder=OUT/job['reservoir_id']
        plan=YearSelection(**read(folder/'selection.json'))
        inputs=pd.read_csv(folder/'observations_si.csv',index_col='date',parse_dates=True,float_precision='round_trip')
        table=pd.read_csv(folder/'row_selection.csv',index_col='date',parse_dates=True)
        pd.testing.assert_frame_equal(plan.table(),table,check_exact=True)
        mask=plan.mask(inputs.index,'calibration');assessment=plan.mask(inputs.index,'assessment')
        assert mask.mean()>=.65 and not (mask&assessment).any() and assessment.any()
        annual=pd.read_csv(folder/'year_selection.csv',index_col='water_year')
        assert annual.loc[annual.role=='calibration','complete'].all()
        assert all(annual.loc[annual.role=='calibration'].regime.eq(r).any() for r in ('dry','normal','wet'))
        raw=pd.read_csv(folder/'raw_generalized_targets_si.csv',index_col='month',float_precision='round_trip')
        bounds=pd.read_csv(folder/'calibration_bounds_si.csv',index_col='month',float_precision='round_trip')
        fitting=inputs.loc[mask]
        for variable,stem,suffix in (('storage_m3','storage','m3'),('release_m3s','release','m3s')):
            assert inputs.loc[mask,variable].notna().sum()/inputs[variable].notna().sum()>=.65
            for month in range(1,13):
                values=fitting.loc[fitting.index.month==month,variable].dropna().to_numpy()
                for level,pair,target in zip(('critical','normal','flood'),((5,35),(35,75),(75,95)),(10,45,85)):
                    key=f'{stem}_{level}_{suffix}'
                    np.testing.assert_array_equal(raw.loc[month,key],np.percentile(values,target,method='hazen'))
                    for side,p in zip(('lower','upper'),pair):np.testing.assert_array_equal(bounds.loc[month,side+'_'+key],np.percentile(values,p,method='hazen'))
        config=read(folder/'model_config.json');constants=config['physical_constants']
        initial=pd.read_csv(folder/'calibration_targets_si.csv',index_col='month',float_precision='round_trip')
        ReservoirParameters(initial,**constants)
        violations=[]
        for month in range(1,13):
            if raw.loc[month,'storage_critical_m3']<=constants['capacity_m3']*constants['dead_storage_fraction']:
                violations.append({'month':month,'issue':'critical storage does not exceed inherited dead storage'})
            if raw.loc[month,'storage_flood_m3']>constants['capacity_m3']:violations.append({'month':month,'issue':'flood storage exceeds capacity'})
        write(folder/'raw_target_physical_review.json',{'raw_targets_preserved':True,'issues':violations,'initialization_kind':job['initialization_kind'],'not_a_corrected_generalized_baseline':bool(violations)})
        rows.append({'reservoir_id':job['reservoir_id'],'GRAND_ID':job['GRAND_ID'],'calibration_fraction_total':float(mask.mean()),'calibration_years':int(annual.role.eq('calibration').sum()),'assessment_years':int(annual.role.eq('assessment').sum()),'initialization_kind':job['initialization_kind']})
    pd.DataFrame(rows).to_csv(OUT/'verified_coverage.csv',index=False,float_format='%.17g')
    fig,axes=plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
    for ax,identifier in zip(axes,('ex5_Garrison','ex6_Oahe')):
        annual=pd.read_csv(OUT/identifier/'year_selection.csv')
        for role,color,marker in (('calibration','tab:blue','o'),('assessment','tab:orange','s'),('warmup','grey','x')):
            part=annual[annual.role==role];ax.scatter(part.water_year,part.inflow_mean_m3s,color=color,marker=marker,label=role,s=65)
        for _,row in annual[annual.role.isin(['calibration','assessment'])].iterrows():ax.annotate(row.regime[0].upper(),(row.water_year,row.inflow_mean_m3s),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
        ax.set_title(identifier+' whole-year selection (D/N/W = relative inflow regime)');ax.set_ylabel('Annual inflow proxy (m³/s)');ax.grid(alpha=.2);ax.legend()
    axes[-1].set_xlabel('Year');fig.savefig(OUT/'garrison_oahe_selection.png',dpi=180);fig.savefig(OUT/'garrison_oahe_selection.pdf');plt.close(fig)
    write(OUT/'scientific_verification.json',{'status':'passed','reservoirs_verified':len(jobs),'minimum_calibration_fraction_total':min(r['calibration_fraction_total'] for r in rows),'all_have_dry_normal_wet_calibration':True,'masked_targets_and_bounds_match_independent_hazen_quantiles':True,'row_masks_disjoint':True,'raw_targets_preserved':True,'physically_valid_optimizer_initializations':True,'simulations':0,'reservoir_optimization_evaluations':0})
    print('Verified twenty whole-year plans, >=65% row/pair coverage and calibration-only quantiles; no simulations or searches.')


if __name__=='__main__':main()
