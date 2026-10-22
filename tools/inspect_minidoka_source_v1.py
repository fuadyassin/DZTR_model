"""Targeted source diagnostics; no calibration, repair or gap filling."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows import extension_assessment_v1 as d

def main():
    folder=ROOT/'outputs/minidoka_source_reconciliation_v1_20261021';folder.mkdir(exist_ok=True)
    source=ROOT/'outputs/reservoir_extension_assessment_v1_20261021/supplementary_sources/horizon_usbrpn_min'
    raw=pd.read_csv(source/'raw.csv',index_col='date',parse_dates=True,float_precision='round_trip')
    frame=d.frame_read(source/'observations_si.csv')
    negative=raw[(raw.s_af<0)|(raw.i_cfs<0)]
    negative.to_csv(folder/'signed_source_records.csv')
    records=[]
    for year,part in raw.groupby(raw.index.year):
        records.append({'year':int(year),'rows':len(part),**{f'{c}_count':int(part[c].notna().sum()) for c in raw.columns},
            'storage_min_af':float(part.s_af.min()),'storage_max_af':float(part.s_af.max()),
            'negative_storage_days':int((part.s_af<0).sum()),'negative_inflow_days':int((part.i_cfs<0).sum()),
            'fractional_release_days':int((np.abs(part.r_cfs-part.r_cfs.round())>1e-8).sum())})
    annual=pd.DataFrame(records);annual.to_csv(folder/'annual_source_support.csv',index=False)
    other=d.frame_read(ROOT/'data/reservoirs_etienne_v1_20261021/etienne_Minidoka/observations_si.csv')
    joined=frame.join(other[['storage_m3','release_m3s']],rsuffix='_etienne')
    pairs=joined.dropna(subset=['storage_m3','storage_m3_etienne'])
    differences=pairs.storage_m3_etienne-pairs.storage_m3
    d.write(folder/'local_source_review.json',{'source_sha256':d.sha(source/'raw.csv'),
        'ranges':{c:{'min':float(raw[c].min()),'max':float(raw[c].max()),'first':str(raw[c].first_valid_index()),'last':str(raw[c].last_valid_index())} for c in raw},
        'paired_storage_days':len(pairs),'storage_source_difference_median_m3':float(differences.median()),
        'maximum_absolute_storage_difference_m3':float(differences.abs().max()),
        'negative_storage_first':str(raw.index[raw.s_af<0].min()),'negative_storage_last':str(raw.index[raw.s_af<0].max()),
        'negative_inflow_first':str(raw.index[raw.i_cfs<0].min()),'negative_inflow_last':str(raw.index[raw.i_cfs<0].max())})
    print((folder/'local_source_review.json').read_text())
    print(annual.loc[annual.i_cfs_count>0].to_string(index=False))

if __name__=='__main__':main()
