"""Batch source intake using the verified inventory and local station crosswalk."""
from pathlib import Path
import json
import math
import re
import sys
import numpy as np
import pandas as pd
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows import extension_assessment_v1 as d


def norm(value):return re.sub('[^a-z0-9]','',str(value).lower())


def main():
    out=ROOT/'outputs/extension_source_intake_v1_20261021';out.mkdir(exist_ok=True)
    base=d.read(ROOT/'data/catalogs/reservoir_collection_v4_20261021.json')
    old=d.read(ROOT/'configs/reservoir_extension_assessment_v1_20261021.json')['decisions']
    excluded={'etienne_Hungry_Horse','etienne_Hugh_Keenleyside','etienne_Revelstoke','etienne_Minidoka'}
    cases=[r for r in base['reservoirs'] if r['reservoir_id'].startswith('etienne_') and r['reservoir_id'] not in excluded];assert len(cases)==17
    grand=pd.read_csv(ROOT/'data/catalogs/grand_v1_1_reference.csv',keep_default_na=False)
    crosswalk_path=Path('E:/1_DataHub/E_Reservoir_data/Z_ReservoirDataNew/ReservoirCollected.xlsx');crosswalk=pd.read_excel(crosswalk_path)
    archive=Path('E:/1_DataHub/E_Reservoir_data/Z_ReservoirDataNew/horizon-master/inst/extdata')
    rows=[];candidates=[];configs=[]
    for record in cases:
        rid=record['reservoir_id'];frame=d.frame_read(ROOT/record['observations_file']);s=int(frame.storage_m3.notna().sum());q=int(frame.release_m3s.notna().sum())
        route='observed_monthly' if s and q else 'storage_only' if s else 'flow_only'
        aliases=[norm(a) for a in re.split(r' / | /|/ ',record['name'])];aliases+=[norm(rid.removeprefix('etienne_').replace('_',' '))]
        def matches(row):return any(alias and alias in norm(row[col]) for alias in aliases for col in ['RES_NAME','DAM_NAME','ALT_NAME'])
        matches_frame=grand[grand.apply(matches,axis=1)];source_count=0
        for _,g in matches_frame.iterrows():
            gid=int(g.GRAND_ID);stations=crosswalk.loc[crosswalk.GRAND_ID==gid,'dam'].dropna().astype(str).tolist()
            if not stations:candidates.append({'reservoir_id':rid,'candidate_GRAND_ID':gid,'candidate_reservoir_name':g.RES_NAME,'candidate_dam_name':g.DAM_NAME,'station':None,'local_file':None,'identity_verified':False,'data_ready':False})
            for station in stations:
                provider,code=station.split('_',1);path=archive/provider/'processed'/f'{code}.csv';exists=path.is_file();entry={'reservoir_id':rid,'candidate_GRAND_ID':gid,
                    'candidate_reservoir_name':g.RES_NAME,'candidate_dam_name':g.DAM_NAME,'station':station,'local_file':str(path) if exists else None,'identity_verified':False,'data_ready':False}
                if exists:
                    data=pd.read_csv(path);source_count+=1;entry['source_sha256']=d.sha(path)
                    for col in ['s_af','i_cfs','r_cfs']:entry[col+'_count']=int(data[col].notna().sum()) if col in data else 0
                candidates.append(entry)
        rows.append({'reservoir_id':rid,'name':record['name'],'response_route':route,'storage_samples':s,'release_samples':q,
            'reported_inflow_samples':int(frame.inflow_m3s.notna().sum()),'local_crosswalk_source_candidates':source_count,
            'status':'source_gated','next_action':old[rid]['next_action'],'candidate_identity_is_not_verified':True})
        configs.append({'reservoir_id':rid,'observation_file':record['observations_file'],'recommended_response_route':route,
            'admission':'blocked_pending_source_evidence','inflow_station_components':[],'release_station_components':[],
            'time_varying_inflow_source':None,'capacity_on_matching_storage_coordinate_m3':None,'storage_reference':None,
            'observation_clock':None,'verified_GRAND_ID':None,'next_action':old[rid]['next_action']})
    pd.DataFrame(rows).to_csv(out/'remaining_17_source_queue.csv',index=False);pd.DataFrame(candidates).to_csv(out/'local_source_candidates.csv',index=False)
    d.write(out/'source_intake_manifest.json',{'kind':'extension_source_intake_v1','cases':configs,'mean_channel_flow_is_not_daily_forcing':True,
        'candidate_matches_do_not_admit_calibration':True,'inputs':{'catalog_sha256':d.sha(ROOT/'data/catalogs/reservoir_collection_v4_20261021.json'),
            'crosswalk_sha256':d.sha(crosswalk_path),'grand_sha256':d.sha(ROOT/'data/catalogs/grand_v1_1_reference.csv')}})
    # Minidoka's exact station crosswalk is corroborated by official coordinates.
    evidence=out/'evidence';evidence.mkdir(exist_ok=True);p=evidence/'minidoka_official_location.json'
    if not p.exists():
        response=requests.get('https://data.usbr.gov/rise/api/location/7191',timeout=45);response.raise_for_status();p.write_bytes(response.content)
    location=json.loads(p.read_text())['data']['attributes'];d.write(out/'minidoka_location_attributes.json',location)
    g=grand.loc[grand.GRAND_ID==413].iloc[0]
    official_lat,official_lon=42.669915,-113.482993
    if str(official_lat) not in p.read_text() or str(official_lon) not in p.read_text():raise ValueError('Official coordinate evidence changed')
    assert crosswalk.loc[crosswalk.dam=='usbrpn_min','GRAND_ID'].tolist()==[413.]
    lat1,lat2=np.deg2rad([official_lat,float(g.LAT_DD)]);dlon=np.deg2rad(float(g.LONG_DD)-official_lon)
    distance=6371*2*np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2))
    assert distance<1 and g.DAM_NAME=='Minidoka' and g.RES_NAME=='Lake Walcott'
    assert not any(r.get('GRAND_ID')==413 for r in base['reservoirs'] if r['reservoir_id']!='etienne_Minidoka')
    d.write(out/'minidoka_identity_verified.json',{'reservoir_id':'etienne_Minidoka','GRAND_ID':413,'station':'usbrpn_min','station_crosswalk_sha256':d.sha(crosswalk_path),
        'grand_name':g.RES_NAME,'grand_dam_name':g.DAM_NAME,'official_lat':official_lat,'official_lon':official_lon,'grand_lat':float(g.LAT_DD),'grand_lon':float(g.LONG_DD),
        'distance_km':float(distance),'official_source':'https://data.usbr.gov/rise/api/location/7191','official_source_sha256':d.sha(p),
        'capacity_verified':False,'storage_coordinate_verified_for_full_period':False})
    lines=['# Remaining extension source intake','','Seventeen additions retain explicit source gates. This intake checks existing observation support, the local station-to-GRAND crosswalk and matching archived source files. Name/crosswalk candidates remain unverified and do not admit calibration.','',
        '[Queue](remaining_17_source_queue.csv), [candidate local records](local_source_candidates.csv), [machine-readable intake](source_intake_manifest.json).','',
        '| Reservoir | Response route | Local source candidates | Required evidence |','|---|---|---:|---|']
    for row in rows:lines.append(f"| {row['name']} | {row['response_route']} | {row['local_crosswalk_source_candidates']} | {row['next_action']} |")
    lines+=['','Empty station-component lists mean no adopted station aggregation, not zero inflow or zero release. Fill these only with verified boundary-compatible stations, units and clocks. Never substitute mean channel flow for daily forcing.','',
        'Minidoka identity is independently linked to GRAND 413 by the exact USBRPN station crosswalk, matching dam/lake names and official coordinates (less than one kilometre apart). This does not certify catalog capacity or the historical storage coordinate. [Identity evidence](minidoka_identity_verified.json).','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8');print(pd.DataFrame(rows)[['reservoir_id','response_route','local_crosswalk_source_candidates']].to_string(index=False));print('Verified Minidoka GRAND_ID 413, position distance km:',distance)


if __name__=='__main__':main()
