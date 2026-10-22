"""Cache a small, explicit set of primary provider responses on D:."""
from pathlib import Path
import hashlib
import json
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/minidoka_source_reconciliation_v1_20261021/evidence'
SOURCES={
    'daily_min_sample.html':'https://www.usbr.gov/pn-bin/daily?station=min&format=html&year=2000&month=7&day=1&year=2000&month=7&day=7&pcode=af&pcode=in&pcode=qd&pcode=qjn&pcode=qjs&pcode=qrd&pcode=fb',
    'rise_record.json':'https://data.usbr.gov/rise/api/catalog-record/8345',
    'rise_storage_item.json':'https://data.usbr.gov/rise/api/catalog-item/131804',
    'usbr_minidoka_dam.html':'https://www.usbr.gov/projects/index.php?id=181',
    'usbr_minidoka_data.html':'https://data.usbr.gov/catalog/8345',
    'usbr_decode.html':'https://www.usbr.gov/pn/hydromet/decod_params.html',
    'usgs_walcott_1966_1970.pdf':'https://pubs.usgs.gov/wsp/2134/report.pdf',
    'usbr_upper_snake_2020.pdf':'https://www.usbr.gov/pn/hydromet/uppersnake/01_BOR%20Upper%20Snake%20Update_12.15.2020.pdf',
}
SOURCES.update({f'rise_item_{i}.json':f'https://data.usbr.gov/rise/api/catalog-item/{i}' for i in range(131805,131813)})
SOURCES['daily_minm_sample.html']='https://www.usbr.gov/pn-bin/daily?station=minm&format=html&year=2000&month=7&day=1&year=2000&month=7&day=7&pcode=af&pcode=in&pcode=qd&pcode=qj&pcode=qjn&pcode=qjs&pcode=qrd&pcode=qu'
SOURCES['daily_min_correct_sample.html']='https://www.usbr.gov/pn-bin/daily?station=min&format=html&year=2000&month=7&day=1&year=2000&month=7&day=7&pcode=af&pcode=id&pcode=qt&pcode=fb&pcode=qj&pcode=qrd'
SOURCES['daily_min_1984_2018.html']='https://www.usbr.gov/pn-bin/daily?station=min&format=html&year=1984&month=10&day=1&year=2018&month=8&day=23&pcode=af&pcode=id&pcode=qt&pcode=fb'

def main():
    OUT.mkdir(parents=True,exist_ok=True);receipts={}
    for name,url in SOURCES.items():
        path=OUT/name
        try:
            if not path.exists():
                response=requests.get(url,timeout=45);response.raise_for_status();path.write_bytes(response.content)
            payload=path.read_bytes();receipts[name]={'url':url,'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
            print(name,len(payload),payload[:50] if not name.endswith('.pdf') else b'%PDF',flush=True)
        except requests.RequestException as exc:receipts[name]={'url':url,'error':str(exc)};print(name,str(exc),flush=True)
    (OUT/'receipts.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
