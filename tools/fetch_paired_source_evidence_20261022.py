"""Preserve targeted primary evidence for the four remaining paired cases."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import requests
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/paired_source_review_v1_20261022';EVIDENCE=OUT/'evidence'
URLS={
 'aishihik_hydrotechnical_2024.pdf':'https://yukonenergy.ca/media/site_documents/AtkinsRealis_Hydrotechnical_Study_2024.pdf',
 'aishihik_levels.html':'https://yukonenergy.ca/environment/current-water-levels/aishihik-water-levels/',
 'aishihik_relicensing.html':'https://yukonenergy.ca/energy-in-yukon/projects-facilities/aishihik-hydro-plant/aishihik-re-licensing-project',
 'seton_monitoring_2015.pdf':'https://www.bchydro.com/content/dam/BCHydro/customer-portal/documents/corporate/environment-sustainability/water-use-planning/lower-mainland/brgmon-6-yr1-2015-03-31.pdf',
 'nechako_operator.html':'https://www.riotinto.com/en/can/canada-operations/bc-works/nechako-reservoir',
 'winnebago_operator.html':'https://www.lrd.usace.army.mil/Missions/Projects/Article/3623884/lake-winnebago/',
 'winnebago_chicago.html':'https://www.lrd.usace.army.mil/chicago/',
 'winnebago_regulation.pdf':'https://www.lrc-wc.usace.army.mil/data/winnebago_strategy/Winnebago_Regulation_Strategy.pdf',
}
class Links(HTMLParser):
    def __init__(self):super().__init__();self.values=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['href','src'] and value:self.values.append({'tag':tag,'attribute':key,'url':value})
def main():
    EVIDENCE.mkdir(parents=True,exist_ok=True);receipts={};links={}
    for name,url in URLS.items():
        path=EVIDENCE/name
        try:
            if not path.exists():
                response=requests.get(url,timeout=40);response.raise_for_status();path.write_bytes(response.content)
            content=path.read_bytes();receipts[name]={'url':url,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
            if name.endswith('.html'):
                parser=Links();parser.feed(content.decode('utf-8',errors='replace'));links[name]=parser.values
            print(name,len(content),flush=True)
        except requests.RequestException as exc:receipts[name]={'url':url,'error':str(exc)};print(name,str(exc),flush=True)
    (EVIDENCE/'receipts.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
    (OUT/'provider_links.json').write_text(json.dumps(links,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
