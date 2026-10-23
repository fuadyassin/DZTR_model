"""Cache primary Winnebago datum and boundary evidence in local ignored output."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import argparse

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/winnebago_source_reconciliation_v1_20261022/evidence'
SOURCES = {
    'usgs_oshkosh_metadata.html': 'https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=04082500',
    'usgs_stockbridge_metadata.html': 'https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=04084255',
    'usgs_appleton_metadata.html': 'https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=04084445',
    'usgs_oshkosh_elevation_2019_2020.json': 'https://waterservices.usgs.gov/nwis/dv/?format=json&sites=04082500&startDT=2019-10-01&endDT=2020-12-31&parameterCd=62615&statCd=00003&siteStatus=all',
    'usgs_stockbridge_elevation_2019_2020.json': 'https://waterservices.usgs.gov/nwis/dv/?format=json&sites=04084255&startDT=2019-10-01&endDT=2020-12-31&parameterCd=62615&statCd=00003&siteStatus=all',
    'usgs_oshkosh_water_year_2010.pdf': 'https://wdr.water.usgs.gov/wy2010/pdfs/04082500.2010.pdf',
    'usgs_stockbridge_water_year_2010.pdf': 'https://wdr.water.usgs.gov/wy2010/pdfs/04084255.2010.pdf',
    'usgs_appleton_water_year_2010.pdf': 'https://wdr.water.usgs.gov/wy2010/pdfs/04084445.2010.pdf',
    'county_pool_management_2019.pdf': 'https://www.co.winnebago.wi.us/sites/default/files/winnebago_waterways_executive_committee_10.24.2019.pdf',
    'usgs_oshkosh_instantaneous_2020_sample.json': 'https://waterservices.usgs.gov/nwis/iv/?format=json&sites=04082500&startDT=2020-06-01&endDT=2020-06-03&parameterCd=00065,62615&siteStatus=all',
    'usgs_stockbridge_instantaneous_2020_sample.json': 'https://waterservices.usgs.gov/nwis/iv/?format=json&sites=04084255&startDT=2020-06-01&endDT=2020-06-03&parameterCd=00065,62615&siteStatus=all',
}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--only', nargs='+', choices=sorted(SOURCES))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT / 'receipts.json'
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    for name, url in SOURCES.items():
        if args.only and name not in args.only:
            continue
        path = OUT / name
        if path.exists():
            if sha256(path.read_bytes()).hexdigest() != receipts[name]['sha256']:
                raise ValueError('Cached evidence changed: ' + name)
            continue
        if name in receipts and 'error' in receipts[name] and not args.retry_errors:
            continue
        try:
            response = requests.get(url, timeout=45); response.raise_for_status()
            if name.endswith('.pdf') and not response.content.startswith(b'%PDF-'):
                raise ValueError('Expected PDF evidence: ' + name)
            series_count = len(response.json()['value']['timeSeries']) if name.endswith('.json') else None
            path.write_bytes(response.content)
            receipts[name] = {'url': url, 'final_url': response.url,
                'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'bytes': len(response.content),
                'sha256': sha256(response.content).hexdigest()}
            if series_count is not None:
                receipts[name]['provider_series_count'] = series_count
            print(name, len(response.content), flush=True)
        except (requests.RequestException, ValueError, AssertionError) as error:
            receipts[name] = {'url': url, 'error': str(error)}
            print(name, type(error).__name__, flush=True)
        receipt_path.write_text(json.dumps(receipts, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
