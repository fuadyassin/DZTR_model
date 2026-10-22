"""Preserve public station records for boundary identification, not calibration."""
from pathlib import Path
import json
import sys

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from workflows import station_inputs_v1 as stations
from workflows import reservoir_v2 as io

OUT = ROOT / 'outputs/paired_source_review_v1_20261022/stations'
QUERIES = {
    '08AA005': ('1995-01-01', '2023-12-31'),
    '08AA012': ('1995-01-01', '2023-12-31'),
    '08AA010': ('1995-01-01', '2023-12-31'),
    '08AA009': ('1995-01-01', '2023-12-31'),
    '08JA023': ('1993-01-01', '2021-12-31'),
    '08JA013': ('1993-01-01', '2021-12-31'),
    '08FE002': ('1993-01-01', '2021-12-31'),
    '08JA017': ('1993-01-01', '2021-12-31'),
    '08ME017': ('1958-01-01', '1978-06-05'),
    '08ME003': ('1958-01-01', '1978-06-05'),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for station, (start, end) in QUERIES.items():
        try:
            pages, receipt = stations._download(station, start, end, OUT / 'raw')
            index = pd.date_range(start, end, name='date')
            frame = pd.DataFrame({
                'level_m': stations._series(pages, station, 'LEVEL', index),
                'discharge_m3s': stations._series(pages, station, 'DISCHARGE', index),
            }, index=index)
            path = OUT / (station + '.csv')
            frame.to_csv(path, float_format='%.17g')
            results[station] = {'receipt': receipt, 'csv_sha256': io.sha(path),
                                'valid_days': frame.notna().sum().to_dict()}
            print(station, results[station]['valid_days'], flush=True)
        except requests.RequestException as exc:
            results[station] = {'error': str(exc)}
            print(station, str(exc), flush=True)
        io.write(OUT / 'receipts.json', results)

    url = 'https://waterservices.usgs.gov/nwis/dv/'
    for site in ['04082500', '04084255', '04084500', '04084445']:
        params = {'format': 'json', 'sites': site, 'startDT': '2007-10-01',
                  'endDT': '2020-12-31', 'parameterCd': '00065,00060',
                  'statCd': '00003', 'siteStatus': 'all'}
        path = OUT / ('usgs_' + site + '.json')
        if not path.exists():
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            path.write_bytes(r.content)
        payload = json.loads(path.read_text(encoding='utf-8'))
        print(site, [(t['name'], len(t['values'][0]['value']))
                     for t in payload['value']['timeSeries']], flush=True)
        io.write(OUT / ('usgs_' + site + '_receipt.json'), {
            'url': requests.Request('GET', url, params=params).prepare().url,
            'sha256': io.sha(path), 'purpose': 'station identification; no storage datum adoption'})


if __name__ == '__main__':
    main()
