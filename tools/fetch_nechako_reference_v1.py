"""Cache primary Nechako reference evidence locally without changing old reviews."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/nechako_source_reconciliation_v1_20261022/evidence'
SOURCES = {
    'operator_storage_reference_2020.pdf': 'https://www.getinvolvednechako.ca/wp-content/uploads/2021/11/Nechako-Reservoir-Licensed-Minimum-Water-Level-20_October_2020-WEI-M15-9_December_2020.pdf',
    'wsc_reference_index.html': 'https://wateroffice.ec.gc.ca/station_metadata/reference_index_e.html?stnNum=08JA023',
    'wsc_level_station.json': 'https://api.weather.gc.ca/collections/hydrometric-stations/items?STATION_NUMBER=08JA023&f=json',
    'operator_flow_naturalization_2022.pdf': 'https://www.getinvolvednechako.ca/wp-content/uploads/2022/08/Ecofish-Nechako-Flow-Naturalization-20221223-cert.pdf',
    'operator_phase1_report_2024.pdf': 'https://www.getinvolvednechako.ca/wp-content/uploads/2024/12/Final_WEI_Phase_1_Report.pdf',
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT / 'receipts.json'
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    for name, url in SOURCES.items():
        path = OUT / name
        if path.exists():
            if sha256(path.read_bytes()).hexdigest() != receipts[name]['sha256']:
                raise ValueError('Cached evidence changed: ' + name)
            continue
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            if name.endswith('.pdf') and not response.content.startswith(b'%PDF-'):
                raise ValueError('Expected a PDF')
            path.write_bytes(response.content)
            receipts[name] = {'url': url, 'final_url': response.url,
                              'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                              'sha256': sha256(response.content).hexdigest(),
                              'bytes': len(response.content)}
            print(name, len(response.content))
        except requests.RequestException as error:
            receipts[name] = {'url': url, 'error': str(error)}
            print(name, type(error).__name__)
        receipt_path.write_text(json.dumps(receipts, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
