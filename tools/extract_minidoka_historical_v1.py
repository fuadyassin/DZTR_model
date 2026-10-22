"""Save the historical coordinate evidence from the cached primary PDF."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.cache/vendor/pypdf'))
from pypdf import PdfReader
folder=ROOT/'outputs/minidoka_source_reconciliation_v1_20261021'
path=folder/'historical_storage_reference_excerpt.txt'
if path.exists():print(path.read_text())
else:
    reader=PdfReader(folder/'evidence/usgs_walcott_1966_1970.pdf');found=[]
    for page in range(165,min(205,len(reader.pages))):
        text=reader.pages[page].extract_text()
        if '13081000' in text and 'WALCOTT' in text.upper():found.append(f'PDF page {page+1}\n'+text)
    if not found:raise ValueError('Review historical reference page location explicitly')
    path.write_text('\n\n'.join(found),encoding='utf-8');print(path.read_text())
