# Provider-computed signed inflow: Minidoka scenario

The completed example uses the official Minidoka `AF`, `ID` and `QT` records.
`ID` is a computed daily mean inflow; `QT` is instantaneous total release.
Negative ID values remain signed net boundary flux without attribution to
individual losses. No interpolation, clipping or newly reconstructed inflow is
applied. Provider independence and historical clock alignment are unverified.

The source stage retains both the previous local archive and current provider
responses. Historical USGS evidence on PDF page 185 of WSP 2134 describes a
115,000 acre-foot storage reference below the reported contents and a 107,240
acre-foot operating range. The explicit sensitivity scenario uses
`gross_storage = (AF + 115000) * 1233.48183754752` and a gross ceiling of 222,240
acre-feet. Those historical quantities are not certified for the whole
1985-2017 experiment. The zero DZTR dead-storage fraction remains an assumption,
separate from the historical power-penstock reference. Original negative
reference-storage values are retained in the source files.

The model integrates daily mean inflow, trapezoidal operating release and a
separate spill volume using the existing equations. That spill volume does not
identify instantaneous spill discharge. The Minidoka scenario consequently
uses **storage NSE only**, with weights storage=1/release=0. Release target
priors and bounds still use calibration-only QT observations. Release scores in
the report have explicit diagnostic/proxy names and do not establish accepted
total-release performance. A future joint calibration requires an appropriate
release observation operator or compatible interval-mean release records.

The fixed protocol uses mixed whole years with at least 65% actual fitting
support, Hazen percentiles, DDS seeds 42/43/44, 512 evaluations per seed,
nearest-previous-storage root selection and 0.0001 m3 volume tolerance. Every
forcing-block start uses frozen calibration-only monthly normal storage;
30 warmup days follow each start. Computed net-flux annual means provide regime
proxies, rather than independently measured naturalized wet/dry-year classes.

From the repository root:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/calibrate_provider_interval_v1.py run --config configs/minidoka_provider_interval_v1_20261021.json --output outputs/minidoka_provider_interval_v1_20261021
python -B tools/calibrate_provider_interval_v1.py replay --output outputs/minidoka_provider_interval_v1_20261021
```

Runs persist independent logs/checkpoints and reuse verified completed reports.
Resume budgets must remain within the fixed DDS horizon. Source files, code,
identity or horizon changes require a new experiment, rather than checkpoint
mutation. The original experiment retains its initially unassigned GRAND ID.
The completion publishes an additional identity-linked registry for verified
GRAND 413, preserving every numerical parameter and the original experiment.

Load parameters using the shared registry:

```python
from workflows.reservoir_v2 import load_profile
metadata, parameters = load_profile(
    "outputs/minidoka_completion_v1_20261021/release/registry.json", grand_id=413
)
```

Use this scenario's signed-net interval adapter for simulation; the native
endpoint-inflow adapter has different forcing semantics. The updated collection
is `data/catalogs/reservoir_collection_v5_20261021.json`, with its 62-row table
at `data/catalogs/reservoir_inventory_v5_20261021.csv`. All previous profiles and
primary observation files remain preserved. The remaining 17 additions have
explicit intake requirements in
`outputs/extension_source_intake_v1_20261021/source_intake_manifest.json`.
