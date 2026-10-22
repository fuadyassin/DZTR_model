# Additional reservoirs from Etienne's archive

The permanent [Etienne collection](../data/reservoirs_etienne_v1_20261021/combined_inventory.json) adds 21 reservoir records to the existing 41-record collection. The combined index therefore contains 62 reservoirs. The original collection, the 37-reservoir paper inventory and completed calibrations remain preserved. American Falls, Williston/Bennett, Ghost, Dickson and St. Mary are recognized as existing identities and are not added twice. Tuyen Quang remains excluded.

The [expansion report](../outputs/reservoir_expansion_v1_20261021/README.md) links the combined inventory and new parameter-estimate table. These observations are additional files in Etienne's archive; they are not all reservoirs in his eight-reservoir published calibration experiment.

## What was imported

Each new reservoir folder contains:

- `raw/`: original observations and available historical statistics, copied without executing archived programs.
- `observations_si.csv`: a daily calendar with storage in m³ and release in m³/s. Missing values stay missing; the unidentified inflow column is NaN.
- `observation_provenance.json`: selected source fields, unit scaling, missing-value declarations, actual coverage and timing limitations.
- `storage_daily_quality.csv` and/or `release_daily_quality.csv`: raw row counts, valid sample counts and within-day extrema.
- `monthly_targets_si.csv`: descriptive monthly P10/P45/P85 targets, using the shared explicit Hazen estimator. Unavailable target components remain blank.
- `monthly_sample_counts.csv`: actual observed days contributing to each month.
- `parameterization.json`: response-data route, source gates, target availability, catalog candidates and readiness.

Seven reservoirs have both target components: Aishihik, Hugh Keenleyside, Minidoka, Nechako, Revelstoke, Seton and Winnebago. Ten have storage only; four have release only. This gives 17 storage-target components, 11 release-target components and seven complete 12-by-6 target-estimate tables.

**Complete target estimates do not establish simulation or calibration readiness.** The new archive records do not identify time-varying inflow or adopted physical/initial-state model settings. Storage definitions and observation clocks also need review. All new profiles explicitly retain `parameters_ready=false`, `simulation_ready=false` and `calibration_ready=false`.

## Normalization decisions

Storage follows the archive reader's explicit million-m³-to-m³ scaling. The underlying storage coordinate remains unresolved: it may represent total, active or datum-relative storage. The conversion does not establish that coordinate.

Negative placeholders are explicitly declared per source, rather than universally removing negatives. Raw values remain in the archived files and placeholder counts are recorded. Other negative values are preserved and block component estimation until interpreted. Valid zero storage and release observations remain in percentile samples.

Hugh Keenleyside's 87 `#VALEUR!` spreadsheet error cells are explicitly declared missing. Other unexpected tokens fail parsing instead of being silently coerced.

Lake Chelan has adjacent cfs and already converted m³/s columns. Its importer selects the m³/s field, preserving the original file. Hungry Horse retains the observed input rates; the separate 0.993 archive experiment factor is not a unit conversion and is not applied.

Wrightsville has numerous within-day records with only a date. Its configured daily arithmetic mean is an **untimed storage proxy**, with original samples, daily counts and extrema preserved. It is not interpreted as end-of-day storage or an exact time-weighted mean. Its historical statistics count within-day samples as days; the new collection reports real calendar coverage.

Storage and release may cover different periods. Each component records its own support, and the inventory reports their actual simultaneous valid days. The collection does not fill gaps, extrapolate dates or reconstruct inflow from the responses.

## Descriptive estimates and future calibration

The current target estimates use **all available records for descriptive parameterization**. They are not fitted/held-out benchmarks and do not claim independent assessment skill. Before calibration:

1. Identify usable independent inflow, or explicitly define and validate a conditional reconstruction.
2. Adopt physical settings and reconcile storage coordinates and observation timing.
3. Resolve any reservoir-to-GRAND mapping. Catalog name matches are stored only as candidates; new verified GRAND IDs remain blank.
4. Select and freeze representative years using the existing at-least-65% calibration policy.
5. Refit observation-derived targets and bounds on calibration years, then use the existing run-manager backend and assess held-out observations.

The seven paired-response reservoirs are the natural first group for those admission checks. Revelstoke's storage record has fewer than two years and receives a short-record warning. Catalog capacities are retained as candidate metadata and diagnostics; they do not automatically become the adopted capacity or rescale observations.

## Reproduce and read the collection

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/import_etienne_reservoirs_v1.py import --config configs/etienne_reservoir_extension_v1.json --collection data/reservoirs_etienne_v1_20261021 --report outputs/reservoir_expansion_v1_20261021
python -B tools/import_etienne_reservoirs_v1.py verify --collection data/reservoirs_etienne_v1_20261021
```

An identical completed import is verified and reused; a differing configuration or changed packaged artifact is rejected. Normal reading/verification uses the archived project files and does not require the original external archive. Re-importing from its configuration still requires those declared source paths.

The combined collection can resolve an existing or new reservoir by ID, or by an existing verified GRAND mapping:

```python
from workflows.etienne_collection_v1 import load_observations

record, observations = load_observations(
    "data/reservoirs_etienne_v1_20261021",
    reservoir_id="etienne_Revelstoke",
)
```

Use the usual project Python path setup for `src` and the repository root, or the installed environment. The combined index points to existing files for the original 41 reservoirs and packaged files for the 21 additions. It keeps parameter-estimate availability separate from model readiness.

The first parser preflight stopped at an undeclared spreadsheet error token. Its partial files were preserved under `data/reservoirs_etienne_v1_20261021_incomplete_source_review`; it has no completed collection manifest and is not the active extension. Use the completed collection above.
