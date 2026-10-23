# Winnebago lake-only relative-storage candidate

The source review distinguishes Lake Winnebago alone from the larger four-lake
managed pool. An explicitly exploratory constant-area candidate uses the lake
area in USGS station metadata, with each gauge transformed into its own NAVD88
reference before averaging. Both gauges are required. Published datum constants
are checked against June 2020 instantaneous pairs, while their extension across
the full archive remains an assumption.

`workflows/lake_level_proxy_v1.py` computes a volume anomaly above a declared
elevation plane. That plane is not a bed elevation; the candidate estimates
neither total volume nor physical capacity. Negative anomalies and missing
gauges remain unchanged. A fixed trailing seven-day triangular filter can be
applied to both storage and release, with a recorded three-day effective delay.
It uses no future values. The filter/difference commutation check verifies the
processing algebra, not independent reservoir forcing or source correctness.

Before fitting on selected years, `eligible_source_windows` requires every input
day in a filtered response to belong to the fitting phase. Differences of
filtered storage also require its preceding window. Recheck the actual 65%
fitting fraction after these exclusions. This prevents smoothing across a
calibration/held-out year boundary from contaminating fitted targets.

The Appleton outlet candidate uses an explicit drainage-area proration and
USGS primary-flow method. Routing and daily observation clocks remain source
decisions. A next-day-Q sensitivity is historical-only and never enters the
parameter profiles. The old extra-field offset is not adopted as true depth.

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/fetch_winnebago_reference_v1.py
python -B tools/review_winnebago_sources_v1.py
python -B tools/review_winnebago_sources_v1.py --verify
```

Configuration: `configs/winnebago_source_reconciliation_v1_20261022.json`.
Default paths require the private archive and preceding local source reviews.
Outputs are local under `outputs/winnebago_source_reconciliation_v1_20261022/`.
Completed artifacts are pinned; new source interpretations require a new version.
Two separate descriptive target profiles use the shared Hazen estimator and
replace none of the existing parameters. Admission still requires compatible
capacity/dead storage, source/clock decisions and suitable forcing. Refit on
frozen mixed calibration years before optimization. A conditional net-flux
experiment cannot demonstrate independent predictive validation.

Primary sources: [USGS pool study](https://pubs.usgs.gov/wri/1980/0107/report.pdf),
[Oshkosh](https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=04082500),
[Stockbridge](https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=04084255),
[Appleton](https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=04084445).
