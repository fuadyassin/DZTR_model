# Nechako source reconciliation

The paired-source review identified total-outlet candidates for Nechako as Skins
Lake Spillway (08JA013) plus Kemano Powerhouse (08FE002). The new source review
compares the archived storage with the operator's documented reference, checks
daily clock sensitivity on common dates, and flags unusual outlet observations.
It does not calibrate or change any existing observation or parameter profile.

Run these commands from the repository, using the project's Python interpreter:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/fetch_nechako_reference_v1.py
python -B tools/review_nechako_sources_v1.py
python -B tools/review_nechako_sources_v1.py --verify
```

The configuration is `configs/nechako_source_reconciliation_v1_20261022.json`.
Prepared artifacts, source receipts, plots and the decision are local under
`outputs/nechako_source_reconciliation_v1_20261022/`. The v7 inventory preserves
all existing monthly parameters. Completed review artifacts are immutable:
rerunning verifies their hashes, and new source decisions require a new version.
The default input paths require the private archive and earlier local reviews.

The [operator reference](https://www.getinvolvednechako.ca/wp-content/uploads/2021/11/Nechako-Reservoir-Licensed-Minimum-Water-Level-20_October_2020-WEI-M15-9_December_2020.pdf)
documents total storage of 23.85 billion m3 at 2800 ft and 7.10 billion m3 of
licensed live storage. The local gauge metadata declares an assumed datum;
the offset to the operator's elevation coordinate remains unresolved. The
licensed minimum is distinct from a physical model dead-storage threshold.
Do not replace the capacity of the archived storage coordinate or translate
the record by matching an observed maximum.

`workflows/source_clock_review_v1.py` computes signed budget diagnostics for
explicit daily clock scenarios. A positive shift uses a preceding daily flow
value; the adjacent-day average is a mean-to-mean approximation. Every scenario
uses identical supported dates, missing components remain missing, and no
scenario is chosen automatically. Fewer negative residuals do not certify
timing or physical forcing. The reconstructed budget depends on observed
storage and release and is never independent inflow. Changing the storage
curve's slope changes these diagnostics and requires a new review.

The [operator-commissioned naturalization study](https://www.getinvolvednechako.ca/wp-content/uploads/2022/08/Ecofish-Nechako-Flow-Naturalization-20221223-cert.pdf)
supports the two outlet pathways and distinguishes downstream routing from
the reservoir boundary. Its downstream naturalized flow must not be substituted
for independently observed reservoir inflow.
