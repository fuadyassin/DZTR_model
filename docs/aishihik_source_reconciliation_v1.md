# Aishihik storage candidate and shared parameterization

`workflows/aishihik_storage_candidate_v1.py` prepares an explicitly exploratory
interpretation of the stage-storage equations in the embedded INRS report and
technical clarification in the [operator-hosted PDF](https://yukonenergy.ca/media/site_documents/AtkinsRealis_Hydrotechnical_Study_2024.pdf).
The southern gauge 08AA005 has a documented 911.565 m reference. The inferred
reduced polynomial coordinate is `x = gauge_level + 911.565 - 912`. It
reproduces the appendix's approximately 0.011% branch discrepancy. The
polynomial coordinate and printed additional factor of 3600 still need source
confirmation; this implementation is not a certified correction.

The converter requires `allow_exploratory=True` and the explicit station
number. It preserves the independent branch fits without adding the reviewer's
proposed lower intercept. It supports elevations 912-916 m, preserves gaps,
and exports daily-mean and causal trailing-three-day mean candidates. Values
represent incremental volume candidates, never total lake storage or verified
endpoint states. The northern gauge cannot inherit the southern gauge datum.

The source-review tool compares gauges and signed water budgets, then generates
separate descriptive monthly targets using the existing shared Hazen estimator.
The one signed outlet subtraction is retained raw and explicitly excluded from
the nonnegative release estimation view. No calibration is admitted or active
parameter profile replaced. Candidate targets use all available data for source
review; they must be refitted on calibration years before optimization.

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/review_aishihik_sources_v1.py
python -B tools/review_aishihik_sources_v1.py --verify
```

Configuration: `configs/aishihik_source_reconciliation_v1_20261022.json`.
Local output: `outputs/aishihik_source_reconciliation_v1_20261022/`.
The default paths require the private records and preceding local reviews.
Completed artifacts are hash-pinned; new source interpretations require a new
version. Capacity, dead-storage reference and observation clocks remain explicit
admission decisions. Conditional net forcing remains response-dependent and
cannot support independent validation claims.
