# Observed storage diagnostics before calibration

Use the observation audit to examine storage assumptions before interpreting a
storage calibration failure. The audit performs no simulation, parameter
estimation or optimization. It never changes packaged observations, fixed model
constants or solver defaults.

From the repository root, run the Ghost and Trinity checks with a fresh output
directory:

```powershell
python examples/review_storage.py --output outputs/my_storage_review
```

For the active collection, including daily and monthly reservoirs:

```powershell
python examples/review_storage.py --all --output outputs/my_collection_storage_review
```

Select individual members with repeated `--reservoir` arguments. Tuyen remains
excluded by the permanent collection manifest. The installed console command is
`dztr-storage-review`; running the example does not require installation. Use
`--no-lags` to omit the descriptive timing comparisons.

## What the residual means

For a native interval between two recorded storage states, the audit computes:

```text
residual = observed storage change - (inflow volume - observed release volume)
```

A positive residual means the recorded storage gain exceeds the net input
calculated under the stated hypothesis. A negative residual means it is smaller.
The units are cubic metres; dividing by the interval duration gives an equivalent
rate in cubic metres per second. Neither sign identifies a cause. Evaporation,
seepage, diversions, precipitation, storage datum, observation derivation and
measurement timing require additional evidence.

The three paired flow conventions are explicit hypotheses:

| Convention | Inflow/release volume between previous and current storage rows |
| --- | --- |
| `endpoint_trapezoid` | Duration × average of previous and current rates |
| `interval_mean_end` | Duration × current row's rate |
| `interval_mean_start` | Duration × previous row's rate |

Storage is treated as an endpoint state for this arithmetic. The source unit
conversion and calendar checks do not establish whether storage was measured at
the start/end of a day, whether each flow is an instantaneous rate or a mean, or
whether inflow was measured independently. Inflow and release may use different
conventions; the three paired hypotheses do not exhaust mixed interpretations.

Monthly records remain monthly. Calendar integration uses the elapsed time
between consecutive month starts; the source fixed duration is reported as a
separate compatibility hypothesis when it differs. A monthly mean storage is not
an endpoint state. Month labels alone cannot resolve which flux belongs between
two storage snapshots. These reports therefore do not establish physical monthly
closure.

## Periods, timing and incomplete data

Each reservoir's configured warmup, calibration and validation records are sliced
**before** integration or shifting. Both storage endpoints and every flow sample
used in an interval stay inside that reporting period. No calibration statistic
depends on validation observations. The interval crossing a period boundary is
excluded; cumulative balances restart in each period.

For each convention the audit compares unshifted flow with four separate timing
perturbations: inflow delayed/advanced one sample, and release delayed/advanced
one sample. At row `j`, lag `+1` reads row `j-1`, while lag `-1` reads row `j+1`.
The five comparisons use their common finite support. Their support can be
smaller than the full baseline ledger, so compare the unshifted **common-support**
result with the shifted results. One sample means one day or one month according
to the native frequency. A lead uses subsequent observed data and is a
descriptive diagnostic, not a forecast or a proposed production transformation.
No preferred lag or convention is selected.

Missing/nonfinite values invalidate affected intervals without being filled.
Invalid intervals break cumulative segments. Seasonal sums and sums over valid
intervals are not whole-period closure when coverage is incomplete. In the JSON,
`complete_closure` means all period intervals are available for the closure
calculation, **not** that the residual is small or zero. The report contains each
complete contiguous segment separately. Missing date rows, duplicates,
nonmidnight times and irregular calendars are rejected rather than interpolated.
Signed flows remain in diagnostic arithmetic and receive explicit review flags.

Raw observed inflow is always reported. Where the configuration differs, a
separate `configured_balance` hypothesis uses:

```text
effective inflow = inflow_correction × (factor × raw inflow + offset_m3s)
```

This is the balance forcing used by the model, not a correction established as
true by the audit. Storage change and observed release are unchanged. The audit
does not assert that external-forcing reservoirs have complete measured inputs.

## Outputs and reusable interface

The review produces `storage_review.json`, a flat `storage_review.csv`, and a
readable `storage_review.md`. Each reservoir/period contains:

| File | Contents |
| --- | --- |
| `intervals.csv` | The three baseline ledgers per declared forcing/clock, native durations, residuals, contiguous segment IDs and cumulative residuals |
| `summary.json` | Coverage, residual magnitude/rates, segment closure, record flags and timing comparisons on common support |
| `monthly.csv` | Seasonal statistics by destination month, without claiming continuous closure across years |
| `extremes.csv` | Ten largest absolute residuals per baseline with their observation context |
| `flags.csv` | Nonfinite/signed records, storage below the configured dead threshold, above capacity and positive release recorded below dead storage |

Storage-limit flags are conflicts with the **configured assumptions**. They do
not automatically establish bad measurements: active versus total storage,
vertical datum, historical capacity changes and daily measurement timing remain
unresolved. Initialization is compared with the first configured simulation
record separately; it is not reset and does not by itself explain bias after
warmup.

The core API is
`dztr.evaluation.storage_audit.audit_storage(period_frame, ...)`, returning a
`StorageAuditResult`. The integration API is
`dztr.integrations.storage_review.run_storage_review(manifest, output, ...)`.
The runner verifies pinned SI inputs, protects collection source files from
output overlap, isolates individual input failures and hashes code/inputs before
and after the review. Successful arithmetic does not certify solver feasibility
or model readiness.

The audit is intended to guide an explicit decision about measurement
interpretations and physical storage constants. A later calibration stage can
use the agreed model assumptions and a fixed tested solver profile, with
validation reserved for reporting.

The completed [Ghost/Trinity diagnostic](../outputs/storage_diagnostics_20261008/README.md)
contains the findings and [figures](../outputs/storage_figures_20261008/README.md).
It audited both reservoirs, without expanding or changing the permanent data
collection. The entire collection can use the same runner when requested.
