# Source admission after the Etienne collection import

Run the whole assessment once, including all 21 decisions, seven signed
water-balance reconstructions and the declared daily-interval baseline probes:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/assess_reservoir_extension_v1.py build --config configs/reservoir_extension_assessment_v1_20261021.json --output outputs/reservoir_extension_assessment_v1_20261021
```

The first build needs the local HORIZON files and identity workbook. Its output
packages the raw supplementary CSVs, SI observations, provenance and identity
evidence; verifying or reusing a completed assessment does not need those
external paths:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/assess_reservoir_extension_v1.py verify --output outputs/reservoir_extension_assessment_v1_20261021
```

Build requires a fresh output directory. An existing completed directory is
verified and reused; a partial directory is never overwritten. Original imports,
calibrations, catalogs and model equations are preserved.

`I_net = Q_mean + diff(S)/86400` retains missing dates and negative values. It
assumes storage endpoints and current-day mean total release. Its numerical
closure is an algebraic identity, not evidence for the true inflow, time clock,
or completeness of outlet observations. In particular, it uses assessment
responses in forcing and cannot supply independent predictive validation.

Supplementary HORIZON records are separate alternatives, never silent merges
with Etienne observations. Exact conversions are 1 acre-foot =
1233.48183754752 m3 and 1 cfs = 0.028316846592 m3/s. Neither HORIZON's later spline
filling nor nonpositive-record removal is applied. Provider-reported inflow can
itself be balance-derived; independence remains unverified.

Baseline probes use all-record descriptive targets, explicitly provisional
capacity references, zero dead fraction and Q99 channel proxies. They run the
existing daily-mean volume-conserving interval solver, including spill, using
`nearest_previous_storage` only to choose among actual valid roots. Complete
consecutive blocks receive observed initial storage anchors and unscored first
rows. The probe stops on the first failed block. It does not choose another
period, alter capacity, fabricate roots or optimize parameters.

The consolidated 62-row table includes all 72 monthly target fields where
available. It distinguishes frozen calibrated profiles from descriptive target
components and leaves unavailable values blank. Bull Lake retains its separate
interval-workflow status rather than being misrepresented as a standard
monthly profile. A verified GRAND identity does not verify capacity or source
quality, and a descriptive target table does not establish readiness.

Once source admission succeeds, use existing DDS/calibration modules with a
frozen whole mixed-year selection covering at least 65% of the data, fit targets
and bounds on calibration observations only, select parameters without held-out
scores, and label all response-conditioned forcing/anchors explicitly. This
assessment command intentionally runs zero optimization evaluations while
source admission remains unresolved.
