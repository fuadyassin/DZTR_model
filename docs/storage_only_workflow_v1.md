# Storage-only DZTR workflow

The run manager now accepts `prepared_storage_v1`. It estimates monthly storage
targets from storage observations (including explicitly prepared level/geometry
proxies) and calibrates them with the scalar cost **1 − NSE(storage)**. Observed
release is physically removed from the training CSV and is used only after
calibration choices have been frozen. The existing numerical equations and
previous workflows remain versioned and unchanged.

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/dztr_run_v7.py plan --config configs/run_manager_storage_only_paper_v1.json
python -B tools/run_storage_cohort_v1.py --config configs/run_manager_storage_only_paper_v1.json --output outputs/managed_storage_only_paper_v1_20261020
```

The persistent runner executes the manager and generates its report after both
experiments finish. Its adjacent `_job.json`, stage status, per-job execution
logs and eight-evaluation checkpoints distinguish running, failed and completed
work. All project cache, inputs, outputs, temporary files and logs stay on D:.

## Release priors and the user's mean-flow proposal

The primary configuration requests all 32 completed monthly paper reservoirs.
It uses the local GRanD v1.1 `DIS_AVG_LS` field as a mean-flow scale wherever a
previously verified GRAND ID and positive discharge exist:

\[
\overline Q_{\mathrm{target}}=\mathrm{DIS\_AVG\_LS}/1000.
\]

`DIS_AVG_LS` is average river discharge in **litres per second**, rather than an
engineering maximum channel discharge. The existing channel capacity is retained
as a separate physical input. See the
[GRanD/GDW attribute documentation](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GDW/GDW_TechDoc_v1_delta.pdf).
The packaged values come from the user's local v1.1 DBF; the newer documentation
is cited only for the field definition, not as a replacement dataset.

For each percentile p = 5, 30 or 97, the release prior is:

\[
Q_{p,\mathrm{target}}=\overline Q_{\mathrm{target}}
\operatorname{median}_{d\ne\mathrm{target}}
\left(\frac{P_p(Q_{\mathrm{naturalized},d})}
{\overline Q_{\mathrm{naturalized},d}}\right).
\]

The seven donor reservoirs are Big Horn, Gardiner, Dickson, E. B. Campbell,
Oldman, St. Mary and Waterton. Their archived simulated naturalized records span
2011–2016. The target reservoir and any donor with the same known GRAND ID are
excluded. At least three unique donors are required. The relationship uses the
same explicit Hazen/linear estimator as the experiment. Valid zero values are
retained, and the observed-flow column in the archived files is ignored.

Station mappings were checked against each named reservoir's archived CR1
coefficient block: its annual P5/P30/P97 equals the matching station statistics.
This check resolved the Oldman/Waterton/St. Mary station assignments. The first
unsealed packaging attempt is explicitly marked incomplete; experiments consume
only `data/storage_release_priors_v1_20261020_final`.

The default primary policy is `grand_then_calibration_inflow`. Where no usable
mean is available for a verified GRAND ID, the scale is a volume-weighted mean
effective inflow over intervals whose **two endpoints are calibration samples**.
It respects the native fixed or calendar time steps and retains signed rates.
A nonpositive scale is rejected. This fallback is labelled
`calibration_effective_inflow_proxy`; it is not reported as a GRAND measurement
or as naturalized output. `grand_only` disables the fallback. No missing GRAND
IDs are guessed or reassigned.

The source-channel limit is explicit: the three release targets are truncated
to the existing channel capacity when necessary. The original targets, modified
targets and number of truncated levels are recorded. The inferred P99 remains
a diagnostic proxy; it does not replace the fixed channel capacity.

This annual shape relationship is an exploratory regional transfer assumption.
It does not create a naturalized target-reservoir time series or establish global
transferability. Donor P10/P90 ratio spread is descriptive, not a confidence
interval. These donors are from the Saskatchewan basin and can be unrepresentative
of other climates or operating purposes. Their independence from historical
observations has not been verified.

## Direct simulated-naturalized option

The same release-prior adapter also reads a dedicated SI CSV with exactly
`date,naturalized_outflow_m3s`, or Etienne's three-column crude format. Sources
must have consecutive daily or month-start dates; no missing dates are filled.
Naturalized values may be missing but must otherwise be nonnegative.

`scope: calibration_overlap` estimates annual P5/P30/P97 from the original
calibration support only. Fitting needs all 12 months and at least 30 daily or
12 monthly samples. Monthly source samples are eligible only when all represented
target samples in that month are calibration samples. A shared parameterization
campaign that already supplies naturalized outflow can hand it over directly,
without duplicating the prior specification.

The secondary experiment requests the seven archived cases. With the current
frozen masks, only Big Horn, Gardiner and E. B. Campbell have usable 2011 overlap.
The other four are explicitly gated for lack of overlap; their original masks
are not changed to admit the prior.

An optional `external_reference` scope requires explicit start/end dates and an
exploratory rationale. It is labelled as a retrospective external climatology
and is not the direct-overlap experiment used in this campaign.

## Estimation, optimization and selection

- Initial monthly storage targets use **P10/P45/P85** of calibration storage.
- Default storage bounds use monthly **P5–P35, P35–P75, P75–P95**. The optional
  `etienne_annual_envelope` uses cross-month envelopes with lower P5/P20/P60 and
  upper P35/P75/P95.
- Bounds are explicitly intersected with the adopted dead-storage/capacity
  constraints, using a documented numerical separation for strict ordering.
  Changed bounds and initializer cells are counted. An empty intersection is
  rejected. Capacity and dead storage are never raised or lowered to fit data.
- Initial storage uses the first row's storage observation if that row is outside
  assessment, finite and within capacity; otherwise it uses the
  calibration-derived normal target.
  Initial release uses the first effective inflow clipped to zero/channel capacity.
- DDS adjusts **36 monthly storage targets**. Release targets stay fixed.
- Original mixed-year masks are preserved; at least **65%** of samples must be
  calibration. Continuous state carries across unscored and assessment years.
- `pooled` and `equal_regime` storage scoring are explicit choices. The primary
  experiment uses pooled raw storage NSE, seeds 42/43/44, perturbation 0.2, a
  64-evaluation checkpoint and a frozen 512-evaluation DDS horizon.
- Each seed and the pooled choice minimize calibration storage cost. Deterministic
  ties use seed then evaluation ID. Held-out storage and hidden release never
  choose a seed or prior relationship.

For geometry-derived storage, this first storage-only adapter uses point proxy
NSE and preserves the conversion/datum/uncertainty receipt. It does not silently
reinterpret daily mean levels as endpoint storage or use the uncertainty band as
a different objective. The separate joint geometry-band workflow remains intact.

Continuation within the frozen horizon uses `tools/calibrate_storage_v1.py run
--evaluation-budget 128` with the unchanged configuration and a preserved copy
of the campaign. Copying a sealed managed campaign before continuation keeps
the original manager receipts verifiable. Changing horizon, bounds, prior,
objective or implementation requires a new campaign.

## Etienne comparison and limitations

The [published study](https://doi.org/10.1139/cjce-2021-0507) explicitly identifies
the storage-observed/outflow-unobserved scenario as missing from its experiments.
Its CR1 naturalized-outflow percentile method supports the prior design. The
local Big Horn `TRY3/readme_TRY3.txt` records a storage-NSE calibration trial,
but that does not establish a storage-only data-availability experiment: archived
initial parameters could still use observed release. We reuse scalar storage NSE,
DDS and the documented bounds as inspiration, while hiding target release here.

The primary mean-flow relationship is the user's new exploratory proposal, not
an Etienne implementation. Reconstructed inflows and adopted physical/channel
metadata may themselves depend on historical observed storage or release. These
are therefore conditional data-availability benchmarks, not independent ungauged
or forecast-skill validation.

Both 37-row GRAND inventories preserve source-gated reservoirs with blank
parameters. No-feasible searches, assessment failures, mean-flow fallback,
target truncation, source identity, fitting years, initial conditions and prior
dependency are exported. The report compares the fixed-prior baseline, per-seed
results and earlier routes from cached trajectories, with mass-balance checks,
seasonal/regime diagnostics and plots. It makes no convergence claim.
