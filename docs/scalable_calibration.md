# Calibration budgets for a reservoir fleet

The scalable workflow needs reservoir-specific inputs, a tested budget policy,
and independent evidence that saved computation does not miss material
calibration improvements. Trinity provides a working example, not a universal
evaluation budget for other reservoirs.

## What is available

The NumPy NSGA-II optimizer supports fixed-budget searches and verified
complete-generation continuation. Saved evaluation archives support
retrospective stopping diagnostics and seed comparisons; the
[adaptive runner](adaptive_calibration.md) checks the same provisional policy
during a search. Fleet preflight validates inputs before expensive computation.
A general manifest-driven fleet scheduler remains further work.

Review a completed study without new optimization evaluations:

```powershell
python -m dztr.calibration.budget_review --study outputs/trinity_convergence --output outputs/trinity_budget_review --policy configs/stopping_policy.example.json
```

The review uses calibration costs only. Candidate selection and validation
reporting follow the existing [optimization policy](optimization.md).

## Objective progress and parameter identification

Generalized percentile parameterization is deterministic for a fixed observation
record, calibration period and quantile method. Evaluation-budget convergence
applies to the optimizer. The stability of empirical monthly quantiles is a
separate question about observation adequacy: a future analysis could compare
calibration subperiods/record lengths or resample hydrological-year blocks
within calibration data while preserving month and season structure. Quantile
resampling is not implemented, and it would not establish a parameter posterior.

A stable release/storage trade-off front does not show that the 72 monthly
targets have been uniquely identified. Several parameter sets can reproduce
similar reservoir behavior. This distinction follows the hydrological concept
of [equifinality](https://doi.org/10.1016/S0022-1694(01)00421-8); multiple
objectives also expose different aspects of model error, rather than resolving
every ambiguity in the targets. See [Gupta et al. (1998)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/97wr03495).

Report objective-space progress and parameter variation separately. Compare
matching representative roles across seeds using the common bound widths, as
in [the Trinity study](convergence.md). Parameter spread is descriptive optimizer
variation, not a confidence interval. Requiring all parameters to agree before
stopping could spend indefinitely on weakly identified coordinates. A future
parameter-identification analysis could combine sensitivity with storage-zone
occupancy to identify targets that the simulated operation rarely informs.

## An illustrative stopping diagnostic

[Official termination guidance](https://pymoo.org/interface/termination.html)
uses recent windows and hard search limits. The repository's diagnostic is its
own combined rule, with the following example settings. These values are for
retrospective testing; they are **not validated production recommendations**.

| Setting | Illustrative value |
| --- | --- |
| Minimum evaluations before considering stagnation | 512 |
| Evaluation interval between checks | 128 |
| Evaluation window compared at each check | 128 |
| Consecutive passing checks required | 3 |
| Hard evaluation limit | 4,096 |
| Fixed reference in negative-NSE coordinates | `[1, 1]` |
| Normalized hypervolume gain limit | 0.002 |
| Full-front movement limit | 0.005 NSE units |
| Release and storage extreme-score change limits | 0.005 NSE units each |

At evaluation count `t`, compare the accumulated feasible nondominated front
with the front at `t - 128`. All three progress conditions must pass:

1. Hypervolume gain divided by `(reference[0] + 1) * (reference[1] + 1)` is
   strictly below 0.002. With `[1, 1]`, the denominator is four, the area from the reference
   to the theoretical negative-NSE ideal `[-1, -1]`. This normalization does not
   measure a percentage of achieved convergence.
2. Symmetric Hausdorff distance between the full fronts is at most 0.005 using
   the maximum absolute coordinate difference, or L-infinity distance. Both
   directions are checked so changes in either front's coverage are visible.
3. The best release NSE and best storage NSE each change by at most 0.005.

Keep the reference and objective units fixed across checks and seeds. A feasible
front point outside the reference rectangle contributes no positive hypervolume
area, so the review must retain its exclusion diagnostics and compare the full
front. Both comparison fronts must have complete reference coverage, and each
window must contain a new finite feasible evaluation. Empty feasible fronts and
insufficient history cannot establish stagnation. Rejected evaluations still
consume budget. Count settings are population aligned. With this example,
512, 640 and 768 are the earliest three checks that can produce a suggestion;
checks below the minimum never accumulate patience. Hypervolume's dependence
on the reference is discussed in [the primary indicator analysis](https://www.cmap.polytechnique.fr/~dimo.brockhoff/publicationListFiles/abbz2009a.pdf).

The assessment reports `insufficient_history`, `continue_search`,
`stopping_candidate` or `budget_exhausted`. It preserves the earliest suggestion
and audits later saved windows, recording `candidate_invalidated` if subsequent
progress or a guard failure contradicts that suggestion. The end of a saved
archive below the cap is not a stopping recommendation. The diagnostic does not
end or extend an optimizer run. The adaptive runner enforces the policy hard
cap independently of stagnation signals, including during audit extensions.

Three small windows remain a heuristic on a correlated search trajectory:
searches can pause temporarily and later improve. Agreement across seeds strengthens a diagnostic
but does not prove the true front has been found. [The official convergence
guide](https://pymoo.org/getting_started/part_4.html) distinguishes progress toward
an observed approximation from convergence to a known front.

## Validate the policy on a representative cohort

The proposed next experiment is larger calibration archives at 2,048 and 4,096
evaluations, followed by a reservoir cohort comparison of populations 16, 32
and 64 at equal evaluation budgets and multiple independent seeds. This
experiment has not been run. Population 16 is a pilot choice; there is no
established fleet rule relating that size to the 72 parameters. The original
[NSGA-II paper](https://dynamics.org/~altenber/UH_ICS/EC_REFS/MULTI_OBJ/DebPratapAgarwalMeyarivan.pdf)
also treated algorithm settings as empirical choices.

Choose reservoirs spanning capacity, storage-to-inflow ratio, seasonality,
operating purpose, observation length and completeness. Estimate percentiles
and bounds from each reservoir's calibration period only. Use calibration
archives to develop the stopping policy, then freeze its settings before an
independent cohort audit.

For apparently stopped searches, continue to a prespecified larger budget and
run additional independent seeds. Record saved evaluations, subsequent
hypervolume/extreme-NSE gains, front disagreement, false-stop frequency and
failure rates. Continued runs test whether a plateau persisted; independent
seeds test whether another search finds materially better trade-offs. Validation
scores must not tune the stopping thresholds, population size or budget.

## Measured retrospective Trinity review

Reviewing the existing 1,024-evaluation archives made no new optimizer calls.
All three seeds report `continue_search`, with no earliest stopping candidate.
The final check compares evaluations 896 and 1,024:

| Seed | Normalized HV gain (% of theoretical area) | Full-front movement | Extreme-score drift | Status |
| --- | --- | --- | --- | --- |
| 42 | 0.1290% | 0.01055 | 0.00257 | `continue_search` |
| 43 | 0.2696% | 0.00548 | 0.00408 | `continue_search` |
| 44 | 0.3046% | 0.01130 | 0.00368 | `continue_search` |

All final full-front movements exceed the illustrative 0.005 threshold, although
each extreme-score drift is smaller. Stable extrema alone would therefore miss
continuing changes in the trade-off front. At 1,024 evaluations, between-seed
front distances are 0.03117 for 42/43, 0.03096 for 42/44 and 0.01462 for 43/44;
none meets the review's separate illustrative agreement tolerance of 0.01.
This motivates a proposed 2,048-evaluation experiment per seed, not a validated
production budget or evidence that convergence will occur at that count.

`checkpoint_diagnostics.csv`, `seed_status.csv` and `cross_seed_fronts.csv`
retain full-precision values. Export their three progress signals with:

```powershell
python examples/plot_budget_review.py --output-dir outputs/trinity_budget_review
```

The resulting `budget_diagnostics.png` shows the thresholds and minimum budget.
Hypervolume gain is displayed as a percentage of the fixed theoretical area;
front movement and extrema drift retain raw NSE-coordinate units.

## Fleet input and output contract

[The example manifest](../configs/fleet_manifest.example.json) references a
separate configuration for each reservoir. Supply physical constants and their
provenance per reservoir; Trinity's initial state, channel capacity and 0.92
inflow factor must not become fleet-wide assumptions.

| Input | Required information |
| --- | --- |
| Identity and sources | Stable reservoir ID and source paths; preserve provenance and computed hashes in outputs |
| Observations | Calendar dates plus storage, inflow and release columns; normalized to m3 and m3/s |
| Units | Explicit source storage/flow units consistent with any unit-bearing columns; SI units for generated outputs |
| Physical configuration | Explicit capacity, initial storage/release and inflow correction; resolved channel capacity, dead-storage fraction and flood mode |
| Experiment | Explicit calibration/validation boundaries and warmup; resolved quantile method, bounds and optimizer settings |
| Task eligibility | Availability of usable observations for each calibration objective and monthly target estimation |

Canonical CSV and Excel observations are supported now. Parquet is a possible
future input adapter. Keep missing observations and missing dates explicit;
do not fill them with zeros. Preflight checks placeholders, coverage, valid
units, objective data, model constraints and bound feasibility.

The Trinity example declares `storage: "mcm"` because its source column is
`Storage(mil_m3)`. Preflight checks this declaration and converts values to m3.
Configuration and policy paths are relative to `--repo`; observation paths can
be relative to that root or absolute. Reservoir IDs are safe directory names
and must be unique ignoring case.

Missing channel capacity uses calibration release P99; omitted dead-storage
fraction and flood mode use the generic model defaults. Every resolved value
has a source status in the provenance file. Current calibration bounds remain
P5–35/P35–75/P75–95; they are recorded rather than supplied by a new custom
bounds field in the manifest.

```powershell
python -m dztr.calibration.fleet --manifest configs/fleet_manifest.example.json --repo . --output outputs/fleet_preflight
```

This command performs preflight, not optimization. It writes `fleet_status.csv`
and a top-level `summary.json`. Each ready reservoir receives
`observations_si.csv`, `resolved_config.json` and `preflight_provenance.json`.
Use a new or empty output directory; preflight preserves existing artifacts
instead of replacing them. The generated configuration points to its canonical
SI copy and can be passed to the existing single-reservoir search or study.
Individual jobs report ready, skipped-placeholder, skipped-unavailable or error
status independently. An unavailable observation record cannot support
release/storage calibration. Simulation with supplied
monthly rules and parameter transfer to unobserved reservoirs are separate
tasks requiring explicit inputs and provenance.

The exploratory checkpoint study now uses one process pool across Trinity seed
jobs and a qualified Gardiner job, with no nested worker pools. It isolates
failures, preserves per-job status and reason, and verifies search identity and
artifact hashes before resuming. Vectors and scores are stored for every trial;
daily trajectories are replayed for selected representatives. Generalizing this
example to a manifest-driven fleet scheduler remains further work.

The [population study](population_study.md) adds a flat population/seed job
matrix with independent receipts and checkpoint recovery. It verifies complete
cached sources before searching, compares equal evaluation budgets, and reports
calibration-only seed ranges and front disagreement. It supplies experimental
evidence for population and budget choices; the current Trinity pilot does not
establish defaults for a fleet of reservoirs.

The [adaptive runner](adaptive_calibration.md) implements complete-generation
checkpoints containing population IDs/order, RNG state, evaluation archive,
counters, settings and input/code hashes. Duplicate keys are reconstructed from
the saved archive. Partial generations are not resumable; work since the latest
checkpoint must be repeated after interruption. Completed archive CSVs alone
do not restore the optimizer.
