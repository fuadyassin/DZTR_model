# Explicit storage–release root policies

The comparison workflow makes ambiguous implicit steps reproducible while
retaining the established strict solver. It is an experimental extension:
permanent data, physical constants, monthly release equations, existing
configurations and old checkpoint identities are preserved.

The [published formulation](https://hess.copernicus.org/articles/23/3735/2019/)
uses an inflow-dependent flood-zone release and a channel-constrained emergency
release. The numerical conventions below are proposed implementation choices;
the paper does not prescribe these root-selection rules.

## Why an implicit step can fail

The trapezoidal balance can be written `S + dt/2 * Q(S) = predictor`. Every
piecewise release branch supplies a candidate, which is checked against the
actual release law and its zone boundaries.

A downward jump in release at the flood/emergency boundary creates overlapping
balance ranges. Two distinct storage–release pairs can satisfy both equations.
At the first ambiguous Glen Canyon and Fort Randall steps, both roots are below
capacity. Clipping storage therefore does not resolve the choice.

An upward jump at the critical/normal boundary can instead leave a gap. At that
boundary the critical release is limited by available water, while the normal
branch starts at the critical release target. When
`critical_storage - dead_storage < dt * critical_release`, these two values
differ. No selection rule can supply an admissible root inside the resulting
balance gap. All four previously identified first no-root failures have this
form. Timestep, target feasibility and release-law changes are separate model
experiments.

## Declared conventions

| Policy | Treatment of multiple admissible roots |
| --- | --- |
| `reject` | Raise the existing strict error; this remains the default. |
| `minimum_storage` | Choose the smallest raw storage, and hence the largest release for that predictor. |
| `maximum_storage` | Choose the largest raw storage, and hence the smallest release for that predictor. |
| `nearest_previous_storage` | Minimize distance to the previous simulated physical storage. |

The nearest rule uses a deterministic minimum-storage tie break. Distances
within `max(atol_storage_m3, 64 * ulp(abs(predictor)))`, with the predictor scale
bounded below by one, count as tied. Current observations and future observations
never enter this choice. It is a continuity heuristic, not an inferred operator
instruction. Minimum storage can increase downstream release, and maximum
storage can retain more water; either choice can alter subsequent states.

Unique roots reuse the original solver exactly. Every convention retains errors
for negative predictors and missing roots. It accepts only physically validated
monthly tables and finite, nonnegative forcing. Source inflow transforms,
sampling frequency, integration duration and operating month remain explicit.

Selection occurs before the existing conservative spill correction. CSVs record
the selected raw storage/release, final physical state, admissibility tolerance,
root count, policy use and tie breaks. The initial row is a configured state,
with no inferred root. No roots are averaged or silently created.

Channel capacity continues to constrain the emergency branch. A flood-zone
release can exceed it under the unchanged formulation; this is counted in the
reports. A minimum-storage choice therefore does not establish downstream safety.

## Compare packaged reservoirs

From the repository root, using a fresh output folder:

```powershell
python examples/review_root_policies.py --reservoir ex1_Glen_canyon --reservoir ex4_Fort_randall --output outputs/root_policy_new
python examples/review_root_policies.py --all --output outputs/root_policy_collection_new
```

The equivalent installed command is `dztr-root-policy-review`. A copied
collection can be selected using `--manifest`. The default comparison includes
all four policies and a `matlab_fixed_point_reference`; repeat `--policy` to
restrict it. `--skip-matlab-reference` omits the expensive fixed-iteration
reference when it is unnecessary.

Targets and bounds are fitted on calibration observations only. Each policy
first runs the warmup/calibration prefix, then continues the identical trajectory
into separately scored validation. A held-out forcing failure cannot alter
calibration targets, scores, policy Pareto lists or finished searches.

The reference uses the same calibrated targets, initial state, native clock and
conservative overflow, with the literal 1,000-update MATLAB-compatible kernel.
It isolates the numerical iteration from parameter differences. Original
full-record coefficients, drifting historical clocks and historical overflow
belong to the separate `--mode legacy` collection workflow. MATLAB itself is not
executed by either Python comparison.

Output files include `root_policy_review.json`, `.csv`, `.md`, calibration
targets/bounds, per-policy trajectories and first-failure iteration diagnostics.
Input errors and scientific gates remain per-reservoir records. Manifest,
consumed inputs and package code are hash checked before/after the review.
`review_completed` means comparison reports were produced; individual policies
can still be blocked or their numerical reference invalid. No policy is promoted
automatically: `policy_selected` remains JSON `null`.

## Small calibration pilots

Both a policy and an evaluation budget must be supplied explicitly:

```powershell
python examples/review_root_policies.py --reservoir ex1_Glen_canyon --reservoir ex4_Fort_randall --policy minimum_storage --pilot-policy minimum_storage --optimize 32 --seed 42 --seed 43 --seed 44 --skip-matlab-reference --output outputs/root_policy_pilot_new
```

This requests 32 search evaluations per reservoir per seed: 192 total for the
two reservoirs and three seeds. Repeat with equal settings for another policy.
Default comparison commands launch no search.

The 72 coordinates, calibration-derived bounds, negative release/storage NSE
objectives and NSGA-II engine are reused. Invalid candidates consume their
evaluation and retain a reason. Even a search with no feasible archive preserves
its trials and vectors. Calibration Pareto representative IDs are fixed before
held-out evaluation, and validation failures remain separate from search status.

Baseline policy Pareto lists and pilot representatives use calibration scores.
Held-out NSE does not choose a policy, candidate, seed or budget. A documented
operating rationale or a predeclared calibration criterion is needed for any
production policy decision.

These bounded pilots exercise the new solver/objective integration. They do not
establish calibration convergence, recommended fleet budgets or useful validation
skill. Fixed-iteration convergence at one timestep is also separate from
convergence of an optimization across parameter evaluations.

This pilot runner does not resume the existing adaptive/checkpoint workflow.
Extending that workflow to a declared root policy requires its own configuration
and checkpoint identity; old checkpoints retain their original solver identity.

## Evidence

The complete 41-reservoir comparison produced the following outcomes. A completed
baseline includes both calibration and continuous held-out simulation with
water balance conserved; it does not establish predictive skill.

| Policy | Completed baselines | Not completed |
| --- | ---: | ---: |
| `reject` | 11 | 30 |
| `minimum_storage` | 24 | 17 |
| `maximum_storage` | 24 | 17 |
| `nearest_previous_storage` | 24 | 17 |

All 13 baselines previously blocked by multiple roots complete under each of the
three selection rules. Among the remaining 17 cases, nine have invalid fitted
physical targets, one needs unresolved external forcing, three are blocked by
signed forcing, and four have a missing root at the critical/normal boundary.
These categories assign each case its first blocking gate; some inputs have
additional issues. Tuyen remains excluded.

Of the 30 cases passing the common input/parameter gates, the MATLAB-compatible
reference completed 16 numerically valid trajectories, produced 12 numerically
invalid trajectories, and could not complete two. Its reported scores are
forensic evidence when its balance or release checks fail.

At the first ambiguous Glen Canyon and Fort Randall steps, fixed iteration
settled on the maximum-storage root. Over the complete Fort Randall trajectory,
the reference nevertheless produced a balance discrepancy of approximately
1.60 billion cubic metres at one timestep. The
[scientific review](../outputs/root_policy_science_review/README.md) examines
the 13 first multiple-root failures, four no-root failures and cycles in archived
MATLAB-compatible source-setting replays.

Equal-budget pilots used all three selection rules, two reservoirs, three seeds
and 32 evaluations per search: 18 searches and 576 evaluations. Of these, 543
were calibration feasible. Each search selected three distinct calibration
Pareto representatives before validation; 49 of the resulting 54 representative
evaluations also completed held-out simulation. A parameter vector can recur
across seeds or policies. The five held-out failures were Fort Randall seed-42
candidates. Neither those counts nor 32 evaluations establish convergence
or justify selecting a policy using held-out performance.

The [study summary](../outputs/root_policy_verification/study_summary.md),
[collection comparison](../outputs/root_policy_collection_final/root_policy_review.md)
and [figure index](../outputs/root_policy_figures/README.md) link the generated
evidence. The [combined audit](../outputs/root_policy_verification/verification.json)
checks 83 completed numerical trajectories and 44 exact comparisons against
previous strict-solver controls, and independently reconstructs the pilot
Pareto representatives from recorded calibration costs.

The complete test suite passed 1,009 tests with one Windows symlink permission
skip. The [offline wheel proof](../outputs/root_policy_verification/wheel/verification.json)
checks all 50 current model modules and nine legacy modules against packaged
source. All 47 pre-existing model modules, permanent inputs, packaging receipts
and historical verification records retain their previous bytes and identities.

## Modules

`simulation/root_policy.py` implements the opt-in step and native-clock runner.
`calibration/root_policy.py` provides an objective with a fixed declared policy.
`integrations/root_policy_review.py` runs comparisons and bounded pilots over
the permanent manifest. The existing simulation, parameterization, optimizer
and checkpoint modules are unchanged.
