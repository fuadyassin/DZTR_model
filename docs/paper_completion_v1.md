# Paper-source completion routes

This extension retains the 37-paper inventory and immutable archived profiles.
It adds source adapters, an enforced admission check, a common profile loader and
checkpointed signed-net calibration without changing the established solvers.

## Commands

Run from the repository in PowerShell, using its Python environment:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/paper_completion_v1.py prepare --output outputs/paper_completion_new
python -B tools/paper_completion_v1.py check --output outputs/paper_completion_new
python -B tools/paper_completion_v1.py run --output outputs/paper_completion_new
python -B tools/paper_completion_v1.py report --output outputs/paper_completion_new
```

Preparation requires a fresh directory. The runner processes the three signed
reservoirs sequentially, with three independent seed processes per reservoir.
Population 32, seeds 42/43/44 and 512 attempted evaluations include the baseline
and rejected candidates. Checkpoints, generation progress, immutable budget
snapshots and logs persist. Resume by repeating `run`; use `--budget 1024` for
an explicitly authorized extension retaining the same identity, bounds and seeds.
Numerical/input drift and shrinking budgets are refused. Exclusive locks prevent
concurrent writes. A stale lock after an abrupt process termination requires
checking that its recorded process is no longer running before removal.

Search selection uses the pooled calibration front and the established normalized
distance-to-ideal compromise. It is frozen before held-out assessment. The report
includes all three calibration-selected seed representatives and a fixed-initializer
comparison. Held-out failure does not select a replacement or undo a completed
search. Completion, skill, source certification and convergence are separate.

## Signed net forcing

Seminoe, Flaming Gorge and Navajo use `SignedNetPolicy`. Every signed source value
is preserved and enters the native endpoint trapezoid literally:

`S[t] - S[t-1] = dt/2 * (net[t] + net[t-1] - Q[t] - Q[t-1])`.

The positive and negative diagnostic components reconstruct net exactly. They
are not independent measurements of gross river inflow and evaporation; no
additional loss is subtracted. Unit inflow correction is mandatory. Seminoe and
Navajo retain storage-driven flood release. Flaming Gorge declares signed net
as the proxy for its existing inflow-dependent rule, without estimating an
unobserved gross inflow. This is an exploratory source/operating hypothesis.

Whole years are classified by signed annual net-flux means, weighted as in the
existing daily/monthly selector. The matched mixed-year protocol retains at least
65% of rows and available response pairs. Observed storage/release enter target
estimation only on calibration support. State is continuous through unscored years.
Nonnegative inputs reproduce native solver states and releases exactly. Only
admissible roots are accepted; nearest previous simulated storage resolves
multiplicity, and negative predictors or missing roots remain failures.

The native endpoint convention is deliberately retained here. It is not a claim
that each provider's flux is an instantaneous endpoint measurement. A later
verified daily-mean route must have a separate experimental identity.

## Bull Lake and the common loader

`load_paper_route(study, reservoir_id, seed=...)` returns route metadata and a
parameter object. Standard profiles resolve to their original verified registries.
Completed signed profiles resolve to the new signed release. Bull Lake requires
an explicit archived seed and returns its exact population-32 targets and cached
calibration trajectory; no new model or optimization call occurs.

Bull Lake remains a segmented conditional interval hindcast with observed clean-
block anchors and computed inflow dependent on observations. Its archived study
has zero held-out validation calls. It must not enter independent predictive
validation summaries or be relabeled as continuous standard calibration.

## Remaining source profiles

Canyon Ferry applies the confirmed inflow calendar crosswalk and calendar-month
rate conversion. Missing October-December 2000 rows remain missing. Copied storage
and release row alignment is a source hypothesis pending stock/flux confirmation.
Release values remain explicitly unverified and are absent from model forcing.
Pre/post October 1998 published capacities are metadata in a declared curve
schedule; their coordinate compatibility and a dynamic-capacity solver remain
open. No simulation bridges the gap or automatically switches capacity.

Trinity stages the pinned 2026-10-08 provider reference, correct raw unit
conversion, original source values and every quality flag. Missing/negative total
storage is unscored, revised/estimated flags remain visible, and surcharge above
normal pool is retained. Signed and missing forcing are not filled. A source
daily-mean closure ledger uses consecutive valid snapshots; it is descriptive
and does not invent losses or establish independent inflow. Quality blocks,
initial anchors, clock applicability, evaporation accounting and surcharge/floor
settings still need a distinct simulation policy before calibration.

Ghost and Bennett preserve source observations and inherited constants as
reference only. Missing storage-coordinate/curve and clock/boundary facts remain
enforced gates. Gross capacity is not substituted into an unidentified active-
storage series, and closure bias is not fitted into an inflow correction.

The all-37 prepared inventory retains blanks for unavailable calibrated targets.
When signed searches complete, a separate all-37 table adds their selected monthly
parameters. It preserves original GRAND mappings; unidentified IDs remain blank.
