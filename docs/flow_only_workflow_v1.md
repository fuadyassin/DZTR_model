# Flow-only calibration through the run manager

This adds release-only calibration to the existing data-availability workflow.
Storage observations are hidden during target estimation, bound estimation,
initialization, optimization and selection. They are reopened only after the
calibration choices have been frozen, to assess storage plausibility.

The cohort configuration requests all **32 completed monthly paper reservoirs**,
including the three signed-net-inflow routes. The other five paper reservoirs
remain visible in both 37-row parameter inventories with their source/route
gates. This feature does not resolve those source decisions.

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/dztr_run_v6.py plan --config configs/run_manager_flow_only_paper_v1.json
python -B tools/dztr_run_v6.py run --config configs/run_manager_flow_only_paper_v1.json --output outputs/managed_flow_only_paper_v1_20261020
```

For a background computational job, `tools/run_flow_cohort_v1.py` runs that same
manager and generates the comparison report automatically after all jobs finish.
Its job receipt is adjacent to the output directory; each reservoir/objective/
seed has an execution log, progress receipt and persistent checkpoint on D:.
The job receipt distinguishes running, completed and failed states. A planned
or prepared route is not a completed calibration.

## Fixed assumptions and adjustable quantities

| Decision | First cohort experiment |
|---|---|
| Storage critical/normal/flood targets | 0.20/0.30/0.97 of adopted capacity, fixed in every month |
| Dickson, E. B. Campbell, Yellowtail | Explicit exploratory adaptation: dead storage + 0.20/0.30/0.97 × (capacity − dead storage) |
| Initial storage | Assumed normal target; never an observed storage value |
| Initial release | First effective inflow clipped to zero/channel capacity |
| Initial monthly release targets | Calibration-only observed P10/P45/P85 |
| Release bounds | Per-month P5–P35, P35–P75, P75–P95 |
| Adjustable parameters | 36 monthly release targets |
| Capacity, channel, dead storage, flood mode | Fixed source metadata |
| Data selection | Original frozen mixed-year masks; at least 65% calibration |
| Simulation | Continuous state; existing native or signed-net solver and root policy |
| Percentiles and zeros | Explicit Hazen estimator; valid zero releases retained |
| Optimizer | Scalar DDS, perturbation 0.2, seeds 42/43/44 |
| Objective arms | Square-root release NSE (predeclared primary) and raw-release NSE |
| Weighting | Pooled; `equal_regime` remains available |
| First checkpoint | 64 total evaluations per search, including the initial solution |
| Frozen DDS horizon | 512 evaluations; the checkpoint is not a convergence claim |

The active-storage adaptation is not a recovered Etienne or paper parameter.
It keeps an assumed critical target above the adopted dead storage without
silently lowering dead storage or raising capacity. Its basis and rationale
are exported. Other source uncertainty remains explicitly unresolved.

## What comes from Etienne's implementation

The archived study used a CR2-style flow-observed route with assumed storage
fractions and monthly release statistics. Some archived calibration cases used
release-only cost, including NSE calculated after square-root transformation.
Their Ostrich setup used DDS with perturbation 0.2. We implement a separately
versioned scalar DDS adapter for this single objective, following
[Tolson and Shoemaker (2007), Figure 1](https://doi.org/10.1029/2005WR004723).
This is not a claim of byte-for-byte Ostrich reproduction or of using its
original 5,000-iteration budget.

The optional `bounds_policy: "etienne_annual_envelope"` uses the minimum across
months of P5/P20/P60 as the three lower limits and the maximum across months of
P35/P75/P95 as upper limits; the flood upper envelope also includes the source
channel value. Such proposals still require ordered, physical targets. The
default first comparison keeps the current month-specific bounds, so the two
objective arms differ only in release transformation. Upstream parameterization
can explicitly choose `linear` percentiles for an Etienne-inspired sensitivity
experiment. This does not reproduce the archive's exclusion of valid zeros.

We retain the existing conservative solver and observation clocks. The archive
also explored target interpolation, extra calibrated physical parameters and
different integration details; these would change additional factors and are
not part of this release-only comparison. See the
[archived implementation review](etienne_review.md) and
[Gaborit, Fortin and Durnford (2022)](https://doi.org/10.1139/cjce-2021-0507).

## Checkpoints, selection and continuation

DDS reduces the number of perturbed coordinates according to its final search
horizon. That horizon is frozen at preparation. Stopping at 64 and continuing
to 128 or 512 therefore matches an uninterrupted run using the same horizon,
seed, bounds and implementation. Changing the horizon changes the experiment.
The tests verify exact continuation and reject changed settings or checkpoint
contents. Checkpoints are saved every eight evaluations and at the requested
budget. An interrupted job may redo at most seven uncommitted trials.

Each seed's best candidate is selected by calibration release cost. The pooled
selection uses the lowest calibration cost across seeds, with deterministic
ties by seed and evaluation ID. Hidden-storage and held-out release scores do
not influence this choice. Undefined or failed assessment scores remain missing.
All-infeasible searches are recorded and do not produce invented parameters.

For a standalone campaign, continuation is:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/calibrate_flow_v1.py run --config configs/flow_only_paper_v1.json --output <campaign> --evaluation-budget 128
```

A managed run seals all stage artifacts. Preserve that managed run before
continuation: copy its `stages/flow_calibration/campaign` tree to a new standalone
campaign directory, then use the command above on the copy. The copied protocol,
relative preparation paths, optimizer state and immutable checkpoints preserve
their identity. The original manager can still verify its original stage.
Do not change configuration files or source code to extend a checkpoint.

## Results and interpretation

Each objective has its own unified profile registry and GRAND-indexed parameter
inventory. All 37 paper rows are retained; parameters remain blank where a
source gate or absence of a feasible search prevents fitting. Missing GRAND IDs
do not drop reservoirs. The first configured objective is the manager's primary
registry; the second is an explicit comparison profile, not a validation winner.

The comparison reports both calibration and held-out raw-release NSE,
square-root-release NSE, hidden-storage NSE, mean bias, PBIAS and sample counts
for every successful seed. Positive PBIAS means overprediction. The baseline
uses the same storage assumptions, cold start, forcing and numerical route,
with unoptimized observed-percentile release targets. Baseline failures and
selected-candidate assessment failures are reported explicitly.

The automatic report adds seasonal biases, dry/normal/wet regime scores,
paired objective comparisons and a cohort plot from sealed cached trajectories.
Previous joint storage/release calibrations are included as context; they have
different data access, initialization, parameter dimensions, optimizers and
budgets, so this is not a controlled optimizer comparison.

This benchmark tests a storage-hidden fitting procedure with the available
forcing. Some inflows were reconstructed using observed storage and release,
and adopted physical metadata are not independently certified. It therefore
does not establish performance at an independently ungauged reservoir.
