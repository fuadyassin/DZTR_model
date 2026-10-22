# World reservoir examples

The local collection at `E:\1_DataHub\E_Reservoir_data\I_world_reservoir`
contains 42 numbered reservoir folders: 28 daily and 14 monthly. The importer
reads the unique unprefixed `SimRIS[M]<number>.m` in each folder, follows the
coefficient filename it actually selects, and checks its release and balance
math. It does not execute MATLAB or select the newest coefficient file.
Other scripts and saved optimization outputs are not presumed to be the last
executed run. Original files are read only and hash checked.

The active collection contains 41 reservoirs, now stored permanently under
[`data/reservoirs/`](../data/reservoirs/README.md), with model settings under
[`configs/reservoirs/`](../configs/reservoirs) and a shared
[manifest](../data/reservoirs/manifest.json). Tuyen Quang is intentionally
excluded at the user's request, as recorded in the
[active scope](../configs/world_reservoir_scope.json). Its existing storage
conflict report remains historical audit evidence and needs no resolution for
this work. The original catalog and results retain all 42 audited folders;
subsequent parameterization, simulation and optimization work uses the permanent
manifest's 41 active configurations. The scope records the original packaging
selection. Number 25 is absent from the collection.

The verified full run completed 40 historical baseline replays and 11 modern
conservative baselines. Thirty calibration objectives/bounds were constructible.
These statuses describe the selected source scripts and the new explicit
protocol, not a reconstruction of every published experiment.

The local [results table](../outputs/world_reservoir_samples_final/cohort.md)
lists all reservoirs and their blockers. The tracked
[catalog](../configs/world_reservoir_collection.json) records source selections,
physical settings, sampling and periods, with paths to generated samples.
The [verification record](../outputs/world_reservoir_samples_final/verification.json)
checks all 41 normalized CSVs against their original MAT values exactly,
all 44 packaged modules against source, and source/code integrity. There were
816 passing tests and one Windows symlink-permission skip. Trinity, Gardiner
and monthly Oahe each completed a 32-evaluation optimizer pilot (96 new search
evaluations total), with separate validation reports and exact portable baseline
values. That historical verification describes the earlier import/code identity.
Permanent inputs are now eligible for version control; fresh simulations and
diagnostics remain in ignored local `outputs/`. No commit or upload is performed
by packaging.

## Run the permanent collection

The permanent inputs contain 41 byte-identical SI observation files plus their
provenance and historical target tables. The configs preserve model constants,
sampling, fitting periods and the protocol exactly; only input references move
to relative paths. Normal runs use CSV/JSON and do not require the E-drive or
SciPy's MAT reader. Provenance paths to original sources are descriptive metadata.

From the repository root:

```powershell
python examples/run_reservoir_collection.py --reservoir Trinity --output outputs/trinity_new
python examples/run_reservoir_collection.py --reservoir ex6_Oahe --output outputs/oahe_new
python examples/run_reservoir_collection.py --all --output outputs/collection_new
```

After installing the package, `dztr-collection` or
`python -m dztr.integrations.collection` accepts the same arguments. Select an
exact ID or reservoir name; repeat `--reservoir` to run several. The default
manifest is `data/reservoirs/manifest.json`; `--manifest` accepts a copied
collection's manifest. Results use a new output folder and include
`collection.json`, `collection.csv`, `collection.md` and per-reservoir
`readiness.json` diagnostics. Hash errors and scientific blockers are isolated
per sample. Any blocked member yields exit code 1 after batch reporting finishes.

```powershell
python examples/run_reservoir_collection.py --reservoir Trinity --mode legacy --output outputs/trinity_legacy_new
python examples/run_reservoir_collection.py --reservoir ex6_Oahe --optimize 32 --output outputs/oahe_search_new
```

The historical mode uses the packaged source settings, raw targets and synthetic
clock. It requires no original MATLAB files. The optional optimization budget
is explicit per selected reservoir; importing, packaging and default baseline
runs perform no search. The existing daily adaptive checkpoint workflow remains
separate from the sampled collection runner.

The [packaging receipt](../data/reservoirs/package_receipt.json) verifies 250
original file hashes and 247 packaged hashes. The observation CSVs and source
evidence are copied without numerical changes. Configurations differ only in
the relocated observation path and four references to packaged evidence/targets.
Use `python examples/package_world_reservoirs.py` to reproduce packaging from the
existing verified imports; differing destination files are never overwritten.

## Permanent collection verification

The [current verification](../outputs/permanent_collection/verification.json)
passed with **878 tests and one Windows symlink-permission skip**. A separate
workspace contained only copies of permanent data and configs. During its runs,
reads from the E-drive and the original output archive were denied. All 41
samples produced readiness reports: 11 modern baselines completed and 30 retained
their scientific blockers. No collection search was launched.

All 11 completed trajectories, calibration targets and bounds reproduced the
earlier values exactly. Of 33 CSV comparisons, 31 were byte-identical; Garrison
and Oahe trajectory CSVs differed only in integer/float formatting of observed
storage. All baseline summaries matched exactly. Trinity and monthly Oahe
historical replay CSVs were byte-identical. The
[offline proof](../outputs/permanent_collection/offline_proof/offline_proof.json)
records those comparisons and the unchanged input/module hashes.

The [wheel check](../outputs/permanent_collection/wheel_proof/verification.json)
verified all 47 current DZTR modules, preserved all 44 earlier modules, and
checked the new collection entry point without installing the package. Reservoir
inputs remain repository data and are not included in the wheel. These proof
files are local artifacts under ignored `outputs/`.

## Readiness review

The new `evaluation/readiness.py` classifies raw measurements, transformed
forcing, calendar coverage, physical targets, empirical bounds and runtime
failures. It fits diagnostic targets and bounds on calibration data only.
Inherited dead storage remains an explicit fixed setting; expressions using
`min(Lc)` retain a documented dependence on the historical full-record table.

The [fresh review](../outputs/permanent_collection/offline_proof/readiness_review.md)
groups issues by affected reservoirs:

| Review gate | Reservoir count |
| --- | ---: |
| Multiple implicit roots | 13 |
| No feasible implicit root | 4 |
| Invalid physical target configuration | 10 |
| Signed transformed forcing | 4 |
| Missing external forcing adapter | 1 |

These categories overlap. Thirty calibration objectives/bounds can be
constructed, while eleven are blocked at that stage; this is separate from
baseline success and does not establish convergence.

At the first reported implicit failure, diagnostics record the predictor,
previous state, operating targets, integration duration and all candidate
release branches. This distinguishes multiple solutions from no solution and
from a capacity/target inconsistency. Two representative failures reproduce
the original reported dates:

| Reservoir | Sampling | First failure | Verified cause |
| --- | --- | --- | --- |
| Glen Canyon | Daily, 86,400 seconds | 1983-06-29 | Two roots below capacity; release drops at the flood boundary. |
| Fort Randall | Monthly, 2,635,200 seconds | 1971-04-01 | Two roots below capacity with the explicit 30.5-day duration; release drops at the flood boundary. |

The [diagnostic figure](../outputs/permanent_collection/figures/implicit_root_ambiguity.png)
plots the two release branches and both balance roots for each case, with an
[exportable PDF](../outputs/permanent_collection/figures/implicit_root_ambiguity.pdf).

These cases need an explicit operating/root-selection policy. Increasing
optimizer evaluations or clipping storage targets does not resolve that
ambiguity. The review preserves the release equations and data; a successful
historical fixed-iteration replay remains distinct from a conservative implicit
baseline with a uniquely defined solution.

The [explicit root-policy workflow](root_policy_review.md) now compares minimum
storage, maximum storage and proximity to the previous simulated storage. It
retains strict rejection as the default, preserves missing-root failures and
separates calibration policy comparisons from held-out reporting. The opt-in
runner records raw selected roots, conservative spill and numerical validity of
the fixed-iteration reference.

## Import or rerun a sample

Install the MATLAB reader dependency and run from the repository root:

```powershell
python -m pip install -e ".[dev,matlab]"
python -m dztr.integrations.world_reservoirs import --source-root "E:\1_DataHub\E_Reservoir_data\I_world_reservoir" --output outputs/world_new
```

`dztr-world import` and `examples/import_world_reservoirs.py` expose the same
command. Use a new output folder for each import; existing results cannot be
overwritten. `--skip-baselines` imports/audits data without simulations.

Each imported reservoir gets `observations_si.csv`, `sample.json`,
`observation_provenance.json`, and `matlab_source.json`. Storage is m3 and flows
are m3/s. Original targets/bounds remain in separate `legacy_*_si.csv` files.
The top-level `cohort.json` and `cohort.csv` include every folder and separate
import, legacy replay, modern baseline and optimizer-readiness statuses.

Run a normalized sample without access to the original source drive:

```powershell
python -m dztr.integrations.world_reservoirs run --config outputs/world_new/reservoirs/ex14_Gardiner/sample.json --output outputs/gardiner_new
```

Use the same command with `ex12_Trinity`, `ex6_Oahe`, or any other imported
reservoir. `examples/simulate_world_reservoir.py` is the equivalent example
entry point. Keep `sample.json` beside its CSV when copying a sample. The runner
checks normalized input hashes before and after the run and records its
code/input identity. Export/import of SI CSV values uses round-trip precision.

An explicitly budgeted search uses the same 72 coordinates, calibration bounds,
negative release/storage NSE objectives and NSGA-II implementation:

```powershell
python -m dztr.integrations.world_reservoirs run --config outputs/world_new/reservoirs/ex6_Oahe/sample.json --output outputs/oahe_pilot --optimize 32 --seed 42
```

Importing never launches a cohort optimization. Small pilots validate integration;
they do not establish convergence or fleet evaluation budgets. Monthly experiments
use `SampledCalibrationObjective`; the existing daily adaptive/checkpoint commands
are not extended to monthly data by this work.

## Two explicit experiments

The historical replay retains the selected coefficient targets, source initial
state, inflow adjustment, fixed iteration count, spill policy, synthetic month
clock, and original excluded warmup rows. It exports physical states and any
shifted reporting series separately. It reproduces the selected base script's
concept, not all saved optimized results or all published scores. MATLAB was
not rerun for this collection.

The modern baseline estimates P10/P45/P85 targets with Hazen percentiles from
calibration measurements only. Its default protocol excludes the first calendar
year as warmup and divides the remaining years chronologically. Validation
continues the simulated state and never influences targets, bounds or search.
These explicit per-sample periods are a new protocol, not recovered paper dates.
A leap-year warmup uses its actual row count. Edit periods explicitly for a
different experiment, retaining sampling-compatible boundaries.

The conservative implicit solver rejects ambiguous roots, discontinuous cases
with no feasible root, invalid storage targets, and unreviewed signed forcing.
Such a sample still has imported data and, when supported, a separate historical
replay. A calibration optimizer can be ready even when its generalized baseline
is infeasible; readiness means the objective/bounds can be constructed, not
that a feasible solution or acceptable validation performance has been found.
Held-out forcing and scores can fail independently of a completed calibration
search and remain separate reporting results.

## Differences recovered from the sources

| Source detail | Python treatment |
|---|---|
| Monthly steps are 30.5 days and synthetic dates drift | Keep 30.5-day integration explicitly; modern operating months use workbook calendar labels. Historical replay uses the drifting MATLAB clock. |
| Central Asian workbooks contain three dekads per month | Preserve the writer's unweighted monthly mean; do not interpolate daily observations. |
| Fractional half-day MAT timestamps | Retain their description as provenance; use verified monthly calendar dates for canonical observations. |
| Gardiner coefficient variants | Use the named base-script file with capacity 9.4 billion m3. |
| Collection Trinity differs from the earlier standalone example | Preserve factor 1.0 and observed initial storage, rather than inheriting standalone factor 0.92 and coefficient initial storage. |
| Bennett multiplies inflow by 1.45; Dickson adds 10; Big Horn uses 1.1 times inflow plus 8 | Apply the forcing transform before balance and release evaluation, separately from `icor`. |
| Travers/Mcgregor use `icor` 0.96/0.90 | Preserve each fixed factor. |
| Some dead-storage fractions derive from minimum critical storage | Evaluate the evidenced expression against the selected historical table; preserve it as a fixed setting. |
| Grand Coulee omits overflow adjustment | Explicit historical `overflow_mode="none"`; modern baseline uses conservative overflow. |
| Chardara shifts release four rows back and storage three rows forward | Export separate reported series; calculate physical balance on the unshifted trajectory. |
| Bhumibol/Sirikit have two mislabeled worksheet dates each | After measurement reconciliation, preserve the writer-generated row calendar and list each discrepancy. |
| Flaming Gorge, Navajo, Bull Lake and Semione have signed inflows | Keep raw values. Historical replay allows them explicitly; default modern forcing requires review. |
| Brazeau replaces observed forcing with external MESH output | Import observations and record the replacement evidence; simulation remains blocked until an external-forcing adapter and valid storage interpretation are supplied. |
| Historical targets can cross dead storage or stated capacity | Keep originals for replay. Modern physical validation does not clip targets or invent capacity. |

The nearby source metadata workbook lists a 37-reservoir subset and some
different periods. It is auxiliary evidence; neither its membership nor its
dates silently override the supplied 42 folders and selected source scripts.
Inherited dead-storage fractions derived from historical full-record targets
remain explicit fixed inputs; only the new six-target estimates and optimization
bounds are fitted from calibration data. Current writer constants also differ
from some selected coefficient headers, so the named stored header governs
baseline replay rather than an inferred replacement.

For actual calendar month durations, set monthly `fixed_step_days` to JSON
`null`. This changes integration and is a new experiment. Monthly dates must be
consecutive first-of-month samples; daily dates must be consecutive days.
Missing measurements remain missing and scoring uses available pairs separately.
Missing forcing or missing sampling intervals blocks a run.

## Modules

`integrations/world_sources.py` discovers settings and evidence;
`integrations/world_observations.py` reconciles measurements/calendars;
`simulation/sampled.py` adds explicit clocks around the shared DZTR kernels;
`calibration/sampled.py` handles sample-based periods, baseline and objective;
`integrations/world_reservoirs.py` creates portable samples and batch reports.
`integrations/packaging.py` publishes the permanent collection;
`integrations/collection.py` runs its independent jobs and checks input identity;
`evaluation/readiness.py` explains data, target and operating-equation blockers.
The established daily simulation and checkpoint implementation remains intact.
