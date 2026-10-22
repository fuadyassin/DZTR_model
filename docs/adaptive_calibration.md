# Checkpointed adaptive calibration

The adaptive runner carries a seeded NSGA-II search forward between requested
budgets and checks the provisional calibration-only stopping policy during the
search. This makes saved optimizer work reusable and allows a sustained
stagnation signal to save actual objective evaluations. It does not establish a
globally optimal front or uniquely identify reservoir targets.

## Run, resume and audit an extension

Start the Trinity seed-42 search at a 512-evaluation stage:

```powershell
python -m dztr.calibration.adaptive --repo . --config configs/trinity_adaptive.json --output outputs/trinity_adaptive/seed_42 --seed 42 --evaluation-budget 512
```

Resume its optimizer state in the same output directory:

```powershell
python -m dztr.calibration.adaptive --repo . --config configs/trinity_adaptive.json --output outputs/trinity_adaptive/seed_42 --resume --evaluation-budget 1024
python -m dztr.calibration.adaptive --repo . --config configs/trinity_adaptive.json --output outputs/trinity_adaptive/seed_42 --resume --evaluation-budget 2048
```

The budget is a cumulative target, not a number of additional calls. Without an
early stop, these stages need 512, 512 and 1,024 new search evaluations,
respectively. Reports must distinguish newly evaluated trials, restored trials
and the cumulative total; selected-candidate calibration/validation replays are
separate reporting work. Different seeds use separate output directories and
fresh initial states.

To investigate whether a provisional stopping suggestion was premature, resume
with `--audit-extension`. This disables stopping on that suggestion while
retaining the policy's hard limit of 4,096 evaluations. The later windows remain
available to confirm or invalidate the earlier suggestion. An audit extension is
additional evidence, not ordinary saved computation.

The Python entry point is:

```python
from dztr.calibration.adaptive import run_adaptive_optimization

report = run_adaptive_optimization(
    "configs/trinity_adaptive.json",
    repo_root=".",
    output_dir="outputs/trinity_adaptive/seed_42",
    evaluation_budget=1024,
    seed=42,
    resume=True,
    honor_stopping=True,
)
```

Configuration adds an `adaptive` section with `checkpoint_interval: 128` and
`stopping_policy_file: "configs/stopping_policy.example.json"`. The target stage
budget, the policy's hard cap and a provisional stagnation suggestion are
different reasons to end a run. Retain the actual reason in its report.

Each reservoir/seed has its own output directory. The existing search tables,
selected SI targets and daily representative simulations remain available,
alongside these continuation artifacts:

| Artifact | Purpose |
| --- | --- |
| `optimizer_checkpoint.npz` | Latest complete saved population, archive and random-generator state |
| `run_identity.json` | Input, settings, code, policy and software identity required for continuation |
| `stopping_history.csv` | Scheduled calibration-only progress signals, patience and reasons |
| `adaptive_manifest.json` | Published search count and checksums for the checkpoint and report artifacts |
| `summary.json` | New/reused counts, termination reason, selected scores and provenance |

A trusted checkpoint newer than the last published report can regenerate owned
derived reports after an interrupted export. Atomic checkpoint and manifest
replacement preserves the previous complete file when a write fails. This does
not make partial generations resumable: only saved boundaries count as reused
search work.

## Restoring state safely and reproducibly

A versioned checkpoint preserves the current population and IDs, RNG state,
evaluated vectors/costs and feasibility, generation/evaluation counters and
recorded diagnostics. Duplicate keys are reconstructed from saved vectors.
The atomic compressed NumPy checkpoint
contains numeric arrays, UTF-8 JSON metadata and SHA-256 content checksums;
loading does not use pickle. It retains no daily trajectories. Restoring only a Pareto front or the selected parameter vector
cannot reproduce the seeded search. [Official checkpoint guidance](https://pymoo.org/misc/checkpoint.html)
illustrates continuation by restoring the algorithm's state; this repository
uses its own NumPy optimizer and checkpoint implementation.

Resume validation checks checkpoint structure and search identity before using
the saved work: observation contents, physical/simulation settings, quantile
method and calibration periods, optimizer seed/operators, stopping policy,
search-code hashes and software versions. Increasing the target budget is an
allowed continuation; silently changing these other inputs is a different
experiment. A damaged or incompatible checkpoint must fail rather than restart
and label the old evaluations as reused.

Checkpoints are taken after the initial full population or a complete generation,
with detached state and the full PCG64 RNG state. Partial final generations do
not produce a resumable checkpoint or run the stopping hook. Use population-
aligned stage budgets and checkpoint intervals, as in the examples above.
Resume verifies deterministic survivor order and rechecks the saved boundary's
stopping condition before evaluating new trials. Saved candidates are not
reevaluated, and old progress/checkpoint events are not emitted again.

Checkpoints retain enough state to preserve generation order and deterministic
ties. Interrupted writes must leave a verifiable completed snapshot. Compare a
staged search against an uninterrupted run of the same seed and settings to
verify candidate vectors, costs, identities and RNG continuation.

The earlier fixed-budget CSV archives contain trial vectors and scores but no
saved RNG/population state. They support retrospective analysis, not direct
continuation through this interface. A fresh checkpointed replay can be compared
with the existing 1,024-evaluation prefixes; it is new search work and must not
be counted as reused work.

## Keep stopping, fit quality and seed agreement separate

The [illustrative stopping policy](scalable_calibration.md) checks fixed-reference
hypervolume gain, full-front movement and both objective extrema. It requires
three consecutive eligible small-change windows after 512 evaluations, using
128-evaluation checks/windows. All decisions read calibration costs only.

A temporary plateau can occur in a poor region of the parameter space. A small
recent change says nothing by itself about whether the reservoir fit is useful.
Report fit quality and physical diagnostics alongside the stopping signal;
do not rename a stalled search a successful calibration.

For example, Trinity seed 42 from 640 to 768 evaluations had normalized
hypervolume gain 0.001844 and extrema drift 0.004038, both below the example
tolerances. Its full-front movement was 0.008313, above the 0.005 tolerance,
so that window did not accumulate stopping patience. The normalized gain is a
fraction of the fixed theoretical area of four, rather than a percentage of
the previously observed hypervolume. Progress in tradeoffs can continue while
the best individual release/storage scores move little.

Independent seeds test whether different searches agree in objective space.
Their front comparisons and audit extensions strengthen evidence but do not
prove global convergence. Validation remains reporting after representatives
are selected, not a means to tune budgets, stopping thresholds or population
size. Objective convergence also does not imply parameter identification;
different acceptable targets may persist through
[equifinality](https://doi.org/10.1016/S0022-1694(01)00421-8).

The staged Trinity experiment uses seeds 42, 43 and 44 with cumulative
budgets 512, 1,024 and 2,048. Actual counts and scores belong in the generated
reports. This guide does not claim that those stages have converged, or that the
illustrative thresholds are validated production settings.

The study helper runs those three Trinity jobs and a Gardiner seed-42 job at
512 then 1,024 evaluations in one process pool:

```powershell
python examples/checkpoint_reservoir_study.py --repo . --workers 3
```

Use `--resume` explicitly to recover an existing study. Its saved stage history
and optimizer checkpoints distinguish completed work from new evaluations. If
online stopping ends a job early, the helper makes a labelled audit extension to
that job's final planned budget. Trinity's first 1,024 trial vectors, costs,
feasibility flags and IDs are compared exactly against the existing archives
after verifying every archived artifact hash and the observation hash. Different
historical code identities remain qualified rather than silently accepted as
identical implementations.

Each job retains `stage_summary.json`; the combined report is saved under
`outputs/checkpoint_reservoir_study/` as `summary.json`, `stages.csv` and
`final_calibration_metrics.csv`. Stage rows record fresh/restored counts,
search elapsed time and separate reporting evaluations. Final calibration-front
diagnostics require no new daily simulations. A failed job is reported while
other jobs finish; the helper then exits with an error. Independent final Trinity
fronts are also compared, using an explicitly illustrative agreement tolerance.
Intermediate validation scores never determine stopping or the extension plan.

After all four jobs complete, export the calibration-only figure and archive
prefix table:

```powershell
python examples/plot_checkpoint_reservoir_study.py --repo .
```

The exporter verifies the final adaptive manifests, every published artifact
hash and checkpoint integrity before reconstructing metrics at the initial
population and each 128-evaluation policy window. It writes `convergence.png`
and `prefix_metrics.csv` in the combined study directory, with separate Trinity
and Gardiner panels and labelled online stops/audit extensions. It performs no
search or daily simulation calls.

## Trinity extension results

All three checkpointed Trinity searches completed 2,048 evaluations. Their
first 1,024 vectors, objective costs, feasibility flags and IDs matched the
verified historical CSV archives bit for bit. Each search used 512 new trials,
then restored 512 and added 512, then restored 1,024 and added 1,024. The earlier
CSV experiment was replayed to create checkpoint state; it was not counted as
restored optimizer work.

| Seed | Hypervolume at 2,048 | Relative gain from 1,024 | Best release NSE | Best storage NSE | Stopping status |
| --- | ---: | ---: | ---: | ---: | --- |
| 42 | 2.951024 | 1.945% | 0.555346 | 0.897395 | continue_search |
| 43 | 2.950895 | 2.483% | 0.563169 | 0.888017 | continue_search |
| 44 | 2.920748 | 2.444% | 0.559232 | 0.873249 | continue_search |

Best release and storage values can belong to different archive members. The
relative hypervolume gains above divide by the observed 1,024-call hypervolume;
they are distinct from the policy's theoretical-area normalization.

No Trinity seed produced a stopping candidate within this experiment, so no
automatic early-stop savings or real stop-to-extension audit occurred. All
three ended because the requested stage budget was reached. Seed 43 had stable
windows at 1,792 and 1,920, followed by a nonstable window at 2,048: normalized
hypervolume gain 0.003502 and front/extrema drift 0.005106 exceeded the policy
tolerances. Two quiet windows would have missed that later progress; this is
one example supporting caution about temporary plateaus, not validation of
three-window patience across reservoirs.

At 2,048, the independent full-front distances were 0.02632 (42/43), 0.02736
(42/44) and 0.01681 (43/44), all above the illustrative 0.01 agreement tolerance.
These results support further investigation rather than a fleet-wide
2,048-evaluation recommendation.

The three calibration-selected balanced representatives still span a median
33.15% of their coordinate-wise calibration bound widths; the maximum is
98.73%. At 1,024 the corresponding ranges were 35.70% and 88.77%. These are
descriptive seed ranges, not confidence intervals or formal identifiability
tests. More stable scores do not guarantee matching monthly targets. The
verified comparison is saved as `trinity_parameter_spread.csv` and
`trinity_parameter_spread.json` under the study output directory; fixed-width
coordinates have zero normalized range.

## Gardiner is a second exploratory case

[Gardiner's configuration](../configs/gardiner_adaptive.json) uses its complete
daily workbook from 1980 through 2011. These inputs support an exploratory
second-reservoir pilot; Gardiner and Trinity do not constitute a diverse,
independent fleet cohort.

| Gardiner input | Value and source qualification |
| --- | --- |
| Capacity | 9,870,000,000 m3; catalog `SMAX` agrees with `CAP_MCM=9870` |
| Channel capacity | 2,500 m3/s from Gardiner's catalog `Q_DS_MAX` |
| Inflow correction | 1.0 from Gardiner's catalog `INFLOW_CORR` |
| Initial storage/release | 7,927,529,906.0127 m3 and 165.26 m3/s, observed on 1980-01-01 |
| Dates | Leap-year 1980 warmup (366 days), 1981-1995 calibration, 1996-2011 validation |

The initial-state policy uses the first warmup observations, which are excluded
from parameter estimation and scores. Catalog `SINIT=6733399841` and
`QINIT=171` are not used because their intended date is undocumented. The
15/16-year calibration/validation split is our explicit exploratory choice,
not a claim about the published Gardiner calibration dates.

Catalog quality is `3: Fair`, with blank `DATA_INFO` and `URL` fields. Its
capacity, channel capacity and inflow correction have local catalog provenance
but have not been independently verified against operating records. Dead
storage 0.1 and storage-driven flood mode are explicitly recorded generic model
assumptions. The configuration records source hashes and these qualifications;
it does not inherit Trinity's physical constants or inflow correction.

The [two-reservoir manifest](../configs/adaptive_cohort_manifest.example.json)
references the Trinity and Gardiner adaptive configurations. Both actual source
workbooks contain `Storage(mil_m3)`, `Inflow(cms)` and `Outflow(cms)`, so both jobs
explicitly use storage `mcm` and flow `m3/s`. These raw storage values are
converted to SI at ingestion.

Run its preflight before the exploratory Gardiner stages:

```powershell
python -m dztr.calibration.fleet --manifest configs/adaptive_cohort_manifest.example.json --repo . --output outputs/adaptive_cohort_preflight
```

Preflight validates data and static feasibility, not operating-rule accuracy.
The other 63 bundled workbooks are
placeholders; additional cohort members require real observations, explicit
physical settings and provenance. The manifest contract is described in
[the scalable-calibration guide](scalable_calibration.md).

Gardiner completed 1,024 evaluations with all trials feasible. Its best
calibration release NSE was 0.547478 and best storage NSE was 0.919271; these
maxima belong to different candidates. The balanced representative had release
NSE 0.529286 and storage NSE 0.918020. Hypervolume improved 2.306% from 512 to
1,024. The final window was stable, but only one consecutive stable window had
accumulated, so its stopping status remained `continue_search`.

## What these experiments support next

The four jobs used 7,168 search evaluations in total, plus separately recorded
selected-candidate calibration and validation replays. All ended at their
requested stage budgets; none claimed convergence or saved evaluations through
an early stop. The output figure and prefix table verify four checkpoints and
48 published artifacts. The 587-test regression suite includes exact staged
continuation, controlled stopping before new calls, interrupted search/report
recovery, incompatible resume rejection and bounded Windows replacement retries.

Use these counts as tested cumulative stages, not production budget defaults.
The [population study](population_study.md) now compares populations 16, 32 and
64 at equal evaluation budgets and multiple seeds, with separate run directories
and matching policies. Population changes cannot resume an existing search. Where
needed, a further cumulative stage up to the illustrative 4,096 cap can test
whether progress continues; reaching that cap is resource exhaustion, not
convergence.

A wider reservoir cohort is still required before choosing fleet tolerances
and budget recommendations. Keep those choices independent of validation
scores, report stopped-but-poor fits separately, and audit provisional stopping
points with later calibration evaluations. Observation adequacy for percentile
parameterization and identification of monthly targets remain separate
questions from optimizer progress.
