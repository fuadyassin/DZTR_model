# Comparing population sizes at equal calibration budgets

The original Trinity comparison is complete: populations 16, 32 and 64, seeds
42, 43 and 44, and 2,048 search trials per job. All nine jobs and their source
proofs were verified. The [results note](../outputs/trinity_population_study/population_results.md)
and [comparison figure](../outputs/trinity_population_study/population_comparison.png)
are published. Cumulative archives provide matched comparisons at 512,
1,024 and 2,048 trials; rejected candidates count toward each budget.

The complete comparison contains 18,432 trials. Its final runner invocation
records 11,136 new trials and 7,296 reused trials, including existing
population-16 archives and restored progress in owned jobs. These are invocation
counts, not a claim about every model call made during the project. Final source
exports separately record 36 selected-candidate calibration replays and 36
validation reporting runs. Earlier exports and runtime experiments are outside
those final-export counts.

## Calibration results and stopping evidence

The table gives seed medians from the full feasible Pareto archives. Hypervolume
uses the fixed `[1, 1]` reference in negative-NSE coordinates; higher is better.
Best release and best storage scores can describe different archive members.

| Population | HV at 512 | HV at 1,024 | HV at 2,048 | Best release NSE at 2,048 | Best storage NSE at 2,048 | Maximum seed-front distance at 2,048 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2.835893 | 2.879390 | 2.950895 | 0.559232 | 0.888017 | 0.027361 |
| 32 | 2.809538 | 2.886425 | 2.940925 | 0.555104 | 0.884399 | 0.031651 |
| 64 | 2.741616 | 2.851771 | 2.960558 | 0.561306 | 0.899274 | 0.078903 |

Population 64 has the highest median hypervolume at 2,048, about 0.33% above
population 16, after trailing at shorter budgets. Its final hypervolume seed
range is 2.951974–2.988246; population 16 spans 2.920748–2.951024. Median
hypervolume still increases by 2.48%, 1.89% and 3.81% from 1,024 to 2,048 for
populations 16, 32 and 64 respectively. The ranking depends on the available
budget and the performance criterion. These three-seed ranges are descriptive,
not confidence intervals or evidence of a fleet-wide winner.

None of the nine final within-population seed pairs meets the illustrative 0.01
full-front agreement tolerance. Balanced monthly targets also differ across
seeds: their median coordinate ranges are 33.15%, 32.20% and 27.51% of the fixed
bound widths for populations 16, 32 and 64; maximum ranges are 98.73%, 95.64%
and 94.62%. Similar objective scores do not establish unique reservoir targets.

Only population 32, seed 42 reaches the provisional `stopping_candidate` status,
exactly at 2,048. No later window was evaluated for that candidate, so the absence
of a recorded invalidation does not validate stopping there. The other eight
jobs finish with `continue_search`. Population 16, seed 43 has two consecutive
quiet windows by 1,920, then breaks that streak at 2,048: front movement and
extrema drift both reach 0.0051063, exceeding the 0.005 tolerance. A hypothetical
patience of two would have suggested stopping prematurely; the configured
patience of three does not flag that run.

Population 16 remains a reasonable provisional choice for shorter Trinity
pilots. Population 64 merits further comparison at larger budgets, especially
for storage fit. A matched 4,096-trial audit should check whether its gains and
seed differences persist and whether population 32, seed 42's candidate survives
later windows. Broader seeds and a reservoir cohort are needed before selecting
fleet defaults. All comparisons and these interpretations use calibration costs;
validation does not choose populations, budgets, representatives or thresholds.

## Preserve and read the completed original study

Production simulation now prepares monthly values in a cache local to each
simulation, as documented in the [simulation performance guide](simulation_performance.md).
The original study retains its original consumed-code identity and immutable
source artifacts. Exact numerical parity does not make its checkpoints compatible
with the changed production source identity.

Do not rerun, resume or invoke the population runner's `--analyze-only` with
`configs/trinity_population_study.json` against the current source tree. That
configuration refers to original-code archives and correctly rejects their reuse.
Historical reporting remains available without running models or comparing old
identities to current source files:

```powershell
python -B examples/finish_population_study.py --repo . --output-dir outputs/trinity_population_study
python -B examples/plot_population_study.py --repo . --output-dir outputs/trinity_population_study
```

These helpers verify the complete study manifest, source receipts, checkpoint
hashes and statistics before publication. The figure shows medians and
three-seed minimum-to-maximum ranges, plus front distances and target variability.
The reporting helper writes the figure and `population_results.md` without
search trials or reservoir simulations.

For exact extensions of the old searches, fork their complete archived run
directories into separate output folders and execute with unchanged original
consumed code and matching recorded software. A preserved original source tree
is available under `outputs/simulation_performance/reference_code/src/dztr`.
Keep the original archives, study receipts and proofs untouched; extending their
source checkpoints in place would invalidate the completed study's proofs.
Never rewrite checkpoint identities to bypass compatibility checks.

## Start a separate comparison with the current implementation

The prepared configuration is `configs/trinity_population_study_prepared.json`.
It retains the same population, seed and budget design, with `reuse_runs=[]`.
Every job therefore starts under the current code identity. This fresh nine-job
experiment has **not been launched**; it would require 18,432 new search trials.

From the repository root:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -B -m dztr.calibration.population_study --repo . --config configs/trinity_population_study_prepared.json --output outputs/trinity_population_study_prepared --workers 3
```

Once that separate study has started, recover interrupted jobs explicitly:

```powershell
python -B -m dztr.calibration.population_study --repo . --config configs/trinity_population_study_prepared.json --output outputs/trinity_population_study_prepared --workers 3 --resume
```

Resume requires an existing study identity and compatible complete-generation
checkpoints. It evaluates only additional trials after the latest saved boundary;
trials after that boundary can be repeated if the process was interrupted.
Completed jobs are verified and reused without search calls. `--resume` and
`--analyze-only` are mutually exclusive. Changing population size, inputs,
operators, policy, consumed source or software requires a new compatible run.
See the [adaptive calibration guide](adaptive_calibration.md).

After the separate current-code matrix completes, its analysis can be regenerated
without simulations, while its recorded source identity remains compatible:

```powershell
python -B -m dztr.calibration.population_study --repo . --config configs/trinity_population_study_prepared.json --output outputs/trinity_population_study_prepared --analyze-only
python -B examples/finish_population_study.py --repo . --output-dir outputs/trinity_population_study_prepared
```

The reporting helper can also wait for that study with `--wait --timeout-hours 24`.
It polls every 30 seconds and reports failed jobs or a timeout explicitly. Paths
supplied to the population API or CLI are relative to `repo_root`/`--repo`;
absolute paths are accepted, and outputs must remain within that repository.

The Python API for a fresh current-code experiment is:

```python
from dztr.calibration.population_study import run_population_study

report = run_population_study(
    "configs/trinity_population_study_prepared.json",
    repo_root=".",
    output_dir="outputs/trinity_population_study_prepared",
    workers=3,
    resume=False,
    analyze_only=False,
)
```

## Keep the comparison controlled

Every population uses the same Trinity observations, calibration dates,
generalized baseline, 72-coordinate calibration bounds, physical constants,
solver, quantile convention and variation operators. Calibration bounds and the
baseline are reconstructed and checked before any search starts, without
simulating the reservoir. The complete planned population/seed matrix is required
for analysis; settings require at least two distinct populations, two seeds and
two aligned checkpoints. Jobs use one process pool, with serial simulations
inside each search.

Metrics use the full cumulative feasible Pareto archives, including candidates
that no longer survive in the retained population. New fixed-budget jobs disable
automatic stopping; the reused original population-16 searches had no early stop
and reached the same count. Retrospective diagnostics use 128-trial windows and
checks, a 512-trial minimum and a 4,096 hard cap; only the policy's population
alignment field differs. Suggested stops neither shorten this experiment nor
save evaluations. The adaptive `audit_extension` flag also marks disabled
stopping on fresh fixed-budget jobs; here it does not imply an earlier real stop.

Population size changes initialization and generation count together:

| Population | Baseline trials | Local perturbations | Latin-hypercube trials | Evolved generations at 2,048 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 7 | 8 | 127 |
| 32 | 1 | 15 | 16 | 63 |
| 64 | 1 | 31 | 32 | 31 |

Matching seed labels supplies reproducibility controls. Populations consume
different random draws, so they do not share matched candidate streams or paired
random variations. This experiment compares complete operational settings,
rather than isolating survivor count from initial sampling or generation count.

## Report artifacts and interpretation

| Artifact | Meaning |
| --- | --- |
| `checkpoint_metrics.csv` | Per-seed full-archive calibration metrics at matched trial counts |
| `population_summary.csv` | Seed minimum, median and maximum scores; normalized balanced-target ranges |
| `front_agreement.csv` | Every within-population seed pair's full-front distance at each checkpoint |
| `parameter_stability.csv` | Monthly coordinate ranges of calibration-selected balanced targets |
| `checkpoint_fronts.csv` | Each prefix's full Pareto archive in release/storage NSE coordinates |
| `stopping_diagnostics.csv` | Calibration-only window signals and retrospective suggestions |
| `job_status.csv` | Job status and separate new/restored trial counts |
| `summary.json` | Design, completed jobs, invocation accounting and qualifications |
| `study_manifest.json` | Analysis hashes and receipts tying reports to source jobs |
| `population_comparison.png` | Calibration medians, descriptive seed ranges and agreement context |
| `population_results.md` | Verified numeric findings, stopping statuses and qualifications |

Hypervolume measures dominated area at its fixed reference. Inspect feasibility
and reference coverage alongside that number. Front agreement uses symmetric
Hausdorff distance with the infinity norm in raw NSE coordinates. Agreement
strengthens reproducibility evidence but does not establish global convergence.

Parameter ranges divide each coordinate's seed range by its fixed calibration
bound width; zero-width coordinates contribute zero. Their median and maximum
summarize 72 coordinates. These describe differences between balanced
representatives, not uncertainty intervals or formal identifiability tests.

Elapsed time requires separate interpretation. Reading an existing archive is
not comparable with a fresh search. Source verification, checkpointing, survivor
checks, reporting and concurrent load all contribute to total runtime. The
integrated monthly cache preserves exact daily outputs and source-version
boundaries; its selected-candidate benchmarks and limitations are documented in
the [simulation performance guide](simulation_performance.md). These replay
ratios do not establish complete optimizer or fleet throughput.
