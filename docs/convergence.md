# Trinity search progress and parameter stability

This study compares seeds 42, 43 and 44 at nested budgets of 96, 384 and 1,024
calibration evaluations per seed. Population size stays at 16, and the candidate
representation, physical configuration, percentile bounds, SBX and mutation
settings stay fixed. It measures progress beyond the initial pilot while
preserving separate release and storage objectives.

There is one maximum-budget search for each seed. Smaller checkpoints are
prefixes of that search's evaluation archive, so the design has three independent
random seeds rather than nine independent experiments. Intermediate checkpoints
end at complete population-sized generations, preserving parity with equivalent
smaller standalone searches. Rejected candidates count toward every budget.

## Run or reanalyze

```powershell
python -m dztr.calibration.study --repo . --output outputs/trinity_convergence
```

`configs/trinity_convergence.json` sets the three seeds, checkpoints, reference
point and three worker processes. Pass `--workers 1` to run serially. Worker
completion order does not determine pooled IDs or tie breaks; seeds and their
evaluation order are sorted before pooling.

Completed seed runs are reused only after their saved input, search-code,
configuration and software-version identity matches and all exported artifact
hashes pass verification. An incomplete run or mismatched manifest raises an
error and requires a new output directory. Search-code identity is separate
from analysis-code provenance: `analysis_code_sha256` records the current
analysis modules used to produce the study tables. Updating this
analysis can reuse unchanged, verified search archives. To require cached
searches and recompute analysis and selected-candidate reporting only:

```powershell
python -m dztr.calibration.study --repo . --output outputs/trinity_convergence --analyze-only
```

This mode performs no new optimization search. Selected calibration and
validation replays remain reporting work. Their counts are separate from search
evaluations, including the reporting done by each original seed run.

The Python entry point is:

```python
from dztr.calibration.study import run_calibration_study

report = run_calibration_study(
    "configs/trinity_convergence.json",
    repo_root=".",
    output_dir="outputs/trinity_convergence",
)
```

## Compare calibration hypervolume with one reference

Hypervolume is the union of dominated rectangles between the feasible
nondominated costs and a predefined reference point. Larger area is better.
This study fixes the reference at `[1, 1]` in
`[-NSE(release), -NSE(storage)]` coordinates before the searches, corresponding
to reference NSE values `[-1, -1]`. Scores and reference coordinates are not
rescaled separately for each seed or checkpoint.

A point at or beyond either reference coordinate contributes no positive area,
even if it is a feasible trade-off elsewhere. Reference choice affects the
indicator's emphasis; retain the chosen point when comparing results. See the
primary analysis of [hypervolume indicators](https://sop.tik.ee.ethz.ch/publicationListFiles/zbt2007a.pdf)
and [reference-point effects](https://www.cmap.polytechnique.fr/~dimo.brockhoff/publicationListFiles/abbz2009a.pdf).

For these negative-NSE objectives, the theoretical ideal is `[-1, -1]` because
NSE cannot exceed one. Hypervolume relative to `[1, 1]` therefore cannot exceed
four in exact arithmetic. This upper bound is an area bound, not a measure of
percentage convergence or evidence that simultaneous perfect fitting is possible.

Each prefix archive includes every feasible nondominated candidate found up to
that budget, including candidates outside the final population. Its hypervolume
cannot decrease as evaluations accumulate. The study also reports maximum
release NSE, maximum storage NSE and the balanced representative's two scores.
Hypervolume gains use the same reference at each checkpoint; relative gains
are undefined when the preceding hypervolume is zero.

The descriptive budget summaries report minimum, median and maximum across
the three seeds. They are not confidence intervals. A flatter curve or a narrow
seed range does not establish a global optimum when the true front is unknown,
as explained in [the official convergence guide](https://pymoo.org/getting_started/part_4.html).
The report explicitly retains `convergence_established: false`.

## Compare targets without treating seed variation as uncertainty

For each seed and checkpoint, the existing calibration-only policy selects
best-release, best-storage and balanced representatives. Balanced selection
uses that checkpoint's own front ranges, not ranges taken from a later budget.
Its position can change because the sampled extrema used in normalization move.

For every role, month and target, the study saves the minimum, median and maximum
target across the three seeds. Storage targets use m3 and release targets m3/s.
Normalized spread is the seed range divided by the common calibration-derived
bound width. Fixed coordinates contribute zero. This avoids comparing raw
storage-volume differences directly with release-rate differences.

These envelopes describe optimizer seed variation among matching representative
roles. They are neither posterior uncertainty nor confidence bounds for actual
reservoir operation. Similar objective scores can coexist with different monthly
targets, indicating limited parameter identification or different trade-offs.
Parameter stability and objective-space progress should therefore be assessed
together.

## Pool calibration candidates, then report validation

The combined archive pools all feasible trials from the maximum-budget runs,
then removes dominated candidates using calibration costs alone. Its three
representative roles follow the same extrema and normalized-distance policy as
the pilot. Pooled IDs preserve a mapping to `(seed, evaluation_id)`, with
deterministic ties resolved by costs and pooled ID.

Representative IDs are fixed before pooled validation evaluation. The report
also retains each seed's maximum-budget representatives and the generalized
baseline. Its selection reads calibration trial costs, not the validation scores
already included in individual seed reports. Validation uses the continuous
simulated trajectory and never resets storage from held-out observations.

The first-year/remaining-halves dates, calibration-only percentile policy and
fixed Trinity configuration remain those documented in
[the baseline notes](calibration_baseline.md). The inherited 0.92 inflow factor
keeps its existing provenance qualification. Validation has already been
inspected during development, so it is not an untouched final test set.

## Saved artifacts

| Artifact | Contents |
| --- | --- |
| `checkpoint_metrics.csv` | Seed/budget calibration indicators, feasible/rejected counts and hypervolume gains |
| `budget_summary.csv` | Descriptive minimum/median/maximum across seeds at each checkpoint |
| `checkpoint_fronts.csv` | Each prefix archive with its seed, budget, evaluation ID and NSE |
| `pooled_front.csv`, `pooled_vectors.csv` | Combined calibration archive and corresponding 72-value vectors |
| `representative_vectors.csv` | Every seed/checkpoint/role candidate with its native vector |
| `representative_targets_si.csv` | Corresponding monthly SI targets |
| `parameter_stability.csv` | Across-seed target ranges, medians, raw bounds and normalized spreads |
| `selected_metrics.csv` | Pooled roles, individual seed roles and baseline, scored separately on calibration and validation |
| `selected_targets_si.csv` | Monthly SI target tables for the unique reported pooled IDs |
| `selected_daily_simulations.csv` | Their continuous trajectories, observations and period labels |
| `bounds_si.csv` | Shared calibration-derived percentile bounds |
| `summary.json`, `resolved_config.json` | Settings, counts, provenance, descriptive summaries and limitations |
| `runs/seed_42/`, `runs/seed_43/`, `runs/seed_44/` | Complete original searches and integrity manifests |

Full-precision numerical results and measured runtime are retained in the saved
reports. `study_elapsed_seconds` measures the current invocation, so a cached
analysis has a shorter elapsed time than the original search. Original seed
search times remain in `runs/seed_*/summary.json` as `search_elapsed_seconds`.
Runtime depends on worker scheduling and hardware; repeated analysis can reuse
verified searches without changing their candidate sequence.

## Measured results

The saved study contains 3,072 search evaluations: 3,055 feasible trials and 17
rejections. Its pooled calibration archive has 19 nondominated candidates and
hypervolume 2.89964. No checkpoint front lost points to the fixed reference
region. Seed 42's first 96 evaluations reproduce the original pilot exactly.
The three parallel seed searches each took approximately 489 to 497 seconds;
cached analysis does not repeat those searches.

| Evaluations per seed | Hypervolume min / median / max | Median best release NSE | Median best storage NSE |
| --- | --- | --- | --- |
| 96 | 2.60491 / 2.63634 / 2.71732 | 0.4639 | 0.8084 |
| 384 | 2.78135 / 2.79573 / 2.83253 | 0.5189 | 0.8463 |
| 1,024 | 2.85108 / 2.87939 / 2.89472 | 0.5400 | 0.8666 |

The objective maxima in the last two columns summarize separate extrema;
they do not describe a single paired release/storage candidate. From 384 to
1,024 evaluations, within-seed hypervolume increased by 2.196%, 2.992% and
2.507% for seeds 42, 43 and 44, respectively. Objective-space progress continues
at the largest budget, so this study does not establish convergence.

The pooled roles were chosen on calibration before reporting these scores:

| Candidate | Seed / evaluation ID | Calibration release NSE | Calibration storage NSE | Validation release NSE | Validation storage NSE |
| --- | --- | --- | --- | --- | --- |
| Generalized baseline | 42 / 0 | 0.3353 | 0.6068 | 0.0995 | 0.3487 |
| Best release | 43 / 1,011 | 0.5426 | 0.8461 | 0.3385 | 0.9139 |
| Best storage | 42 / 1,019 | 0.4915 | 0.8799 | 0.3458 | 0.9324 |
| Balanced | 42 / 1,017 | 0.5290 | 0.8729 | 0.3682 | 0.9346 |

For the three seed-specific balanced representatives at 1,024 evaluations,
the median parameter range spans 35.70% of its calibration-bound width, and
the largest range spans 88.77%, across 72 free coordinates. Improved fits
therefore coexist with considerable variation in some monthly targets.
The plotted coordinate medians summarize those representatives; they are not
an additional optimized candidate or an uncertainty estimate.

## Plot search progress and target ranges

```powershell
python -m pip install -e ".[plots]"
python examples/plot_trinity_convergence.py --output-dir outputs/trinity_convergence
```

`trinity_convergence.png` shows calibration hypervolume, maximum release NSE and
maximum storage NSE against nested budget, with a separate line for each seed
and a generalized-baseline reference. Every panel uses calibration quantities.

`trinity_parameter_stability.png` shows all six monthly target series for the
maximum-budget balanced representatives. It displays individual seed curves,
their median and descriptive range, together with the raw percentile bounds.
Storage is displayed in million m3 and release in m3/s. The shaded range is
descriptive seed variation rather than a confidence band.

The subsequent [budget and fleet guide](scalable_calibration.md) adds a
retrospective, calibration-only stopping diagnostic and common reservoir input
manifest. The diagnostic can audit these saved archives without new searches
or daily replays; its thresholds remain illustrative.
The [adaptive calibration guide](adaptive_calibration.md) describes online
checks, resumable checkpoints and larger exploratory Trinity/Gardiner stages.
Historical CSV studies remain evidence for prefix comparison; they do not
contain the optimizer state needed for continuation.
