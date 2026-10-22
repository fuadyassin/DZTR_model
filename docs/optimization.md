# Trinity multi-objective calibration pilot

The calibration layer now runs a seeded NSGA-II search over the 72 monthly
storage and release targets. The two costs remain separate:

```text
[-NSE(release), -NSE(storage)]
```

The search uses the existing candidate decoder, empirical bounds and reservoir
simulator. Capacity, initial conditions, dead storage, channel capacity, inflow
factor and flood mode remain fixed configuration. This NumPy implementation is
a new optimizer adapter; the historical study used AMALGAM.

The initial configuration in `configs/trinity_optimization.json` uses a
population of 16, a budget of 96 search evaluations and seed 42. This deliberately
small run verifies the complete workflow. Its nondominated archive describes
the candidates found within that budget; it does not establish convergence,
a global Pareto front or reliable performance across random seeds.

## Run the pilot

From the repository root, after installing the package:

```powershell
python -m dztr.calibration.search --repo . --output outputs/trinity_optimization
```

The runner accepts `--config`, `--seed`, `--population-size` and
`--evaluation-budget`. Command-line overrides are saved with the resolved
configuration. For example:

```powershell
python -m dztr.calibration.search --repo . --seed 42 --population-size 16 --evaluation-budget 96 --output outputs/trinity_optimization
```

The Python entry point is:

```python
from dztr.calibration.search import run_configured_optimization

report = run_configured_optimization(
    "configs/trinity_optimization.json",
    repo_root=".",
    output_dir="outputs/trinity_optimization",
)
print(report["selected_candidate_ids"])
```

## Search variables and operators

The candidate vector preserves MATLAB's six twelve-month blocks: critical,
normal and flood storage fractions, then critical, normal and flood release
rates in m3/s. The optimizer works in unit coordinates relative to each
variable's empirical lower and upper limits. This makes variation independent
of the numerical scale difference between storage fractions and release rates.
Variables with identical lower and upper limits stay fixed.

Initialization includes the generalized baseline, candidates near that baseline
and a Latin hypercube sample across the empirical box. The configuration records
the local fraction and scale. With the initial settings, the other fifteen
population members comprise seven Gaussian perturbations with normalized
standard deviation 0.05 and eight Latin hypercube points. Operator proposals
are bounded to the empirical box; subsequent physical validation still applies.
The search uses simulated binary crossover (SBX), with probability 0.9 and
distribution index 15, and polynomial mutation with distribution index 20.
Mutation probability per coordinate is one divided by the number of free
coordinates. Settings are recorded in the resolved configuration.

NSGA-II survival combines parents and offspring, sorts feasible candidates by
nondomination and uses normalized crowding distance to choose among candidates
on a partially retained front. Tournament selection uses rank and crowding.
The archive retains nondominated feasible candidates from all search evaluations,
including candidates that leave the current population.

The raw percentile limits remain P5–35/P35–75/P75–95 for both storage and release,
estimated only from calibration observations. Points inside this box still need
to satisfy the existing physical ordering and simulation checks. A rejected
candidate has an explicit reason and receives no fabricated NSE. Rejected
candidates cannot enter the feasible archive or representative selection.

Every search evaluation counts toward the budget, including the starting
baseline and rejected candidates. The final generation can be partial to honor
the requested evaluation count. Exact duplicate vectors are not reevaluated;
bounded duplicate retries can terminate a degenerate search early, with the
actual count and reason recorded. Replays of selected calibration candidates and
their validation simulations are reporting work and are counted separately.
Unexpected programming failures propagate rather than becoming candidate scores.

## Freeze representative selection before validation

Three descriptive roles are selected using the calibration archive alone:

| Role | Selection rule |
| --- | --- |
| Best release | Smallest negative release NSE, then negative storage NSE, then evaluation ID |
| Best storage | Smallest negative storage NSE, then negative release NSE, then evaluation ID |
| Balanced | Smallest Euclidean distance to the normalized calibration ideal point; ties use release cost, storage cost and evaluation ID |

For the balanced role, each objective is normalized using its minimum and maximum
over the found calibration archive. A constant objective contributes zero.
These extrema approximate the ideal and nadir of the sampled archive; they are
not known global extrema. The balanced role is an explicit post-processing
choice and does not scalarize the search's two objectives.

Evaluation IDs follow deterministic search evaluation order. Several roles can
select the same ID, and a small archive can contain fewer than three unique
candidates. The report preserves every role while evaluating shared IDs once.
The generalized baseline is also reported, whether or not it remains
nondominated.

The runner fixes representative IDs before calculating their validation scores.
Validation carries the simulated state continuously from spin-up through
calibration and any intervening dates. Observed validation storage does not
reset the model. Validation scores neither choose representatives nor control
search termination. Calibration improvement can coexist with poorer validation
performance; both periods remain visible in the report.

The initial dates are 1970 spin-up, 1971–1985 calibration and 1986–2000 validation.
Their interpretation and the inherited 0.92 inflow-factor qualification remain
documented in [the baseline notes](calibration_baseline.md). Target and bound
estimation exclude both spin-up and validation observations.

## Saved artifacts

Files are written under the selected output directory:

| Artifact | Contents |
| --- | --- |
| `summary.json` | Resolved settings, counts, timing, software versions, data/code hashes, selection policy and separate scores |
| `resolved_config.json` | Configuration including optimizer defaults and command-line overrides |
| `evaluations.csv` | Every search evaluation, feasibility, reason and calibration costs/NSE |
| `evaluated_vectors.csv` | All native 72-value candidates, indexed by evaluation ID |
| `pareto_front.csv` | Feasible nondominated calibration archive and its scores |
| `pareto_vectors.csv` | Corresponding native candidate vectors |
| `history.csv` | Lightweight generation progress and search counts |
| `selected_metrics.csv` | Baseline and representative scores for calibration and validation, with paired counts |
| `selected_targets_si.csv` | Monthly SI target tables for each unique selected ID |
| `selected_daily_simulations.csv` | Continuous trajectories, observations and period labels for those IDs |
| `bounds_si.csv` | Calibration-derived monthly SI lower and upper limits |

Rejected costs and undefined auxiliary metrics use JSON `null`, accompanied by
feasibility and reasons. They are excluded from front normalization. The pilot
results and measured runtime are saved in `summary.json`; inspect
`selected_metrics.csv` for daily NSE/KGE and other diagnostics.

## Initial pilot results

The seed-42 run reached its 96-evaluation budget: 95 candidates were feasible,
one was rejected for invalid storage-target ordering, and five candidates formed
the nondominated archive. Search time in this environment was approximately
45.35 seconds, excluding selected-candidate replays and validation reporting.

| Representative | Evaluation ID | Calibration release NSE | Calibration storage NSE | Validation release NSE | Validation storage NSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generalized baseline | 0 | 0.33535 | 0.60684 | 0.09945 | 0.34867 |
| Best release | 87 | 0.46392 | 0.74183 | 0.32245 | 0.78435 |
| Best storage | 35 | 0.35732 | 0.80269 | 0.22580 | 0.82284 |
| Balanced | 86 | 0.44266 | 0.77774 | 0.29934 | 0.82048 |

Each scored period contains 5,479 paired daily release values and 5,479 paired
storage values. The table rounds the saved scores; CSV/JSON artifacts retain
their full precision. All three representatives improve both scores relative
to the generalized baseline in this run, including held-out validation. These
results describe one small pilot and do not establish convergence or expected
performance for another seed, budget or reservoir.

## Plot calibration and held-out scores

```powershell
python -m pip install -e ".[plots]"
python examples/plot_trinity_optimization.py --output-dir outputs/trinity_optimization
```

The standalone figure is `trinity_pareto.png`. Its calibration panel shows all
feasible search evaluations, the connected nondominated archive, selected roles
and the generalized baseline. The validation panel shows only the fixed
representatives and baseline. It does not draw or label a validation Pareto
front. Both panels use release NSE horizontally and storage NSE vertically;
higher values are better. Shared roles use one marker and a combined legend
label with their evaluation ID.

## Reproducibility and verification

The run records its NumPy PCG64 seed, optimizer settings, requested and actual
evaluation counts, rejection reasons, source/configuration hashes, package
versions and Python-code hashes. Preserve the generated configuration and
vectors with a result. Repeating the seed with identical code, inputs and
software reproduces the candidate sequence and deterministic scores; measured
runtime can vary. The [seed and budget study](convergence.md) now compares
nested budgets and independent seeds, with fixed-reference hypervolume and
monthly parameter spread.

The [adaptive runner](adaptive_calibration.md) extends this search with atomic
complete-generation checkpoints and provisional online stopping. Increasing a
cumulative stage budget restores the trial archive, population order and RNG
state; the earlier CSV-only workflow cannot provide exact continuation.

The verification suite checks independent Pareto definitions, crowding distance,
exact budgets with partial generations, seed reproducibility, invariance to
release-unit rescaling, fixed variables, rejected candidates, deterministic
representative ties and exclusion of validation from the search. Existing
scientific checks cover the model equations, water conservation, input period
isolation and saved MATLAB simulation/objective references.

The algorithm follows the nondominated sorting and elitist survival formulation
in [Deb et al. (2002)](https://ieeexplore.ieee.org/document/996017).
[Official NSGA-II documentation](https://pymoo.org/algorithms/moo/nsga2.html)
explains rank and crowding selection, and
[the official decision-making guide](https://pymoo.org/getting_started/part_3.html)
explains normalization using approximate archive extrema. These sources describe
the algorithm and decision-making conventions; this repository's runner uses
its own NumPy implementation.
