# DZTR model

[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23631633.svg)](https://doi.org/10.5281/zenodo.23631633)

A modular Python implementation of the dynamically zoned target release reservoir model described by [Yassin et al. (2019)](https://doi.org/10.5194/hess-23-3735-2019). Observation normalization, generalized parameterization, daily reservoir simulation, evaluation, a calibration/validation baseline and a budgeted NSGA-II calibration search are implemented. The simulation and objective port is checked against saved Trinity MATLAB outputs.

**Start here:** [researcher documentation](docs/index.md),
[portable application quick start](docs/portable_application_v1.md), and
[scientific assessment](docs/scientific_assessment_v1.md). The stable `dztr`
command applies self-contained frozen Gardiner, Oahe and Garrison profiles using
inflow alone and delegates configured parameterization/calibration jobs to the
existing versioned run managers. Public examples require no private `outputs/`
archives. Version 0.3.1 uses the MIT license and includes cross-platform release
verification. See the [documentation](https://fuadyassin.github.io/DZTR_model/)
and [release/citation guide](docs/release_v0_3_1.md). Cite version 0.3.1 with its
[software DOI](https://doi.org/10.5281/zenodo.23631633). The model paper is cited
separately. The [GitHub release](https://github.com/fuadyassin/DZTR_model/releases/tag/v0.3.1)
provides the source bundle, wheel, rendered documentation and checksums.

## Install and test

Use Python 3.10 or later. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
New-Item -ItemType Directory -Force outputs | Out-Null
.venv\Scripts\python -m pytest -q
```

The distribution retains the name `dztr_param`. New functionality is under `dztr`; existing imports from `dztr_param` remain available. Bundled workbooks remain repository data rather than being included in the wheel.

The [world reservoir workflow](docs/world_reservoirs.md) expands the examples to
the 42 original MATLAB folders, with per-reservoir settings, reconciled daily
or monthly observations, historical baseline replay, and calibration-only
Python baselines. It reports data/model blockers individually and exposes an
explicitly budgeted optimizer for normalized samples.

The **full local collection of 41 active samples** has permanent SI inputs in
[`data/reservoirs/`](data/reservoirs/README.md), fixed settings in
[`configs/reservoirs/`](configs/reservoirs), and a shared
[manifest](data/reservoirs/manifest.json). They run without the original E-drive
folders. From the repository root:

```powershell
python examples/run_reservoir_collection.py --reservoir Trinity --output outputs/trinity_new
python examples/run_reservoir_collection.py --all --output outputs/collection_new
```

The collection reports input errors and scientific blockers per reservoir.
Tuyen Quang is excluded. Outputs remain separate from permanent data; packaging
has not committed or uploaded these files.

## Shared example data

Only **Gardiner, Trinity, Oahe and Garrison** observation examples are included
in the shared repository. Other inflow, storage, release and level series,
station-download archives, prepared source datasets and donor flow records
remain local and ignored by Git. Gardiner also includes the selected level and
geometry inputs needed to explain conversion from level to storage. The
[public data guide](data/public_examples/README.md) lists every exception.

```powershell
& .\tools\Set-ProjectCache.ps1
python -B examples/run_reservoir_collection.py --manifest data/public_examples/manifest.json --reservoir Gardiner --output outputs/public_gardiner_new
```

Use the explicit public manifest for a fresh checkout. Commands using the full
local manifest, collection-wide audits, archive-dependent tests or saved
`outputs/` paths require the corresponding private inputs or newly prepared
results. These data-sharing choices preserve all existing local records and
scientific settings.

## Estimate parameters

```python
from dztr.data import read_reservoir_observations
from dztr.parameterization import estimate_generalized_parameters, estimate_percentiles

observations = read_reservoir_observations("ResDataInflowStorageRelease/Gardiner_253.xlsx")
print(observations.quality)
quantiles = estimate_percentiles(observations)
parameters = estimate_generalized_parameters(
    observations,
    channel_capacity_m3s=2500,  # Known catalog value; omit to use release P99.
)
print(parameters.monthly_targets)  # Twelve months, six targets per month.
```

Normalized storage and target storage use **m³**; inflow, release and target release use **m³/s**. Unit-bearing source headers establish their units. For populated bare `Storage`, `Inflow`, `Outflow` columns, specify units explicitly:

```python
from dztr.data import normalize_observations

observations = normalize_observations(
    dataframe, storage_unit="million_m3", flow_unit="m3/s"
)
```

The quality report distinguishes `available`, `placeholder`, and `unavailable` datasets. Missing observations and calendar gaps stay missing. Duplicate daily dates, invalid measurements and ambiguous populated units raise errors. Placeholder or unavailable datasets cannot produce parameters.

Percentiles use linear interpolation and non-exceedance probabilities. The shared default includes 5–100 in steps of five **plus 99**, with sample counts for each variable/month. Generalized targets default to P10/P45/P85. Every month must have storage and release observations to construct a complete target set.

## Reproduce the bundled estimates

```powershell
.venv\Scripts\python -m dztr.parameterization.benchmark --repo . --output outputs/parameterization
```

This checks all **780 saved quantiles per reservoir** for Gardiner (253) and Trinity (131), calculates the new P99 estimates, and audits all 65 observation workbooks. It writes `summary.json`, `dataset_audit.csv`, `quantiles_si.csv`, and `monthly_targets_si.csv` without changing any source workbook. The expected audit is **2 available datasets and 63 placeholders**. Outputs are ignored by Git.

Gardiner contains daily observations for 1980–2011; Trinity for 1970–2000. Reproduction intentionally uses each complete historical record. The calibration/validation baseline below estimates targets only from its calibration period.

## Existing workbook workflow

```python
from pathlib import Path
import pandas as pd
from dztr_param import update_extended_data

saved = pd.read_excel("Extended_GRanD_Dams_v1_1_DZTR.xlsx")
updated = update_extended_data(saved, [253, 131], folder="ResDataInflowStorageRelease")
Path("outputs").mkdir(exist_ok=True)
updated.to_excel("outputs/updated_parameters.xlsx", index=False)
```

Compatibility functions retain storage in **million m³**, matching the existing workbooks. The updater returns a copy, adds missing percentile columns, preserves metadata, and skips placeholders with a warning. Existing exporters are `create_com_csv`, `create_com_csv_hylakes`, and `create_com_txt_hylakes`. Their external format and unit conventions still need verification before simulation integration; the required HydroLAKES workbook is not bundled.

## Simulate using the current parameterization

```python
from dztr.data import read_reservoir_observations
from dztr.parameterization import estimate_generalized_parameters
from dztr.schemas import ReservoirParameters
from dztr.simulation import simulate_reservoir
from dztr.evaluation import evaluate_simulation

observations = read_reservoir_observations("ResDataInflowStorageRelease/Trinity_131.xlsx")
estimates = estimate_generalized_parameters(observations, channel_capacity_m3s=300)
parameters = ReservoirParameters.from_generalized(
    estimates,
    capacity_m3=2533500000,
    initial_storage_m3=1593000000,
    initial_release_m3s=10,
    inflow_correction=0.92,
    flood_mode="storage",
)
result = simulate_reservoir(observations, parameters)
metrics = evaluate_simulation(observations, result, warmup_steps=365)
```

These constants are the verified Trinity MATLAB configuration. Use reservoir-specific
values for other reservoirs. `examples/simulate_trinity.py` is a complete runnable
example using the repository's observations and new generalized parameterization.

The default solver solves the coupled piecewise-linear trapezoidal balance
algebraically and conserves water when handling capacity overflow. For the
historical loop, select `solver="matlab_fixed_point", overflow_mode="matlab"`.
Results include water-balance residuals and overflow volumes. Simulation requires
consecutive daily dates and complete inflow; gaps are never silently filled.

## Verify the MATLAB port

```powershell
.venv\Scripts\python -m pip install -e ".[dev,matlab]"
.venv\Scripts\python -m dztr.simulation.benchmark --matlab-folder "E:\1_DataHub\E_Reservoir_data\ex12_Trinity" --repo . --output outputs/trinity_simulation
```

The benchmark reads the original files without editing them, imports their
coefficient layout into the shared monthly target table, and compares three
saved evaluation members over all **11,323 days**. It also compares the legacy
annual output and baseline performance metrics, then runs the current `linear`
and MATLAB-compatible `hazen` parameter estimates. The ordinary test suite
includes a portable two-year reference fixture, so it does not require the E drive.

The default percentile method remains `linear` for the existing Python workbooks.
Use `quantile_method="hazen"` to reproduce MATLAB's historical `prctile` algorithm.
See [the MATLAB port notes](docs/matlab_port.md) for parameter mapping, numerical
choices, reference provenance and the inferred optimization calendar discrepancy.

After the benchmark, generate the observed and simulated Trinity comparison:

```powershell
.venv\Scripts\python -m pip install -e ".[plots]"
.venv\Scripts\python examples/plot_trinity_simulation.py
```

The figure is saved as `outputs/trinity_simulation/trinity_comparison.png`.

## Establish the calibration/validation baseline

```powershell
.venv\Scripts\python -m dztr.calibration.benchmark --repo . --output outputs/trinity_baseline
```

The initial Trinity settings use 1970 for spin-up, 1971–1985 for calibration and
1986–2000 for validation. These dates interpret the paper's first-year/remaining-
halves protocol for the available record; they are configurable in
`configs/trinity_baseline.json`.

Targets use calibration-only `hazen` percentiles. Spin-up is excluded from
estimation and scoring, and validation carries the simulated state forward
without resetting. The run writes daily values, monthly targets, separate
scores, paired counts and balance diagnostics. The runnable example is
`examples/calibration_baseline_trinity.py`; see [the baseline notes](docs/calibration_baseline.md)
for scientific choices, current results and the optional comparison plot.

## Prepare calibration candidates and objectives

```powershell
.venv\Scripts\python -m dztr.calibration.review --repo . --output outputs/trinity_calibration
```

This prepares 72 variables in MATLAB order, estimates P5–35/P35–75/P75–95 bounds
from calibration observations, validates physical constraints and evaluates the
starting candidate with `[-NSE(release), -NSE(storage)]`. The callable uses only
calibration data; held-out evaluation is separate. Invalid candidates receive
explicit rejection reasons. No optimizer runs at this stage.

Use `--matlab-folder "E:\1_DataHub\E_Reservoir_data\ex12_Trinity"` with the MATLAB
extra to check three saved full-record objective references. Historical bounds
and candidates differ from the new calibration-derived search space. See
[the candidate and objective notes](docs/calibration_objectives.md) for API usage,
reference provenance and the optional bounds plot.

## Run a small calibration search

```powershell
.venv\Scripts\python -m dztr.calibration.search --repo . --output outputs/trinity_optimization
.venv\Scripts\python examples/plot_trinity_optimization.py
```

The configured NSGA-II pilot uses seed 42, a population of 16 and 96 search
evaluations, including rejected candidates. It starts with the generalized
baseline and searches the two NSE objectives separately. All variables are
normalized for variation, then decoded into the existing 72-value representation.

The run saves every evaluated vector, rejection reasons, generation history and
the feasible nondominated archive. Release-focused, storage-focused and balanced
representatives are selected from calibration scores before validation is scored.
Selected monthly SI targets and continuous daily runs are exported alongside
source/code hashes and resolved settings. This small run checks the workflow;
it does not establish convergence. See [the optimization guide](docs/optimization.md)
for the API, selection rule, artifacts and limitations.

## Compare seeds and evaluation budgets

```powershell
.venv\Scripts\python -m dztr.calibration.study --repo . --output outputs/trinity_convergence --workers 3
.venv\Scripts\python examples/plot_trinity_convergence.py
```

The study runs seeds 42, 43 and 44 to 1,024 evaluations each, with nested
checkpoints at 96 and 384 evaluations. All searches use the same population and
calibration-only bounds. A fixed hypervolume reference measures improvement in
the two objectives; descriptive seed ranges and monthly target spreads show
search and parameter variability. Pooled representatives are chosen using
calibration costs, then scored continuously on validation.

Completed seed runs are reused after settings, data, code, version and artifact
hash checks. `--analyze-only` requires verified runs and recomputes study
reporting. See [the convergence study guide](docs/convergence.md) for indicators,
artifacts and the limits of these comparisons.

## Review budgets and prepare multiple reservoirs

```powershell
.venv\Scripts\python -m dztr.calibration.budget_review --study outputs/trinity_convergence --output outputs/trinity_budget_review --policy configs/stopping_policy.example.json
.venv\Scripts\python examples/plot_budget_review.py
.venv\Scripts\python -m dztr.calibration.fleet --manifest configs/fleet_manifest.example.json --repo . --output outputs/fleet_preflight
```

The retrospective diagnostic compares hypervolume gains, full-front movement
and objective extrema over consecutive calibration-only windows. Its thresholds
are illustrative and require extension and reservoir-cohort audits. Trinity's
three saved searches all require continued searching at 1,024 evaluations under
this example policy; the review starts no new searches or simulations.

The fleet manifest references separate reservoir configurations. Preflight
normalizes observations to SI, checks coverage and physical/bound feasibility,
and records readiness or a failure reason independently for each reservoir.
It writes reusable per-reservoir configurations and a fleet status table.
See [the scalable calibration guide](docs/scalable_calibration.md) for the
manifest contract and provisional stopping thresholds.

## Resume searches and check stopping during calibration

```powershell
.venv\Scripts\python -m dztr.calibration.adaptive --repo . --config configs/trinity_adaptive.json --output outputs/trinity_adaptive_prepared/seed_42 --evaluation-budget 512
.venv\Scripts\python -m dztr.calibration.adaptive --repo . --config configs/trinity_adaptive.json --output outputs/trinity_adaptive_prepared/seed_42 --resume --evaluation-budget 1024
```

The second stage restores the population, trial archive and RNG state. Its
budget is cumulative: completing these stages requires 512 new trials each.
Input, configuration, policy, code and software identities must match before
saved work is reused. Complete-generation checkpoints use checked numeric
arrays and JSON metadata without pickle.

Online stopping checks calibration-only changes every 128 evaluations under
the example policy. A provisional stopping candidate can end a search early;
`--audit-extension` continues from its checkpoint to test later improvement.
The policy hard cap remains 4,096. Validation is reported after calibration
representatives are fixed and never controls stopping.

`examples/checkpoint_reservoir_study.py` runs three Trinity seeds through
512/1,024/2,048 cumulative stages and a qualified Gardiner pilot through
512/1,024 in one process pool. It verifies Trinity's historical 1,024-trial
prefixes and records any early stop followed by an audit extension. This is an
exploratory two-reservoir workflow; a general fleet scheduler and a diverse
cohort audit remain further work. See [the adaptive calibration guide](docs/adaptive_calibration.md).

## Compare population sizes at equal budgets

```powershell
.venv\Scripts\python -m dztr.calibration.population_study --repo . --config configs/trinity_population_study_prepared.json --output outputs/trinity_population_study_prepared --workers 3
.venv\Scripts\python examples/plot_population_study.py --output-dir outputs/trinity_population_study_prepared
```

This experiment compares populations 16, 32 and 64 across three seeds at
512/1,024/2,048 cumulative evaluations. The completed original-code comparison
reused three verified population-16 archives and ran six new searches. The
fresh prepared-code configuration above starts all nine jobs with the current
source identity; that new experiment has not been launched. Each new search runs to the same final budget,
and calibration-only stopping diagnostics are reported retrospectively.

The runner records each job independently and restores interrupted jobs with
`--resume`. `--analyze-only` recomputes reports from complete verified sources.
Calibration scores, front agreement and monthly target variability are kept
separate. Larger populations also change initialization and the number of
generations; three seeds describe variability rather than establish a universal
fleet default. See [the population study guide](docs/population_study.md).

For automatic reporting after a long run, start
`examples/finish_population_study.py --wait` in another terminal. It verifies
all completed sources before publishing a comparison figure and results note;
it performs no simulations. The original local comparison is complete; its
[verified results](outputs/trinity_population_study/population_results.md) and
[figure](outputs/trinity_population_study/population_comparison.png) cover 18,432
search trials. Population 64 led median hypervolume at 2,048 trials by about
0.33% over population 16, while none met the seed-agreement tolerance. These
results do not establish convergence or a fleet default.

Daily simulation now reuses monthly targets within each call. All 712 tests
passed, and selected Trinity/Gardiner calibration replays matched exactly while
using about 7.3 times less CPU. This timing does not estimate complete optimizer
throughput. Historical checkpoint identities are preserved; use new output
folders for the changed source. See [the simulation performance guide](docs/simulation_performance.md).

## Package responsibilities

```text
src/dztr/
  schemas.py          shared observations and parameter contracts
  data/               local readers, dates, units, quality reports
  parameterization/   percentiles, monthly targets, reproducibility check
  model/              release equations and coupled balance solver
  simulation/         daily time stepping, diagnostics and MATLAB benchmark
  evaluation/         NSE, KGE and the MATLAB objective metrics
  calibration/        splits, objectives, NSGA-II search and convergence studies
  integrations/       MATLAB observation, coefficient and vector adapters
configs/              reproducible reservoir selections
examples/             parameter estimation, simulation and comparison plots
tests/                input edge cases, model equations and saved MATLAB references
docs/                 scientific conventions and staged development
```

See [the parameterization conventions](docs/parameterization.md) and [the staged architecture](docs/architecture.md) for details and open scientific questions.

