# Unified reservoir workflow

Use `tools/reservoir_workflow.py` for configurable calibration and versioned profiles.
The previous experiments, numerical modules, Trinity example and calibrated registry are preserved.

## Configuration

`configs/workflow_twenty_mixed_pooled.json` declares all twenty reservoirs and their verified
comparison archives. Defaults for new configurations are mixed years, pooled NSE, seeds
42/43/44, population 32, budget 512 and three independent worker processes.

```json
{
  "schema_version": 1,
  "year_selection": "mixed",
  "scoring": "pooled",
  "optimization": {
    "seeds": [42, 43, 44],
    "population": 32,
    "evaluation_budget": 512,
    "workers": 3
  },
  "reservoirs": [
    {
      "reservoir_id": "my_reservoir",
      "grand_id": null,
      "observations_csv": "observations.csv",
      "model_config": "model.json"
    }
  ]
}
```

Paths are relative to the configuration file; output paths are relative to the calling directory.
`year_selection` accepts `mixed` or `chronological`; `scoring` accepts `pooled` or `equal_regime`.
All four combinations work for new calibration. Archive import supports only combinations
actually evaluated in that archive and requires matching seeds, population and budget.
Unknown settings, duplicate IDs, duplicate non-null GRAND_IDs and unsafe ID paths are rejected.

The default input CSV columns are `date`, `inflow_m3s`, `storage_m3`, `release_m3s`.
Calibration needs observations; missing storage/release pairs may be represented by blank cells.
Dates must be sorted, unique and have a complete native daily/monthly sampling timeline.
Inflows must already be normalized **model inputs**, finite and nonnegative. Apply source
corrections once before this workflow; the physical model's `inflow_correction` remains explicit.

Alternative column names and units can be declared per reservoir:

```json
{
  "columns": {"date": "Date", "inflow": "I", "storage": "S", "release": "Q"},
  "units": {"inflow": "m3/day", "storage": "Mm3", "release": "m3/s"}
}
```

Supported flow units are `m3/s` and `m3/day`; storage units are `m3` and `Mm3`.
Unit conversion occurs before classification, fitting or simulation. Raw negative inflows are
not silently clipped or interpreted as withdrawals.

The model JSON declares `physical_constants`, `sampling`, `root_policy` and `atol_storage_m3`.
The existing `outputs/representative_years_v1_20261012/<id>/model_config.json` files are examples.
Physical constants use SI units and the existing validated DZTR parameter schema. Target
percentiles use the verified Hazen method. Solver policies and integration clocks are preserved.

## New calibration

Run commands from the repository root. Substitute the Python interpreter used by the project.

```powershell
python -B tools/reservoir_workflow.py prepare --config configs/my_reservoirs.json --output outputs/my_workflow_v1
python -B tools/reservoir_workflow.py run --config configs/my_reservoirs.json --output outputs/my_workflow_v1
```

`run` prepares a fresh directory if needed, then executes independent checkpointed seed jobs,
selects parameters using calibration evidence, assesses excluded years and exports a registry.
Repeating `run` resumes unfinished jobs and reuses verified completed results.

The prepared protocol pins normalized inputs, year masks, fitting bounds, physical constants,
sampling settings and code. Seeds and population are fixed in checkpoint identities. Budgets
must be multiples of population size so saved states represent complete generations.

Continue the same searches to a higher checkpoint:

```powershell
python -B tools/reservoir_workflow.py run --config configs/my_reservoirs.json --output outputs/my_workflow_v1 --evaluation-budget 1024
```

The preparation remains fixed. Budget-specific search results and releases are retained as
`jobs/<id>/seed_<seed>/budget_512` and `releases/budget_512`, alongside later versions.
Budgets cannot shrink below the current optimizer checkpoint; old release registries remain usable.
Changed code, inputs, scoring, masks, seeds or population require a new preparation/version.

Independent stages are also available:

```powershell
python -B tools/reservoir_workflow.py calibrate --output outputs/my_workflow_v1 --reservoir-id my_reservoir --seed 42
python -B tools/reservoir_workflow.py finish --output outputs/my_workflow_v1
```

Run/job locks prevent duplicate computation. After forced process termination, verify the
recorded PID is no longer active before removing a stale lock. Logs and checkpoints persist.
Preparation failures stop with the original reason; reservoirs are not silently omitted.

## Year selection and scientific safeguards

Both year selectors use at least 65% of all recorded rows and available storage/release pairs,
with one complete initial warmup year. Calendar years are indivisible. The tested comparison
selector retains a common later assessment set and a later training opportunity where possible.
Mixed years cover forcing regimes and antecedent/time variation. Short or incomplete records
that cannot satisfy the minimum fail preparation explicitly.

Targets and empirical bounds use calibration observations only. Simulation retains continuous
forcing/state through warmup, unscored and assessment years; no observed storage is assimilated.
Raw generalized targets are preserved even if invalid. A separate feasible bounds midpoint may
initialize optimization; it is not represented as a corrected generalized estimate.

Selection pools calibration nondominated fronts across seeds and chooses the normalized
distance-to-ideal compromise with deterministic cost/seed/evaluation ties. Seed compromises
are also frozen before assessment. No assessment score chooses or replaces a profile.
Independent trapezoidal balance reconstruction checks stored/simulated trajectories.

Scoring changes calibration objectives: pooled NSE uses all selected observations together;
equal-regime NSE averages separate dry/normal/wet NSE scores. Assessment reports both pooled
and represented-regime scores. Missing assessment regimes, tiny holdouts, failed seed replays
and negative NSE remain limitations rather than being hidden or repaired using alternate seeds.
512/1024 evaluations are checkpoints, not convergence certificates or prospective validation.

## Reuse the completed twenty-reservoir comparison

Importing verified results runs **zero optimization evaluations and zero simulations**:

```powershell
python -B tools/reservoir_workflow.py adopt --config configs/workflow_twenty_mixed_pooled.json --output outputs/unified_workflow_v1_20261013
```

The implemented release at that location contains all twenty mixed-year/pooled profiles.
Import recomputes calibration-only choices from frozen fronts, checks archived hashes and target
vectors, preserves assessment evidence and verifies mass balance. Existing registries are not
overwritten. Fort Randall retains a null GRAND_ID.

`adopt` imports profiles and evidence, not optimizer checkpoint identities from the older
comparison runners. Those checkpoints remain in their original archives. Checkpoint continuation
through `run` applies to jobs prepared by this unified workflow; importing profiles does not
silently convert an earlier experiment into a new search.

Registries use paths relative to their version directory and include profiles by reservoir ID
and verified GRAND_ID, a wide monthly parameter table, scores, seed diagnostics and provenance.
Profiles remain exploratory and do not claim certified source inputs or optimizer convergence.
Numerical/workflow source and release artifacts are pinned; an intentional future code change
requires a new verified release rather than silently mutating an existing one.

```powershell
python -B tools/reservoir_workflow.py verify --registry outputs/unified_workflow_v1_20261013/registry.json
```

## Simulate a versioned profile

The forcing CSV uses `date,inflow_m3s` with finite, nonnegative model-input inflows in m³/s.
Sampling and solver settings come from the exported profile.

```powershell
python -B tools/reservoir_workflow.py simulate --registry outputs/unified_workflow_v1_20261013/registry.json --grand-id 753 --forcing-csv outputs/representative_years_v1_20261012/ex5_Garrison/observations_si.csv --output outputs/garrison_unified_replay
```

Alternatively use `--reservoir-id ex5_Garrison`. For a different forcing start, supply
`--initial-state initial.json`, containing `{"storage_m3": ..., "release_m3s": ...}`.
The archived initial state is automatically used only at its original date. A legacy synthetic
target clock requires a dedicated profile for a different start; initial-state overrides do not
silently change that clock. A physically invalid trajectory fails rather than changing solvers.

The Python API is `workflows.reservoir.load_profile`, `simulate`, `prepare`, `calibrate`,
`finish` and `adopt`. The old `FrozenProfiles` reader continues to serve its original release;
use this API or the unified CLI to consume the new registries.
