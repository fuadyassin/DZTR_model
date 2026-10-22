# Optional scalar GA with DDS retained

The release-only and storage-only v2 calibration backends accept an explicit
optimizer choice. **DDS remains the default** when `optimizer` is omitted.
The original v1 DDS modules, NSGA-II modules, saved experiments and checkpoint
identities are preserved. Existing configuration versions still use their
original runners through `tools/dztr_run_v9.py`.

## Choose the optimizer

Use `dztr_flow_calibration_v2` or `dztr_storage_calibration_v2`, schema version 2.
Both retain the existing objectives, fitting masks, bounds, initialization and
physical solver. The flow case fits 36 release coordinates; the storage case
fits 36 storage coordinates. The unobserved response remains excluded from
target fitting, initialization, bounds, scoring and parameter selection.

DDS configuration:

```json
"optimization": {
  "seeds": [42, 43, 44],
  "horizon": 512,
  "evaluation_budget": 64,
  "workers": 4,
  "perturbation": 0.2
}
```

The equivalent explicit choice adds `"optimizer": "dds"`.

GA configuration:

```json
"optimization": {
  "optimizer": "ga",
  "seeds": [42, 43, 44],
  "horizon": 512,
  "evaluation_budget": 64,
  "workers": 4,
  "ga": {
    "population_size": 32,
    "crossover_probability": 0.9,
    "crossover_eta": 15.0,
    "mutation_eta": 20.0,
    "local_initialization_fraction": 0.5,
    "local_initialization_scale": 0.05
  }
}
```

GA controls shown above are the recorded defaults; only overrides need to be
supplied under `ga`. Mutation probability is fixed at one divided by the active
dimension per gene, matching the existing NSGA-II operator. DDS `perturbation`
is rejected for GA, and GA options are rejected for DDS, so no setting is ignored
silently. Capacity, channel limits and other physical parameters are unaffected.

## GA algorithm and shared implementation

`strategies/ga_v1.py` reuses the existing NSGA-II bounded simulated-binary
crossover, polynomial mutation and Latin-hypercube functions. It supports any
nonempty vector dimension, including the 36-coordinate calibration cases.
This is an elitist scalar genetic algorithm with parent-plus-offspring survival,
not a call to the existing two-objective NSGA-II optimizer.

The supplied initial parameter vector is trial zero. Remaining initial
candidates combine bounded local normal draws and a global Latin hypercube.
Binary tournaments choose parents by scalar cost, with evaluation ID breaking
ties. Feasible candidates rank before rejected candidates. Survivors are the
best candidates from the parents and evaluated offspring. All previously
evaluated duplicate vectors are excluded. Fixed coordinates remain fixed;
fixed-bound or representationally exhausted searches stop with an explicit
reason instead of fabricating new evaluations. A hydrological rejection remains
an infeasible trial and counts toward the evaluation budget.

This is the same broad GA structure described by the
[pymoo GA documentation](https://pymoo.org/algorithms/soo/ga.html), with shared
variation operators from our frozen implementation of
[NSGA-II](https://pymoo.org/algorithms/moo/nsga2.html). No new dependency is installed.

## Run-manager and direct commands

The new manager configuration is `dztr_run_manager_v7`, schema 7. Use calibration
backend `prepared_flow_v2` or `prepared_storage_v2`. Earlier backends remain
available. The configuration example below plans four Gardiner comparisons;
running it executes new jobs, so choose it deliberately:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/dztr_run_v9.py plan --config configs/run_manager_optional_ga_gardiner_v1.json
python -B tools/dztr_run_v9.py run --config configs/run_manager_optional_ga_gardiner_v1.json --output outputs/my_optional_ga_run
```

Individual configurations are `configs/flow_ga_gardiner_v2.json`,
`configs/storage_ga_gardiner_v2.json` and their `_dds_` counterparts. For a
standalone GA campaign:

```powershell
python -B tools/calibrate_flow_v2.py run --config configs/flow_ga_gardiner_v2.json --output outputs/my_flow_ga
python -B tools/calibrate_flow_v2.py run --config configs/flow_ga_gardiner_v2.json --output outputs/my_flow_ga --evaluation-budget 128
```

Use `calibrate_storage_v2.py` for storage. Continuation keeps the configuration,
seed, horizon, optimizer controls, priors and bounds unchanged. The requested
budget may grow within the frozen horizon. After a managed run is sealed,
preserve it and copy its campaign to a new versioned directory before extending
its optimizer checkpoints; otherwise the manager's completed-stage receipt
would no longer match its artifacts.

GA saves queued proposals, partially evaluated batches, survivors and complete
PCG64 state. Resume reconstructs proposal/survivor lineage from cached costs
without reservoir evaluations and rejects altered state. A checkpoint can stop
inside initialization or a generation and continue exactly. Checkpoints are
saved every eight evaluations and at termination. DDS keeps its existing
continuation schedule. Changing optimizer requires a fresh campaign, and old
v1 checkpoints continue through their original v1 backend.

## Reporting and scientific limits

Registries, profiles, search/phase scores, summaries and GRAND tables identify
`GA` or `DDS`, optimizer settings, requested budget and actual evaluation counts.
GA population size and fixed horizon are explicit. A GA profile does not claim a
DDS horizon. Selection still minimizes calibration cost with seed/evaluation-ID
tie breaking; held-out observations remain assessments. Physical ordering,
solver rejection and independent mass balance remain unchanged.

The small real-data verification uses Gardiner with one seed per route and
continues GA from 32 to 64 evaluations. It reuses the exact old DDS reference
searches after checking objective inputs, bounds and initial targets. Its scores
verify functionality; they do not establish optimizer superiority or convergence.
The original four-route report remains a frozen report of its original DDS
campaigns. GA experiments receive separate versioned registries and reports.

For a later optimizer comparison, match reservoir membership, objective,
parameter bounds, initial reference, year masks and total objective-evaluation
budget. Include initialization and rejected candidates in that budget. Equal
generation counts are not equal computation. Compare multiple seeds before
changing the preferred optimizer. All project cache and temporary files stay on D:.
