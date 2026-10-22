# Complete workflow for the twenty calibrated reservoirs

The working output is `outputs/twenty_reservoir_workflow_20261012`. Run from the repository:

```powershell
python -B tools/complete_twenty_pipeline.py
```

The command prepares the frozen plan, continues the sixty archived512-evaluation checkpoints to1024, freezes pooled calibration selections, assesses their held-out performance, writes comparisons and diagnostics, checks the archived science, and publishes reports and parameter tables. Completed stages are reused. Failed continuation jobs retain their checkpoints and can resume through `tools/twenty_workflow.py --resume-job RESERVOIR SEED`. A partial postprocessing stage is preserved for inspection rather than silently overwritten.

```mermaid
flowchart LR
    A[Verified 512-trial archives] --> B[Continue identical searches to 1024]
    B --> C[Pool three calibration Pareto fronts]
    C --> D[Freeze equal-weight compromise]
    D --> E[Held-out assessment and diagnostics]
    E --> F[GRanD tables and selected profiles]
```

The selection rule minimizes equal-weight range-normalized distance from the pooled calibration ideal. Cost values, seed and trial ID break ties deterministically. All twenty selected identities are saved before pooled-choice validation. Validation scores do not choose a seed, parameter vector, root policy or model constants. These are reproducible research profiles, not automatically accepted operational parameters.

`baseline_comparison.csv` contains all three seeds at512 and1024 evaluations, in both calibration and held-out periods. `seasonal_bias.csv`, `trajectory_diagnostics.csv` and `zone_diagnostics.csv` report seasonal bias, storage-error drift, native release peaks and stored physical-storage zones. Sparse/constant-support NSE remains undefined. Peaks of monthly mean samples are not instantaneous flood peaks. Calibration P95 release thresholds use the configured percentile method and no held-out fitting.

`observed_balance_diagnostics.csv` uses the working model's integration/forcing convention on finite within-period observation intervals. It is a stated accounting hypothesis; a residual does not establish a missing flux or justify an inflow correction. Numerical simulation conservation is independently checked and remains a separate claim.

Convergence diagnostics compare fixed-reference hypervolume at512,896 and1024, report reference-excluded points and between-seed front distances, and apply an illustrative screen: last128 relative area gain below0.5%, seed-front distance at most0.02 NSE, and no reference exclusions. This screen is not a validated stopping rule. `parameter_variation.csv` separately reports bound-normalized variations among seed compromise vectors; neither this variation nor objective stability is a parameter-uncertainty estimate. No searches beyond1024 are scheduled automatically.

The GRanD workbook has an updated parameter catalog, twenty selected rows, monthly SI targets, a crosswalk, baseline comparisons and the untouched historical reference values. Optimized storage/release targets have explicit new field names; existing empirical percentile fields retain their meaning. Capacities and operating floors are inherited from the reviewed model profiles, not silently replaced by GRanD volumes.

The portable metadata snapshot is `data/catalogs/grand_v1_1_reference.csv`, with its external-workbook checksum and export provenance alongside it. Explicit IDs are maintained in `configs/calibrated_twenty_grand_ids.json`; simulation profile lookups are in `configs/calibrated_twenty.json`. Nineteen local v1.1 records match. US Fort Randall remains unresolved because the same-name2356 entry is Canadian. Its parameters remain accessible by internal reservoir ID. The source catalog's unique-ID semantics are described in the [GRanD v1.1 documentation](https://water-future.org/wp-content/uploads/2017/11/gwsp_GRanD_Technical_Documentation_v1_1.pdf).

To load a selected profile into the existing model contracts:

```python
import json
from pathlib import Path
import pandas as pd
from dztr.schemas import ReservoirParameters
from dztr.simulation.sampled import SamplingSettings

root = Path.cwd()  # repository root
registry = json.loads((root / "configs/calibrated_twenty.json").read_text())
profile_file = root / registry["profiles_by_GRAND_ID"]["253"]
profile = json.loads(profile_file.read_text())
targets = pd.read_csv(profile_file.parent / profile["monthly_targets_file"],
                      index_col="month", float_precision="round_trip")
parameters = ReservoirParameters(targets, **profile["physical_constants"])
sampling = SamplingSettings(**profile["sampling"])
root_policy = profile["root_policy"]
```

Use these three objects with the existing root-policy simulation API and explicitly supplied normalized forcing. Initial states and forcing support must suit the intended simulation period. The saved selected trajectories already reproduce the historical experiment and need not be rerun merely to inspect results.

`other_reservoirs_pending.csv` retains the twenty uncalibrated reservoirs' actions plus Bull Lake's separate cached interval case. They are parked for later work, as requested. The numerical modules, canonical inputs, original GRanD/reference workbooks and512-evaluation archives remain unchanged.
