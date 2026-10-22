# One workflow for four data-availability cases

The versioned gateway `tools/dztr_run_v8.py` adds two stages to the existing
manager: `no_observations` generates and runs fixed profiles; `compare_routes`
joins them with frozen joint, release-only and storage-only results. Older
configuration versions continue to use their original runners.

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/dztr_run_v8.py plan --config configs/run_manager_availability_paper_v1.json
python -B tools/dztr_run_v8.py run --config configs/run_manager_availability_paper_v1.json --output outputs/my_four_route_run
python -B tools/dztr_run_v8.py run --config configs/run_manager_availability_paper_v1.json --output outputs/my_four_route_run --resume
```

`plan` performs no simulation or optimization. The paper run requests 32 monthly
reservoirs and retains all 37 paper inventory rows, including Bull Lake's
separate interval route and the four source-gated cases. This delivery runs
fixed profiles only; it does not resume existing optimizer jobs.

## Parameter generation without storage or release

For `ungauged_mean_flow`, storage targets are fixed to 20%, 30% and 97% of the
adopted capacity. The explicitly configured active-storage adaptation uses
`dead_storage + fraction * (capacity - dead_storage)` instead; the paper
configuration preserves that existing assumption for Dickson, E. B. Campbell
and Yellowtail. Neither capacity nor dead storage is adjusted to make targets
valid. An invalid target set remains gated.

Release targets use the existing shared mean-flow prior: the target's verified
GRAND DIS_AVG_LS divided by 1,000 gives m³/s. Where explicitly permitted, a
volume-weighted mean effective inflow over intervals wholly within calibration
is the fallback. The mean is multiplied by median external-donor P5/P30/P97
to mean ratios, excluding the target. These three annual targets repeat across
all twelve months. Channel limiting and donor provenance remain explicit.

Initial storage is the assumed normal target; initial release is the first
effective inflow clipped to zero and channel capacity. The estimator receives
an inflow-only frame. Storage and release observations are kept separately and
read for assessment after parameters have been frozen. The original mixed-year
masks and at least 65% calibration support are retained. State remains
continuous across years. No optimizer is available for this no-response recipe.

Inflow, capacity and channel metadata may themselves depend on historical
observations. External donor simulations represent one region and are not
independently certified. This is an exploratory conditional benchmark, not proof
of global ungauged or forecast skill.

## Automatic recommendation and explicit method choice

The recommendation depends on usable calibration observations and source gates:

| Usable data | Recommended method | Fitted target responses |
|---|---|---|
| Storage and release | `observed_monthly` | Both |
| Prepared level-derived storage and release | `level_geometry_monthly` | Both, with the geometry provenance |
| Release only | `flow_only` | Release |
| Storage only and a usable release prior | `storage_only` | Storage |
| Neither response and a usable release prior | `ungauged_mean_flow` | None |

Every numerical route also needs inflow and valid physical inputs. A source
gate remains a gate. The recommendation selects an already prepared profile;
an absent recommended profile is reported explicitly. It does not switch to
another route because assessment scores look better, and it does not start
optimization implicitly. Level-derived profiles still need their existing
reference, timing and conversion checks before being usable.

The comparison's `route_registry.json` links all available comparison profiles
to their original sealed registries. The default application selects the
recommended method. An explicit weaker-data comparison uses `method`:

```json
{
  "id": "application",
  "type": "simulate",
  "config": {
    "registry": "@comparison.registry",
    "forcing_csv": "inputs/new_prepared_inflow.csv",
    "reservoir_id": "my_reservoir",
    "method": "ungauged_mean_flow"
  }
}
```

Omit `method` for the automatic recommendation. A verified `grand_id` can replace
`reservoir_id`. A different forcing start requires an `initial_state` JSON with
`storage_m3` and `release_m3s`; the legacy clock restrictions remain active.
The simulation gateway expects prepared SI forcing, preserving the declared
source transformation and physical inflow correction. Responses are not read
for application. Missing GRAND IDs stay blank and remain selectable by exact
reservoir ID.

## Comparison and interpretation

The report recalculates metrics from saved trajectories on the same observations,
native calendars and original masks. It rejects changed response data, year
selections, profile identities and physical-balance checks. Missing cached
support is reported without rerunning old models. Per-case phase scores,
paired differences, seasonal biases, plots, parameters and failures are saved.

Available-case medians retain each route's actual membership; a separate
common-successful-case summary uses identical membership across all four
routes. Profiles and simulation success are separate fields. A valid assumed
profile can still fail at a timestep; that failure is not converted into a
successful run or removed from the inventory.

The existing joint calibration used a 512-evaluation NSGA-II search; the two
single-response routes use the 64-evaluation DDS checkpoint of a frozen
512-evaluation horizon. The no-response route uses zero optimizer evaluations.
Different data access, initial states and target dimensions prevent attributing
performance differences to the optimizer alone. Convergence remains unproven.

The parameter table has one row per reservoir and method, with profile source
paths for full provenance. The route inventory has one row per reservoir and
the recommended method. Existing registries and earlier scientific outputs
remain unchanged. Completed jobs and known failures have persistent receipts;
`--resume` verifies and reuses them. All project caches and temporary files
remain on D:.
