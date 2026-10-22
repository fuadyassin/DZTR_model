# Shared level–geometry parameter generation and calibration

Geometry-derived storage is now an explicit input route for the shared estimators. The configured sequence is: prepare observations/reference → generate targets → simulate a fixed baseline → prepare/calibrate → assess/export by GRAND ID → replay the selected profile. Existing observed-storage workflows and frozen scientific releases remain usable through their original versions.

The complete Gardiner example is [run_manager_gardiner_shared_geometry_v1.json](../configs/run_manager_gardiner_shared_geometry_v1.json). Use the v5 gateway:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/dztr_run_v5.py plan --config configs/run_manager_gardiner_shared_geometry_v1.json
python -B tools/dztr_run_v5.py run --config configs/run_manager_gardiner_shared_geometry_v1.json --output outputs/my_shared_geometry_run
python -B tools/dztr_run_v5.py run --config configs/run_manager_gardiner_shared_geometry_v1.json --output outputs/my_shared_geometry_run --resume
```

The example deliberately uses one seed, population 8 and **16 evaluations** to verify the integration. This is not a recommended calibration budget or evidence of convergence. The completed, continued Gardiner demonstration is [here](../outputs/managed_gardiner_shared_geometry_v1_20261019/README.md).

## Estimator and storage-source choices

The shared data-availability configuration uses `kind=dztr_data_availability_v2`, `schema_version=2`. Its `methods` list still selects the existing numerical target estimators: `observed_monthly`, `level_geometry_monthly`, `flow_only`, `storage_only` and `ungauged_cr1`. Their target equations are unchanged. For geometry-derived storage, choose:

```json
{
  "methods": ["level_geometry_monthly"],
  "supplemental_inputs": {
    "ex14_Gardiner": {
      "levels_csv": "path/to/levels_si.csv",
      "level_time_basis": "daily_mean_level",
      "allow_mean_level_proxy": true,
      "geometry_csv": "path/to/curve.csv",
      "geometry_format": "regeom",
      "geometry_grand_id": 253,
      "geometry_evidence": "Verified reservoir/GRAND mapping",
      "level_reference": {
        "method": "calibration_storage_alignment",
        "level_datum": "PRAIRIE FARM REHABILITATION ADMINISTRATION DATUM"
      },
      "storage_uncertainty": {
        "kind": "calibration_offset_quantiles",
        "probabilities": [0.05, 0.95]
      }
    }
  }
}
```

Paths are relative to the configuration containing them. Normal run-manager stage references also work: for example `levels_csv: "@acquire.levels_csv"` and `level_provenance_json: "@acquire.provenance"`. Existing Canadian station-list acquisition can sum separately declared inflow and outflow gauges; one level gauge is selected separately. `datum_metadata_json` optionally pins station metadata and rejects a published datum that disagrees with the declared level datum. It does not certify historical datum continuity.

The estimator uses the frozen mixed-year selection, with at least 65% of the whole calendar assigned to calibration. Reference fitting, uncertainty fitting, monthly percentiles and search bounds use calibration rows only. The model runs continuously; year masks select scoring and fitting rows rather than resetting reservoir state.

## Reference methods

| `level_reference.method` | Required information | Interpretation |
|---|---|---|
| `documented_bottom` | `zero_level_m`, `level_datum`, `reference_datum`, `evidence`; an explicit `level_to_reference_offset_m` and `datum_conversion_evidence` when datums differ | Documented depth reference; surveyed depth requires suitable surveyed geometry as well |
| `documented_level_depth` | `anchor_level_m`, `anchor_depth_m`, `level_datum`, `evidence`; optional `reference_datum`, `datum_offset_m`, `datum_evidence` | A documented matching level/depth anchor |
| `calibration_storage_alignment` | `level_datum`, overlapping usable calibration storage and levels; optional `min_pairs` (default 30) | Median effective zero from level minus inverse-geometry depth; requires storage observations |
| `full_supply_capacity` | `anchor_level_m`, `level_datum`, `evidence`, `allow_exploratory=true`; optional explicit datum conversion | Assumes the operating level corresponds to adopted capacity at its unique inverse-curve depth |
| `full_supply_top` | Same anchor information and explicit exploratory opt-in | Assumes the operating level equals the actual tabulated curve top |
| `calibration_maximum` | `level_datum`, `evidence`, `allow_exploratory=true` | Assumes the calibration maximum equals curve top; never a full-record maximum |

The latter three are available for controlled scenarios, not automatic defaults. Gardiner's top/maximum assumptions previously failed physical target checks. A constant datum/reference is required over the applicable record; unresolved historical datum changes remain a source issue. The converter neither infers a surveyed bottom from dam height nor changes capacity to make a scenario work.

## Explicit uncertainty and calibration options

`storage_uncertainty` is required for the new reference interface:

| `kind` | Configuration | Meaning |
|---|---|---|
| `none` | No additional fields | Point proxy; no uncertainty adjustment |
| `fixed_offset_band` | `half_width_m`, `evidence` | Explicit scenario perturbation of the geometry zero |
| `calibration_offset_quantiles` | Two `probabilities` straddling 0.5; calibration storage-alignment reference required | Empirical spread of equivalent-zero residuals on calibration pairs |

Each band endpoint is independently converted through the bounded geometry curve. Missing/out-of-domain band endpoints on scored calibration proxy support block calibration. There is no extrapolation, volume rescaling or capacity repair. Offset bands omit other uncertainties unless represented by the declared scenario; they are **not confidence intervals** for true volume, bottom elevation or the geometry product.

Use the calibration stage backend `prepared_geometry_v1` and configuration kind `dztr_geometry_calibration_v1`. Declare `scoring` as `pooled` or `equal_regime`, and `storage_objective` as:

- `proxy_nse`: standard negative NSE against the geometry-derived storage center.
- `reference_band_nse`: negative compatibility score, with squared error only outside the declared band. The score is `1 − sum(distance(simulation, band)^2) / sum((proxy − mean(proxy))^2)`. Zero-width bands reduce to point-proxy NSE. Equal-regime scoring averages the separate regime scores.

Release uses observed release NSE. Raw proxy NSE remains in the diagnostics even when the band objective drives storage calibration. The shared `level_geometry_monthly` route requires release observations; storage-only and signed/interval calibration continue to need their respective supported adapters. Their parameter generators remain available, but this new calibration adapter does not silently convert them to joint standard-native jobs.

## Calibration, selection and export safeguards

Geometry calibration has its own checkpoint identity, including the prepared observations, reference/uncertainty receipt, frozen year mask, source code, storage objective and optimizer settings. Existing observed-storage checkpoints cannot be substituted. Seeded continuation preserves prior trials/vectors and immutable budget snapshots. Independent reservoir/seed jobs retain execution logs and progress files.

All candidate and seed choices use calibration evidence only. The release records the pooled selected profile and per-seed calibration choices before assessing them. Existing supplied storage is preserved separately and used for common assessment whenever available. Proxy assessment is reported separately; missing independent storage remains unavailable. Calibration-only input provenance does not establish independent ungauged forcing or initialization.

The GRAND-indexed parameter table and profile JSON carry the estimator, reference method/datums/effective zero, uncertainty policy, objective, geometry signature and fitting-year signature. Missing GRAND IDs retain reservoir identifiers. No newly generated profile replaces the original paper cohort automatically. Physical ordering, fixed observation support, unchanged solver equations and mass-balance checks apply throughout.

## Continue and verify

To extend a geometry campaign, use the same configuration and preparation with a larger complete-population checkpoint budget:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/calibrate_geometry_v1.py run --config path/to/resolved_config.json --output path/to/campaign --evaluation-budget 32
```

This produces a new immutable budget release, without changing earlier snapshots. The managed run is frozen to its configured budget; a direct extension does not rewrite its completed stage receipt. Preserve that run and use the new calibration release explicitly for subsequent applications.

The demonstrated Gardiner run continued 8 → 16 evaluations under its predeclared 16-evaluation configuration. Verify its saved numerical outputs without another model run:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/verify_shared_geometry_v1.py --output outputs/managed_gardiner_shared_geometry_v1_20261019
```
