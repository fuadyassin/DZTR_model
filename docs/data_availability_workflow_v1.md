# Data-availability preparation and fixed-profile baselines

The first two deliveries in the [integrated plan](data_availability_implementation_plan_v1.md) are implemented in new versioned modules. The workflow inventories all paper reservoirs, prepares geometry-derived storage when the required inputs are supplied, composes five target recipes, validates physical targets, and executes persistent fixed-profile baselines. It does not create optimizer jobs or modify the existing 32 calibrations.

The [completed paper campaign release](../outputs/data_availability_paper_v1_20261017/report/implementation_release.md) records the executed baselines, source dependencies, independent numerical checks and full regression results.

## Paper campaign

The geometry package under [data/geometry_profiles_v1_20261017](../data/geometry_profiles_v1_20261017/README.md) archives only curves corresponding to the cohort's 18 verified GRanD mappings. Each original CSV and SI curve is hash-pinned. Missing IDs are not inferred from names. No level series or datum reference is invented from curve availability.

The [paper configuration](../configs/data_availability_paper_v1.json) reads the frozen 37-row catalog and retains its masks, forcing transformations, physical constants, native sampling and root policies. All benchmark profiles are labeled conditional controlled comparisons because forcing and initialization have not been certified independent of the observations.

From the repository root:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/data_availability_v1.py prepare --config configs/data_availability_paper_v1.json --output outputs/data_availability_new
python -B tools/data_availability_v1.py run-baselines --campaign outputs/data_availability_new
python -B tools/data_availability_v1.py report --campaign outputs/data_availability_new
```

Preparation requires a fresh output directory. To execute one reservoir's eligible baselines, add `--reservoir-id ex14_Gardiner` to `run-baselines`. There is one receipt/lock per profile. Repeating the command verifies completed or failed receipts and reuses them. It does not rerun successful simulations or retry known failures under the same identity. A changed recipe/source/objective requires a new campaign.

Preparation writes the all-reservoir inventory, profile registry, input/code hashes, original and optional geometry-derived observation frames, fixed year roles, target CSVs and physical-validation results. The report includes all requested reservoir/method rows, including unavailable recipes, with blank exported parameter values unless physical validation succeeds. The permanent calibrated parameter export remains unchanged.

`target_estimation_ready` means the required estimators and inputs exist; `parameters_ready` additionally means the existing physical parameter contract passes; `simulation_ready` also requires usable forcing and its numerical interpretation. `calibration_ready` stays false for these new recipe profiles: objective/bound preparation and optimizer job identities belong to the next delivery. Existing calibrations remain complete under their original registries.

## Target recipes

- `observed_monthly`: calibration-only monthly storage and release P10/P45/P85.
- `level_geometry_monthly`: the same estimator using explicitly converted storage plus observed release.
- `flow_only`: fixed storage fractions 0.20/0.30/0.97, with monthly observed release targets.
- `storage_only`: observed or level-derived storage targets plus an explicitly supplied naturalized-release prior. Missing prior is a dependency, not invented data.
- `ungauged_cr1`: fixed storage fractions and annual naturalized-outflow P5/P30/P97. Raw inflow is never silently substituted.

All generators return the existing twelve-by-six SI target contract. Hazen or linear percentiles are explicit. `channel_policy=source_defined` retains the declared model channel target; `fitting_p99` uses P99 of the fitting-period release source and labels it as an empirical proxy. An infeasible target set is reported without expanding capacity, moving physical floors, changing bounds or switching root policies.

The default paper configuration has no supplied level series/reference pairs or independently identified naturalized-outflow series. Those recipes therefore report dependencies. This is different from claiming those observations do not exist elsewhere or that all geometry-supported reservoirs are already calibration-ready.

## Supplying levels and naturalized outflow

Add an entry under `supplemental_inputs` for an existing reservoir. CSV paths are relative to the configuration file. Example shape, using illustrative references rather than recommended real-world values:

```json
{
  "ex14_Gardiner": {
    "levels_csv": "inputs/gardiner_levels.csv",
    "level_time_basis": "end_of_day_level",
    "reference": {
      "level_m": 100.0,
      "depth_m": 50.0,
      "basis": "source_reference",
      "evidence": "Replace with the documented correspondence between gauge and geometry coordinates"
    },
    "naturalized_outflow": {
      "csv": "inputs/gardiner_naturalized_outflow.csv",
      "source_kind": "simulated_naturalized_outflow",
      "evidence": "Identify the natural-lake/hydrologic simulation and its forcing",
      "independent_of_target_responses": true
    }
  }
}
```

Level CSV columns are `date,level_m`; naturalized-flow CSV columns are `date,naturalized_outflow_m3s`. Naturalized flow must use the original full input calendar. Levels are reindexed to that calendar without filling gaps. Do not assign a reference just to improve model performance.

For an explicitly exploratory reference inferred only from calibration levels, omit `reference` and set `allow_calibration_maximum=true`. The adapter freezes the calibration maximum against the actual supported curve top. Held-out higher levels are flagged out of range; they cannot move the reference or earlier storage estimates. This assumption is not a surveyed datum.

Daily/monthly mean levels need `allow_mean_level_proxy=true`. The resulting series is labeled a transformation-of-mean-level proxy. No daily mean storage or end-of-day storage is fabricated. Exact interval-mean observation operators remain unsupported by this native baseline interface.

To provide a curve directly, use `geometry_csv`, `geometry_format` (`regeom` or `surveyed_si`) and `geometry_evidence`. ReGeom also needs a known `geometry_grand_id` if the reservoir has no verified cohort mapping. Declaring a geometry link does not assign a new paper GRanD ID. Surveyed SI CSV columns must be `depth_m,storage_m3` and optionally `area_m2`, in that order.

Conversion uses `depth = reference_depth + level - reference_level`. Depths are monotone and unique; storage/area are finite, nonnegative and monotone. The curve is loaded once and applied to the batch. There is no implicit extrapolation, appended flat endpoint or capacity rescaling. Missing/negative height metadata stays missing; a valid independent reference can still use the tabulated coordinate. Area/volume closure and curve-top capacity differences are diagnostics, not silently repaired geometry.

## Additional reservoirs without reservoir-specific code

A standalone configuration uses `reservoirs` instead of `catalog`:

```json
{
  "kind": "dztr_data_availability_v1",
  "schema_version": 1,
  "methods": ["observed_monthly", "flow_only", "storage_only", "ungauged_cr1"],
  "reservoirs": [{
    "reservoir_id": "my_reservoir",
    "GRAND_ID": null,
    "name": "My reservoir",
    "observations_csv": "inputs/observations_si.csv",
    "model_config": "inputs/model.json",
    "forcing_stage": "prepared",
    "source_gates": []
  }]
}
```

The observations use `date` and available SI columns `inflow_m3s,storage_m3,release_m3s`. Missing responses stay missing. Model JSON declares the existing world-sample `model` physical fields, `sampling`, `root_policy`, `atol_storage_m3` and optional `inflow_preprocessing`. Explicit signed forcing needs a `signed_net_native_v1` route and `forcing_policy`; a root-selection policy alone cannot establish the meaning of negative inflow.

Without `selection_json`, the existing mixed representative-year selector derives and freezes masks from declared adjusted forcing. It requires its existing minimum record length and at least 65% calibration coverage. With `selection_json`, supply an existing `YearSelection` object serialized as JSON on the exact calendar. Source gates are retained across all recipes and prevent promotion to usable parameters.

## Assessment and reproducibility

Fixed-profile runs preserve continuous native forcing and state through warmup/context years up to the last scored date. Only trailing unscored context is omitted and its length is recorded. There are no resets from observed storage. The existing solver's independent mass-balance/storage checks remain in force.

Scored support depends only on observations and frozen phase roles. A nonfinite/negative simulated response on that support fails assessment instead of dropping the day. Valid zero release is retained. Scores include raw NSE, PBIAS, RMSE and square-root release NSE, with undefined values left blank.

Where supplied storage exists, it remains the common assessment reference for every recipe, including geometry-derived target recipes. Geometry proxy scores are explicitly labeled where no supplied storage exists; they are not compared numerically with archived independent-storage NSE. Original supplied storage is preserved separately.

Report differences are **new fixed baseline minus archived calibrated profile** on common native support. They are not results of new optimization or proof of a fully ungauged application. Method medians use completed subsets and cannot rank methods without accounting for failures and common cases.

The test suite checks batch conversion, supported domains, units, placeholders, fixed references, held-out independence, reproduction of existing quantiles, missing response/prior handling, fixed scored support, shared assessment references, persistent job reuse and hash tampering. Run focused tests during changes and the complete regression suite at the end of a substantial delivery.
