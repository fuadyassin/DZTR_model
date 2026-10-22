# Datum-aware Gardiner level-to-storage pilot

Use the new configured pilot through the gateway. Existing v1–v3 runs and frozen scientific releases are preserved.

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/dztr_run_v4.py plan --config configs/run_manager_gardiner_geometry_v1.json
python -B tools/dztr_run_v4.py run --config configs/run_manager_gardiner_geometry_v1.json --output outputs/my_gardiner_geometry_run
python -B tools/dztr_run_v4.py run --config configs/run_manager_gardiner_geometry_v1.json --output outputs/my_gardiner_geometry_run --resume
python -B tools/dztr_run_v4.py verify-level-storage --output outputs/my_gardiner_geometry_run
```

The pilot uses the archived historical levels rather than repeatedly downloading them. The existing station acquisition stage in `workflows/station_inputs_v1.py` produced the input package in `data/gardiner_levels_v1_20261017/`; raw provider pages, flags, query and retrieval time are preserved there. A completed resume verifies file hashes and performs no new simulations. Earlier workflow and batch configurations continue through this gateway with their existing arguments.

## What datum information does

Gauge height, elevation above a vertical datum, depth at a particular point, and the depth coordinate of an approximate reservoir curve are different quantities. Use

```
geometry_depth = reported_level + datum_conversion_offset - geometry_zero_level
storage = geometry_curve(geometry_depth)
```

For a surveyed curve the geometry zero may be a documented bottom elevation. For a regular-shape approximation it is an effective reference that aligns the curve to the observations; it does not establish the actual bed. Both heights must share a vertical datum and consistent units. `DepthReference` rejects unexplained datum mismatches and records the sign and evidence of explicit conversions.

For Gardiner, ECCC's station metadata identifies PFRA datum. The archived datum page gives −0.313 m to CGVD2013:EPOCH2010, published in 2022 following a 2021 GNSS occupation. Applying the same verified offset to both water elevation and bottom/reference elevation leaves depth unchanged. This metadata alone does not establish the historical datum continuity of a 1980–2011 record. The pilot keeps the levels and references in PFRA coordinates and records the modern conversion for later reconciliation.

Prefer references in this order, according to evidence:

1. A surveyed curve or documented level–depth/bottom reference in the same datum, with any documented conversion and applicable dates.
2. An effective geometry reference inferred from overlapping usable level/storage observations, fitted on calibration years only. This route requires storage observations and is conditional on their definition and timing.
3. A documented operating level with an explicit assumed connection to curve depth or the curve volume at adopted capacity. This is exploratory when the geometry tie or vertical datum is unverified.
4. Calibration maximum aligned to curve top, explicitly exploratory. A short record need not reach full supply; this is not an automatic global default.

Unknown reference or unresolved datum changes remain a source gate. Do not infer a datum or a surveyed bottom merely from the numerical magnitude of a level, dam height, reservoir capacity or full-record maximum. Epoch-specific datum changes require separately documented references for each affected period; the pilot does not invent a historical correction.

## Gardiner experiment

The four predefined reference scenarios are:

| Scenario | Reference rule | Uses storage responses? |
|---|---|---|
| `full_supply_top` | Operator FSL 556.87 m ↔ actual tabulated curve top, depth 65.50 m | No |
| `full_supply_capacity` | Operator FSL ↔ inverse-curve depth at the unchanged model capacity, 9.4 billion m³ | Capacity metadata only |
| `calibration_maximum` | Maximum level in the frozen calibration years ↔ tabulated curve top | No |
| `calibration_storage_alignment` | Median of `level − inverse_curve(provided_storage)` over calibration pairs | Yes |

`calibration_storage_alignment` was declared as the primary exploratory reference before this run. Held-out scores do not select the reference, parameters, seeds or assumptions. The FSL scenarios assume its datum matches the station; the operator page itself does not establish this. The actual top is 65.50 m; the ReGeom header height of 65.55 m is not appended as another curve row.

Rounded flat-volume segments make inverse depth ambiguous at their exact volume. Those pairs are reported and excluded from reference fitting; supported volumes elsewhere remain usable. No extrapolation or volume rescaling is performed.

The existing mixed-year mask is preserved exactly: 7,669 calibration days (65.6143% of all 11,688 days), 1,460 held-out days, 366 warmup days and 2,193 context days. The context designation belongs to the existing frozen selection; it is not additional held-out evidence. The model runs continuously across roles.

Targets use calibration-only monthly P10/P45/P85 with Hazen percentiles. Each geometry scenario passes the existing physical validation before simulation. The observed-storage baseline is reused only after checking its targets, observations, selection, physical/numerical settings and mass balance. Solver equations, forcing, initial states, capacity, channel limit and strict root policy are preserved. Optimizer evaluations: zero.

Daily mean levels transformed through nonlinear geometry are labelled `volume_of_daily_mean_level_proxy`. They are not exact mean volumes or end-of-day stocks. Provider symbols, including estimated and partial-day observations, remain in the raw archive and diagnostic table. The current experiment retains the published values and reports their performance separately.

## Outputs and interpretation

The completed pilot folder contains:

- `README.md` and `storage_diagnostics.png`: interpretation, readiness decision and plots.
- `references.json`: numerical anchors, datums, fitting dependencies and evidence.
- `conversion_metrics.csv`, seasonal/annual/quality diagnostic CSVs: conversion error against the supplied storage reference.
- `reference_sensitivity.csv`: ±0.25/0.5/1 m reference perturbations, labelled as scenarios, not confidence intervals.
- Each reference folder: full conversion table, receipt, monthly targets, physical validation and a persistent simulation receipt.
- `baseline_comparison.csv`: same-support assessment of target recipes using the unchanged DZTR solver.
- `geometry_observations_si.csv`: separately labelled primary geometry-derived storage with the existing inflow/release columns. Original supplied storage remains preserved.
- `plan.json` and `manifest.json`: pinned input, source-code and artifact identities.

Strict conversion scores fail when any estimate is missing on fixed observed support. Explicit `supported_*` columns provide subset diagnostics; they do not disguise missing predictions. Model results always use the same supplied storage and release assessment reference.

The geometry pilot does not promote these profiles into the existing calibrated export. A separate proxy-aware objective/bounds/calibration handoff is still required. A fully ungauged claim also requires independent inflow and initialization, not merely hiding storage/release from target estimation.

Run independent arithmetic verification without another simulation:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/verify_level_storage_pilot_v1.py --output outputs/gardiner_level_storage_pilot_v1_20261017
```

This independently checks raw provider levels, explicit interpolation weights, calibration-only reference estimation, Hazen order statistics, mass balance and preservation of the previous completed paper release.

## Completed Gardiner results

See the [completed report](../outputs/gardiner_level_storage_pilot_v1_20261017/README.md) and [diagnostic figure](../outputs/gardiner_level_storage_pilot_v1_20261017/storage_diagnostics.png).

The calibration-only effective zero is 493.668440 m in PFRA coordinates, yielding an effective geometry-depth range of 53.965–63.232 m over the full record. Held-out conversion NSE is 0.979677, bias −38.74 million m³ and RMSE 113.23 million m³. These are agreement scores against the supplied storage record, not proof of an independently surveyed depth or volume.

The full-supply/top and calibration-maximum references fail physical target validation; neither was repaired or simulated. Two full-supply/top days also exceed the curve's supported depth and remain missing. The capacity-aligned reference has two slightly above-capacity volume estimates, which remain flagged without clipping, although its monthly target table passes validation. The primary calibration-aligned estimates remain inside the adopted capacity throughout.

The valid fixed simulations illustrate a separate issue:

| Monthly target source | Held-out model storage NSE | Held-out model release NSE |
|---|---:|---:|
| Supplied observed storage | 0.895185 | 0.591742 |
| Geometry with capacity-aligned FSL | 0.909487 | 0.566183 |
| Geometry with calibration-fitted reference | 0.865254 | 0.598971 |

Good level-to-volume conversion does not automatically improve the operating model. Keep the fitted reference as an exploratory conversion profile; do not choose a different reference using these held-out simulation scores. The next modeling task is an explicit proxy-aware calibration handoff, with storage timing/approximation uncertainty preserved and suitable independent checks before expansion to other reservoirs.
