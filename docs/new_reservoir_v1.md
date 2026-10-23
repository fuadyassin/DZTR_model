# Add a reservoir to the workflow

Start with `configs/portable/gardiner_workflow.json`, give the reservoir a stable
internal ID, and retain a null GRAND ID until the identity is verified. One
workflow can contain multiple reservoir entries; existing worker settings limit
independent calibration jobs.

## Prepare usable inputs

1. Declare capacity, dead-storage definition, channel capacity, units and their
   evidence. Mean river discharge is not engineering channel capacity.
2. Supply consecutive daily or month-start SI records. Keep missing responses
   missing; complete forcing is required for a continuous simulation block.
3. Establish the boundary: sum independent inflow/outlet components only where
   appropriate. Account for diversions and travel time explicitly. Serial river
   gauges are not independent inflow components. Levels are never summed.
4. For levels, supply geometry and a gauge-to-geometry reference. Where the
   reference is fitted, use calibration data only and freeze it before assessment.
   Carry mean-level timing and approximation uncertainty into the profile.
5. Mark unresolved source decisions in `source_gates`. They prevent promotion
   to an admitted application/calibration profile.

## Select a parameter source

| Usable information | Method | Main assumption |
|---|---|---|
| Storage and release | `observed_monthly` | Monthly calibration-response percentiles |
| Prepared level-derived storage and release | `level_geometry_monthly` | Declared reference and geometry uncertainty |
| Release only | `flow_only` | Fixed storage fractions; release-only fitting |
| Storage only | `storage_only` | Storage fitting with an explicit release prior |
| Neither response, simulated naturalized flow supplied | `ungauged_cr1` | Capacity fractions and naturalized-flow percentiles |
| Neither response, mean-flow/donor fallback supplied | `ungauged_mean_flow` | Regional annual percentile-pattern transfer |

Every route still needs suitable forcing and physical inputs. The mean-flow
fallback does not generate a naturalized time series. A missing response is
not replaced by a fabricated observation. Existing prepared-calibration adapters
have method-specific bounds/objectives; changing a method string alone cannot
turn one calibration problem into another.

Use [four-route guidance](four_route_workflow_v1.md), [flow-only details](flow_only_workflow_v1.md),
[storage-only details](storage_only_workflow_v1.md) and [geometry guidance](shared_geometry_workflow_v1.md)
for the corresponding stage configurations and prior requirements. The portable
source bundle includes the backends; private donor files are intentionally not
distributed. Supply an explicit local prior/donor source when that route needs it.

## Fit, select and apply

Choose mixed years covering at least 65% usable calibration data. Decide the
application priority before examining held-out results: storage, release or
their weighted compromise. Keep numerical policies and bounds explicit. Use
checkpoint continuation to assess progress; do not call a preset budget converged.

After calibration, freeze the selected target table and settings. Export from
the verified native registry through the stable command:

```powershell
python -B -m dztr export-profile --registry outputs/my_calibration/registry.json --reservoir-id my_reservoir --preprocessing-json inputs/preprocessing.json --method observed_monthly --output outputs/my_portable_profile.json
```

Use the actual registry emitted by the calibration stage. The preprocessing
JSON must declare exactly `factor` and `offset_m3s`; the exporter does not guess
whether the source data was transformed. A verified `--grand-id` can replace
`--reservoir-id`. Export reads the registry and target table, not observations,
and verifies the original source/artifact identities before writing the profile.

The portable profile format embeds the settings and provenance; application
does not need the original archive paths. `tools/prepare_researcher_examples_v1.py`
documents the separate verified three-public-example release. Exporting a new
solver route requires its own validation; the portable endpoint adapter rejects
unsupported interval profiles.

For subsequent application, pass inflow and declared initial conditions through
`dztr simulate` or `dztr.portable.run_profile`. The Python function accepts an
inflow-only DataFrame and returns the existing `SimulationResult` plus independent
checks. This is a time-series coupling interface, not an incremental routing
network or a new hydrological inflow model.
