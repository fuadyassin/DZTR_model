# Workflow extensions: explicit initializers and short records

Use `tools/reservoir_workflow_v2.py` with schema-version-2 configurations for these
extensions. The original V1 runner, twenty-reservoir release, numerical equations,
Hazen percentiles, bounds and checkpoints remain unchanged. V2 reads verified V1
profile registries; resume each optimizer preparation with its original runner.
Checkpoint identities are never migrated silently.

## Explicit calibration initializer

`configs/workflow_tyuyamuyun_v2.json` is a complete example. A reservoir entry can
declare `initializer` with exactly five fields: `csv`, `sha256`, `basis`,
`selection_sha256` and `provenance`. Paths are relative to the configuration file.
The basis must be `calibration_only`. The CSV has months 1–12 and the six native
SI target columns shown in the example initializer. Its content hash and selected
calibration-year signature must match. Physical ordering and the unchanged
calibration-derived empirical bounds must also pass; the runner does not clip,
reorder targets or widen bounds to accept the file.

The accepted file is copied into the prepared inputs and used as the actual
optimizer baseline. The release retains its provenance. This enables Tyuyamuyun's
previously verified sufficient-gap starting point with the declared
`nearest_previous_storage` root policy. That policy is a continuity heuristic;
it does not establish the historical operator's intent.

```powershell
python -B tools/reservoir_workflow_v2.py run --config configs/workflow_tyuyamuyun_v2.json --output outputs/tyuyamuyun_calibration_v2_20261013
```

This run has completed. Repeating it at the same budget reuses its snapshots.
A later authorized continuation uses the same command with
`--evaluation-budget 1024`, preserving populations, seeds, bounds and RNG state.
The 512-evaluation release remains immutable; continuation produces a new release.

## Six-year short-record protocol

Set `selection_protocol` to `short_record_v1` for exactly six complete native daily
or monthly calendar years. The first year is warmup. Annual mean inflow classifies
the other five years into dry, normal and wet regimes. Mixed selection chooses the
latest whole assessment year that leaves all three calibration regimes and at
least 65% of all rows and available storage/release pairs for calibration. The
choice uses forcing and observation availability, not observed fit. Continuous
forcing and simulated state pass through every year, including unscored years.

For Mcgregor this prepares calibration years **2003, 2004, 2005 and 2007**,
assessment **2006**, and warmup **2002**: **66.68%** calibration coverage.
Holding out 2007 would remove the only normal year from calibration, so a
chronological selection fails this protocol. One dry assessment year supplies no
wet/normal assessment and does not establish prospective forecast skill.

Post-selection diagnostics flag constant observation years without changing the
selection. Such records require source review before optimization. Equal-regime
NSE also fails preparation if any calibration regime has undefined NSE; it must
not silently drop a regime or invent a finite score.

Mcgregor is currently blocked: all 365 storage observations in 2005 equal
326.270712 million m³. The original hash-matched Excel workbook contains these
values, and its MATLAB writer copies column 4 with a million-to-m³ conversion.
The flat year is upstream of the Python port. Until its meaning is established,
the prepared short-record plan is retained and no production optimization runs.
The dry-storage regime NSE is undefined because its calibration observations
have zero variance.

## Results and programme tracking

[Verification and comparison report](../outputs/workflow_extensions_verification_20261013/README.md)
links the new Tyuyamuyun release, all three seeds, checkpoints, source evidence,
combined parameter table and updated status of all 41 active reservoirs.
The two extension records retain null GRAND IDs pending verified source identity.
All calibrated profiles remain exploratory; completion at 512 evaluations does
not establish convergence or resolve every source interpretation.
