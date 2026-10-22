# Run the completed reservoir cohort through one batch gateway

The [paper batch configuration](../configs/run_manager_paper_batch_v1.json)
includes every one of the 37 paper reservoirs. It imports 29 standard native
profiles and three exploratory signed-inflow profiles. Bull Lake's conditional
interval study and four source-gated cases remain explicit inventory rows.
Travers is outside this paper cohort; Tuyen remains excluded.

This operation reuses completed calibration results. It does not start searches,
extend budgets, generate new simulations or choose new parameters.

The [completed paper batch](../outputs/managed_paper_batch_v1_20261017_final/release/README.md)
and [all-reservoir parameter table](../outputs/managed_paper_batch_v1_20261017_final/release/reservoir_parameterization_by_GRAND_ID.csv)
are available now. Inspect the completed run with:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/dztr_run_v3.py status --output outputs/managed_paper_batch_v1_20261017_final
```

To create a fresh import:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/dztr_run_v3.py plan --config configs/run_manager_paper_batch_v1.json
python -B tools/dztr_run_v3.py run --config configs/run_manager_paper_batch_v1.json --output outputs/my_verified_paper_batch
python -B tools/dztr_run_v3.py status --output outputs/my_verified_paper_batch
```

For a completed or interrupted batch, use the same command with `--resume`.
Imported files and original source/checkpoint hashes are verified before reuse.
`--until ex14_Gardiner` stops after the named configured reservoir; resume then
completes the remaining entries.

Resume writes one final status update after verifying existing receipts.
Transient Windows reader/scanner locks on atomic status replacement are retried
at most five times. Persistent errors are reported, and the manager lock is
released even if status cannot be saved. This behavior does not change numerical
code, scientific inputs, checkpoint identities or imported profiles.

`tools/dztr_run_v3.py` also accepts the existing v1/v2 workflow configurations and
delegates them to the unchanged v2 manager. The older runners, numerical modules,
checkpoints and verified releases are preserved.

## What the manifest controls

Each reservoir entry declares:

- Reservoir ID, verified GRanD ID or null, numerical route and import/defer action.
- Prepared observations, frozen year selection, model configuration, empirical
  bounds and initial target table, with exact SHA256 digests.
- Expected physical constants, initial conditions, sampling, root policy,
  mass-balance tolerance, forcing interpretation and preprocessing.
- The observation-based monthly calibration method and already-prepared input
  stage. Transformations are not applied again during import.

The current protocol retains the established mixed-year, pooled-NSE, Hazen
formulation, at least 65% calibration coverage, seeds 42/43/44, population 32 and
512 evaluations per seed. These are compatibility requirements for these archived
experiments, not universal stopping recommendations.

To create a fresh manifest from another compatible verified cohort or select an
explicit subset:

```powershell
python -B tools/dztr_run_v3.py init-batch --catalog outputs/calibrated_paper_cohort_v1_20261017_final/catalog.json --assessment outputs/calibrated_paper_cohort_v1_20261017_final --reservoir-id ex14_Gardiner --config configs/my_batch.json
```

Omit `--reservoir-id` to retain the full paper inventory. Paths in these manifests
are explicitly relative to the repository root. Generating a manifest never
certifies changed or corrupt source files: the original catalog, release
manifests, assessment evidence and consumed source/checkpoint hashes are checked
first.

## Compatibility and identity checks

Before importing each profile, the workflow verifies:

1. The requested observations, year-mask file, model, bounds and initializer
   match the archived experiment, including declared file hashes.
2. The model agrees with the selected profile and the checkpoint's objective,
   year-selection signature, numerical settings and signed-forcing policy.
3. Every original checkpoint is complete. Its evaluated vectors/costs match
   the stored trial tables; its lower/upper bounds and initial vector match the
   prepared bound and target files.
4. The selected checkpoint vector reconstructs the selected monthly target
   table, and the frozen choice reproduces the calibration-only pooled-front
   selection rule. Assessment scores never choose a seed or replace parameters.

The imported parameter JSON and selected target CSV are copied byte for byte.
Original seed-identity JSON files are retained exactly. The original checkpoint
NPZ files are referenced by path and hash, rather than copied or given a new
experiment identity. Optimizer settings and original source kinds remain in the
import receipts.

A different requested input or model is marked **incompatible**, remains in the
inventory and receives no imported parameter values. It requires an explicitly
prepared new calibration. Corrupt original archives or imported files stop the
batch rather than being treated as reusable evidence.

Source-gated cases cannot be promoted by changing the manifest's action. Existing
exploratory assumptions remain assumptions after import.

## Consolidated outputs

The new run contains:

| Output | Contents |
|---|---|
| `plan.json`, `status.json` | Frozen batch decisions and per-reservoir progress/results |
| `jobs/<id>/receipt.json` | Compatibility evidence, original experiment identities and copied-file hashes |
| `release/README.md` | Cohort counts, outcomes, source limitations and output links |
| `release/reservoir_parameterization_by_GRAND_ID.csv` | Every configured reservoir; available parameters plus original assumptions and import status |
| `release/registry.json` | Imported profiles indexed by reservoir ID and verified GRanD ID |
| `release/catalog.json` | Ready/deferred inventory and standard/signed simulation dispatch |
| `release/report/reservoir_status.csv` | Per-reservoir readiness, route, original status and next action |
| `release/report/baseline_comparison_all_seeds_and_selected.csv` | Calibration/held-out scores and baseline gains for all seeds and the frozen selected profile |
| `release/report/*diagnostics.csv` | Cached checkpoint, seasonal, zone and regime diagnostics |
| `release/manifest.json` | Immutable imported-release files and scientific source hashes |

Missing GRanD IDs never remove a reservoir. Deferred/incompatible monthly
parameter values stay blank. Bull Lake retains its separate-study reference;
Bennett retains its documented 1.45 source inflow multiplier and unresolved
storage-definition gate. Yellowtail and Nurek retain unavailable baseline scores.

The baseline can be a generalized profile or a documented feasible initializer.
Positive NSE and a successful import do not establish suitability, independent
forcing, source certification or optimizer convergence.

## Use an imported profile

```powershell
python -B tools/dztr_run_v3.py simulate --catalog outputs/my_verified_paper_batch/release/catalog.json --grand-id 253 --output outputs/my_gardiner_application
python -B tools/dztr_run_v3.py simulate --catalog outputs/my_verified_paper_batch/release/catalog.json --reservoir-id ex8_Flaming_gorge --output outputs/my_flaming_gorge_application
```

These explicit simulation commands run a model; batch import itself does not.
The catalog dispatches the three signed cases through their original signed-net
solver and retains their explicit exploratory net-boundary interpretation.
Archived prepared forcing is the default. A supplied forcing CSV requires
`--forcing-stage canonical` or `prepared`; a different start date requires an
explicit SI `--initial-state` JSON. Observed storage/release do not initialize or
reset the simulation.

Independent read-only verification of a completed import is available through:

```powershell
python -B tools/verify_batch_import_v1.py --run outputs/my_verified_paper_batch
```

The verifier compares copied parameters, identities, physical settings, all
GRanD mappings, cached reports and baseline-gain arithmetic with their original
sources. It runs no model or optimizer.
