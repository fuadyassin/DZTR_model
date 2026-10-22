# Gardiner from parameterization through calibration and export

Use [run_manager_gardiner_v2.json](../configs/run_manager_gardiner_v2.json) with
`tools/dztr_run_v2.py`. This configuration completes four operations in order:

1. Estimate the observed monthly targets using the existing Gardiner input and
   its exact frozen mixed-year selection.
2. Run a fixed baseline using those targets.
3. Hand those exact observations, targets, year roles and physical/numerical
   settings into checkpointed calibration. Assess the calibration-selected
   results and export a GRanD-indexed parameter table and profile registry.
4. Replay the selected exported profile through the simulation command.

The [completed Gardiner run](../outputs/managed_gardiner_complete_v2_20261017/workflow_completion.md)
is available for inspection. Its three continued searches reproduce all earlier
verified 512-evaluation trials exactly. To inspect or verify/reuse that completed
run without starting new searches:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/dztr_run_v2.py status --output outputs/managed_gardiner_complete_v2_20261017
python -B tools/dztr_run_v2.py run --config configs/run_manager_gardiner_v2.json --output outputs/managed_gardiner_complete_v2_20261017 --resume
```

The following fresh-directory command starts a new experiment:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/dztr_run_v2.py plan --config configs/run_manager_gardiner_v2.json
python -B tools/dztr_run_v2.py run --config configs/run_manager_gardiner_v2.json --output outputs/my_gardiner_complete_v2
python -B tools/dztr_run_v2.py status --output outputs/my_gardiner_complete_v2
```

Add `--until baseline` to prepare and inspect the inputs before any search, or
`--resume` to verify/reuse completed stages and resume interrupted optimizer jobs.
The v1 commands, configurations, numerical sources and verified results are
preserved. V2 accepts v1 configuration files in a fresh output directory, but
does not migrate a v1 managed-run identity or its checkpoints.

## What crosses the handoff

The calibration stage uses backend `prepared_v1`:

```json
{
  "id": "calibration",
  "type": "calibrate",
  "backend": "prepared_v1",
  "config": {
    "kind": "dztr_prepared_calibration_v1",
    "schema_version": 1,
    "campaign": "@targets.campaign",
    "profiles": [{"reservoir_id": "ex14_Gardiner", "method": "observed_monthly"}],
    "scoring": "pooled",
    "optimization": {
      "seeds": [42, 43, 44],
      "population": 32,
      "evaluation_budget": 512,
      "workers": 3
    }
  }
}
```

The source must be a verified data-availability preparation. The adapter:

- Copies the prepared observations, selection JSON and monthly target table
  byte for byte; it never calls a year selector.
- Retains the exact calendar, warmup/context roles and at least 65% calibration
  coverage established by `YearSelection`.
- Retains capacity, channel capacity, dead storage, initial state, sampling,
  root policy and balance tolerance. It never repeats inflow preprocessing.
- Reproduces the target table from calibration rows and derives the existing
  empirical percentile bounds from those rows only. It verifies the supplied
  starting profile against those bounds without clipping/repairing it.
- Creates a native v2 optimizer preparation with an explicit initializer. The
  adapter and all scientific sources are included in the protocol identity.
  The existing optimizer, solver, objective, checkpoint serialization and
  calibration selection functions are used without modification.
- Rejects source-gated profiles and unsupported methods, timing, signed/interval
  routes, percentile conventions or inconsistent physical definitions.

`pooled` and `equal_regime` scoring remain available. Both retain separate storage
and release objectives. The cross-seed selection rule remains the normalized
distance to the ideal on the pooled nondominated **calibration** fronts. Held-out
scores assess the frozen selection; they do not choose the seed or parameters.

The source availability registry still describes the earlier parameterization
delivery and is not rewritten. Calibration readiness for a specific supported
profile is recorded in the new handoff's `preparation.json`.

## Persistent calibration jobs

Every reservoir/seed is a separate process with its own checkpoint, progress
file, identity, log and immutable budget snapshots. The configured worker count
limits concurrency. There is no progress polling loop in the manager.

The independent adapter command is also available:

```powershell
python -B tools/calibrate_prepared_v1.py run --config inputs/my_handoff.json --output outputs/my_calibration --evaluation-budget 256
python -B tools/calibrate_prepared_v1.py run --config inputs/my_handoff.json --output outputs/my_calibration --evaluation-budget 512
```

Both calls use the same configuration, seeds, population, bounds, observations,
year mask and code. The second continues saved generations. A budget must contain
whole populations and cannot shrink below a saved checkpoint. Existing budget
snapshots and published releases remain immutable. Checkpoints from historical
campaigns have a different protocol identity and are not relabeled as this run.

A completed manager run seals its stage artifacts. Continue checkpoints before
sealing that stage, as when resuming an interrupted job. Advancing an optimizer
externally after its manager stage is complete changes the sealed stage and is
deliberately detected on manager resume. Published parameter registries remain
independently verifiable and usable after a later budget is produced.

## Baseline comparison and final outputs

The calibration release contains:

| File | Purpose |
|---|---|
| `registry.json` | Profiles addressed by reservoir ID or verified GRanD ID |
| `reservoir_parameterization_by_GRAND_ID.csv` | Selected targets, physical constants, source profile and year-mask provenance |
| `baseline_comparison.csv` | Baseline versus calibrated storage/release NSE for every seed and the pooled selected profile, in calibration and assessment |
| `seed_scores.csv` | Per-seed assessments and explicit failures |
| `selection.json` | Calibration-only selection rule and chosen evaluation IDs |
| `profiles/<id>/parameters.json` | Complete model settings, targets link, provenance and assessment status |
| `diagnostics/<id>/simulation.csv` | Continuous selected-profile assessment simulation |
| `baseline_reuse.json` | Whether a verified baseline was reused; no new baseline is simulated during publication |
| `manifest.json` | Scientific source and artifact hashes |

Missing/failed baselines remain explicit; they are never replaced with assumed
scores. Successful fixed baseline simulations are reused to compute pooled and
per-regime reference NSE on the same observation support. A positive improvement
means calibrated NSE minus baseline NSE is positive.

The native release under `releases/budget_<n>` is retained. The adapter creates
an enriched, atomically published release under `published/budget_<n>` and exposes
its registry, parameter table and assessment report to subsequent manager stages.
Source inputs and previously published releases are preserved.

The final `replay` stage uses `@calibration.registry` and, for a single-reservoir
handoff, `@calibration.forcing_csv`. This checks that the exported parameters are
usable through the ordinary native simulation interface. Replaying the original
forcing is a reproducibility check, not a new out-of-sample forecasting test.

## Scope

This delivery supports the observed-monthly native joint storage/release route.
It does not turn flow-only, storage-only, level-derived or ungauged recipes into
joint calibration problems. Those require their own bounds/objectives and
timing checks. Existing signed-forcing and interval studies retain their
separate scientific identities.

Gardiner's three 512-evaluation searches demonstrate the completed workflow.
They do not establish optimizer convergence. Source certification and a real
level-to-storage comparison remain separate work.
