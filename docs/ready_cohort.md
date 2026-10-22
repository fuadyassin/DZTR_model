# Work on eligible reservoirs while source questions remain open

The [2026-10-08 cohort experiment](../outputs/ready_cohort_20261008/README.md) defers Ghost and Trinity and preserves Tuyen's exclusion. It screens the remaining records and runs independent, checkpointed pilot calibrations for Gardiner, Oahe and High Aswan. Nine samples pass the exploratory calibration gates. Their source assumptions are explicit; eligibility does not certify storage curves, measurement clocks or physical constants.

For a new batch, retain a hash-pinned parent collection and write a separate worklist with selected IDs and per-reservoir reasons. Use calibration data to check forcing, target adequacy, bounds and strict baseline feasibility. Report held-out data issues independently so that validation performance cannot tune the worklist or operating parameters.

Choose a bounded job matrix of reservoir IDs, seeds, population sizes and evaluation budgets before running it. The saved pilot uses three reservoir IDs, two seeds, population 16 and 128 evaluations per job. Each job preserves native daily/monthly sampling and its own fixed constants and periods; targets and bounds use calibration records only. The [artifact runner](../outputs/ready_cohort_20261008/run_pilots.py) demonstrates a flat process pool, isolated per-job outputs, atomic generation checkpoints and selection before validation. It intentionally refuses to overwrite existing results. No production scheduler or new default budget is installed by this experiment.

For an eligible individual or selected group, the existing collection command accepts reservoir IDs and a fixed budget, for example in a fresh output directory:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -B -m dztr.integrations.collection --manifest data/reservoirs/manifest.json --reservoir ex5_Garrison --output outputs/garrison_next --optimize 128 --seed 42
```

The collection command checks its pinned inputs but does not enforce the separate cohort review automatically. Review the current worklist first. It exports the whole calibration Pareto front for held-out reporting; the cohort artifact runner additionally saves three calibration-selected representative roles and checkpoint identities. Existing results and metadata stay separate from a new experiment.

Budget convergence requires longer archives and additional seeds. Continue only complete-generation checkpoints whose code, inputs, model settings, bounds, baseline vector and optimizer variation settings still match. Preserve the parent identity and distinguish a budget extension from a new experiment. The optimizer API supports an increased evaluation budget with unchanged search settings; archived CSVs alone do not reconstruct the RNG and population state. The pilot's 128 evaluations establish neither a final fleet budget nor adequacy of its 72 monthly parameters.
