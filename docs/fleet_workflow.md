# Fleet calibration and reporting

The fleet controller replaces separate reservoir-by-reservoir preparation. The prepared worklist in `outputs/fleet_workflow_20261010` covers all 41 active reservoirs. Tuyen remains excluded.

Nine reservoirs pass the inherited calibration-only exploratory gates: Garrison, Oahe, Amistad, Gardiner, Bhumibol, High Aswan, Sirikit, Oroville and Nurek. Each receives population 32, a 512-search-evaluation cap and seeds 42, 43 and 44. This schedules 27 jobs and at most 13,824 search evaluations. The inherited runner also makes at most eight additional baseline/reporting calls per job, including held-out reporting; these are separate from the search budget. This budget is provisional and does not establish convergence.

Bull Lake's three verified population-32 archives are reused under its separate computed-net, segmented interval policy. The other 31 reservoirs retain explicit deferral reasons in `reservoir_status.csv`. Failed source or parameter gates are never silently bypassed. Historical held-out scores do not determine admission or parameter selection.

After execution is authorized, run from the repository with its Python environment:

```powershell
python -B tools/fleet_pipeline.py --run-authorized
```

This command runs three independent processes at a time and then produces common reports for completed jobs. Each job retains its native generation checkpoints, trials, vectors, selected parameters, calibration and validation outputs, and a process log. An individual failure does not cancel the other jobs; successful results are still reported. Input, numerical-code and runner hashes are checked before execution. Existing job directories are never overwritten. The controller preserves incomplete checkpoints but does not automatically resume them; continuation must respect their recorded identity.

Reports include observed and simulated storage/release, monthly targets, all seed Pareto fronts and calibration/validation metrics. The lowest seed provides a fixed illustrative trajectory; held-out fit is never used to choose it. Saved-artifact reporting makes zero model calls. Existing reports for Gardiner, Oahe, High Aswan and Bull Lake are under `reports/archived_reviewed` in the prepared output directory.

Numerical conservation checks do not certify physical source definitions. Monthly models retain the inherited 30.5-day clock. High Aswan's inherited channel capacity remains provisional. Bull Lake's population-32 archives have no held-out scores. These limitations are retained in each applicable report. The numerical model and old experiment archives remain unchanged.

The current worklist is prepared, not executed: the earlier instruction to avoid further evaluations remains in effect until the pending execution question is answered. Reports and preparation are already complete without new model evaluations.
