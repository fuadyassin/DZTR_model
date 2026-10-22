# Twenty-reservoir strategy comparison

The Garrison/Oahe pilot is preserved in `outputs/strategy_comparison_pilot_20261012`.
The remaining eighteen reservoirs use `outputs/strategy_comparison_fleet_v2_20261012`.
The earlier fleet directory is a parked preparation/search draft, not the final comparison.

Each reservoir has three strategies, seeds 42/43/44, population 32, and a 512-evaluation budget per search:

1. Earliest whole calibration years, pooled storage/release NSE.
2. Forcing-stratified mixed whole years, pooled storage/release NSE.
3. The same mixed years, equal-weight dry/normal/wet calibration NSE.

Calibration covers at least 65% of all recorded rows and available storage/release pairs.
The number of whole calibration years is matched between strategies. Daily leap-year row counts
can differ slightly. Initial warmup and other unscored years retain continuous forcing/state;
observations in these years do not reset simulated storage.

Common assessment years are chosen before optimization, using inflow features only, from years
after the chronological calibration period. Reserve up to four such years, retaining at least
one later training opportunity when more than one later year exists. Favor representation of
available dry, normal and wet years, then additional dry/normal/wet coverage. Dry years take
priority if the short later record cannot supply all three regimes. Missing regimes and the
assessment sample counts are explicit in `design.json` and each reservoir report.

This adaptation avoids identical chronology/mixed masks for seventeen of the eighteen new
reservoirs. Dickson retains identical masks under the 65% minimum and common holdout; it
cannot independently test year-selection effects under this design. Its weighting comparison
is still retained. Small assessment sets and missing regimes limit interpretation. Scores are
retrospective on previously inspected records, not blind prospective validation.

Targets, percentile bounds and optimizer initialization are estimated exclusively from each
strategy's calibration years. Thus year-selection effects include refitting these quantities.
The weighting contrast holds dates, bounds and initial populations fixed. An invalid raw
generalized estimate is distinguished from a feasible optimizer initialization, which is
never labeled a corrected generalized baseline. Physical constants and numerical equations
remain unchanged.

Three calibration fronts are pooled to choose a compromise by the established normalized
distance-to-ideal rule. Assessment scores never select seeds or parameters. All three
seed-balanced profiles are assessed as well. The previous calibrated registry remains intact;
comparison scores do not automatically promote replacement profiles.

The revised preparation reuses 24 completed draft searches (12,288 evaluations). The adapter
requires byte-identical prepared inputs, equal objective identities, identical runtime versions,
equivalent worker code except for output location, and exactly preserved checkpoint vectors
and costs. `checkpoint_reuse.json` records the old and new checkpoint hashes. The original
checkpoints and logs remain available.

The persistent runner executes searches, freezes parameter selection, evaluates common years,
creates reports/figures for all twenty including the reused pilot, and runs the complete
regression suite. It writes stage/failure information to `workflow_status.json`, job-specific
logs/checkpoints, a test receipt, and a final verification manifest. It does not poll or interpret
incomplete optimization scores.

To inspect progress, read `outputs/strategy_comparison_fleet_v2_20261012/workflow_status.json`.
After checking that no workflow is already running, resume an interrupted workflow with:

```powershell
& 'C:\Users\YassinF\AppData\Local\anaconda3\python.exe' -B tools\finish_comparison_fleet_v2.py
```

An exclusive `workflow.lock` prevents duplicate finishers. If a process was forcibly terminated,
verify its recorded PID is no longer active before removing the stale lock. Completed searches
are reused; unfinished searches continue from their checkpoints. A failed assessment is recorded
without inventing a score. A regression failure leaves the workflow failed for investigation.

Expected final entry points:

- `README.md`: overall outcome counts and twenty-reservoir score table.
- `reservoir_reports/<reservoir_id>/README.md`: strategy scores, changes, seed ranges,
  assessment years, missing regimes, and comparison figures.
- `reservoir_outcomes_by_GRAND_ID.csv`: comparable strategy scores and outcomes keyed by
  GRAND_ID where verified; Fort Randall remains null.
- `all_twenty_scores.csv`: every assessed calibration-selected profile and initialization/reference.
- `all_twenty_regime_diagnostics.csv`: NSE, RMSE, bias and observed storage variability.
- `all_twenty_seed_agreement.csv`: optimizer seed medians/ranges, not confidence intervals.

512 evaluations is a matched comparison budget, not a convergence certificate. Historical
profiles use different training periods/budgets and are descriptive references, not matched arms.
