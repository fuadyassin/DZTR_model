# Calibration pilots under reviewed daily interval policies

The [interval solver and objective](interval_simulation.md) separate reported daily mean volumes from modeled endpoint releases. The new `dztr.calibration.interval_search` module adds a calibration-only search adapter. It uses the existing NumPy NSGA-II optimizer and convergence diagnostics; it does not reproduce the historical MATLAB AMALGAM search.

## Inputs and source policy

`load_interval_problem(policy_path)` returns `(problem, baseline_vector, validation_period, context)`. The context supplies the calibration fit, sample support, bounds, prepared masks, configuration, declared periods, and source hashes. The loader requires explicit SI units, nominal daily mean-end labels, an observed calendar-month clock, matching canonical data hashes, supported preprocessing, and a reviewed source policy. Computed net inflow requires unit scaling, no additional fitted loss, and the storage-only flood rule.

Storage targets use clean calibration storage rows; release targets use eligible calibration interval means. The input policy, reservoir physics, quarantine dates, and nominal observed-storage initialization stay fixed. Only the existing 72 monthly target coordinates are optimized. Strict multiple-root rejection remains the solver policy.

## One seed, counted attempts

`run_interval_seed(policy_path, output_dir, settings=..., checkpoints=..., stopping_policy=...)` searches calibration only. The first trial is the generalized baseline. Failed candidates count against the same evaluation budget as successful candidates; duplicate proposals do not cause a model evaluation. The output directory must be fresh.

The output includes every attempted vector, feasibility and raw dated rejection reason, the calibration Pareto front, generation history, nested-prefix checkpoint metrics, sample support, targets, bounds, source/code/software identity, and a resumable optimizer checkpoint at the last complete generation. The checkpoint preserves survivor order and random-generator state. A partial trailing generation remains visible in trial CSVs but cannot replace the last complete checkpoint.

The [example](../examples/calibrate_interval_policy.py) runs one calibration seed:

```powershell
python -B examples/calibrate_interval_policy.py --seed 42 --evaluations 128 --output outputs/bull_lake_interval_search_42
```

Use a fresh output directory for each seed. This command computes no validation scores. Source and numerical-module hashes are checked before the completed search is written.

## Progress and stopping interpretation

Prefix metrics reuse the saved trial stream instead of repeating smaller searches. Hypervolume uses a fixed negative-NSE reference `(1, 1)` and the fixed theoretical area four for normalization. Objective extrema, full-front movement, reference coverage, new feasible attempts, and rejected counts accompany that indicator. A balanced representative can change as front ranges change; its scores alone do not establish stagnation.

Stopping assessment is diagnostic in this pilot. It does not shorten the fixed budget. The Bull Lake protocol uses population 16, comparisons every 16 attempts across a 64-attempt window, a minimum of 96 attempts, and patience two. The earliest provisional candidate is 112 attempts. The windows overlap, so successive stable checks are not independent confirmations. Small recent objective changes do not demonstrate global convergence or parameter identification.

Two seeds permit descriptive pairwise front comparisons. Both share the same generalized baseline, so apparent agreement can simply retain that common point. Formal cross-seed agreement in the existing budget-review workflow requires at least three seeds. Bounds-normalized parameter spreads describe searched-front variability, not confidence limits.

## Selection and held-out reporting

Pool feasible calibration trials using distinct IDs that retain their source seed and local evaluation ID. Select best release, best storage, and balanced representatives from the pooled nondominated archive using the existing deterministic selection rule. Keep the generalized baseline as a separate comparison even when dominated. Write selected identities and vectors, and hash them, before the first validation evaluation.

Only then call `problem.evaluate_validation(vector, validation_period)`. A held-out failure remains a failure with its reason and partial diagnostic trajectory; do not choose a replacement using validation scores. Selected calibration replays must reproduce archived costs, and feasible validation runs must reproduce the calibration prefix exactly.

The validation label remains **segmented conditional hindcast** because clean blocks use provisional observed-storage anchors. Water conservation applies within blocks. Neither optimization nor quarantine establishes the physical source clock, repairs an unknown source anomaly, or turns computed net forcing into an independent inflow measurement.

The [Bull Lake experiment configuration](../configs/reviews/bull_lake_interval_pilot.json) declares both seeds, fixed budgets, checkpoints, tolerances, and selection policy. Its [report](../outputs/bull_lake_interval_pilot_20261009/README.md) records the results and the next experiment justified by calibration progress.
