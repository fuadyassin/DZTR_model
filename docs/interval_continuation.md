# Verified continuation of reviewed interval calibration searches

The new `dztr.calibration.interval_resume` adapter extends a saved interval search without repeating its inherited evaluations. It uses the existing optimizer's complete-generation checkpoint. The [initial calibration workflow](interval_calibration.md) and [interval simulation contract](interval_simulation.md) retain their original behavior.

## Checkpoint and identity checks

`resume_interval_seed(previous_seed_dir, policy_path, output_dir, total_evaluation_budget=256, ...)` validates the prior experiment's artifact hashes, source/configuration identity, numerical-module hashes, software versions, baseline vector, empirical bounds, and saved trial/history tables before new model evaluations. It loads the checkpoint using its original saved identity.

Introducing the continuation adapter adds one numerical-module file. The migration permits that explicitly named addition, records its hash, and requires every previous module to retain its original hash. It does not silently accept a changed solver, objective, optimizer, preprocessing rule, or source record. Later chains preserve the historical module hashes and adapter hash, record the previous identity hash, and generate a new bridge digest for the extension.

The optimizer restores the prior survivor order, random-generator state, duplicate tracking, and archived costs. Only the total evaluation budget grows. The inherited baseline and successful or failed trials are reused. The reporting artifacts retain the old CSV prefixes and append new attempts, and the final complete-generation checkpoint is saved for a later verified extension.

## Count totals and new work separately

The Bull Lake extension moves each seed from 128 to 256 total attempts. This makes 128 new search evaluations per seed, 256 new evaluations across both seeds, and 512 cumulative attempts across their full histories. Reporting replays and held-out evaluations are separate calls outside that search budget.

The [example](../examples/resume_interval_policy.py) continues one existing seed into a fresh directory:

```powershell
python -B examples/resume_interval_policy.py --previous-seed outputs/bull_lake_interval_pilot_20261009/seed_42 --total-evaluations 256 --output outputs/bull_lake_interval_resume_example_42
```

The previous seed's parent directory must contain its immutable verification manifest. The helper performs calibration search only; it computes no validation scores. Use a fresh output directory and matching numerical/source identity.

## Compare prefixes and audit stopping suggestions

Keep the reference `(1, 1)`, normalization area four, stopping tolerances, 64-attempt comparison windows, and 16-attempt check frequency fixed. Extending the hard budget cap is a resource change, not evidence of convergence. Assess the full inherited-and-new stream rather than resetting patience at the checkpoint boundary.

An early provisional stopping suggestion can be invalidated by later windows. Report the earliest suggestion, any invalidation, and the current final status together. Archive hypervolume cannot decrease as feasible trials accumulate, but full-front movement and objective tradeoffs can persist despite a small hypervolume gain.

For cross-stage comparisons, preserve historical pooled candidate IDs and their source seed/local-trial provenance. Select pooled calibration representatives and freeze their vectors before held-out reporting. Held-out behavior cannot change those choices or the stopping policy.

The same held-out period has already been reported in the pilot, so later scores remain descriptive comparisons; they do not create a fresh untouched validation sample. The physical interpretation remains a segmented conditional hindcast with computed net forcing and provisional observed-storage restarts. Conservation holds within clean blocks, and spill-volume changes must accompany the aggregate scores.

See the [continuation protocol](../configs/reviews/bull_lake_interval_continuation.json) and [experiment report](../outputs/bull_lake_interval_continuation_20261009/README.md).
