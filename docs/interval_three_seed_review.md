# Three-seed review of interval calibration budgets

The [Bull Lake protocol](../configs/reviews/bull_lake_interval_three_seed.json) extends two verified 256-attempt histories to 512 attempts and starts seed 44 at the same 512-attempt budget. The [experiment report](../outputs/bull_lake_interval_three_seed_20261009/README.md) retains the results. This is a controlled calibration budget experiment under the existing reviewed daily input policy.

## Count cumulative attempts and new evaluations separately

Seeds 42 and 43 each reuse 256 archived trials and make 256 new evaluations. Seed 44 makes 512 new evaluations, including its generalized starting trial. The full experiment therefore contains 1,536 cumulative attempts and 1,024 new search evaluations. Failed evaluations consume budget; duplicate proposals within a seed do not call the model. Reporting replays and held-out evaluations are separate work.

All three searches retain population 16, the same 72 monthly coordinates, calibration-derived Hazen bounds, source masks, physical settings, strict root rejection, nominal observed-storage anchors, and software versions. Distinct random seeds change the random stream while retaining these shared assumptions and the generalized initial candidate.

Historical pooled IDs 0–511 preserve their previous seed and local-trial origins. New seed-42 trials occupy IDs 512–767, new seed-43 trials 768–1023, and seed-44 trials 1024–1535. Seed/local trial identity is the primary provenance; pooled IDs remain stable across reports.

## Apply the complete stopping and extension history

The [review script](../outputs/bull_lake_interval_three_seed_20261009/compare_budgets.py) adapts the existing `calibration/budget_review.py` decision to the verified interval artifacts. It does not create generic-search manifests. It consumes calibration archives only.

The predeclared stopping-candidate decision requires all of the following:

- At least three distinct seeds with equal complete budgets and matching mathematical, numerical, source and nonseed settings.
- A current stopping candidate in every seed under the existing 16-attempt checks, 64-attempt window, minimum 96 attempts, patience two, normalized hypervolume gain below 0.0025, front movement at most 0.01 NSE units, and extrema drift at most 0.005 NSE units.
- Full reference coverage and every pair's symmetric full-front Hausdorff distance at most 0.01 in raw negative-NSE coordinates.
- No later invalidation of an earliest stopping suggestion, and an earliest-suggestion-to-final extension that still satisfies the same progress tolerances.
- At least one nonbaseline feasible trial strictly dominating the shared baseline in every seed, with the baseline absent from all final fronts.

Hypervolume retains reference `(1, 1)` and theoretical normalization area four. The range of final hypervolume across seeds is descriptive; it is not a new stopping threshold. The stability windows overlap and are not independent confirmations.

Do not restart patience at the continuation boundary or discard an invalidated early suggestion. Seed 42's earlier suggestion at 160 was already invalidated in the previous stage. That fact remains part of its extension audit even if later windows become stable. A new stable epoch would need a separately declared and independently checked review rule.

If all gates pass, the label is `stopping_candidate_requires_independent_audit`. Otherwise the completed 512-attempt experiment is `hard_cap_reached_review_required`. Neither outcome establishes a global optimum, identified parameters, adequate predictive fit, or a production evaluation budget. The protocol makes no automatic extension beyond 512 attempts.

## Freeze calibration selection before reporting

Pool the complete feasible calibration fronts and use the preserved deterministic best-release, best-storage and range-scaled balanced selection. Retain the generalized baseline separately. Freeze selected identities and vectors before the first held-out evaluation. Report failures without substituting a candidate based on held-out results.

The 2008–2013 period has already been reported. Additional scores are repeated descriptive comparisons, not a new untouched validation sample. They remain segmented conditional hindcasts: observed storage initializes clean blocks and computed net forcing depends on source accounting. Verify interval and block balances and report phase-specific spill volumes alongside release and storage scores. Observed restart changes across excluded intervals are state initialization, not modeled water fluxes.

## Checkpoint compatibility in this fixed-code experiment

The two continued seeds retain their validated identity chains and can be verified by the existing continuation adapter. Fresh seed 44 saves a complete, validated optimizer checkpoint under all 59 current numerical-module hashes. Its native search identity contains no continuation bridge.

The current continuation adapter requires such a bridge whenever its own module is already present in a prior identity. It therefore cannot directly resume the fresh seed-44 checkpoint. This experiment preserves that original identity and checkpoint rather than rewriting provenance. Supporting native checkpoints from searches started after the adapter was added needs a separate compatibility change and verification before any later seed-44 continuation. This limitation does not affect the completed 512-attempt calibration search.
