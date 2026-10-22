# Versioned source-capacity profiles

A documented correction to a reservoir capacity belongs in a separate source
profile. Preserve canonical observations, previous model configurations and
published experiment releases. Capacity must come from independently identified
source definitions, rather than observed maxima, optimizer feasibility or
held-out performance.

The first such profile is Charvak:

- [Observations and provenance](../data/source_profiles_v1_20261013/ex37_Charvak)
- [Model configuration](../configs/source_profiles_v1_20261013/ex37_Charvak_model.json)
- [V2 workflow configuration](../configs/workflow_charvak_source_v2.json)
- [Completed release](../outputs/charvak_calibration_source_v2_20261013/releases/budget_512/README.md)
- [Source review, baseline comparison and verification](../outputs/source_limit_review_v1_20261013/README.md)

The profile retains the original observation bytes and every physical setting
except the independently documented gross capacity. The assumed DZTR floor,
observation aggregation, model step and declared root policy remain explicit.
Source provenance distinguishes gross/active capacity and records limitations.
Passing simulation and balance gates does not certify observation timing.

```powershell
python -B tools/reservoir_workflow_v2.py run --config configs/workflow_charvak_source_v2.json --output outputs/charvak_calibration_source_v2_20261013
```

This run has completed. The same budget reuses verified snapshots; an explicitly
planned continuation adds `--evaluation-budget 1024` without changing seeds,
population, bounds or source definitions. Its 512 release stays immutable.
See [V2 instructions](workflow_extensions_v2.md) for checkpoint identity rules.

Keep the generalized source-profile baseline alongside the calibrated result.
Evaluate it only after freezing parameter selection and report improvements and
trade-offs on both objectives. A positive NSE does not establish improvement;
one represented held-out regime does not provide assessment of the others.

Calendar or stock/flow repairs are separate source profiles with their own
provenance, complete timeline checks and scientific validation. The Canyon Ferry
crosswalk in the review is evidence for such a repair, not a simulation-ready
dataset. Never silently fill its gap or reuse old checkpoints under new dates,
capacity definitions or units.
