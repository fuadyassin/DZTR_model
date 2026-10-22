# Preparing daily mean inputs and source-quality exclusions

Daily mean forcing needs explicit interval volumes. The existing MATLAB replay and simulation paths retain their endpoint conventions; the new preparation API supplies a separate input contract for an interval-aware solver.

`dztr.data.intervals.prepare_daily_intervals` accepts a finite normalized SI frame on consecutive timezone-naive calendar dates. The caller must declare computed net inflow explicitly to accept negative values. The default physical-inflow interpretation rejects negatives. Storage and observed release must be nonnegative under either interpretation. Input units, calendar gaps, timezone evidence, and physical source definitions should be reconciled before preparation.

```python
import json
from pathlib import Path

import pandas as pd

from dztr.data.intervals import DailyIntervalPolicy, prepare_daily_intervals

root = Path.cwd()  # Run from the repository root.
policy_path = root / "configs/reviews/bull_lake_interval_policy.json"
review = json.loads(policy_path.read_text(encoding="utf-8"))
observations_path = (policy_path.parent / review["observations_file"]).resolve()
frame = pd.read_csv(observations_path, parse_dates=["date"],
                    index_col="date", float_precision="round_trip")
policy = DailyIntervalPolicy(
    flow_semantics=review["interval_inputs"]["flow_semantics"],
    quarantined_interval_end_dates=review["quarantine"]["interval_end_dates"],
)
prepared = prepare_daily_intervals(frame, policy=policy)
```

Each end-labelled daily mean has a nominal reported volume `inflow_m3s × 86400`. The reported values remain visible on all source dates, including excluded rows. `usable_inflow_volume_m3` is available only for eligible intervals with clean storage endpoints; the initial row has no preceding interval in the supplied frame. The dates represent an adopted nominal calendar convention, not independently confirmed physical measurement times or daylight-saving durations.

For a flagged transition ending on date i, both storage endpoints i−1 and i are quarantined. Every interval touching either endpoint becomes ineligible. The preparation retains all dates and values, then identifies contiguous clean storage blocks. Their first observations are provisional initial states, excluded from storage and release scores. This prevents a later objective from awarding an exact initialization match as a simulated prediction.

Optional `period_labels` must share the input index. Period boundaries preserve state continuity: a clean warmup-to-calibration or calibration-to-validation transition remains usable for simulation. `source_audit_interval_eligible` additionally restricts arithmetic evidence to endpoint pairs in the same phase. Splitting the simulation at every scoring-phase boundary would introduce extra observed-state resets, so the preparation does not do that.

| Output | Meaning |
| --- | --- |
| `storage_target_eligible` | Clean source storage rows eligible for phase-restricted descriptive estimation |
| `simulation_interval_eligible` | Both interval endpoints are clean and a preceding row exists |
| `source_audit_interval_eligible` | Eligible interval whose endpoints share a reporting phase |
| `storage_score_eligible`, `release_score_eligible` | Destinations of simulated intervals; initialization rows are excluded |
| `source_transition_flagged` | The originally flagged transition, before conservative endpoint expansion |
| `storage_endpoint_quarantined` | Either storage endpoint of a flagged transition |
| `segment_id`, `segment_initialization` | Clean contiguous blocks and their provisional start rows |

Preparation masks express eligibility, not certified accuracy. The caller must separately exclude warmup from scoring, restrict parameter estimation to calibration, and use modeled interval-average release when comparing daily-mean discharge. Blocks can span multiple scoring phases. Never pass the rows obtained by deleting masked dates to the existing continuous simulator: that changes the calendar and does not implement the restart plan.

For Bull Lake, the [selected policy](../configs/reviews/bull_lake_interval_policy.json) preserves signed computed net flux and fixes its scale at one. It retains the storage-only release mode; a gross-inflow surrogate is unnecessary. Negative values do not identify evaporation, infiltration, measurement error, or a unique physical loss. Adding a separate loss for the same negative flux would double-count that deficit.

The future solver should initialize endpoint release from the operating rule at each provisional storage anchor, integrate modeled release over the matching forcing interval, carry state through clean protocol boundaries, and report physical/root infeasibility explicitly. Observed daily-mean release cannot supply an endpoint rate without another interpretation. Applying the selected signed policy by merely enabling `allow_signed_inflow` in the current solver would retain its endpoint averaging.

After quarantine, validation with observed storage restarts is a **segmented conditional hindcast**. Pool eligible observation/prediction pairs within each phase when calculating an objective; report segment support separately. Per-block NSE can be undefined in a one-interval block and should not be averaged. Computed inflow and storage/release diagnostics are statistically dependent.

The policy predeclares nominal initial storage and ±1% of capacity as future robustness perturbations. These perturbations are not measurement uncertainty bounds; they must not be selected using validation fit. No calibration should begin until the interval solver and segmented objective consume the preparation contract. The old simulation API and reservoir configurations remain available for their existing conventions.
