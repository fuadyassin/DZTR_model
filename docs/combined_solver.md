# Ghost: combined substeps and root selection

The combined experiment completes Ghost's entire 1990–2011 simulation under
each of three explicit root policies at both 30- and 15-minute durations.
All six trajectories pass physical-state, integrated water-balance and
recorded internal-quality checks. The daily controls still reproduce the
original missing-root failure. This establishes numerical feasibility for
these settings and inputs; storage prediction remains a separate concern.

## Which zones cause the root failures?

The implementation numbers the zones as follows:

| Zone | Storage region |
| ---: | --- |
| 0 | Dead storage |
| 1 | Critical storage |
| 2 | Normal storage |
| 3 | Flood storage |
| 4 | Emergency storage |

The diagnosed failures occur at **boundaries**, rather than only within zones
1 and 4:

| Boundary | Release mismatch | Consequence for implicit balance |
| --- | --- | --- |
| 1 → 2, critical → normal | Available-water-limited critical release can be below the normal branch's starting critical target. | An upward jump can leave no root. |
| 3 → 4, flood → emergency | Inflow-dependent flood release can exceed the emergency branch's starting release. | A downward jump can create multiple roots. |

At the critical boundary, continuity requires
`critical_storage - dead_storage >= internal_duration * critical_release`.
Ghost's calibrated May targets require at least 29 equal substeps per day;
48 and 96 satisfy that condition. This addresses the identified gap but does
not resolve a downward flood-boundary jump.

At flood storage, the flood branch ends at `max(flood_release, inflow)` in
inflow mode, or `flood_release` in storage mode. The emergency branch starts
just above the boundary at `min(flood_release, channel_capacity)`. Thus high
inflow can cause the observed drop; a channel limit below the flood target
can also cause it in storage mode. With valid ordered targets, the dead to
critical and normal to flood joins are continuous. Negative balance predictors,
invalid inputs/targets and historical fixed-point iteration failures are
separate problems and are not confined to these boundaries.

## Protocol and results

Parameters are estimated from calibration observations only and remain fixed
across policies and durations. Initial state, raw-inflow interpolation,
destination-month targets and conservative spill follow the existing
[substep conventions](substeps.md). The policies choose only admissible roots.
No optimization, policy selection or default change is performed.

| Root policy | Daily control | 30-minute full run | 15-minute full run |
| --- | --- | --- | --- |
| `nearest_previous_storage` | No root on 1991-05-06 | Completed | Completed |
| `minimum_storage` | No root on 1991-05-06 | Completed | Completed |
| `maximum_storage` | No root on 1991-05-06 | Completed | Completed |

Nearest-storage and minimum-storage produce byte-identical saved native
trajectories at each duration in these Ghost runs. This does not imply that the
policies are equivalent for other parameters or reservoirs. Subdaily states
for the entire record were not stored; their equality is not independently
asserted from native equality.

All six runs have calibration endpoint-release NSE around 0.953 and held-out
release NSE around 0.946. Held-out storage NSE is approximately -0.263 to -0.267,
so storage prediction is worse than the observed-period mean benchmark.
Completing the integration does not establish adequate reservoir prediction.
Interval-mean release scores remain diagnostic because native observation
averaging conventions are unresolved.

Maximum-versus-minimum policy differences decrease with refinement:

| Difference between policies | 30 minutes | 15 minutes |
| --- | ---: | ---: |
| Calibration storage RMSE difference (m³) | 71,560 | 33,806 |
| Held-out storage RMSE difference (m³) | 79,012 | 38,600 |
| Calibration endpoint-release RMSE difference (m³/s) | 0.5773 | 0.1603 |
| Held-out endpoint-release RMSE difference (m³/s) | 1.0636 | 0.3393 |

These quantify sensitivity with the declared timestep-dependent release law.
Two successful durations do not establish convergence, an acceptable error
threshold or the correct operating policy. Held-out results do not choose
settings.

## Modeled release changes

Daily endpoint/mean-release variation and ambiguity exposure are reported
separately for warmup, calibration and validation. Ambiguity exposure is the
number of accepted multiple-root parts weighted by their duration; it is
policy/path dependent and is not an oscillation metric.

Six small event continuations use May 27, 1990, the date of the corresponding
archived strict refined failures. Each starts from its own full run's preceding
native state. The preceding interval has no spill, and the event's ending state
and interval diagnostics exactly match the full trajectory. This is distinct
from the earlier common-state critical-gap replay.

During this event, nearest/minimum have no significant release-direction
reversals or flood/emergency crossings. Maximum selection crosses into the
emergency zone and produces two release-direction reversals, including one
large release drop:

| Sampled event diagnostic | Nearest/minimum, 30 min | Nearest/minimum, 15 min | Maximum, 30 min | Maximum, 15 min |
| --- | ---: | ---: | ---: | ---: |
| Largest absolute release step (m³/s) | 0.9735 | 0.4869 | 151.1170 | 139.8908 |
| Release-direction reversals | 0 | 0 | 2 | 2 |
| Flood to emergency crossings | 0 | 0 | 1 | 1 |

These modeled subdaily rates use assumed linear forcing. They do not establish
real gate behavior or the absence of oscillations elsewhere in the record.
Every full native endpoint/mean release is within Ghost's channel capacity;
that check alone does not establish subdaily downstream safety.

## Reproduce

From the repository root, use a fresh output directory for each policy:

```powershell
python examples/review_substeps.py --reservoir ex26_Ghost --substeps 1 --substeps 48 --substeps 96 --root-policy nearest_previous_storage --output outputs/ghost_nearest_new
python examples/review_substeps.py --reservoir ex26_Ghost --substeps 1 --substeps 48 --substeps 96 --root-policy minimum_storage --output outputs/ghost_minimum_new
python examples/review_substeps.py --reservoir ex26_Ghost --substeps 1 --substeps 48 --substeps 96 --root-policy maximum_storage --output outputs/ghost_maximum_new
```

The declared protocol, all outputs, local-event audit and figures are linked
from the [study summary](../outputs/combined_solver_verification_20261008/study_summary.md).
The [figures](../outputs/combined_solver_figures_20261008/README.md) compare fit,
policy sensitivity and the sampled flood transition. Artifacts under `outputs`
are local and ignored by Git; this guide provides persistent reproduction
instructions.

## Implication for the next stage

The tested combination resolves Ghost's diagnosed numerical blockers without
changing the release equations. Before fleet use, assess duration requirements
and operating conventions per reservoir. The current storage discrepancy is
much larger than the tested refinement differences, so examine storage
residuals and forcing/balance assumptions, then calibrate with an explicitly
declared solver profile. The profile and any timestep-feasibility constraints
must be fixed for calibration and carried into validation and checkpoint
identity. No fleet-wide duration, policy or evaluation budget follows from
this single-reservoir experiment.
