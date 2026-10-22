# Declared substep comparison

This experiment retains the direct implicit solver and compares explicitly
declared internal durations. It uses packaged collection Trinity as a control
and Ghost as a missing-root case. Collection Trinity has different initial
storage and inflow settings from the standalone `examples/simulate_trinity.py`;
that example is preserved. The existing production defaults and checkpoint
identities are unchanged.

## Conventions

Each native observation interval is divided into equal parts. The first study
uses 1, 24, 48 and 96 parts per day: daily, hourly, 30-minute and 15-minute steps.
The command records the entire list before evaluating any trajectory. It does
not retry failures at a smaller duration or choose a duration from validation.

- Raw inflow is interpolated linearly between consecutive input rows. Exact
  source values are assigned at interval endpoints. The existing correction
  factor applies in balance; inflow-dependent release still sees raw inflow.
- The destination observation row's operating month is held throughout its
  interval. This preserves the established current-row clock, including month
  transitions. These inferred substeps are not new observed records.
- The configured initial release is used once. Every subsequent substep carries
  the preceding physical release, including any spill increment. Release is not
  recalculated at the beginning of a day or reset from observed release/storage.
- The internal duration enters the unchanged DZTR equations, including the
  available-water limits. This deliberately changes the discrete release law.
- Each substep solves all affine branches and validates accepted roots, then
  applies the existing conservative trapezoidal overflow correction. The
  default root policy remains `reject`. Any alternative requires an explicit
  `--root-policy`; no implicit fallback or averaging occurs.

For a substep of duration `h`, the accepted raw root obeys
`S + h/2 * Q(S) = predictor`. Above capacity, endpoint release increases by
`2 * excess / h`, and storage is capped. The previous corrected release remains
the preceding endpoint of the next trapezoid.

## What is conserved and reported

The native table retains `release_m3s` as the final endpoint release. It also
reports `integrated_release_volume_m3`, the sum of substep trapezoidal volumes,
and `interval_mean_release_m3s`, that volume divided by the native duration.
The corresponding integrated inflow volume uses the declared linear forcing.

Native water balance is checked against these integrated volumes:

```text
storage_end - storage_start = integrated_inflow - integrated_release
```

A single native trapezoid using only the start/end release rates generally
differs from the integrated substep trajectory. Its discrepancy is reported as
`native_endpoint_balance_residual_m3`, separately from conserved integrated
balance. It is not used to reject a correctly conserved substep trajectory.

`overflow_volume_m3` sums raw storage clipping corrections. It is distinct from
`integrated_spill_volume_m3`, which integrates the spill increments at both
endpoints of every substep. Carrying a preceding spill increment into the next
trapezoid means the two totals need not match.

Initial integrated volumes and mean release are undefined (`NaN`), because the
first row is a configured state. Cumulative volume ledgers start at zero. Fluxes
use accurate interval summation and compensated cumulative summation. Reports
check physical states, interval and internal balance, pre-overflow release-rule
residuals, and cumulative closure against a recorded roundoff bound.

Both endpoint fit metrics and diagnostic interval-mean release metrics are
reported separately for calibration and validation. Input metadata does not
establish whether every recorded daily release is an instantaneous endpoint or
a daily mean; the mean-release score therefore cannot establish a superior
observation convention. Storage and release observations never reset states.

## Critical-boundary feasibility

The critical and normal release branches meet continuously when
`critical_storage - dead_storage >= h * critical_release`. Each comparison
exports all 12 months' available volumes, maximum durations and required
substep counts. Zero critical release imposes no bound from this condition.

With Ghost's calibration targets, May requires at least 29 equal parts per day
to remove this particular critical/normal gap; its maximum duration is about
51.1 minutes. June requires at least two parts; the other months permit one.
Trinity permits one for all 12 months under this specific condition. These
bounds do not guarantee root uniqueness, nonnegative predictors, or valid
complete trajectories. Other release discontinuities can persist.

On a one-substep missing-root failure, the workflow also repeats that isolated
interval from its recorded previous state at every declared resolution. This
separates the local failure mechanism from the changed path of a complete
substep simulation. The isolated replay is labeled explicitly and does not
constitute a complete reservoir trajectory.

## Run the comparison

Use a fresh output directory from the repository root:

```powershell
python examples/review_substeps.py --output outputs/substep_comparison_new
```

The installed entry point is `dztr-substep-review`. Selection and durations can
be supplied explicitly:

```powershell
python examples/review_substeps.py --reservoir ex12_Trinity --reservoir ex26_Ghost --substeps 1 --substeps 24 --substeps 48 --substeps 96 --root-policy reject --output outputs/substep_explicit_new
```

The one-substep control is required and must exactly match all existing
root-policy simulation columns on its completed native prefix. A failure
records the date, internal part, current inputs/state, reason and completed
prefix; calibration can remain complete even if held-out integration fails.
The first reporting workflow accepts daily observed-calendar inputs. The core
API also supports existing monthly clocks, but the interpretation of monthly
inflow/release statistics needs a separate study before fleet use.

Outputs include `substep_review.json`, `.csv`, `.md`, unchanged calibration-only
targets across resolutions, the critical-boundary table, native trajectories,
partial trajectories and first-failure context. `--record-internal` additionally
saves every accepted internal part; it is disabled by default to keep memory
and output size proportional to native records. Isolated failure replays always
retain their small internal tables.

## Import the experimental API

```python
from dztr.simulation.substeps import SubstepSettings, simulate_substep_reservoir

result = simulate_substep_reservoir(
    observations, parameters,
    settings=SubstepSettings(substeps_per_interval=48, root_policy="reject"),
)
daily = result.simulation.frame
```

Failures raise `SubstepSimulationError`, whose `detail` records the failing part
and whose `partial` retains completed intervals. Setting `record_internal=True`
also retains accepted parts of the unfinished interval, separately from native
endpoints. No existing simulator or calibration objective is changed to use
this API automatically. This runner does not resume old search checkpoints.

## Scientific limits

Differences between adjacent successful resolutions quantify sensitivity of
this explicitly timestep-dependent scheme. They do not prove convergence to
the original daily map, establish a universal operating policy, or justify
choosing a duration from held-out scores. No optimization is performed. The
manifest, consumed inputs and package code are hash checked before and after
each review; old experiment outputs retain their original identities.

## Completed Trinity and Ghost comparison

The declared comparison is saved in
[`outputs/substep_comparison_final`](../outputs/substep_comparison_final/substep_review.md).
It uses packaged collection configurations, calibration-only targets and strict
root rejection. No optimization or setting selection was performed.

| Internal duration | Trinity full trajectory | Ghost full trajectory | Ghost isolated May 6, 1991 interval |
| --- | --- | --- | --- |
| Daily | Completed | No root, May 6, 1991 | No root |
| Hourly | Completed | Multiple roots, May 26, 1990 | No root at part 9 of 24 |
| 30 minutes | Completed | Multiple roots, May 27, 1990 | Completed |
| 15 minutes | Completed | Multiple roots, May 27, 1990 | Completed |

Trinity's adjacent calibration storage differences decrease from 100,212 m³
RMSE for hourly to 30-minute steps to 50,344 m³ for 30- to 15-minute steps.
Endpoint-release differences also decrease. Release NSE does not improve:
calibration NSE changes from 0.2168 daily to 0.1443 at 15 minutes, while held-out
NSE changes from -0.0842 to -0.1371. Numerical refinement and predictive fit
therefore require separate assessment.

Ghost's finer full trajectories stop during the 1990 warmup at a downward
release jump near flood storage. Both candidate roots satisfy the implicit
balance and literal release law. There is no complete Ghost calibration or
validation result. The successful 30- and 15-minute replays establish repair
of the specific critical-boundary interval from a fixed preceding state; they
do not establish a successful full Ghost simulation or convergence.

Quality and accepted-substep counts for a blocked trajectory describe completed
native intervals. Accepted parts of its unfinished interval are recorded
separately in `first_failure`. Zero accepted multiple-root steps under `reject`
does not imply there was no ambiguity: the rejected failure records it.

The full regression run passed 1,140 tests, skipped one Windows symlink test,
and encountered an access-denied directory rename in one existing portability
test. That exact test passed unchanged on retry, giving 1,141 passing distinct
tests across the run and retry. The new solver/workflow account for 132 tests.
An isolated offline wheel verifies the API and CLI; all 50 prior model modules,
nine legacy modules and permanent input files retain their previous bytes.

See the [study summary and verification](../outputs/substep_verification/study_summary.md)
and [comparison figures](../outputs/substep_figures_20261007/README.md).
Output artifacts are local and ignored by Git; `docs/substeps.md`, source code,
the example and portable tests provide the repository implementation.

The follow-up [combined solver study](combined_solver.md) tests explicit root
selection with 30- and 15-minute steps. All six Ghost policy/duration runs
complete; the daily controls retain the missing-root failure. The strict
comparison above remains the reference for separating the two mechanisms.
