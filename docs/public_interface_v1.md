# Public commands and scientific contracts

| Command | Purpose | Starts optimization? |
|---|---|---|
| `dztr plan --config FILE` | Validate and describe ordered workflow stages | No |
| `dztr run --config FILE --output DIR` | Execute declared acquisition, targets, baseline, calibration or application stages | Only if a calibration stage is reached |
| `dztr run ... --until STAGE --resume` | Stop at a stage or reuse verified completed work | According to the declared stages |
| `dztr status --output DIR` | Read saved stage/application status | No |
| `dztr inspect --profile FILE` | Verify and describe a standalone profile | No |
| `dztr export-profile --registry FILE --reservoir-id ID --preprocessing-json FILE --method METHOD --output FILE` | Export a verified native profile without observations | No |
| `dztr simulate --profile FILE --forcing CSV --forcing-stage STAGE --output DIR` | Apply a fixed standalone profile | No |

`python -m dztr` is equivalent. Workflow commands delegate to the existing v1-v8
configuration backends through the current versioned gateway; no archived
scientific implementation is renamed or rewritten. Workflow outputs stay under
the source bundle's `outputs/`. Standalone applications accept fresh output
directories elsewhere and are usable from an installed wheel.

## Inputs

- **Profile JSON:** complete 12-month, six-target table in m3 and m3/s; capacity,
  dead-storage fraction, initialization, channel capacity, inflow correction,
  flood mode, sampling, target-month clock, root policy, absolute balance
  tolerance, preprocessing, method and provenance.
- **Application forcing CSV:** exactly `date,inflow_m3s`. Unique, ascending,
  timezone-naive midnight dates; consecutive days or consecutive month starts,
  as declared by the profile. At least two rows. Missing/invalid values stop
  the run. Signed net rates require a separately declared signed-net profile.
- **Initial-state JSON:** exactly `storage_m3` and `release_m3s`, required when
  the forcing starts on a different date. Storage must be within capacity and
  release nonnegative. Application reads no response observations.
- **Workflow JSON:** paths are relative to the containing configuration file;
  `@stage.output` links earlier stages. Includes station arithmetic where used,
  units, source gates, year-selection policy, recipe, objective and optimizer.

Hash checks detect changed parameters, sources, inputs and completed output
files. Scientific code hashes normalize Python source line endings to LF so a
Windows/Unix checkout does not itself change a portable profile's identity.
Hashes are consistency checks, not third-party signatures or data certification.
Runtime Python, NumPy and pandas versions are recorded in the application receipt.

## Numerical interpretation

The portable endpoint solver retains the established balance:

\[
S_t-S_{t-1}=\frac{\Delta t}{2}
\{c(I_t+I_{t-1})-(Q_t+Q_{t-1})\}.
\]

Release depends on storage and its DZTR zone, so storage and release must be
solved together. The existing piecewise-linear solver finds valid implicit
solutions algebraically. A root policy only selects among actual valid roots.
`nearest_previous_storage` chooses the root nearest the preceding state; ties
follow the existing deterministic convention. It cannot repair a timestep with
no valid root. Capacity overflow is handled by the existing conservative release
adjustment. Neither tolerances nor capacity change during application.

Daily-mean observations are not automatically endpoint states/rates. The
separate daily-interval studies retain their own solver, source and checkpoint
identities. This portable profile format currently supports standard and signed
native endpoint routes; it rejects unsupported interval profiles.

## Calibration interpretation

Mixed whole-year selection represents dry, normal and wet years while keeping
at least 65% calibration support. Model state remains continuous through
unscored and assessment years. Calibration masks control response fitting and
scoring; they do not concatenate selected years or reset storage each year.
Filtered responses additionally need fitting-pure source windows.

Pooled NSE scores all eligible response pairs together. Equal-regime scoring
averages separately calculated dry/normal/wet NSE values. It is not the same
as selecting mixed years. Scalar full-data calibration minimizes:

\[
J=w_S(1-\mathrm{NSE}_S)+w_Q(1-\mathrm{NSE}_Q),\qquad w_S+w_Q=1.
\]

Nonnegative user weights are normalized by the existing backend. DDS and GA
are optional scalar optimizers; NSGA-II retains separate storage/release
objectives. A zero storage weight in a full-data experiment does not hide
storage from empirical targets or bounds. Use the separate flow-only route
when storage must be unavailable during fitting.

Parameters, seeds and routes are selected using calibration evidence. Held-out
scores assess the frozen choice. Optimizer checkpoints preserve the declared
horizon, bounds, objectives and seeds; changing those creates a new experiment.

## Outputs and failures

`simulation.csv` contains predicted storage/release, forcing, zone, overflow,
water-balance and root diagnostics. `monthly_targets_si.csv` and `profile.json`
retain the applied parameters. `simulation_receipt.json` records decisions,
environment and independent balance checks; `manifest.json` seals the result.
The initial row anchors state and is excluded from interval balance checks.
`--plot` adds `simulation.png`.

A missing root, invalid calendar, changed checksum, unresolved admission gate
or unsupported numerical route produces an error and a nonzero command exit.
Use a fresh output directory after changed decisions. `--resume` verifies a
completed standalone result rather than silently rerunning it.
