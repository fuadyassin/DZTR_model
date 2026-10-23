# Standalone frozen examples

Gardiner (253), Oahe (870), and Garrison (753) each contain:

- `profile.json`: all frozen parameters, clocks, numerical choices and provenance.
- `forcing_si.csv`: inflow-only historical forcing derived from the already-public example.
- `forcing_restart_si.csv` and `initial_state_restart.json`: a later-start example initialized from the frozen model trajectory.
- `expected_checks.json`: reference row counts and scientific checks.

See the [quick start](../../docs/portable_application_v1.md). Neither fitting
responses nor private experiment archives are read by the application command.
The first forcing record anchors the state; it is not an integrated interval.
The supplied restarts are reproducibility checks, not new independent forecasts.
The profiles retain their original source qualifications and unproven convergence.
