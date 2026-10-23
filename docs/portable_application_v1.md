# Portable DZTR application

The public application contains three frozen, calibration-selected profiles:
**Gardiner (GRAND 253), Oahe (870), and Garrison (753)**. Each profile includes
all 72 monthly targets, physical constants, numerical choices and provenance.
Its simulation needs only inflow. No private calibration archive, observation
workbook, station download or E-drive path is required at runtime.

## Install

From a checkout or the extracted application bundle, use Python 3.10 or later:

On Windows, extract to a short workspace path. Existing workflow stages create
nested preparation/profile folders and can exceed Windows' path-length limit
under a deeply nested extraction directory. The portability verifier uses a
fresh, checked short workspace for that reason.

```powershell
& .\tools\Set-ProjectCache.ps1
python -B -m pip install ".[dev,plots]"
python -B -m dztr --help
```

Use a dedicated environment if desired. Project-managed caches and temporary
files are redirected to this checkout's `.cache`. Both the `dztr` console command
and `python -m dztr` invoke the same interface. The package distribution retains
its existing name, `dztr_param`; version 0.3.0 is distributed through GitHub
release files. It has not been uploaded to PyPI.

## Apply a frozen profile

```powershell
python -B -m dztr inspect --profile examples/portable/gardiner/profile.json
python -B -m dztr simulate --profile examples/portable/gardiner/profile.json --forcing examples/portable/gardiner/forcing_si.csv --forcing-stage prepared --output outputs/my_gardiner_application --plot
python -B -m dztr status --output outputs/my_gardiner_application
```

Replace `gardiner` with `oahe` or `garrison` for the other examples. Gardiner is
daily; Oahe and Garrison preserve their archived fixed **30.5-day monthly steps**.
Monthly samples must not be relabelled as daily or assumed to use calendar-month
integration. The first record anchors the initial state; each later step uses
the profile's existing endpoint trapezoidal balance.

Every run writes `simulation.csv`, `monthly_targets_si.csv`, a copy of the frozen
`profile.json`, `simulation_receipt.json`, and a checksum manifest. `--plot` adds
storage, flow and balance plots. Total release includes the solver's conservative
overflow adjustment; the diagnostic overflow volume must not be added again to
the reported release when checking balance.

Resume verifies the same profile, forcing, initial conditions, plotting choice
and completed files, then reuses the result:

```powershell
python -B -m dztr simulate --profile examples/portable/gardiner/profile.json --forcing examples/portable/gardiner/forcing_si.csv --forcing-stage prepared --output outputs/my_gardiner_application --plot --resume
```

## Start at another date

Use an explicit state in SI units. The supplied restart example uses an earlier
model prediction, not observed storage or release:

```powershell
python -B -m dztr simulate --profile examples/portable/gardiner/profile.json --forcing examples/portable/gardiner/forcing_restart_si.csv --forcing-stage prepared --initial-state examples/portable/gardiner/initial_state_restart.json --output outputs/my_gardiner_restart
```

For your own inflow, supply a CSV with exactly `date,inflow_m3s` and an initial
state JSON containing `storage_m3` and `release_m3s`. Keep its sampling and clock
compatible with the profile. A new start date without a state is rejected.
Changing a legacy synthetic target clock requires a separately prepared profile.

`canonical` means the profile's declared source factor/offset still needs to be
applied; `prepared` means it has already been applied. Physical inflow correction
is a separate model constant. The example forcing is **prepared**. No gap filling,
response assimilation or automatic parameter changes occur.

## Estimate parameters through the same gateway

The public workflow configuration uses only distributed Gardiner observations
and a distributed model configuration:

```powershell
python -B -m dztr plan --config configs/portable/gardiner_workflow.json
python -B -m dztr run --config configs/portable/gardiner_workflow.json --output outputs/my_gardiner_parameterization
python -B -m dztr run --config configs/portable/gardiner_workflow.json --output outputs/my_gardiner_parameterization --resume
```

It selects wet, normal and dry whole years with at least 65% calibration data,
fits monthly targets on those years, and evaluates fixed observed-data and
flow-only recipes. Source transformations, state continuity, gaps and scientific
failures retain their existing backend meanings. This fresh preparation has a
new selection identity; it is not relabelled as the archived calibration.

Optional configurations expose new optimization explicitly:

| Configuration | Behavior when run |
|---|---|
| `gardiner_calibration.json` | Joint NSGA-II; three seeds, 512 evaluations each |
| `gardiner_scalar_calibration.json` | DDS and GA; release-only and configurable weighted objectives |

Use `plan` first. `run --until baseline` stops the joint configuration before
optimization. `--until targets` stops either configuration after parameterization.
These example budgets are reproducible settings, not evidence of convergence.
The released frozen examples reuse completed searches and start none.

The installed wheel supports standalone frozen-profile application anywhere.
Workflow orchestration additionally needs the source/application bundle because
the established backends live in `workflows/`, `strategies/` and `tools/`. From
another directory, supply `--project-root PATH` for `plan` and `run`.

## What the demonstrations establish

The three historical replays and later-start restarts reproduce saved storage
and release exactly and pass independent water-balance checks. This establishes
export/application reproducibility. It does not establish independent forecast
skill, forcing independence, source certification or optimizer convergence.
See [the assessment](scientific_assessment_v1.md) before selecting a method.
