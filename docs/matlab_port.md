# Trinity MATLAB port

The [world reservoir adapter](world_reservoirs.md) now applies the same core
equations to the supplied 42 source folders. It preserves per-folder settings
and distinguishes daily/monthly clocks, historical baseline replay, and modern
calibration-only experiments. Collection Trinity has different inflow/initial
settings from the standalone example described below.
The [permanent collection manifest](../data/reservoirs/manifest.json) now runs
41 active reservoirs from repository CSV/JSON inputs, with Tuyen Quang excluded.
`examples/run_reservoir_collection.py` supports individual or batch execution
and records per-reservoir readiness diagnostics.

The [root-policy comparison](root_policy_review.md) adds explicit experiments
for ambiguous conservative steps, with separate calibration/validation reports
and fixed-iteration convergence/cycle diagnostics. It preserves this standalone
MATLAB-compatible example and the established strict default.

## Source chain and parameter mapping

The supplied example at `E:\1_DataHub\E_Reservoir_data\ex12_Trinity` contains
`SimRISM14.m` (the actual filename), `DZTR.m`, `main_script.m`,
`reservoir_system.m`, `obj_fun.m`, parameterization scripts and saved outputs.
No files in that folder are modified. Source hashes are recorded in the
benchmark report and portable test fixture provenance.

`res_info.xlsx` is a headerless daily table: year, month, day, storage in million
m³, inflow, release, then constants. Its first six columns exactly match the
repository's Trinity observation workbook. `res_info_out.mat` holds the same
observations in SI units with MATLAB serial dates.

The coefficient TXT contains four scalar values and a 19 by 12 table:

| MATLAB location | Python representation |
| --- | --- |
| First four values: capacity, initial release, initial storage, channel capacity | Explicit `ReservoirParameters` fields |
| Table row 1 | Months 1–12 |
| Rows 2–4: critical, normal, flood storage | Three `storage_*_m3` columns |
| Rows 5–7: critical, normal, flood release | Three `release_*_m3s` columns |
| Rows 8–13: release lower and upper bounds | `read_matlab_bounds` release columns |
| Rows 14–19: storage lower and upper bounds | `read_matlab_bounds` storage columns |

`read_matlab_coefficients` imports the physical configuration and monthly targets.
`parameters_from_matlab_vector` converts a 72-value AMALGAM candidate: three
12-month storage-fraction blocks, then three release blocks. Storage fractions
are multiplied by the explicit capacity. Imported and estimated parameters use
the same validated monthly table and simulator.

Trinity's MATLAB configuration is capacity 2,533,500,000 m³, initial storage
1,593,000,000 m³, initial release 10 m³/s, channel capacity 300 m³/s, dead-storage
fraction 0.1, inflow adjustment 0.92 and storage-based flood-zone release. The
initial storage is deliberately different from the first observed storage.
The `simRISMcom.m` variant uses observed initial storage and an inflow factor
of 1.0; callers can specify those choices explicitly.

## Percentile methods

The historical MATLAB `prctile` uses plotting positions `(k - 0.5)/n`, equivalent
to NumPy `hazen`. NumPy `linear`, retained for the current workbook workflow,
uses a different convention. MATLAB's newer documentation calls its historical
method `midpoint`, but **NumPy `midpoint` is not equivalent**.
[MATLAB documentation](https://www.mathworks.com/help/matlab/ref/prctile.html),
[NumPy documentation](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)

Against the full-precision 2018 coefficient targets, regenerated Hazen estimates
differ only by printed rounding (at most 0.464 m³ and 0.000477 m³/s). Linear
estimates differ by up to about 3.64 million m³ and 0.31744 m³/s. Selecting Hazen
is explicit and does not change the earlier Python workbook regression checks.

## Operating rules and time stepping

`dztr_release` reproduces the five `DZTR.m` branches, their exact `<=` boundaries,
and downstream channel limiting only in the emergency zone. `flood_mode="storage"`
maps to `cse=1`; `"inflow"` uses raw current inflow in that zone, as MATLAB does.
The balance uses adjusted inflow at both endpoints. Monthly targets switch
according to the **actual current observation date**.

The coupled daily equations are:

```text
A = S_previous + dt/2 * (factor*(I_current + I_previous) - Q_previous)
S_current = A - dt/2 * Q_current
Q_current = DZTR(S_current, I_current, current_month)
```

The default `implicit` solver finds the algebraic candidates for every linear
release segment and validates them against the actual rule. It reports missing
or multiple feasible roots instead of accepting a failed solver result.
Discontinuities can occur with inflow-dependent release or restrictive channel
capacity, so a globally continuous root function is not assumed.

`matlab_fixed_point` reproduces exactly 1,000 iterations by default, including
the original final-iterate staggering and possible negative storage. It reports
numerical issues rather than silently clipping them. `main_script.m` instead
calls `fsolve` on the same equations, but does not check its convergence status.

For storage exceeding capacity, regular `conservative` handling caps storage
and adds `2*excess/dt` to endpoint release. Legacy `matlab` handling adds
`excess/dt`, leaving a trapezoidal balance error of `-excess/2`. The compatibility
choice is explicit. Total release includes uncontrolled capacity overflow and
may exceed channel capacity; diagnostics expose the overflow volume separately.
Precipitation/evaporation are not separate forcings in the supplied implementation.

## Evaluation and reference provenance

`objective_metric` ports all ten `obj_fun.m` criteria. NSE and KGE evaluate
storage and release independently after excluding matched NaN pairs. KGE uses
sample standard deviations; PBIAS is positive for overprediction. Undefined
metrics return NaN. `warmup_steps=365` corresponds to MATLAB `wrmp=366`.

The primary parity references are `flowSIM_eval.mat` and `stoSIM_eval.mat`, each
11,323 by 100. Their script calls `S2Q1JAN`, whose source is absent. Its operating
rule was independently checked against the supplied `DZTR.m` at all 1,132,200
postinitial saved states; differences are below 4.4e-12 m³/s. The full-record
Python benchmark compares members 1, 50 and 100 with these saved outputs.
Ordinary tests carry a 54 KB fixture covering 1970–1971 for those three members,
including January/March/month changes. MATLAB was not rerun on this machine.

`anual.mat` corresponds to the rounded 2019/2020/2024 baseline coefficient
targets. `FA.mat` instead corresponds to the more precise 2018 version. The
benchmark respects that distinction. The annual artifact comparison preserves
MATLAB's first-365-rows-of-each-year averaging only to reproduce the saved file;
that historical policy misaligns dates after February in leap years.

Saved `amalgam_out.mat` `sim`/`F` differs from the actual-calendar evaluation
outputs. Reconstruction matches a leap-year-start calendar rather than the
1970-start data, indicating a calendar defect in the unavailable calibration
implementation. This is an inference from the saved outputs; the specific
leap start year cannot be established. These arrays are excluded from primary
simulation parity checks. The new calibration implementation uses the
observation calendar directly.

## Reproducible checks

```powershell
python -m pytest -q
python -m dztr.simulation.benchmark --matlab-folder "E:\1_DataHub\E_Reservoir_data\ex12_Trinity" --repo . --output outputs/trinity_simulation
```

The benchmark writes baseline and current-parameter simulations, linear/Hazen
monthly targets, metrics, errors and source hashes. Source MATLAB data are
needed for the full-record benchmark; portable tests only require this repo.
The [calibration/validation baseline](calibration_baseline.md) now estimates targets
from calibration observations and scores continuous held-out simulations.
The [NSGA-II pilot](optimization.md) adds calibration optimization, and the
[adaptive runner](adaptive_calibration.md) adds verified optimizer continuation
and calibration-only stopping. Hydrological-model coupling remains further work.
The [candidate and objective layer](calibration_objectives.md) now prepares
calibration-only bounds and verifies negative-NSE costs against saved evaluation
outputs, while preserving the distinction between historical and current bounds.

## Verified simulation performance

The daily runner now prepares each visited month's targets once per simulation
and reuses them in the solver, release diagnostic and zone classification.
Release equations, arithmetic order, overflow corrections and MATLAB fixed
iterations are unchanged. A fresh call observes parameter edits made between
simulations. Targets must remain unchanged during a simulation.

All 621 prior regressions, 51 exact-parity tests and 40 integration/recovery
tests passed. Sixteen saved Trinity/Gardiner candidate pairs matched in full
daily values and calibration scores; the selected replay benchmark used about
7.3 times less CPU. This ratio does not describe full optimization throughput.

The completed [population comparison](../outputs/trinity_population_study/population_results.md)
still uses its original source identity. Its checkpoints and results were
preserved when integrating the faster runner. New optimization runs use fresh
folders; old checkpoints cannot resume under changed source. The
[performance guide](simulation_performance.md) describes the independent
references, reproduction commands and verification records.

A fresh Trinity verification also passed after integration: a 128-trial stage
continued to 256 with exactly 128 new search trials, matching an uninterrupted
256-trial run in vectors, costs, survivor order and RNG state. Both reproduced
the original archive's first 256 trials exactly. The old checkpoint correctly
rejected continuation under changed source, and every original archive byte
remained unchanged. This verification performed 512 new search trials across
two searches, plus 12 calibration and 12 validation reporting runs; it did not
assess convergence or choose budgets. Its
[saved verification](../outputs/simulation_performance/integrated_search_check/verification.json)
records the exact source identity and separate accounting.
