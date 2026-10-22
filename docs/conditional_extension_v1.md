# Conditional extension experiments

This batch retains all 20 additions remaining after Hungry Horse. Two have
admitted exploratory scenarios: Hugh Keenleyside and Revelstoke. The other five
paired records and 13 weaker-data records keep their individual source gates.
This is a numerical route for declared scenarios, separate from the observation
availability routes used to estimate targets.

The scenario file is `configs/conditional_extension_cohort_v1_20261021.json`.
It records source paths, dates, assumed capacity and dead storage, nominal flow
clock, splitting method, seeds, weights and the decision for every addition.
Targets and percentile bounds are fitted from supported calibration samples
only, using Hazen percentiles. Channel capacity uses calibration Q99 as a
benchmark proxy, rather than a verified engineering limit. Raw source records
are retained without clipping, datum shifts or capacity inflation.

The forcing is `Q(t) + [S(t)-S(t-1)]/86400`. It includes assessment responses.
Consequently an assessment period here is a **conditional consistency test**,
not independent validation. Direct fitting views hide assessment responses;
that does not remove their indirect contribution through the forcing. Changing
assessment responses and reconstructing forcing changes the experiment.

Hugh Keenleyside uses complete mixed years with dry/normal/wet classes based
on reconstructed annual net-flux means. These classes are proxies. Revelstoke
has only about 20 months of storage; it uses an explicitly separate chronological
split with all 12 fitting months and at least 65% of the whole calendar actually
scored for fitting. Its later assessment does not cover a complete seasonal
cycle. Never describe it as the standard mixed-regime benchmark.

Simulation uses the existing daily interval equations, nearest-previous-storage
root selection and a 0.0001 m3 balance tolerance. Every finite forcing block
starts from frozen calibration-only monthly normal storage. Thirty days after
each start are excluded from fitting/assessment scores. Missing forcing remains
missing. This first adapter admits only nonnegative reconstructed flux;
signed-flux cases need their separate source decision and numerical adapter.
Representative trajectories must match the canonical computed-net-inflow
simulator exactly before publication.

From the repository root in PowerShell:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/calibrate_conditional_cohort_v1.py run --config configs/conditional_extension_cohort_v1_20261021.json --output outputs/conditional_extension_cohort_v1_20261021
```

The run manager launches three independent jobs at a time, persists logs and
checkpoints every 16 evaluations, and freezes the minimum calibration-cost seed
before calculating assessment metrics. DDS uses weights 0.5 storage and 0.5
release, seeds 42/43/44 and 512 total evaluations per seed. Jobs can resume at
the same horizon without changing the proposal schedule. Changing the horizon,
bounds, source code, records or source decisions requires a new experiment;
the current 512 horizon cannot silently become 1024.

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/calibrate_conditional_cohort_v1.py verify --output outputs/conditional_extension_cohort_v1_20261021
python -B tools/calibrate_conditional_cohort_v1.py replay --output outputs/conditional_extension_cohort_v1_20261021 --reservoir-id etienne_Hugh_Keenleyside
```

Use this replay adapter for these profiles. The native endpoint-flow application
adapter has different timing semantics. Profiles use the shared parameter
registry and retain a distinct `conditional_response_net_daily_v1` route.
Unverified GRAND candidates are not registry keys; both profiles are loaded by
reservoir ID. The collection v4 index adds scenario profiles without changing
the original observation entries or Hungry Horse and paper profiles.

Completed results are in
`outputs/conditional_extension_cohort_v1_20261021/report/README.md` and
`outputs/conditional_extension_completion_v1_20261021/README.md`, including
per-seed comparisons, plots, selection coverage, seasonal/zone/peak diagnostics
and the consolidated table of all 62 reservoirs.
