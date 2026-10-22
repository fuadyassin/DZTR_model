# Calibrated paper cohort: assessment and application

The cohort assessment joins the 32 completed paper profiles while preserving each
frozen registry, source treatment, native sampling, root policy and year mask.
Bull Lake's conditional interval study and four source-gated cases remain in the
37-row inventory. Their ordinary native monthly targets are not fabricated.

From the repository in PowerShell:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/assess_calibrated_cohort_v1.py --output outputs/cohort_report_new --diagnostic-cache outputs/cohort_diagnostic_cache_v1_20261017
```

Use a fresh report directory. The versioned diagnostic cache contains scientific
trajectories and identity receipts, so it stays in `outputs`, separate from tool
caches. Existing trajectories are reused. A fixed-profile replay is required only
where an archived held-out trajectory stops before later calibration support or
an initializer assessment is absent. It cannot change a selected parameter or
source treatment. No search is run by this command.

The report provides a 32-row summary, all 96 seed checkpoints and 192 seed/phase
comparison rows, monthly and simulated-zone diagnostics, regime support, release
peak bias, within-contiguous-block storage error trends, and standalone PNG/PDF
gain plots. Initial-reference failures remain visible with blank baseline NSE
and gains. Feasible initializers are not relabeled as generalized estimates.

Screening thresholds are descriptive and post-fit. Regime/zone NSE can have low
within-group variance; short assessment periods limit drift and seasonal
interpretation. Prefix hypervolume gain and seed-front agreement do not establish
global or parameter convergence. Held-out evidence never chooses a replacement
seed, treatment, calibration mask or parameter vector.

The generated `catalog.json` is the entry point for application:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B examples/simulate_profile.py --catalog outputs/cohort_report_new/catalog.json --grand-id 753 --output outputs/garrison_application_new
python -B examples/simulate_profile.py --catalog outputs/cohort_report_new/catalog.json --reservoir-id ex40_Semione --output outputs/seminoe_application_new
```

These examples use the original prepared forcing and frozen initial states. The
GRAND lookup uses verified mappings only. A missing GRAND ID requires the exact
reservoir ID. Standard profiles use the existing native root-policy solver;
signed-native profiles use their explicit `SignedNetPolicy`. Neither route
creates roots, clips signed forcing or reads observed responses to initialize or
reset state. Bull Lake and source-gated records are refused by this native route.

For an external SI CSV with `date` and `inflow_m3s`, declare its preprocessing
stage and, for a different initial date, both initial states:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B examples/simulate_profile.py --catalog outputs/cohort_report_new/catalog.json --reservoir-id ex30_Big_horn --forcing path/to/inflow.csv --forcing-stage canonical --initial-storage-m3 600000000 --initial-release-m3s 50 --output outputs/big_horn_application_new
```

The illustrative initial states above are user inputs, not recommended values.
`canonical` means before the declared source transform; `prepared` means already
transformed. The transform is applied once and remains separate from the physical
`inflow_correction`. Input must match the profile's daily/monthly clock and SI
units; missing timesteps/forcing are rejected. SI output, parameter targets and
an independent balance/provenance receipt are written to a fresh directory.

Profile/catalog checksums, source hashes and prepared-input identities are
verified on load. These preserve computational reproducibility, not independent
physical certification. Scientific source changes require a new profile version.
