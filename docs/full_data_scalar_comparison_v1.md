# Full-data scalar calibration: GA and DDS

Use this workflow when inflow, storage and release observations are available. It fits all 72 monthly DZTR targets: three storage and three release targets for each month. Capacity, dead-storage fraction, channel capacity, initial conditions, forcing adjustments and solver settings remain fixed.

The supplied Gardiner experiment compares two algorithms and two objectives, using seeds 42, 43 and 44 and 512 evaluations per search: 12 searches, 6,144 evaluations. GA uses population 32; DDS uses perturbation 0.2 and a frozen 512-evaluation horizon. Rejected candidates consume evaluations. The existing empirical initialization is included in every search's budget.

## Set the objective weights

Edit `configs/full_data_ga_dds_gardiner_v1.json`:

```json
"objectives": [
  {"id": "release_only", "weights": {"storage": 0, "release": 1}},
  {"id": "weighted_joint", "weights": {"storage": 0.5, "release": 0.5}}
]
```

The minimized scalar cost is

`L = w_storage × (1 − NSE_storage) + w_release × (1 − NSE_release)`.

Weights are finite, nonnegative and normalized to sum to one; at least one must be positive. `{"storage": 2, "release": 1}` assigns two-thirds to storage. NSE is dimensionless, so m³ and m³/s do not directly determine this balance. Equal weights express an application preference, not proof that both responses are equally reliable. Raw release NSE emphasizes large errors; square-root release NSE is available in the separate flow-only workflow, not this experiment.

In **this full-data release-only arm**, storage observations still determine storage-target initialization and bounds. The optimizer can adjust all 72 targets. Storage NSE is omitted from the loss when its weight is zero. This differs from `flow_only`, which hides storage and fixes assumed storage fractions. The two objectives here share the same parameter freedom, allowing a controlled comparison of their loss functions.

`"scoring": "pooled"` computes each response's NSE over all calibration dates. Optional `"equal_regime"` first averages its dry/normal/wet calibration NSEs. Regime weighting and storage/release weighting are separate decisions. A positive-weight response must have a valid NSE in every active scoring group.

## Run through the gateway

From the repository in PowerShell:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/dztr_run_v10.py plan --config configs/run_manager_full_data_ga_dds_gardiner_v1.json
python -B tools/dztr_run_v10.py run --config configs/run_manager_full_data_ga_dds_gardiner_v1.json --output outputs/managed_full_data_ga_dds_gardiner_v1_20261020_final
```

Use the project's installed Python executable if `python` points elsewhere. Add `--resume` to reuse an interrupted or completed managed run. Older configuration versions continue through the earlier gateway implementation.

The calibration stage exports a campaign, combined GRAND-indexed parameter table, report and, for a single reservoir, prepared forcing. Its `registry` output identifies the **first declared objective and first declared algorithm**. All four registries are available separately under `campaign/releases/budget_512/<objective>/<algorithm>/registry.json`; select the required registry explicitly for application.

## Standalone jobs and continuation

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/compare_full_data_v1.py run --config configs/full_data_ga_dds_gardiner_v1.json --output outputs/full_data_ga_dds_gardiner_v1 --evaluation-budget 256
python -B tools/compare_full_data_v1.py run --config configs/full_data_ga_dds_gardiner_v1.json --output outputs/full_data_ga_dds_gardiner_v1 --evaluation-budget 512
```

Checkpoints persist per reservoir, objective, algorithm and seed. They include protocol and runtime identities, normalized weights, RNG state, optimizer settings and trial history. GA also preserves its population and pending proposals. Continuation within the frozen horizon reproduces a fresh run exactly, including partial generations. Changing weights, bounds, source code or seeds requires a new campaign. Changing DDS's horizon changes its perturbation schedule and is a new experiment, rather than exact continuation. A completed manager stage is sealed: continue a standalone campaign or fork it before increasing its budget.

Preparation reuses the existing observed-monthly estimators and bounds; it runs no NSGA-II searches. It requires a ready native observed-monthly source, Hazen percentiles and the existing strict empirical bounds. Older profiles that omit the route label may be explicitly identified as standard only after checking finite nonnegative forcing, unsigned sampling and absence of a signed-forcing adapter; that decision is recorded in the handoff. Signed-inflow, interval and unresolved source cases are not automatically admitted. The previous completed reservoir calibrations and exports remain separate.

## Scientific checks and outputs

The existing mixed-year selection is preserved, with at least 65% calibration rows. Monthly initial targets and bounds use calibration responses only. Optimization objects discard all noncalibration responses. State is simulated continuously using forcing through the final calibration date, without observed-state resets. Initial state and source-derived forcing remain explicitly inherited conditions.

Each algorithm/objective selects its best vector per seed and then its best seed using **calibration loss only**, with ties broken by seed and evaluation ID. Assessment data are read after this decision is frozen. Full-period failures remain visible and cannot select another seed. Every reported trajectory receives the existing independent water-balance and capacity checks.

The completed release contains:

- `README.md` and `figures/`: comparison summary, seed convergence, assessment scatter and selected trajectories.
- `baseline_comparison.csv`: both periods, all seeds, S/Q NSE, baseline improvements, biases and paired observation counts.
- `paired_seed_comparison.csv`: GA minus DDS for matching objective/seed/period.
- `convergence.csv`: running best calibration loss at every evaluated candidate.
- `search_status.csv` and `scientific_checks.csv`: feasibility counts, failures and independent balance checks.
- `selection.json`: calibration-only selection rule and all chosen vectors.
- `selected_parameterization_by_GRAND_ID.csv`: selected profiles across all arms, identified by reservoir, GRAND ID, objective and optimizer. Missing verified GRAND IDs remain blank.
- `<objective>/<algorithm>/registry.json`: applicable parameter profiles and their provenance.
- `trajectories/`: full trajectories for every seed, including those not exported as the selected seed.

A fixed budget and three seeds do not establish convergence or a universal optimizer ranking. Use the comparison to assess this case, and choose operational weights from scientific/application priorities rather than held-out performance. The experiment is conditional on the prepared forcing and initial conditions; it is not an independent ungauged test.
