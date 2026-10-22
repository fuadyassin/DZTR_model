# Daily interval volumes and segmented conditional evaluation

The [interval preparation contract](interval_inputs.md) supplies explicit signed or physical daily mean volumes and source-quality masks. The new `dztr.simulation.intervals` and `dztr.calibration.intervals` modules consume that contract through separate APIs. Existing MATLAB replay, root-policy, continuous calibration, and substep defaults retain their established behavior.

For a nominal daily interval ending at row i, the new balance is:

`S[i] = S[i−1] + Vin[i] − 0.5 × dt × (Qendpoint[i−1] + Qendpoint[i]) − Vspill[i]`

`Vin[i]` uses the current reported daily mean times 86,400 seconds, with the explicitly permitted scale. The solver never averages adjacent observed daily means. Its numerical approximation is one trapezoid of **modeled endpoint operating release**, evaluated from the literal monthly DZTR release law. This is an explicit new input/integration convention, not exact reproduction of the original MATLAB daily map.

Computed net inflow requires the storage-only flood rule and inflow scale one. A negative volume retains its reported sign without assigning a physical loss cause. Physical inflow retains its nonnegative interpretation; an inflow-dependent rule uses the current mean as its declared operating driver. Neither input mode invents an additional loss process or clips a signed deficit to zero.

## Roots, storage limits, and spill

Each linear DZTR branch supplies a candidate, checked against the actual rule and water balance. The solver retains physically bounded roots and adds a capacity-bound candidate with nonnegative spill when feasible. It considers all physical candidates before selecting a root. The default rejects multiple roots; minimum storage, maximum storage, and nearest previous storage require explicit selection. Missing roots, negative predictors, invalid initial states, and nonfinite arithmetic remain failures.

Spill is an explicit interval volume at capacity. The final endpoint operating release still equals the rule evaluated at the final physical storage. Spill does not become an extra operating endpoint rate carried into the next interval. The outputs distinguish:

| Output | Interpretation |
| --- | --- |
| `release_endpoint_m3s` | Rule-consistent operating release at the stored endpoint |
| `integrated_release_volume_m3` | Trapezoidal operating-release volume over the interval |
| `interval_mean_release_m3s` | Mean operating release, excluding explicit spill |
| `spill_volume_m3` | Explicit overflow volume over the interval |
| `interval_mean_total_outflow_m3s` | Mean operating release plus spill volume divided by duration |
| `integrated_inflow_volume_m3` | The declared current interval's signed or physical input volume |
| `balance_residual_m3` | Endpoint storage change minus input plus operating release plus spill volumes |

Total mean outflow is the comparison variable for Bull Lake's observed daily mean total discharge. A different measurement location or discharge definition would need its own mapping. Capacity spill enforces the model's storage bound; it does not recover an observed overflow route or an independent source release record.

## Provisional block initialization

The preparation retains every source row. Quarantined storage rows remain outside simulated blocks. At the first clean row of each block, the simulator uses observed storage plus the declared initialization offset and evaluates endpoint release from the operating rule. It does not use observed daily-mean release as an endpoint rate. Initial storage must remain within physical bounds; it is never silently clipped.

Initial rows have no scored interval-mean release and cannot earn a storage prediction score. Clean phase boundaries carry simulated storage and endpoint release; calibration and validation labels alone do not restart the model. Month selection follows preserved destination-row calendar months, including the rule evaluation at each provisional anchor. The release law's timestep-dependent limits retain the nominal daily timestep.

The simulator revalidates policy-derived structure, volumes, and inputs used within the requested horizon. A shorter calibration horizon does not inspect held-out forcing or score values. A predeclared quarantine immediately after the horizon can still invalidate the terminal storage endpoint; this is source-quality metadata, not validation-performance selection. A failed step retains an explicitly incomplete trajectory for diagnosis and rejects the candidate. Failure does not trigger a skipped interval or an unannounced observed-state reset.

## Fitting and scoring

`fit_interval_targets(preparation, calibration_period, ...)` estimates existing monthly targets from calibration dates only. Storage uses clean storage rows; release uses eligible mean-flow intervals. Variables have independent exclusion masks. Inflow is omitted from the target-estimation view because the six targets use storage and release, and a signed net forcing is not an empirical nonnegative inflow sample.

`estimate_interval_bounds(preparation, calibration_period, template, ...)` estimates the existing P5–35, P35–75, and P75–95 target bounds from those same masked calibration views. The empirical box and reservoir physics remain separate checks. Validation observations do not enter target or bound estimation.

`IntervalCalibrationObjective` supplies the existing optimizer interface: `evaluate(vector)` returns feasibility, two minimization costs, metrics, paired counts, and a trajectory. It decodes the existing 72-value monthly target vector. Calibration evaluates only its warmup/calibration prefix; held-out evaluation requires a separate `evaluate_validation(vector, validation_period)` call.

Storage uses modeled endpoint storage; release uses modeled interval mean **total** outflow. Eligible pairs are pooled within each phase before calculating negative NSE costs. Block NSE values are never averaged. Initialization rows, quarantine intervals, and warmup are excluded from scored pairs. Root or balance failure rejects the complete candidate rather than scoring a surviving prefix. Undefined objectives remain explicit infeasibility.

Validation with observed storage restarts is a **segmented conditional hindcast**. It cannot establish continuous free-running prediction. Computed source inflow is also dependent on storage/release accounting, so fit diagnostics do not independently validate those measurements.

## Bull Lake baseline protocol

The [selected source policy](../configs/reviews/bull_lake_interval_policy.json) and [baseline protocol](../configs/reviews/bull_lake_interval_baseline.json) retain their separately dated decisions. The baseline uses strict root rejection and three prespecified initial-storage cases: nominal observed anchors and offsets of ±1% of capacity. All cases share the same masked calibration-derived targets. The offsets are robustness stresses, not measurement confidence limits or scenarios chosen by validation performance.

The [baseline report](../outputs/bull_lake_interval_baseline_20261009/README.md) records feasibility, conditional scores, root/spill diagnostics, and preservation hashes. It contains zero optimization evaluations. The stage 4 policy retains its original readiness snapshot; later readiness is reported by the subsequent experiment, not by rewriting the earlier record.

The [example](../examples/simulate_interval_policy.py) runs the same nominal conditional baseline from the repository root:

```powershell
python -B examples/simulate_interval_policy.py --output outputs/bull_lake_interval_example
```

Choose a fresh output directory; the example rejects existing directories. It writes masked targets and bounds, sample support with explicit month labels, both trajectories, and a feasibility/score summary. It performs no optimization. Its default source policy is Bull Lake's explicit computed-net interpretation; using another reservoir requires an appropriate source policy rather than inheriting that interpretation automatically.
