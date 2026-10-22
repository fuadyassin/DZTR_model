# Calibration and validation baseline

The baseline estimates generalized DZTR parameters from calibration observations
and evaluates one continuous simulation on separate calibration and validation
periods. It establishes the reference that a later optimizer must improve.
No optimization is performed in this stage.

## Trinity periods and their provenance

[Yassin et al. (2019), Section 3.6](https://hess.copernicus.org/articles/23/3735/2019/)
uses the first simulation year for spin-up, then the first half of the remaining
record for calibration and the second half for validation. For the repository's
1970–2000 Trinity record, the initial configuration interprets that protocol as:

| Purpose | Inclusive dates | Daily rows |
| --- | --- | ---: |
| Spin-up | 1970-01-01 to 1970-12-31 | 365 |
| Calibration: parameter estimation and scoring | 1971-01-01 to 1985-12-31 | 5,479 |
| Validation: scoring with fixed parameters | 1986-01-01 to 2000-12-31 | 5,479 |

These Trinity-specific dates are our calendar-year interpretation of the paper's
protocol; the text does not state the exact Trinity split dates. All selections
are explicit in `configs/trinity_baseline.json` and can be changed there.

The paper does not explicitly establish which period supplied the CDFs for its
split-sample generalized parameters. This implementation explicitly excludes
spin-up and validation storage/release observations from percentile estimation.
It does not claim that this was the original study's estimation policy.

## Parameter estimation and simulation

- Monthly storage/release P10/P45/P85 and the reported P99 are estimated from
  calibration observations only. `hazen` is the baseline default, matching the
  historical MATLAB percentile convention. Existing general estimation and
  workbook APIs retain their `linear` default.
- Channel capacity is fixed at Trinity's known 300 m³/s. If omitted or set to
  null in a custom configuration, it is inferred from calibration release P99.
- Capacity, initial storage/release, dead-storage fraction, flood mode and the
  0.92 inflow factor retain the verified MATLAB configuration. Their values are
  fixed assumptions here. The provenance of the factor's original tuning is
  unknown; exclusion of validation observations applies to our new target
  estimation, rather than proving every inherited setting independent of them.
- The simulation starts `warmup_steps` calendar days before calibration begins.
  The configured initial state applies on that first simulated date. Changing
  the calibration start also changes this initial-state date; review the
  configuration when using a different period or reservoir.
- A single continuous run covers spin-up, calibration and validation. Validation
  inherits simulated storage and release. It does not restart from observations,
  and none of its days are discarded as an additional warmup.
- If periods have a gap, its daily inflows are simulated to carry state forward,
  but those dates are neither fitted nor scored. Every requested simulation date
  and finite inflow must exist; dates are never fabricated or silently shortened.
- Daily scoring excludes missing storage/release pairs independently and reports
  paired sample counts. Undefined metrics become JSON null in saved artifacts.

The regular baseline uses the implicit solver and conservative capacity-overflow
handling. Every day includes balance and release-rule diagnostics. Numerical
compatibility options remain explicit as described in [the MATLAB port notes](matlab_port.md).

## Run and review

After installation, from the repository root:

```powershell
python -m dztr.calibration.benchmark --repo . --output outputs/trinity_baseline
```

Alternatively run `python examples/calibration_baseline_trinity.py`. A custom
settings file can be selected with `--config path/to/settings.json`.

The output folder contains:

| Artifact | Contents |
| --- | --- |
| `daily_simulation.csv` | Simulated and observed storage/release, period labels and diagnostics |
| `monthly_targets_si.csv` | Calibration-only monthly targets in SI units |
| `calibration_quantiles_si.csv` | Calibration quantiles with variable/month sample counts |
| `metrics.csv` | Separate calibration and validation scores |
| `summary.json` | Periods, scores, paired counts, balance diagnostics and provenance |
| `resolved_config.json` | Complete settings used for the run |

Source/configuration hashes and Python/package versions are recorded in the
summary. Outputs are ignored by Git; source observations remain unchanged.

For a figure of the whole experiment:

```powershell
python -m pip install -e ".[plots]"
python examples/plot_calibration_baseline.py
```

The figure displays monthly means for readability; all metrics use daily values.

## Initial baseline results

For the included settings, the 11,323-day run gives:

| Period | Release NSE | Release KGE | Storage NSE | Storage KGE |
| --- | ---: | ---: | ---: | ---: |
| Calibration | 0.3353 | 0.6342 | 0.6068 | 0.5653 |
| Validation | 0.0995 | 0.5427 | 0.3487 | 0.4805 |

Both variables have 5,479 scored pairs in each period. No capacity overflow
occurs; the largest absolute daily balance residual is approximately
8.54 × 10⁻⁷ m³. These scores use new calibration-only parameters and different
periods from the previous whole-record MATLAB reproduction, so they are not
directly comparable to that report as an optimizer performance claim.

## Next stage

The [candidate and objective layer](calibration_objectives.md) now provides a
constrained 72-value representation, calibration-derived bounds and verified
release/storage objective evaluation. The [NSGA-II pilot](optimization.md) now
adds a budgeted search and calibration-only representative selection. Historical
MATLAB bounds remain available for reproduction;
their estimation period must not be assumed to match this split.
