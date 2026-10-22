# Calibration candidates, bounds and objectives

The calibration layer defines the search variables and objective evaluation
used by the [NSGA-II adapter](optimization.md). The review command on this page
does not run an optimizer.
The generalized baseline is the verified starting candidate.

## Candidate representation

`parameters_to_vector` and `parameters_from_vector` use six blocks of twelve
calendar months, preserving the MATLAB order:

| Zero-based positions | Quantity | Candidate unit |
| --- | --- | --- |
| 0–11 | Critical storage | Fraction of capacity |
| 12–23 | Normal storage | Fraction of capacity |
| 24–35 | Flood storage | Fraction of capacity |
| 36–47 | Critical release | m³/s |
| 48–59 | Normal release | m³/s |
| 60–71 | Flood release | m³/s |

For example, storage fraction 0.6 represents `0.6 * capacity_m3`. Decoding
produces the existing monthly SI target table; the simulator does not consume
a differently structured parameter file. The MATLAB-vector importer now uses
this same decoder.

Capacity, initial conditions, dead storage, inflow factor, channel capacity and
flood mode stay fixed in the template. Candidate values replace only monthly
storage/release targets. Inputs must contain exactly 72 finite real numbers.
Values are never silently clipped, sorted, flattened or repaired.

## Bounds and physical constraints

`estimate_calibration_bounds(observations, period, template)` selects the fitting
period internally. Spin-up and validation observations cannot supply percentile
limits. The baseline defaults to Hazen percentiles.

Both storage and release use the ranges in
[Yassin et al. (2019), Section 3.3](https://hess.copernicus.org/articles/23/3735/2019/):

| Target | Lower percentile | Upper percentile |
| --- | ---: | ---: |
| Critical | P5 | P35 |
| Normal | P35 | P75 |
| Flood | P75 | P95 |

The returned `CalibrationBounds.monthly_bounds` table retains these raw limits
in SI units. Its `lower_vector`/`upper_vector` properties convert storage limits
to fractions for the optimizer representation. Period, method, source and
percentile definitions are recorded in its summary.

Box membership and physical feasibility are separate:

- `bounds.contains(vector)` checks shape, finite values and inclusive percentile
  limits. It does not guarantee physical feasibility.
- `bounds.validate(vector, template)` also requires dead storage < critical
  storage < normal storage < flood storage ≤ capacity, and nonnegative critical
  release ≤ normal release ≤ flood release.
- Bounds construction rejects any month whose box contains no physically
  feasible target combination. Missing calibration months also raise errors.
- Adjacent percentile intervals share endpoints, so a vector can be inside the
  box while having equal storage targets. Such a candidate is rejected rather
  than assigned an arbitrary zone width.
- Release targets are not globally limited to channel capacity: the original
  DZTR equation applies that cap only in the emergency zone. A physically ordered
  candidate may still cause a missing/ambiguous implicit solution and be rejected
  during simulation.

The supplied 2018 AMALGAM metadata and older coefficient files use different
ranges: P5–25/P30–65/P70–95. Current MATLAB parameterization scripts and the
published ranges agree with the defaults above. Historical bounds remain
importable for reproduction; their estimation period is not assumed to be the
new calibration window.

## Objective evaluation

`CalibrationObjective` minimizes two costs in this fixed order:

```text
[-NSE(release), -NSE(storage)]
```

This matches the sign convention in `SimRISM14_eval.m` and the paper's Pareto
plots. Perfect NSE scores give costs `[-1, -1]`; lower costs are better.
The objectives remain separate rather than being combined with arbitrary weights.

The ordinary callable simulates only spin-up plus calibration and evaluates
only calibration dates. It neither requires future validation inflows nor uses
validation observations to choose parameters. `evaluate_validation` is a separate
explicit call: it reruns continuously from the same initial date through the
validation end and scores every validation day without resetting state.

Expected candidate failures return `CandidateEvaluation(feasible=False)` with
infinite costs and a reason. These include malformed vectors, percentile-box
violations, invalid physical ordering, missing/ambiguous simulation roots,
negative simulated storage and failed balance convergence. Static configuration
errors such as missing dates/inflows or nonconstant-observation requirements
raise before candidate evaluation. A finite release/storage NSE is required;
undefined auxiliary KGE values do not invalidate otherwise defined objectives.

The default uses implicit solving and conservative overflow handling. In MATLAB
overflow mode, balance validation removes only the documented `-excess/2`
discrepancy before checking convergence; selecting compatibility mode does not
permit unconverged iterations. Original candidate penalty conventions cannot
be recovered because the `SimRISM12` optimizer wrapper is absent, so the explicit
rejection policy is a choice in this port.

Observations, the template and bound tables are copied when a problem is created,
making repeated evaluations independent of later caller edits. The fixed 0.92
Trinity inflow factor retains its earlier provenance qualification described
in [the baseline notes](calibration_baseline.md).

## Example API

```python
from dztr.calibration import CalibrationObjective, estimate_calibration_bounds, parameters_to_vector

# observations: normalized source record
# split: CalibrationValidationSplit
# template: calibration-only generalized ReservoirParameters
bounds = estimate_calibration_bounds(observations, split.calibration, template)
problem = CalibrationObjective(
    observations, split.calibration, template,
    warmup_steps=split.warmup_steps, bounds=bounds,
)
starting_vector = parameters_to_vector(template)
costs = problem(starting_vector)
calibration = problem.evaluate(starting_vector)
validation = problem.evaluate_validation(starting_vector, split.validation)
```

For optimizer use, supply calibration-derived bounds. `bounds=None` supports
physical-only evaluation of historical references; it does not waive ordering
or reservoir-capacity constraints.

## Reproducible review

From the installed repository:

```powershell
python -m dztr.calibration.review --repo . --output outputs/trinity_calibration
```

To also reproduce full-record saved MATLAB objectives:

```powershell
python -m pip install -e ".[matlab]"
python -m dztr.calibration.review --repo . --output outputs/trinity_calibration --matlab-folder "E:\1_DataHub\E_Reservoir_data\ex12_Trinity"
```

`examples/review_trinity_calibration.py` accepts the same flags. The optional
MATLAB check uses its original physical settings and the actual-calendar
`flowSIM_eval.mat`, `stoSIM_eval.mat` and `FA_eval.mat` outputs. It excludes
the previously identified calendar-shifted `amalgam_out.mat` `F/sim` arrays.
MATLAB itself is not rerun. Ordinary objective tests use the portable reference
fixture and need no E drive or SciPy.

The output folder contains:

| Artifact | Contents |
| --- | --- |
| `bounds_si.csv` | Twelve monthly lower/upper storage and release limits |
| `candidate_space.csv` | 72 named variables, months, units, bounds and starting values |
| `baseline_targets_si.csv` | Calibration-only generalized monthly targets |
| `candidate_vectors.csv` | Baseline and optional historical reference vectors |
| `candidate_checks.csv` | Box/physical validity and reference roles |
| `reference_checks.csv` | Optional historical objective and trajectory errors |
| `summary.json` | Signed objectives, scores, source hashes and provenance |
| `resolved_config.json` | Settings used for the review |

The baseline costs are `[-0.335349, -0.606841]` in calibration and
`[-0.099452, -0.348675]` in validation. Saved MATLAB members 1, 50 and 100 match
over 11,323 simulated days, with maximum signed-objective discrepancy below
3 × 10⁻¹⁵. All three historical vectors fall outside the current calibration-only
percentile box. They establish numerical reproduction, not new optimizer results
or independent validation performance under this split.

For a monthly bounds figure:

```powershell
python -m pip install -e ".[plots]"
python examples/plot_calibration_bounds.py
```

## Optimization

The [NSGA-II pilot](optimization.md) adds seeded initialization, normalized
variation, an explicit evaluation budget, rejected-candidate accounting and
saved Pareto solutions. It searches calibration costs only, then scores selected
solutions on validation without using validation scores for search or selection.
