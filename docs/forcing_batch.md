# Supplied-forcing batch workflow, version 1.1

Run the twenty calibrated reservoir profiles on new inflow records using a JSON batch manifest. Profiles retain their calibrated monthly targets, physical constants, root-selection policies and integration clocks. Initial storage and release are supplied explicitly for the first CSV date. No optimization, parameter fitting or observation-based state correction occurs.

From the repository root:

```powershell
python -B tools/forcing_batch.py examples/forcing_batch/gardiner_synthetic.json --output outputs/gardiner_synthetic_run
python -B tools/forcing_batch.py path/to/my_batch.json --output outputs/my_forcing_batch
```

Use a new output directory under `outputs`. The [Gardiner example](../examples/forcing_batch/gardiner_synthetic.json) contains invented future inflows for demonstrating the interface; it is not a forecast.

## Inputs

The manifest has `schema_version: 1`, `kind: "dztr_forcing_batch"` and a nonempty `jobs` list. Each job declares:

| Field | Meaning |
| --- | --- |
| `job_id` | Unique path-safe name for this reservoir/scenario run. Separate jobs may use the same reservoir with different forcing. |
| `grand_id` or `reservoir_id` | Exactly one identifier from the twenty-profile registry. Fort Randall uses `ex4_Fort_randall`. |
| `forcing.csv` | CSV path relative to the manifest directory, or an absolute path. |
| `forcing.columns` | Map `date`, `inflow`, and optional observed `storage`/`release` to CSV column names. |
| `forcing.units` | One unit for every mapped measurement: flows `m3/s` or `m3/day`; storage `m3` or `Mm3`. Flows are rates, not interval volumes. |
| `forcing.stage` | `raw_source` applies the reservoir's inherited source factor/offset once. `model_input` means those adjustments have already been applied, or the supplied record is directly expressed as model inflow. The separate physical `inflow_correction` is always retained. |
| `initial` | `storage`, `storage_unit`, `release`, `release_unit`; values apply at the first simulated timestamp. |
| `sampling` | Copy the selected profile's entire sampling object. Different clocks require separate scientific assessment and a new profile version. |
| `period` | Optional exact CSV `start`/`end` date boundaries to simulate. |
| `evaluation_periods` | Optional named objects containing exact `start`/`end` boundaries for observation-based metrics. |

Dates must be sorted, unique timezone-naive calendar dates at midnight, with consecutive daily samples or consecutive month-start samples. The runner does not interpolate missing intervals, infer units or clip negative inflows. At least two timestamps are required. Adjusted inflows must be finite and nonnegative. Missing observed storage/release is allowed; infinities and negative observations are rejected.

The existing trapezoidal solver uses adjacent supplied inflow rates and releases. A CSV's first row supplies the initial inflow and boundary state; the first integration step ends at the second row. Monthly fixed 30.5-day profiles retain that integration duration even though actual months differ in length. Supplying monthly averages adopts the inherited approximation; the interface does not turn them into independently verified endpoint flows.

Observed storage and release are optional and affect metrics only. With observations and no `evaluation_periods`, the runner assesses all rows after the initial boundary row. Named periods include precisely their specified dates. Missing pairs are omitted independently per variable; undefined metrics become JSON `null`, with pair counts recorded. No observations means no performance metrics.

## Outputs and checks

Each job writes its request, normalized SI input, exact monthly SI parameters, simulation, metrics and result JSON. The simulation includes storage, total release, the existing conservative overflow-volume diagnostic, DZTR zone, integration duration, root-selection diagnostics and water-balance residuals. Release already includes the conservative spill adjustment; overflow volume must not be added to release again.

The runner independently reconstructs trapezoidal mass balance and fails a job if its inherited absolute tolerance is exceeded. A solver failure records its date and reason through the numerical solver's error. Other jobs continue. The batch exits unsuccessfully if any job fails, and records all statuses in JSON/CSV summaries. Its execution manifest pins the submitted batch, supplied CSV bytes, consumed profile/model files and outputs. Files remain in the repository; there is no external-source dependency for the twenty historical profiles.

The [historical verification release](../outputs/forcing_workflow_v1_1_20261012/README.md) runs all twenty through this interface and checks exact trajectories and calibration/held-out NSE. It does not perform new optimization.

Successful numerical checks do not establish predictive accuracy. The existing held-out diagnostics and unresolved source questions still apply. Simulating later periods assumes the calibrated operating rules remain applicable; these profiles retain their exploratory status.
