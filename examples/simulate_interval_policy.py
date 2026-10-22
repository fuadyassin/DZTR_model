"""Run a reviewed daily interval policy in a fresh output directory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dztr.calibration.baseline import json_ready
from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.intervals import IntervalCalibrationObjective, estimate_interval_bounds, fit_interval_targets
from dztr.calibration.periods import DatePeriod
from dztr.data.intervals import DailyIntervalPolicy, prepare_daily_intervals
from dztr.schemas import ReservoirParameters
from dztr.simulation.intervals import IntervalSimulationSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/reviews/bull_lake_interval_policy.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/bull_lake_interval_example")
    parser.add_argument("--initial-offset", type=float, default=0.0, help="Capacity fraction added to each provisional observed storage anchor.")
    parser.add_argument("--root-policy", choices=("reject", "minimum_storage", "maximum_storage", "nearest_previous_storage"), default="reject")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output directory; existing results are preserved.")
    policy_path = args.policy.resolve()
    policy_json = json.loads(policy_path.read_text(encoding="utf-8"))
    temporal = policy_json["interval_inputs"]
    if temporal["flow_alignment"] != "interval_mean_end" or temporal["nominal_step_seconds"] != 86400:
        parser.error("This example supports declared end-labelled nominal daily means.")
    config_path = (policy_path.parent / policy_json["reservoir_config_file"]).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["sampling"]["frequency"] != "daily":
        parser.error("This example requires a reviewed daily-mean policy.")
    preprocessing = config.get("inflow_preprocessing", {})
    if preprocessing.get("factor", 1) != 1 or preprocessing.get("offset_m3s", 0) != 0:
        parser.error("Review and implement the configured forcing preprocessing before using this example.")
    source_path = (policy_path.parent / policy_json["observations_file"]).resolve()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_hash != policy_json["observations_sha256"]:
        parser.error("Observation file does not match the reviewed policy hash.")
    source = pd.read_csv(source_path, parse_dates=["date"], index_col="date", float_precision="round_trip")
    cal = DatePeriod(**config["periods"]["calibration"])
    val = DatePeriod(**config["periods"]["validation"])
    labels = pd.Series("warmup", index=source.index)
    labels.loc[cal.start:cal.end] = "calibration"
    labels.loc[val.start:val.end] = "validation"
    policy = DailyIntervalPolicy(flow_semantics=policy_json["interval_inputs"]["flow_semantics"],
                                quarantined_interval_end_dates=policy_json["quarantine"]["interval_end_dates"])
    prepared = prepare_daily_intervals(source, policy=policy, period_labels=labels)
    fit = fit_interval_targets(prepared, cal, channel_capacity_m3s=config["model"]["channel_capacity_m3s"],
            target_percentiles=tuple(config["target_percentiles"]), quantile_method=config["quantile_method"])
    params = ReservoirParameters.from_generalized(fit.estimates, **{
        k: config["model"][k] for k in ("capacity_m3", "initial_storage_m3", "initial_release_m3s",
                                       "dead_storage_fraction", "inflow_correction", "flood_mode")})
    bounds = estimate_interval_bounds(prepared, cal, params, quantile_method=config["quantile_method"])
    settings = IntervalSimulationSettings(root_policy=args.root_policy,
                                          initial_storage_offset_capacity_fraction=args.initial_offset)
    objective = IntervalCalibrationObjective(prepared, cal, params, settings=settings, bounds=bounds)
    vector = parameters_to_vector(params)
    evaluations = {"calibration": objective.evaluate(vector), "validation": objective.evaluate_validation(vector, val)}
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
    output.mkdir(parents=True, exist_ok=False)
    fit.estimates.monthly_targets.to_csv(output / "monthly_targets_si.csv")
    fit.support.to_csv(output / "target_sample_support.csv", index_label="month")
    bounds.monthly_bounds.to_csv(output / "monthly_bounds_si.csv")
    summary = {"policy_id": policy_json["policy_id"], "source_sha256": source_hash,
               "validation_label": "segmented conditional hindcast", "root_policy": args.root_policy,
               "initial_offset_capacity_fraction": args.initial_offset, "optimization_evaluations": 0, "evaluations": {}}
    for phase, evaluation in evaluations.items():
        summary["evaluations"][phase] = {"feasible": evaluation.feasible, "reason": evaluation.reason,
                                        "metrics": evaluation.metrics, "paired_counts": evaluation.paired_counts}
        if evaluation.simulation is not None:
            evaluation.simulation.frame.to_csv(output / f"{phase}_simulation.csv", index_label="date")
    contents = json.dumps(json_ready(summary), indent=2, allow_nan=False) + "\n"
    (output / "summary.json").write_text(contents, encoding="utf-8")
    print(contents)


if __name__ == "__main__":
    main()
