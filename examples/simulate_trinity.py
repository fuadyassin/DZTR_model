"""Run Trinity from current parameterization after pip install -e ."""

import argparse
import json
from pathlib import Path

import pandas as pd

from dztr.data import read_reservoir_observations
from dztr.evaluation import evaluate_simulation
from dztr.parameterization import estimate_generalized_parameters
from dztr.schemas import ReservoirParameters
from dztr.simulation import simulate_reservoir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantile-method", choices=("linear", "hazen"), default="linear")
    parser.add_argument("--output", type=Path, default=Path("outputs/trinity_current"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root / "configs/trinity_simulation.json").read_text(encoding="utf-8"))
    catalog = pd.read_excel(root / "GRanD_Dams_v1_1_DZTR.xlsx").set_index("GRAND_ID").loc[131]
    observations = read_reservoir_observations(root / "ResDataInflowStorageRelease/Trinity_131.xlsx")
    estimates = estimate_generalized_parameters(observations,
        channel_capacity_m3s=float(catalog["Q_DS_MAX"]), quantile_method=args.quantile_method)
    parameters = ReservoirParameters.from_generalized(estimates,
        capacity_m3=float(catalog["SMAX"]),  # Verified against the MATLAB coefficient capacity.
        initial_storage_m3=float(catalog["SINIT"]), initial_release_m3s=float(catalog["QINIT"]),
        **{key: settings[key] for key in ("dead_storage_fraction", "inflow_correction", "flood_mode")})
    result = simulate_reservoir(observations, parameters)
    metrics = evaluate_simulation(observations, result, warmup_steps=settings["warmup_steps"])
    args.output.mkdir(parents=True, exist_ok=True)
    result.frame.to_csv(args.output / "daily_simulation.csv", index_label="date")
    estimates.monthly_targets.to_csv(args.output / "monthly_targets_si.csv")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
