"""Estimate one reservoir after installing the project with pip install -e ."""

from pathlib import Path

from dztr.data import read_reservoir_observations
from dztr.parameterization import estimate_generalized_parameters


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    observations = read_reservoir_observations(root / "ResDataInflowStorageRelease/Gardiner_253.xlsx")
    parameters = estimate_generalized_parameters(observations, channel_capacity_m3s=2500)
    print(observations.quality)
    print(parameters.monthly_targets)


if __name__ == "__main__":
    main()
