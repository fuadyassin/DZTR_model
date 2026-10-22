"""Run a configured reservoir baseline with separate calibration and validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from dztr import __version__
from dztr.data import read_reservoir_observations

from .baseline import json_ready, run_baseline
from .periods import CalibrationValidationSplit, DatePeriod


def run_configured_baseline(config_path: str | Path, repo_root: str | Path, output_dir: str | Path) -> dict:
    """Read explicit settings, execute the baseline, and write reproducible artifacts."""
    config_path, root, output = Path(config_path), Path(repo_root), Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selection = config["periods"]
    split = CalibrationValidationSplit(
        DatePeriod(**selection["calibration"]), DatePeriod(**selection["validation"]),
        warmup_steps=selection["warmup_steps"],
    )
    source = root / config["observations_file"]
    artifact_names = ("daily_simulation.csv", "monthly_targets_si.csv", "calibration_quantiles_si.csv",
                      "metrics.csv", "resolved_config.json", "summary.json")
    if any((output / name).resolve() in {source.resolve(), config_path.resolve()} for name in artifact_names):
        raise ValueError("The output folder would overwrite source observations or configuration; choose another folder.")
    observations = read_reservoir_observations(source)
    simulation_settings = config.get("simulation", {})
    result = run_baseline(observations, split, **config["model"],
                          target_percentiles=tuple(config.get("target_percentiles", (10, 45, 85))),
                          quantile_method=config.get("quantile_method", "hazen"), **simulation_settings)
    report = result.summary()
    report["reservoir"] = config.get("reservoir")
    report["protocol"] = config.get("protocol")
    report["source_sha256"] = {
        "observations": hashlib.sha256(source.read_bytes()).hexdigest(),
        "configuration": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    report["configuration"] = config
    report["software_versions"] = {"python": platform.python_version(), "numpy": np.__version__,
                                   "pandas": pd.__version__, "dztr": __version__}
    report = json_ready(report)
    # Finish all computation before creating artifacts; never write source workbooks.
    output.mkdir(parents=True, exist_ok=True)
    result.daily_table().to_csv(output / "daily_simulation.csv", index_label="date")
    result.estimates.monthly_targets.to_csv(output / "monthly_targets_si.csv")
    result.quantiles.frame.to_csv(output / "calibration_quantiles_si.csv", index=False)
    pd.DataFrame.from_dict(report["metrics"], orient="index").rename_axis("period").to_csv(output / "metrics.csv")
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, help="JSON settings; defaults to <repo>/configs/trinity_baseline.json.")
    parser.add_argument("--output", type=Path, default=Path("outputs/trinity_baseline"))
    args = parser.parse_args()
    report = run_configured_baseline(args.config or args.repo / "configs/trinity_baseline.json", args.repo, args.output)
    print(pd.DataFrame.from_dict(report["metrics"], orient="index").round(4).to_string())
    print(f"Artifacts: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
