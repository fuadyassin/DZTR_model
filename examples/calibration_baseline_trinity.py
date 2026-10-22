"""Run the Trinity split-sample baseline after pip install -e ."""

from pathlib import Path

import pandas as pd

from dztr.calibration.benchmark import run_configured_baseline


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/trinity_baseline"
    report = run_configured_baseline(root / "configs/trinity_baseline.json", root, output)
    print(pd.DataFrame.from_dict(report["metrics"], orient="index").round(4).to_string())
    print(f"Saved: {output}")
