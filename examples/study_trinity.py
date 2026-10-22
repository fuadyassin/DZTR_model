"""Run the configured Trinity seed and evaluation-budget comparison."""

from pathlib import Path
import sys

from dztr.calibration.study import main


if __name__ == "__main__":
    if "--repo" not in sys.argv:
        sys.argv.extend(["--repo", str(Path(__file__).resolve().parents[1])])
    raise SystemExit(main())
