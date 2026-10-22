"""Run the configured Trinity NSGA-II calibration pilot."""

from pathlib import Path
import sys

from dztr.calibration.search import main


if __name__ == "__main__":
    if "--repo" not in sys.argv:
        sys.argv.extend(["--repo", str(Path(__file__).resolve().parents[1])])
    raise SystemExit(main())
