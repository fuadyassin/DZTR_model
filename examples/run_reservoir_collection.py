"""Run permanent samples: --reservoir Trinity or --all, plus --output NEW_FOLDER."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dztr.integrations.collection import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
