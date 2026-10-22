"""Compare Trinity/Ghost daily substeps; use --output with a new directory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dztr.integrations.substep_review import main

if __name__ == "__main__":
    raise SystemExit(main())
