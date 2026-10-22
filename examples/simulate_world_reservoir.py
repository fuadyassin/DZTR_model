"""Run a normalized sample: --config SAMPLE_JSON --output NEW_FOLDER [--optimize N]."""

import sys
from dztr.integrations.world_reservoirs import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
