"""Import original MATLAB folders: --source-root PATH --output NEW_FOLDER."""

import sys
from dztr.integrations.world_reservoirs import main

if __name__ == "__main__":
    raise SystemExit(main(["import", *sys.argv[1:]]))
