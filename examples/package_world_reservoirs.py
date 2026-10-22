"""Copy the reviewed scope into permanent data/config folders; no simulation."""

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dztr.integrations.packaging import package_world_samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scope", default="configs/world_reservoir_scope.json")
    parser.add_argument("--manifest", default="data/reservoirs/manifest.json")
    args = parser.parse_args(argv)
    manifest = package_world_samples(args.repo_root, scope_file=args.scope, manifest_file=args.manifest)
    print(json.dumps({"manifest": args.manifest, "reservoir_count": manifest["reservoir_count"],
                      "counts": manifest["counts"], "excluded_reservoirs": manifest["excluded_reservoirs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
