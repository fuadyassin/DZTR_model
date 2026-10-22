"""Export archived initialization/selected monthly targets in explicit SI units."""
from pathlib import Path

import numpy as np
import pandas as pd

from policy_fleet_job import ROOT, read, sha, write
from dztr.schemas import MONTHLY_TARGET_COLUMNS


def main():
    fleet = ROOT / "outputs/policy_fleet_20261012"
    if read(fleet / "scientific_verification.json")["status"] != "passed":
        raise ValueError("Scientific verification required before parameter export.")
    out = fleet / "parameters"
    out.mkdir(exist_ok=False)
    jobs = []
    for job in read(fleet / "execution_summary.json")["jobs"]:
        if job["status"] != "completed":
            continue
        source = fleet / "pilots" / job["reservoir_id"] / f"seed_{job['seed']}"
        result, config = read(source / "result.json"), read(source / "resolved_config.json")
        vectors = pd.read_csv(source / "vectors.csv", index_col="evaluation_id", float_precision="round_trip")
        folder = out / job["reservoir_id"] / f"seed_{job['seed']}"
        folder.mkdir(parents=True)
        roles = {"initialization": 0, **{r["role"]: r["evaluation_id"] for r in result["representatives"]}}
        files = {}
        for role, identifier in roles.items():
            targets = vectors.loc[identifier].to_numpy().reshape(6, 12).T.copy()
            targets[:, :3] *= config["model"]["capacity_m3"]
            frame = pd.DataFrame(targets, columns=MONTHLY_TARGET_COLUMNS, index=pd.Index(range(1, 13), name="month"))
            destination = folder / f"{role}_monthly_targets_si.csv"
            frame.to_csv(destination, float_format="%.17g")
            np.testing.assert_array_equal(pd.read_csv(destination, index_col="month", float_precision="round_trip").to_numpy(), targets)
            files[role] = {"evaluation_id": identifier, "file": destination.relative_to(out).as_posix(), "sha256": sha(destination)}
        metadata = {"reservoir_id": job["reservoir_id"], "seed": job["seed"], "root_policy": job["root_policy"],
                    "units": {"storage_targets": "m3", "release_targets": "m3/s"},
                    "physical_constants": config["model"], "sampling": config["sampling"], "periods": config["periods"],
                    "initialization_source": result["initialization"]["selected_candidate"],
                    "roles": files, "calibration_only_selection": True,
                    "source_result_sha256": sha(source / "result.json"), "source_vectors_sha256": sha(source / "vectors.csv")}
        write(folder / "parameters.json", metadata)
        jobs.append(metadata)
    write(out / "parameter_index.json", {"status": "passed", "jobs": jobs, "monthly_tables": sum(len(r["roles"]) for r in jobs), "new_model_evaluations": 0, "csv_roundtrip_verified": True, "exporter_sha256": sha(Path(__file__))})
    print(f"Exported {sum(len(r['roles']) for r in jobs)} verified monthly SI tables from {len(jobs)} archived jobs.")


if __name__ == "__main__":
    main()
