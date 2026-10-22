"""Add explicit policy/clock metadata to a fresh copy of completed reports."""
from pathlib import Path
import shutil

from policy_fleet_job import ROOT, read, sha, write


def main():
    fleet = ROOT / "outputs/policy_fleet_20261012"
    original = fleet / "reports/completed_fleet"
    out = fleet / "reports/reviewed_fleet"
    if out.exists():
        raise ValueError("Preserve existing reviewed reports.")
    proof = read(original / "report_manifest.json")
    for name, digest in proof["artifact_sha256"].items():
        if sha(original / name) != digest:
            raise ValueError("Original generated report changed.")
    shutil.copytree(original, out)
    metadata = []
    for record in proof["reservoirs"]:
        identifier = record["reservoir_id"]
        seed = record["trajectory_seed"]
        job = fleet / "pilots" / identifier / f"seed_{seed}"
        config, result = read(job / "resolved_config.json"), read(job / "result.json")
        folder = out / identifier
        text = (folder / "README.md").read_text(encoding="utf-8")
        fixed = config["sampling"].get("fixed_step_days")
        # The generic archived reporter calls this older field timestep_days;
        # fresh config files use fixed_step_days. Correct only the new copy.
        if config["sampling"]["frequency"] == "monthly":
            text = text.replace("timestep of None days", f"timestep of {fixed} days")
        period = config["periods"]
        policy = result["root_policy"]
        text += f"\nDeclared root policy: `{policy}`. The nearest-storage convention, when used, refers to previous simulated storage and uses the existing minimum-storage tie rule.\n\nCalibration: {period['calibration']['start']} to {period['calibration']['end']}; held-out validation: {period['validation']['start']} to {period['validation']['end']}. The plotted seed is fixed in advance. Plots use saved trajectories from the completed calibration jobs; plotting adds no model calls.\n"
        (folder / "README.md").write_text(text, encoding="utf-8")
        item = {"reservoir_id": identifier, "root_policy": policy, "trajectory_seed": seed,
                "sampling": config["sampling"], "periods": period,
                "result_sha256": sha(job / "result.json"), "config_sha256": sha(job / "resolved_config.json")}
        write(folder / "report_metadata.json", item)
        metadata.append(item)
    write(out / "report_manifest.json", {"status": "reviewed", "new_model_evaluations": 0,
        "source_report_manifest_sha256": sha(original / "report_manifest.json"), "reservoirs": metadata,
        "reviewer_sha256": sha(Path(__file__)),
        "artifact_sha256": {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file() and p.name != "report_manifest.json"}})
    print(f"Reviewed clock/policy captions for {len(metadata)} reservoir reports; original charts preserved byte-for-byte.")


if __name__ == "__main__":
    main()
