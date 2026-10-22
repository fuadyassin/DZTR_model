"""Verify stored replay evidence without repeating simulations or optimization."""
import argparse

import numpy as np
import pandas as pd

from simulate_calibrated_reservoirs import ROOT, ARCHIVE, read, sha, resolve, compare_frames, write


def verify(output):
    manifest = read(output / "release_manifest.json")
    if sha(ARCHIVE / "final_verification.json") != manifest["archive_receipt_sha256"]:
        raise ValueError("Calibration archive receipt changed")
    for relative, expected in manifest["artifact_sha256"].items():
        if sha(output / relative) != expected:
            raise ValueError(f"Replay artifact changed: {relative}")
    for relative, expected in manifest["consumed_sha256"].items():
        if sha(resolve(relative)) != expected:
            raise ValueError(f"Replay source changed: {relative}")
    # Reconcile every consumed input with the original calibration protocol,
    # including batches executed before this extra preflight check was added.
    parent = ROOT / "outputs/policy_fleet_20261012"
    protocol = read(ARCHIVE / "protocol.json")
    if sha(parent / "execution_verification.json") != protocol["parent_execution_sha256"]:
        raise ValueError("Parent receipt changed")
    if sha(parent / "protocol.json") != read(parent / "execution_verification.json")["artifact_sha256"]["protocol.json"]:
        raise ValueError("Parent protocol changed")
    source_inputs = {p.replace("\\", "/"): h for p, h in read(parent / "protocol.json")["inputs_sha256"].items()}
    for relative, expected in manifest["consumed_sha256"].items():
        if relative.startswith(("data/reservoirs/", "configs/reservoirs/")) and source_inputs[relative] != expected:
            raise ValueError(f"Replay input differs from calibration: {relative}")
    summary = read(output / "batch_summary.json")
    if summary["status"] != "passed" or summary["optimization_evaluations"] != 0:
        raise ValueError("Batch did not pass")
    registry = read(ROOT / "configs/calibrated_twenty.json")
    identifiers = [item["reservoir_id"] for item in summary["jobs"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate replay jobs")
    maximum, rows = 0., 0
    for item in summary["jobs"]:
        identifier = item["reservoir_id"]
        saved = resolve(registry["profiles_by_reservoir_id"][identifier]).parent
        frame = pd.read_csv(output / identifier / "simulation.csv", index_col="date", parse_dates=True, float_precision="round_trip")
        reference = pd.read_csv(saved / "simulation.csv", index_col="date", parse_dates=True, float_precision="round_trip")
        assert all(value == 0 for value in compare_frames(frame, reference).values())
        profile = read(output / identifier / "parameters.json")
        assert profile == read(saved / "parameters.json")
        s,q,i,dt = [frame[c].to_numpy() for c in ("storage_m3","release_m3s","inflow_m3s","step_seconds")]
        residual = np.diff(s)-dt[1:]/2*(profile["physical_constants"]["inflow_correction"]*(i[1:]+i[:-1])-q[1:]-q[:-1])
        np.testing.assert_array_equal(residual, frame.balance_residual_m3.to_numpy()[1:])
        maximum = max(maximum, float(np.abs(residual).max()))
        assert np.abs(residual).max() <= 1e-4
        rows += len(frame)
    return {"status":"passed", "reservoirs_verified":len(identifiers), "trajectory_rows_verified":rows,
            "all_twenty_verified":set(identifiers)==set(registry["profiles_by_reservoir_id"]),
            "exact_replay":True, "maximum_balance_residual_m3":maximum,
            "verification_simulations":0, "optimization_evaluations":0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=__import__("pathlib").Path)
    args = parser.parse_args()
    result = verify(args.output)
    write(args.output / "scientific_verification.json", result)
    print(result)


if __name__ == "__main__":
    main()
