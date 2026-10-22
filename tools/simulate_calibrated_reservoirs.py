"""Replay frozen calibrated profiles through the exported GRAND_ID table.

No optimization or held-out parameter selection occurs in this command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dztr.calibration.periods import DatePeriod
from dztr.calibration.sampled import select_sampled_period
from dztr.integrations.collection import load_collection, _verified_inputs
from dztr.integrations.world_reservoirs import _forcing
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters
from dztr.simulation.sampled import SamplingSettings
from dztr.simulation.root_policy import simulate_root_policy_reservoir

ARCHIVE = ROOT / "outputs/twenty_reservoir_workflow_20261012"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve(relative):
    path = (ROOT / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Registry path escapes repository")
    return path


def select_ids(registry, grand_ids=(), reservoir_ids=(), all_reservoirs=False):
    if all_reservoirs and (grand_ids or reservoir_ids):
        raise ValueError("Use --all or explicit IDs")
    if all_reservoirs:
        return sorted(registry["profiles_by_reservoir_id"])
    selected = list(reservoir_ids)
    for identifier in grand_ids:
        key = str(identifier)
        if key not in registry["profiles_by_GRAND_ID"]:
            raise ValueError(f"Unknown calibrated GRAND_ID: {key}")
        path = registry["profiles_by_GRAND_ID"][key]
        matches = [i for i, p in registry["profiles_by_reservoir_id"].items() if p == path]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous registry alias: {key}")
        selected.extend(matches)
    if not selected:
        raise ValueError("Select --all, --grand-id or --reservoir-id")
    for identifier in selected:
        if identifier not in registry["profiles_by_reservoir_id"]:
            raise ValueError(f"Unknown calibrated reservoir_id: {identifier}")
    return list(dict.fromkeys(selected))


def table_targets(row, profile, targets):
    """Require exact SI round-trip and identity agreement before any simulation."""
    if row.reservoir_id != profile["reservoir_id"]:
        raise ValueError("Profile/table reservoir identity differs")
    grand = None if pd.isna(row.GRAND_ID) else int(row.GRAND_ID)
    if grand != profile["grand_id"]:
        raise ValueError("Profile/table GRAND_ID differs")
    for name, value in profile["physical_constants"].items():
        if row[name] != value:
            raise ValueError(f"Profile/table physical constant differs: {name}")
    if row.root_policy != profile["root_policy"]:
        raise ValueError("Profile/table root policy differs")
    if not targets.index.equals(pd.Index(range(1, 13), name="month")):
        raise ValueError("Targets require ordered months 1..12")
    matrix = np.array([[row[f"{column}_month{month:02d}"] for column in MONTHLY_TARGET_COLUMNS]
                       for month in range(1, 13)], dtype=float)
    if not np.array_equal(matrix, targets.loc[:, list(MONTHLY_TARGET_COLUMNS)].to_numpy()):
        raise ValueError("Profile/table monthly SI targets differ")
    return pd.DataFrame(matrix, index=targets.index, columns=MONTHLY_TARGET_COLUMNS)


def compare_frames(actual, expected):
    pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_dtype=False,
                                  check_freq=False)
    return {name: float(np.max(np.abs(actual[name] - expected[name])))
            for name in ("storage_m3", "release_m3s", "balance_residual_m3")}


def run(output, *, grand_ids=(), reservoir_ids=(), all_reservoirs=False):
    receipt = read(ARCHIVE / "final_verification.json")
    consumed = {}

    def checked(path):
        path = Path(path)
        relative = path.relative_to(ROOT).as_posix()
        if path.is_relative_to(ARCHIVE):
            expected = receipt["artifact_sha256"][path.relative_to(ARCHIVE).as_posix()]
        else:
            expected = receipt["source_sha256"][relative]
        if sha(path) != expected:
            raise ValueError(f"Frozen artifact changed: {relative}")
        consumed[relative] = expected
        return path

    registry = read(checked(ROOT / "configs/calibrated_twenty.json"))
    identifiers = select_ids(registry, grand_ids, reservoir_ids, all_reservoirs)
    wide = pd.read_csv(checked(resolve(registry["table"])), float_precision="round_trip")
    if not wide.reservoir_id.is_unique:
        raise ValueError("Duplicate parameter table reservoir IDs")
    wide = wide.set_index("reservoir_id", drop=False)
    protocol = read(checked(ARCHIVE / "protocol.json"))
    for relative, expected in protocol["code_sha256"].items():
        if sha(resolve(relative)) != expected:
            raise ValueError(f"Numerical source changed: {relative}")
        consumed[relative] = expected
    _, manifest, manifest_path, manifest_sha = load_collection(ROOT / "data/reservoirs/manifest.json")
    parent = ROOT / "outputs/policy_fleet_20261012"
    parent_receipt = parent / "execution_verification.json"
    if sha(parent_receipt) != protocol["parent_execution_sha256"]:
        raise ValueError("Parent execution receipt changed")
    parent_protocol = parent / "protocol.json"
    if sha(parent_protocol) != read(parent_receipt)["artifact_sha256"]["protocol.json"]:
        raise ValueError("Parent input protocol changed")
    consumed[parent_receipt.relative_to(ROOT).as_posix()] = sha(parent_receipt)
    consumed[parent_protocol.relative_to(ROOT).as_posix()] = sha(parent_protocol)
    input_protocol = read(parent_protocol)
    expected_inputs = {key.replace("\\", "/"): value for key, value in input_protocol["inputs_sha256"].items()}
    if manifest_sha != expected_inputs[manifest_path.relative_to(ROOT).as_posix()]:
        raise ValueError("Input collection changed since calibration")
    consumed[str(manifest_path.relative_to(ROOT)).replace("\\", "/")] = manifest_sha
    entries = {item["reservoir_id"]: item for item in manifest["reservoirs"]}
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / "outputs") or output.exists():
        raise ValueError("Use a new directory inside repository outputs")
    output.mkdir(parents=True)
    results = []
    for identifier in identifiers:
        folder = output / identifier
        folder.mkdir()
        try:
            path = checked(resolve(registry["profiles_by_reservoir_id"][identifier]))
            profile = read(path)
            targets = pd.read_csv(checked(path.parent / profile["monthly_targets_file"]),
                                  index_col="month", float_precision="round_trip")
            targets = table_targets(wide.loc[identifier], profile, targets)
            config_path, config, observations, hashes = _verified_inputs(ROOT, entries[identifier])
            consumed.update({p.relative_to(ROOT).as_posix(): h for p, h in hashes.items()})
            for input_path, digest in hashes.items():
                if expected_inputs[input_path.relative_to(ROOT).as_posix()] != digest:
                    raise ValueError("Input changed since calibration")
            resolved = read(checked(ARCHIVE / "pilots" / identifier / f"seed_{profile['seed']}" / "resolved_config.json"))
            for key in ("model", "sampling", "periods", "inflow_preprocessing"):
                if config[key] != resolved[key]:
                    raise ValueError(f"Input configuration differs from calibration: {key}")
            if profile["physical_constants"] != resolved["model"] or profile["sampling"] != resolved["sampling"] or profile["periods"] != resolved["periods"]:
                raise ValueError("Profile assumptions differ from calibration")
            sampling = SamplingSettings(**profile["sampling"])
            period = DatePeriod(profile["periods"]["simulation_start"], profile["periods"]["validation"]["end"])
            forcing = select_sampled_period(_forcing(observations, config["inflow_preprocessing"]), period, sampling)
            parameters = ReservoirParameters(targets, **profile["physical_constants"])
            frame = simulate_root_policy_reservoir(forcing, parameters, sampling=sampling,
                        policy=profile["root_policy"], atol_storage_m3=resolved["atol_storage_m3"]).frame.copy()
            for column in ("storage_m3", "release_m3s"):
                frame["observed_" + column] = observations.frame[column].reindex(frame.index)
            frame["period"] = "warmup"
            for label in ("calibration", "validation"):
                dates = profile["periods"][label]
                frame.loc[dates["start"]:dates["end"], "period"] = label
            expected = pd.read_csv(checked(path.parent / "simulation.csv"), index_col="date", parse_dates=True,
                                   float_precision="round_trip")
            differences = compare_frames(frame, expected)
            s, q, i, dt = [frame[c].to_numpy() for c in ("storage_m3", "release_m3s", "inflow_m3s", "step_seconds")]
            residual = np.diff(s) - dt[1:]/2 * (parameters.inflow_correction*(i[1:]+i[:-1])-q[1:]-q[:-1])
            np.testing.assert_array_equal(residual, frame.balance_residual_m3.to_numpy()[1:])
            maximum = float(np.max(np.abs(residual)))
            if maximum > resolved["atol_storage_m3"]:
                raise ValueError("Mass balance tolerance exceeded")
            frame.to_csv(folder / "simulation.csv", index_label="date", float_format="%.17g")
            targets.to_csv(folder / "monthly_targets_si.csv", float_format="%.17g")
            write(folder / "parameters.json", profile)
            result = {"reservoir_id": identifier, "GRAND_ID": profile["grand_id"], "status": "passed",
                      "trajectory_rows": len(frame), "maximum_balance_residual_m3": maximum,
                      "maximum_replay_difference": differences, "simulation_reproducible": True,
                      "physical_source_certified": profile["physical_source_certified"],
                      "convergence_established": profile["convergence_established"],
                      "validation_release_nse": profile["validation"]["metrics"]["release_nse"],
                      "validation_storage_nse": profile["validation"]["metrics"]["storage_nse"],
                      "production_readiness": "not_established"}
        except Exception as error:
            result = {"reservoir_id": identifier, "status": "failed", "reason": str(error), "simulation_reproducible": False}
        results.append(result)
        write(folder / "result.json", result)
        print(f"{identifier}: {result['status']}", flush=True)
    write(output / "batch_summary.json", {"schema_version": 1, "workflow_version": "1.0",
          "status": "passed" if all(r["status"] == "passed" for r in results) else "failed",
          "selection_uses_validation": False, "optimization_evaluations": 0, "jobs": results})
    pd.DataFrame([{k:v for k,v in r.items() if k != "maximum_replay_difference"} for r in results]).to_csv(output / "readiness.csv", index=False, float_format="%.17g")
    write(output / "release_manifest.json", {"schema_version": 1, "workflow_version": "1.0",
          "archive_receipt_sha256": sha(ARCHIVE / "final_verification.json"),
          "consumed_sha256": consumed, "runner_sha256": sha(__file__),
          "artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    return all(r["status"] == "passed" for r in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--grand-id", nargs="+", default=[])
    parser.add_argument("--reservoir-id", nargs="+", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return 0 if run(args.output, grand_ids=args.grand_id, reservoir_ids=args.reservoir_id, all_reservoirs=args.all) else 1


if __name__ == "__main__":
    raise SystemExit(main())
