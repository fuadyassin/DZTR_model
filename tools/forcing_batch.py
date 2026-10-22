"""Batch simulations using supplied forcing and frozen calibrated DZTR profiles."""
import argparse
import csv
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from simulate_calibrated_reservoirs import (
    ROOT, ARCHIVE, read, sha, write, resolve, select_ids, table_targets,
)
from dztr.calibration.sampled import _observations
from dztr.evaluation import evaluate_simulation
from dztr.schemas import ReservoirParameters
from dztr.simulation.sampled import SamplingSettings, validate_sampling_index
from dztr.simulation.root_policy import simulate_root_policy_reservoir

FLOW_UNITS = {"m3/s": 1., "m3/day": 1./86400.}
STORAGE_UNITS = {"m3": 1., "Mm3": 1e6}


def converted(values, unit, factors, name):
    if unit not in factors:
        raise ValueError(f"Unsupported {name} unit: {unit!r}; choose {list(factors)}")
    return pd.to_numeric(values, errors="raise") * factors[unit]


class FrozenProfiles:
    """Check consumed artifacts once; never change the archived parameter set."""
    def __init__(self):
        self.receipt = read(ARCHIVE / "final_verification.json")
        self.hashes = {"outputs/twenty_reservoir_workflow_20261012/final_verification.json": sha(ARCHIVE / "final_verification.json")}
        self.registry = read(self.checked(ROOT / "configs/calibrated_twenty.json"))
        self.wide = pd.read_csv(self.checked(resolve(self.registry["table"])), float_precision="round_trip").set_index("reservoir_id", drop=False)
        if not self.wide.index.is_unique:
            raise ValueError("Duplicate parameter table identities")
        protocol = read(self.checked(ARCHIVE / "protocol.json"))
        for relative, digest in protocol["code_sha256"].items():
            if sha(resolve(relative)) != digest:
                raise ValueError(f"Numerical source changed: {relative}")
            self.hashes[relative.replace("\\", "/")] = digest
        # The helper imported above is itself part of the frozen replay release.
        release = ROOT / "outputs/twenty_workflow_release_v1_20261012/final_verification.json"
        helper = "tools/simulate_calibrated_reservoirs.py"
        if sha(ROOT / helper) != read(release)["source_sha256"][helper]:
            raise ValueError("Frozen profile helper changed")
        self.hashes[helper] = sha(ROOT / helper)
        self.hashes[release.relative_to(ROOT).as_posix()] = sha(release)

    def checked(self, path):
        relative = path.relative_to(ROOT).as_posix()
        expected = (self.receipt["artifact_sha256"][path.relative_to(ARCHIVE).as_posix()]
                    if path.is_relative_to(ARCHIVE) else self.receipt["source_sha256"][relative])
        if sha(path) != expected:
            raise ValueError(f"Frozen artifact changed: {relative}")
        self.hashes[relative] = expected
        return path

    def load(self, job):
        selectors = set(job) & {"reservoir_id", "grand_id"}
        if len(selectors) != 1:
            raise ValueError("Declare exactly one reservoir_id or grand_id")
        identifier = select_ids(self.registry, grand_ids=[job["grand_id"]] if "grand_id" in job else (),
                                reservoir_ids=[job["reservoir_id"]] if "reservoir_id" in job else ())[0]
        path = self.checked(resolve(self.registry["profiles_by_reservoir_id"][identifier]))
        profile = read(path)
        targets = pd.read_csv(self.checked(path.parent / profile["monthly_targets_file"]), index_col="month", float_precision="round_trip")
        targets = table_targets(self.wide.loc[identifier], profile, targets)
        config = read(self.checked(ARCHIVE / "pilots" / identifier / f"seed_{profile['seed']}" / "resolved_config.json"))
        if profile["physical_constants"] != config["model"] or profile["sampling"] != config["sampling"]:
            raise ValueError("Profile assumptions differ from calibration")
        return profile, ReservoirParameters(targets, **profile["physical_constants"]), config


def load_forcing(job, base, profile, config):
    spec = job["forcing"]
    columns, units = spec["columns"], spec["units"]
    if not {"date", "inflow"} <= set(columns) or set(columns) - {"date", "inflow", "storage", "release"}:
        raise ValueError("Forcing columns require date/inflow and optional storage/release")
    if set(units) != set(columns) - {"date"} or len(set(columns.values())) != len(columns):
        raise ValueError("Declare one explicit unit per unique measurement column")
    if job["sampling"] != profile["sampling"]:
        raise ValueError("Sampling must match the frozen profile; changing clocks requires a new profile")
    sampling = SamplingSettings(**job["sampling"])
    path = (base / spec["csv"].replace("\\", "/")).resolve(strict=True)
    content = path.read_bytes()
    text = content.decode("utf-8-sig")
    header = next(csv.reader(StringIO(text)))
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column names")
    frame = pd.read_csv(StringIO(text), float_precision="round_trip")
    dates = pd.DatetimeIndex(pd.to_datetime(frame[columns["date"]], errors="raise"), name="date")
    frame.index = dates
    if "period" in job:
        period = job["period"]
        start, end = pd.Timestamp(period["start"]), pd.Timestamp(period["end"])
        if start > end or start not in dates or end not in dates:
            raise ValueError("Requested period requires exact existing date boundaries")
        frame = frame.loc[start:end]
    validate_sampling_index(frame.index, sampling.frequency)
    normalized = pd.DataFrame(index=frame.index)
    normalized["inflow_m3s"] = converted(frame[columns["inflow"]], units["inflow"], FLOW_UNITS, "inflow")
    for name, target, factors in (("storage", "storage_m3", STORAGE_UNITS), ("release", "release_m3s", FLOW_UNITS)):
        normalized[target] = converted(frame[columns[name]], units[name], factors, name) if name in columns else np.nan
        values = normalized[target].to_numpy()
        if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
            raise ValueError(f"Observed {name} must be nonnegative or missing")
    stage = spec["stage"]
    if stage == "raw_source":
        preprocessing = config["inflow_preprocessing"]
    elif stage == "model_input":
        preprocessing = {"factor": 1., "offset_m3s": 0.}
    else:
        raise ValueError("Forcing stage must be raw_source or model_input")
    normalized["inflow_m3s"] = normalized.inflow_m3s * preprocessing["factor"] + preprocessing["offset_m3s"]
    if not np.isfinite(normalized.inflow_m3s).all() or (normalized.inflow_m3s < 0).any():
        raise ValueError("Adjusted inflow must be finite and nonnegative")
    return _observations(normalized, str(path)), sampling, {"path": str(path), "sha256": __import__("hashlib").sha256(content).hexdigest(), "stage": stage, "preprocessing": preprocessing}


def initial_parameters(parameters, initial):
    if set(initial) != {"storage", "storage_unit", "release", "release_unit"}:
        raise ValueError("Initial state requires storage/release values and units")
    for name in ("storage", "release"):
        if isinstance(initial[name], bool) or not isinstance(initial[name], (int, float)) or not np.isfinite(initial[name]):
            raise ValueError("Initial values must be finite numbers")
    storage = float(converted(initial["storage"], initial["storage_unit"], STORAGE_UNITS, "initial storage"))
    release = float(converted(initial["release"], initial["release_unit"], FLOW_UNITS, "initial release"))
    return replace(parameters, initial_storage_m3=storage, initial_release_m3s=release)


def period_metrics(observations, result, periods):
    rows = []
    for name, period in periods.items():
        start, end = pd.Timestamp(period["start"]), pd.Timestamp(period["end"])
        if start > end or start not in result.frame.index or end not in result.frame.index:
            raise ValueError(f"Metric period {name!r} requires exact existing date boundaries")
        observed = observations.frame.loc[start:end]
        simulated = result.frame.loc[start:end]
        from types import SimpleNamespace
        values = evaluate_simulation(_observations(observed), SimpleNamespace(frame=simulated))
        rows.append({"period": name, "start": str(start.date()), "end": str(end.date()),
                     "storage_pairs": int(observed.storage_m3.notna().sum()),
                     "release_pairs": int(observed.release_m3s.notna().sum()),
                     **{key: float(value) if np.isfinite(value) else None for key, value in values.items()}})
    return rows


def simulate_job(job, base, store):
    if set(job) - {"job_id", "reservoir_id", "grand_id", "forcing", "initial", "sampling", "period", "evaluation_periods"}:
        raise ValueError("Unknown job fields; solver and physical constants cannot be overridden")
    profile, parameters, config = store.load(job)
    observations, sampling, source = load_forcing(job, base, profile, config)
    parameters = initial_parameters(parameters, job["initial"])
    result = simulate_root_policy_reservoir(observations, parameters, sampling=sampling,
              policy=profile["root_policy"], atol_storage_m3=config["atol_storage_m3"])
    frame = result.frame.copy()
    s,q,i,dt = [frame[c].to_numpy() for c in ("storage_m3", "release_m3s", "inflow_m3s", "step_seconds")]
    residual = np.diff(s)-dt[1:]/2*(parameters.inflow_correction*(i[1:]+i[:-1])-q[1:]-q[:-1])
    np.testing.assert_array_equal(residual, frame.balance_residual_m3.to_numpy()[1:])
    maximum = float(np.abs(residual).max())
    if maximum > config["atol_storage_m3"]:
        raise ValueError("Mass balance tolerance exceeded")
    for column in ("storage_m3", "release_m3s"):
        frame["observed_" + column] = observations.frame[column]
    periods = job.get("evaluation_periods")
    if periods is None:
        periods = ({"assessment": {"start":str(frame.index[1].date()), "end":str(frame.index[-1].date())}}
                   if observations.frame[["storage_m3", "release_m3s"]].notna().any().any() else {})
    metrics = period_metrics(observations, result, periods)
    summary = {"job_id":job["job_id"], "reservoir_id":profile["reservoir_id"], "GRAND_ID":profile["grand_id"],
               "status":"passed", "rows":len(frame), "start":str(frame.index[0].date()), "end":str(frame.index[-1].date()),
               "maximum_balance_residual_m3":maximum, "sum_overflow_volume_m3":float(frame.overflow_volume_m3.sum()),
               "zone_counts":{str(int(z)):int(n) for z,n in frame.zone.value_counts().sort_index().items()},
               "multiple_root_steps":int((frame.root_count>1).sum()), "solver_policy":profile["root_policy"],
               "sampling":job["sampling"], "forcing_source":source, "initial_storage_m3":parameters.initial_storage_m3,
               "initial_release_m3s":parameters.initial_release_m3s, "metrics":metrics, "issues":list(result.issues),
               "operating_rules_assumed_stationary":True, "parameter_status":"exploratory",
               "production_readiness":"not_established", "selection_uses_supplied_observations":False}
    return frame, observations, parameters, summary


def run_batch(manifest_path, output):
    manifest_path = Path(manifest_path).resolve(strict=True)
    # Read once so the receipt hashes precisely the manifest that was executed.
    content = manifest_path.read_bytes()
    manifest = json.loads(content.decode("utf-8-sig"))
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "dztr_forcing_batch":
        raise ValueError("Expected version 1 dztr_forcing_batch")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Batch requires at least one job")
    ids = [job.get("job_id") for job in jobs]
    if any(not isinstance(i,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*",i) for i in ids) or len({i.casefold() for i in ids}) != len(ids):
        raise ValueError("Job IDs must be unique path-safe names")
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / "outputs") or output.exists():
        raise ValueError("Use a new directory inside repository outputs")
    store = FrozenProfiles()
    output.mkdir(parents=True)
    (output / "submitted_batch.json").write_bytes(content)
    results = []
    for job in jobs:
        folder = output / job["job_id"]
        folder.mkdir()
        write(folder / "request.json",job)
        try:
            frame, observations, parameters, summary = simulate_job(job, manifest_path.parent, store)
            frame.to_csv(folder / "simulation.csv", index_label="date", float_format="%.17g")
            observations.frame.to_csv(folder / "normalized_inputs_si.csv", index_label="date", float_format="%.17g")
            parameters.monthly_targets.to_csv(folder / "monthly_targets_si.csv", float_format="%.17g")
            pd.DataFrame(summary["metrics"]).to_csv(folder / "metrics.csv", index=False, float_format="%.17g")
        except Exception as error:
            summary = {"job_id":job["job_id"],"status":"failed","reason":str(error),"error_type":type(error).__name__}
        write(folder / "result.json",summary)
        results.append(summary)
        print(f"{job['job_id']}: {summary['status']}",flush=True)
    passed = all(row["status"] == "passed" for row in results)
    write(output / "batch_summary.json",{"schema_version":1,"workflow_version":"1.1","status":"passed" if passed else "failed", "optimization_evaluations":0,"jobs":results})
    pd.DataFrame([{key:value for key,value in row.items() if not isinstance(value,(dict,list))} for row in results]).to_csv(output / "batch_summary.csv",index=False,float_format="%.17g")
    write(output / "execution_manifest.json",{"manifest_path":str(manifest_path),"manifest_sha256":__import__("hashlib").sha256(content).hexdigest(),
          "frozen_source_sha256":store.hashes,"runner_sha256":sha(__file__),
          "artifact_sha256":{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    return 0 if run_batch(args.manifest,args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
