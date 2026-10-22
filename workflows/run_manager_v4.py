"""Shared-estimator gateway with explicit geometry calibration handoff.

This module orchestrates existing scientific implementations. It does not
change equations, checkpoint identities, observation timing or route readiness.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from workflows import reservoir_v2 as w
from workflows import station_inputs_v1 as stations
from workflows.geometry_v1 import digest
from workflows.atomic_io_v1 import write as atomic_write

ROOT = w.ROOT
REFERENCE = re.compile(r"^@([A-Za-z][A-Za-z0-9_-]*)\.([A-Za-z_]+)$")
PATH_KEYS = {"catalog", "geometry_index", "observations_csv", "model_config", "selection_json",
             "levels_csv", "geometry_csv", "csv", "registry", "forcing_csv", "initial_state", "campaign",
             "level_provenance_json", "datum_metadata_json"}
TYPES = {"acquire", "parameterize", "baseline", "calibrate", "simulate"}


def now():
    return datetime.now(timezone.utc).isoformat()


def walk(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)
    else:
        yield value


def _paths(value, base, pinned, key=None):
    if isinstance(value, dict):
        return {k: _paths(v, base, pinned, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_paths(v, base, pinned, key) for v in value]
    if isinstance(value, str) and not value.startswith("@") and key in PATH_KEYS:
        path = (base / value).resolve(strict=True)
        if not path.is_file() and key != "campaign":
            raise ValueError(f"Expected input file: {path}")
        if path.is_file():
            pinned[str(path)] = w.sha(path)
        return str(path)
    if isinstance(value, str) and value.startswith("@") and not REFERENCE.fullmatch(value):
        raise ValueError("Stage references must be complete @stage.artifact values")
    return value


def plan(config_path):
    path = Path(config_path).resolve(strict=True)
    cfg = w.read(path)
    if set(cfg) - {"kind", "schema_version", "description", "stages"} or (cfg.get("kind"), cfg.get("schema_version")) != ("dztr_run_manager_v4", 4):
        raise ValueError("Unknown run-manager configuration")
    if not isinstance(cfg.get("stages"), list) or not cfg["stages"]:
        raise ValueError("Declare ordered stages")
    pinned, result, seen = {str(path): w.sha(path)}, [], {}
    available_outputs = {"acquire": {"observations_csv", "levels_csv", "provenance"},
                         "parameterize": {"campaign", "parameter_table", "profile_registry"},
                         "baseline": {"summary"}, "calibrate": {"registry", "campaign", "parameter_table", "assessment_report", "forcing_csv"},
                         "simulate": {"simulation_csv", "campaign"}}
    for stage in cfg["stages"]:
        if set(stage) - {"id", "type", "config", "backend"}:
            raise ValueError("Unknown stage fields")
        identifier, kind = stage.get("id"), stage.get("type")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", identifier) or identifier in seen:
            raise ValueError("Stage IDs must be unique safe names")
        if kind not in TYPES:
            raise ValueError("Unknown workflow stage type")
        backend = stage.get("backend", "v2")
        if backend not in {"v1", "v2", "prepared_v1", "prepared_geometry_v1", "prepared_flow_v1"} or ("backend" in stage and kind not in {"calibrate", "simulate"}) or (backend in {"prepared_v1", "prepared_geometry_v1", "prepared_flow_v1"} and kind != "calibrate"):
            raise ValueError("Choose native v1/v2, or prepared_v1 for an exact calibration handoff")
        spec, base = stage["config"], path.parent
        if isinstance(spec, str):
            template = (base / spec).resolve(strict=True)
            pinned[str(template)] = w.sha(template)
            spec, base = w.read(template), template.parent
        if not isinstance(spec, dict):
            raise ValueError("Stage config must be a JSON object or a JSON file")
        spec = _paths(spec, base, pinned)
        dependencies = set()
        for value in walk(spec):
            match = REFERENCE.fullmatch(value) if isinstance(value, str) else None
            if match:
                source, artifact = match.groups()
                if source not in seen or artifact not in available_outputs[seen[source]]:
                    raise ValueError("References must name an earlier stage's supported output")
                dependencies.add(source)
        if kind == "acquire":
            stations.validate(spec)
        elif kind == "baseline" and (set(spec) - {"campaign", "reservoir_id"} or "campaign" not in spec):
            raise ValueError("Baseline stage requires campaign and optional reservoir_id")
        elif kind == "parameterize" and spec.get("kind") not in {"dztr_data_availability_v1", "dztr_data_availability_v2"}:
            raise ValueError("Parameterization requires a data-availability configuration")
        elif kind == "calibrate" and backend == "prepared_flow_v1":
            if spec.get("kind") != "dztr_flow_calibration_v1" or spec.get("schema_version") != 1:
                raise ValueError("Flow calibration requires its explicit release-only handoff")
        elif kind == "calibrate" and backend == "prepared_geometry_v1":
            if spec.get("kind") != "dztr_geometry_calibration_v1" or spec.get("schema_version") != 1:
                raise ValueError("Geometry calibration requires its explicit proxy/band handoff configuration")
        elif kind == "calibrate" and backend == "prepared_v1":
            if spec.get("kind") != "dztr_prepared_calibration_v1" or spec.get("schema_version") != 1:
                raise ValueError("Prepared calibration requires its explicit handoff configuration")
        elif kind == "calibrate" and spec.get("schema_version") not in (1, 2):
            raise ValueError("Calibration requires the existing native workflow configuration")
        elif kind == "simulate":
            if set(spec) - {"registry", "forcing_csv", "reservoir_id", "grand_id", "initial_state"} or not {"registry", "forcing_csv"} <= set(spec):
                raise ValueError("Simulation needs registry and forcing_csv")
            if ("reservoir_id" in spec) == ("grand_id" in spec):
                raise ValueError("Select exactly one reservoir_id or grand_id")
        result.append({"id": identifier, "type": kind, "backend": backend,
                       "config": spec, "depends_on": sorted(dependencies)})
        seen[identifier] = kind
    code = {**w.source_hashes(), **{p: w.sha(ROOT / p) for p in (
        "workflows/run_manager_v1.py", "workflows/station_inputs_v1.py", "tools/dztr_run.py",
        "workflows/run_manager_v2.py", "tools/dztr_run_v2.py",
        "workflows/prepared_calibration_v1.py", "tools/calibrate_prepared_v1.py",
        "workflows/data_availability_v1.py", "strategies/data_availability_v1.py",
        "workflows/geometry_v1.py", "workflows/observation_operators_v1.py", "workflows/profile_application_v1.py")}}
    from workflows import geometry_calibration_v1 as geometry
    code.update(geometry.sources())
    from workflows import flow_calibration_v1 as flow
    code.update(flow.sources())
    code.update({p: w.sha(ROOT / p) for p in ("workflows/run_manager_v4.py", "tools/dztr_run_v6.py")})
    body = {"kind": "dztr_run_plan_v4", "stages": result, "input_sha256": pinned, "code_sha256": code,
            "description": cfg.get("description", ""),
            "notes": ["Plan performs no download, simulation or optimization.",
                      "Prepared observed-monthly calibration retains the supplied year mask and initial targets.",
                      "Geometry profiles use explicit datum references and proxy/band calibration objectives; held-out records never select references or parameters."]}
    return {**body, "signature": digest(body)}


def substitute(value, results):
    if isinstance(value, dict):
        return {k: substitute(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, results) for v in value]
    match = REFERENCE.fullmatch(value) if isinstance(value, str) else None
    if match:
        source, artifact = match.groups()
        if artifact not in results.get(source, {}):
            raise ValueError(f"Required stage output unavailable: {value}")
        return results[source][artifact]
    return value


def _command(args, folder):
    with (folder / "execution.log").open("a", encoding="utf-8") as log:
        completed = subprocess.run([sys.executable, "-B", *map(str, args)], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if completed.returncode:
        raise ValueError(f"Stage command failed; inspect {folder / 'execution.log'}")


def execute(stage, folder, results):
    spec = substitute(stage["config"], results)
    template = folder / "resolved_config.json"
    if template.exists() and w.read(template) != spec:
        raise ValueError("Resolved stage decisions changed")
    atomic_write(template, spec)
    kind = stage["type"]
    campaign = folder / "campaign"
    if kind == "acquire":
        return stations.acquire(spec, folder / "inputs")
    if kind == "parameterize":
        if spec["kind"] == "dztr_data_availability_v2":
            from workflows import data_availability_v2 as availability
        else:
            from workflows import data_availability_v1 as availability
        if campaign.exists():
            availability.verify(campaign)
        else:
            availability.prepare(template, campaign)
        return {"campaign": str(campaign), "parameter_table": str(campaign / "report/reservoir_profiles_by_GRAND_ID.csv"),
                "profile_registry": str(campaign / "profile_registry.json")}
    if kind == "baseline":
        from workflows import data_availability_v1 as availability
        summary = availability.run_baselines(spec["campaign"], reservoir_id=spec.get("reservoir_id"))
        atomic_write(folder / "baseline_execution.json", summary)
        return {"summary": str(folder / "baseline_execution.json")}
    script = ROOT / "tools" / ("reservoir_workflow.py" if stage["backend"] == "v1" else "reservoir_workflow_v2.py")
    if kind == "calibrate" and stage["backend"] == "prepared_flow_v1":
        _command([ROOT / "tools/calibrate_flow_v1.py", "run", "--config", template, "--output", campaign], folder)
        budget = spec["optimization"]["evaluation_budget"]
        release = campaign / "releases" / f"budget_{budget}"
        arm = release / spec["objectives"][0]
        outputs = {"campaign": str(campaign),
                   "parameter_table": str(arm / "reservoir_parameterization_by_GRAND_ID.csv"),
                   "assessment_report": str(release / "baseline_comparison.csv")}
        if (arm / "registry.json").exists(): outputs["registry"] = str(arm / "registry.json")
        if len(spec["profiles"]) == 1:
            outputs["forcing_csv"] = str(campaign / "prepared" / spec["profiles"][0]["reservoir_id"] / "observations_si.csv")
        return outputs
    if kind == "calibrate" and stage["backend"] == "prepared_geometry_v1":
        _command([ROOT / "tools/calibrate_geometry_v1.py", "run", "--config", template, "--output", campaign], folder)
        budget = spec["optimization"]["evaluation_budget"]
        registry = campaign / "releases" / f"budget_{budget}" / "registry.json"
        outputs = {"campaign": str(campaign), "registry": str(registry),
                   "parameter_table": str(registry.parent / "reservoir_parameterization_by_GRAND_ID.csv"),
                   "assessment_report": str(registry.parent / "baseline_comparison.csv")}
        if len(spec["profiles"]) == 1:
            outputs["forcing_csv"] = str(campaign / "prepared" / spec["profiles"][0]["reservoir_id"] / "observations_si.csv")
        return outputs
    if kind == "calibrate" and stage["backend"] == "prepared_v1":
        _command([ROOT / "tools/calibrate_prepared_v1.py", "run", "--config", template, "--output", campaign], folder)
        budget = spec.get("optimization", {}).get("evaluation_budget", 512)
        registry = campaign / "published" / f"budget_{budget}" / "registry.json"
        outputs = {"campaign": str(campaign), "registry": str(registry),
                   "parameter_table": str(registry.parent / "reservoir_parameterization_by_GRAND_ID.csv"),
                   "assessment_report": str(registry.parent / "baseline_comparison.csv")}
        if len(spec["profiles"]) == 1:
            outputs["forcing_csv"] = str(campaign / "prepared" / spec["profiles"][0]["reservoir_id"] / "observations_si.csv")
        return outputs
    if kind == "calibrate":
        _command([script, "run", "--config", template, "--output", campaign], folder)
        registry = campaign / "releases" / f"budget_{spec.get('optimization', {}).get('evaluation_budget', 512)}" / "registry.json"
        return {"campaign": str(campaign), "registry": str(registry)}
    registry_kind = w.read(spec["registry"])
    if registry_kind.get("provenance", {}).get("method") == "flow_only":
        script = ROOT / "tools/calibrate_flow_v1.py"
    args = [script, "simulate", "--registry", spec["registry"], "--forcing-csv", spec["forcing_csv"], "--output", campaign]
    for key in ("reservoir_id", "grand_id", "initial_state"):
        if key in spec:
            args.extend(["--" + key.replace("_", "-"), spec[key]])
    # A simulation output is immutable; inspect partial runs before retrying.
    if campaign.exists():
        raise ValueError("Incomplete simulation directory exists; preserve it and use a new run directory")
    _command(args, folder)
    return {"campaign": str(campaign), "simulation_csv": str(campaign / "simulation.csv")}


def _artifacts(stage, outputs, folder):
    paths = [folder / "resolved_config.json"]
    if stage["type"] == "acquire":
        paths.extend(p for p in (folder / "inputs").rglob("*") if p.is_file())
    elif stage["type"] == "parameterize":
        campaign = Path(outputs["campaign"])
        paths.extend([campaign / "protocol.json", campaign / "config.json", campaign / "profile_registry.json"])
        paths.extend(p for p in (campaign / "prepared").rglob("*") if p.is_file())
    elif stage["type"] in {"simulate", "calibrate"}:
        campaign = Path(outputs["campaign"])
        paths.extend(p for p in campaign.rglob("*") if p.is_file() and not p.name.endswith(".log"))
    else:
        paths.append(Path(outputs["summary"]))
        campaign = Path(w.read(folder / "resolved_config.json")["campaign"])
        paths.extend(p for p in (campaign / "jobs").rglob("*") if p.is_file() and p.name != "run.lock")
    return {str(p.resolve()): w.sha(p) for p in paths}


def _report(out, state):
    from workflows.geometry_calibration_v1 import write
    write(out / "status.json", state)
    lines = ["# DZTR managed run", "", f"Run status: **{state['status']}**", "",
             "| Stage | Operation | Status |", "|---|---|---|"]
    for item in state["stages"]:
        lines.append(f"| {item['id']} | {item['type']} | {item['status']} |")
    lines += ["", "Scientific outputs, effective configuration and logs are stored per stage.",
              "A completed orchestration stage does not certify the sources or imply every reservoir succeeded.",
              "Inspect parameter readiness and individual baseline/search receipts before using a profile.", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def verify_inputs(design):
    for path, expected in design["input_sha256"].items():
        if w.sha(path) != expected:
            raise ValueError(f"Run input changed during execution: {path}")
    for relative, expected in design["code_sha256"].items():
        if w.sha(ROOT / relative) != expected:
            raise ValueError(f"Run code changed during execution: {relative}")


def run(config_path, output, *, until=None, resume=False):
    design = plan(config_path)
    ids = [s["id"] for s in design["stages"]]
    if until is not None and until not in ids:
        raise ValueError("--until must name a configured stage ID")
    out = Path(output).resolve()
    if not out.is_relative_to(ROOT / "outputs") or out == ROOT / "outputs":
        raise ValueError("Managed runs must be inside repository outputs")
    if out.exists():
        if not resume:
            raise ValueError("Existing run requires --resume")
        if w.read(out / "plan.json") != design:
            raise ValueError("Run inputs, decisions or code changed; use a new output directory")
    else:
        out = w.fresh_output(out)
        atomic_write(out / "plan.json", design)
    lock = out / "manager.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    state = {"status": "running", "updated_utc": now(), "signature": design["signature"],
             "stages": [{"id": s["id"], "type": s["type"], "status": "pending"} for s in design["stages"]]}
    results = {}
    try:
        for stage, row in zip(design["stages"], state["stages"]):
            row["status"] = "running"
            verify_inputs(design)
            folder = out / "stages" / stage["id"]
            folder.mkdir(parents=True, exist_ok=True)
            receipt = folder / "receipt.json"
            if receipt.exists():
                proof = w.read(receipt)
                if proof["plan_signature"] != design["signature"]:
                    raise ValueError("Stage receipt belongs to different decisions")
                for path, expected in proof["artifact_sha256"].items():
                    if w.sha(path) != expected:
                        raise ValueError(f"Completed stage artifact changed: {path}")
                results[stage["id"]] = proof["outputs"]
                row.update(status="reused", outputs=proof["outputs"])
            else:
                row["status"] = "running"
                _report(out, state)
                started = now()
                outputs = execute(stage, folder, results)
                verify_inputs(design)
                proof = {"plan_signature": design["signature"], "stage_id": stage["id"], "started_utc": started,
                         "completed_utc": now(), "outputs": outputs, "artifact_sha256": _artifacts(stage, outputs, folder)}
                atomic_write(receipt, proof)
                results[stage["id"]] = outputs
                row.update(status="completed", outputs=outputs)
            print(json.dumps({"stage": stage["id"], "status": row["status"]}), flush=True)
            if stage["id"] == until:
                break
        state["status"] = "stopped_at_requested_stage" if any(s["status"] == "pending" for s in state["stages"]) else "completed"
    except Exception as exc:
        state.update(status="failed", reason=str(exc))
        for row in state["stages"]:
            if row["status"] == "running":
                row.update(status="failed", reason=str(exc))
        raise
    finally:
        state["updated_utc"] = now()
        try: _report(out, state)
        finally: lock.unlink(missing_ok=True)
    return state
