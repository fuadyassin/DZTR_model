"""Import additional Etienne observations and descriptive DZTR target estimates.

This is a data collection extension. No inflow is reconstructed, no geometry or
storage datum is inferred, and descriptive all-record estimates are not promoted
to calibration-ready parameters. Existing collections remain immutable.
"""
from datetime import datetime
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from dztr.schemas import MONTHLY_TARGET_COLUMNS
from strategies.data_availability_v1 import empirical_targets
from workflows import reservoir_v2 as native
from workflows import profile_application_v1 as application
from workflows.atomic_io_v1 import write
from workflows.geometry_v1 import digest

ROOT = native.ROOT


def sources():
    return {**native.source_hashes(), **{p: native.sha(ROOT / p) for p in (
        "workflows/etienne_collection_v1.py", "strategies/data_availability_v1.py", "workflows/geometry_v1.py",
        "workflows/atomic_io_v1.py", "workflows/profile_application_v1.py", "tools/import_etienne_reservoirs_v1.py")}}


def normalized(value): return re.sub(r"[^a-z0-9]", "", value.lower())


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value): raise ValueError("Unsafe reservoir identifier")
    return value


def source_path(archive, relative):
    path = (Path(archive) / relative).resolve(strict=True)
    if not path.is_relative_to(Path(archive).resolve()) or not path.is_file(): raise ValueError("Source escapes the archive")
    return path


def config(path):
    path = Path(path).resolve(strict=True); cfg = native.read(path)
    required = {"kind", "schema_version", "archive", "base_collection", "base_catalog", "grand_reference", "reservoirs", "existing_aliases", "quantile_method"}
    if set(cfg) != required or (cfg["kind"], cfg["schema_version"]) != ("dztr_etienne_collection_v1", 1): raise ValueError("Unknown collection import configuration")
    for key in ("archive", "base_collection", "base_catalog", "grand_reference"):
        cfg[key] = str((path.parent / cfg[key]).resolve(strict=True))
    if cfg["quantile_method"] not in {"hazen", "linear"}: raise ValueError("Declare percentile estimator")
    if not isinstance(cfg["reservoirs"], list) or not cfg["reservoirs"]: raise ValueError("Declare additional reservoirs")
    ids, identities = [], set()
    aliases = set()
    for old in cfg["existing_aliases"]:
        if set(old) != {"alias", "reservoir_id"}: raise ValueError("Declare existing identity aliases explicitly")
        aliases.add(normalized(old["alias"]))
    used = set()
    for entry in cfg["reservoirs"]:
        if set(entry) - {"reservoir_id", "name", "catalog_terms", "storage", "release"} or not {"reservoir_id", "name", "catalog_terms"} <= set(entry):
            raise ValueError("Invalid reservoir import specification")
        ids.append(safe_id(entry["reservoir_id"]))
        identity = normalized(entry["name"])
        if identity in aliases or identity in identities: raise ValueError("Duplicate existing or new reservoir identity")
        identities.add(identity)
        if not {"storage", "release"} & set(entry): raise ValueError("A new record must include observations")
        for variable in ("storage", "release"):
            if variable not in entry: continue
            spec = entry[variable]
            required_source = {"file", "value_column", "factor", "source_units", "missing_values", "missing_tokens", "duplicate_policy", "evidence", "statistics"}
            if set(spec) != required_source or not spec["evidence"]: raise ValueError("Declare complete observation normalization evidence")
            if isinstance(spec["value_column"], bool) or not isinstance(spec["value_column"], int) or spec["value_column"] < 1: raise ValueError("Value column includes date as column zero")
            if isinstance(spec["factor"], bool) or not isinstance(spec["factor"], (int, float)) or not np.isfinite(spec["factor"]) or spec["factor"] <= 0: raise ValueError("Invalid SI conversion")
            if spec["duplicate_policy"] not in {"reject", "daily_arithmetic_mean_proxy"}: raise ValueError("Declare duplicate-day treatment")
            if variable == "release" and spec["duplicate_policy"] != "reject": raise ValueError("Untimed discharge cannot be converted to a volume-weighted daily mean")
            if not isinstance(spec["missing_values"], list) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in spec["missing_values"]): raise ValueError("Invalid missing-value declaration")
            if not isinstance(spec["missing_tokens"], list) or any(not isinstance(v, str) or not v for v in spec["missing_tokens"]): raise ValueError("Invalid missing-token declaration")
            if spec["file"] in used: raise ValueError("A source file cannot silently represent two reservoirs/variables")
            used.add(spec["file"])
    if len(ids) != len(set(ids)): raise ValueError("Duplicate reservoir IDs")
    return path, cfg


def read_observations(path, spec):
    """Select an explicitly declared field and retain daily missing support."""
    dates, values, stats = {}, {}, {"blank_lines": 0, "missing_rows": 0, "declared_sentinel_rows": 0,
        "source_rows": 0, "extra_field_rows": 0, "uninterpreted_negative_rows": 0, "declared_missing_text_rows": 0}
    previous, parsed = None, None
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields: stats["blank_lines"] += 1; continue
            try:
                if fields[0] != previous:
                    parsed = pd.Timestamp(datetime.strptime(fields[0], "%Y-%m-%d" if "-" in fields[0] else "%Y%m%d"))
                    previous = fields[0]
            except ValueError as exc: raise ValueError(f"Invalid date at {Path(path).name}:{number}") from exc
            stats["source_rows"] += 1
            stats["extra_field_rows"] += int(len(fields) > spec["value_column"] + 1)
            if parsed in dates and spec["duplicate_policy"] == "reject": raise ValueError(f"Duplicate day at {Path(path).name}:{number}; no observation time was provided")
            dates[parsed] = dates.get(parsed, 0) + 1
            token = fields[spec["value_column"]] if len(fields) > spec["value_column"] else "nan"
            if token in spec.get("missing_tokens", []):
                stats["declared_missing_text_rows"] += 1; token = "nan"
            try: raw = float(token)
            except ValueError as exc:
                if token.lower() in {"na", "n/a", "--", "---"}: raw = float("nan")
                else: raise ValueError(f"Unknown observation token at {Path(path).name}:{number}") from exc
            if np.isinf(raw): raise ValueError(f"Infinite observation at {Path(path).name}:{number}")
            if raw in spec["missing_values"]:
                stats["declared_sentinel_rows"] += 1; raw = float("nan")
            if np.isnan(raw): stats["missing_rows"] += 1
            if raw < 0: stats["uninterpreted_negative_rows"] += 1
            values.setdefault(parsed, []).append(raw * spec["factor"])
    if not dates: raise ValueError("Observation file contains no dates")
    index = pd.date_range(min(dates), max(dates), freq="D", name="date")
    daily, counts = [], []
    for day in index:
        sample = np.asarray(values.get(day, []), float); valid = sample[np.isfinite(sample)]
        daily.append(float(valid.mean()) if len(valid) else np.nan)
        counts.append({"source_rows": dates.get(day, 0), "valid_samples": len(valid),
            "minimum_si": float(valid.min()) if len(valid) else np.nan, "maximum_si": float(valid.max()) if len(valid) else np.nan})
    result = pd.Series(daily, index=index, dtype=float)
    quality = pd.DataFrame(counts, index=index)
    stats.update(first_date=index[0].date().isoformat(), last_date=index[-1].date().isoformat(), distinct_source_dates=len(dates))
    stats.update(calendar_days=len(index), missing_calendar_days=int(sum(d not in dates for d in index)),
        valid_days=int(result.notna().sum()), zero_days=int(result.eq(0).sum()),
        repeated_date_rows=sum(dates.values()) - len(dates),
        observation_basis="daily_arithmetic_mean_of_untimed_samples_proxy" if spec["duplicate_policy"] != "reject" else "source_daily_date_label_timing_unverified")
    return result, quality, stats


def target_components(frame, method):
    mask = pd.Series(True, index=frame.index, dtype=bool)
    targets = pd.DataFrame(np.nan, index=pd.Index(range(1, 13), name="month"), columns=MONTHLY_TARGET_COLUMNS)
    counts = pd.DataFrame(index=targets.index)
    for variable, column, cols in (("storage", "storage_m3", MONTHLY_TARGET_COLUMNS[:3]), ("release", "release_m3s", MONTHLY_TARGET_COLUMNS[3:])):
        values = frame[column]
        if values.dropna().lt(0).any():
            counts[variable + "_samples"] = [int(values[(values.index.month == month) & values.ge(0)].notna().sum()) for month in range(1, 13)]
            continue  # retain unresolved negatives in observations; no silent rectification
        estimates, n = empirical_targets(values, mask, monthly=True, percentiles=(10, 45, 85), method=method)
        targets.loc[:, list(cols)] = estimates; counts[variable + "_samples"] = n
    return targets, counts


def catalog_candidates(frame, terms):
    columns = ("RES_NAME", "DAM_NAME", "ALT_NAME")
    found = []
    for row in frame.to_dict("records"):
        if any(normalized(term) in normalized(str(row.get(col, ""))) for term in terms for col in columns):
            found.append({k: (None if pd.isna(v) else v) for k, v in row.items()})
    # These are search leads, never verified ID assignments or adopted capacities.
    return found


def verify(collection):
    out = Path(collection).resolve(strict=True)
    proof = native.read(out / "manifest.json")
    if proof["kind"] != "etienne_reservoir_collection_extension_v1": raise ValueError("Wrong collection kind")
    for name, expected in proof["artifact_sha256"].items():
        path = (out / name).resolve(strict=True)
        if not path.is_relative_to(out) or native.sha(path) != expected: raise ValueError("Collection artifact changed: " + name)
    return out, proof


def load_observations(collection, *, reservoir_id=None, grand_id=None):
    out, _ = verify(collection)
    if (reservoir_id is None) == (grand_id is None): raise ValueError("Select one reservoir ID or verified GRAND ID")
    rows = native.read(out / "combined_inventory.json")["reservoirs"]
    found = [r for r in rows if r["reservoir_id"] == reservoir_id] if reservoir_id is not None else [r for r in rows if r.get("GRAND_ID") == grand_id]
    if len(found) != 1: raise ValueError("No unique verified reservoir mapping")
    record = found[0]; path = (ROOT / record["observations_file"]).resolve(strict=True)
    if not path.is_relative_to(ROOT) or native.sha(path) != record["observations_sha256"]: raise ValueError("Observation file changed")
    frame = pd.read_csv(path, index_col="date", parse_dates=True, float_precision="round_trip")
    return record, frame


def build(config_path, collection, report):
    path, cfg = config(config_path)
    out = Path(collection).resolve()
    if out.exists():
        _, proof = verify(out)
        if proof["config_signature"] != digest(cfg): raise ValueError("Existing collection has different import decisions")
        return write_report(out, report)
    if not any(out.is_relative_to(ROOT / name) and out != ROOT / name for name in ("data", "outputs")): raise ValueError("Use a fresh project data/output directory")
    base = native.read(cfg["base_collection"]); catalog = application.read_catalog(cfg["base_catalog"])
    if base["kind"] != "reservoir_collection": raise ValueError("Wrong base collection")
    known = {r["reservoir_id"] for r in base["reservoirs"]}
    if any(e["reservoir_id"] in known for e in cfg["reservoirs"]): raise ValueError("New ID collides with an existing reservoir")
    known_names = {normalized(r["name"]) for r in base["reservoirs"]}
    if any(normalized(e["name"]) in known_names for e in cfg["reservoirs"]): raise ValueError("New name duplicates an existing reservoir")
    if any(a["reservoir_id"] not in known for a in cfg["existing_aliases"]): raise ValueError("Existing alias has no known reservoir")
    for record in base["reservoirs"]:
        for rel, expected in record["files_sha256"].items():
            if native.sha(ROOT / rel) != expected: raise ValueError("Base input changed before import: " + rel)
    original = {str(Path(cfg[key])): native.sha(cfg[key]) for key in ("base_collection", "base_catalog", "grand_reference")}
    original[str(path)] = native.sha(path)
    grand = pd.read_csv(cfg["grand_reference"], keep_default_na=False)
    out.mkdir(parents=True)
    write(out / "config.json", cfg)
    source_folder = out / "source_evidence"; source_folder.mkdir()
    reader = source_path(cfg["archive"], "compute_monthly_statistics.py")
    original[str(reader)] = native.sha(reader); shutil.copyfile(reader, source_folder / reader.name)
    shutil.copyfile(cfg["grand_reference"], source_folder / "grand_reference.csv")
    registry, parameter_rows = [], []
    for entry in cfg["reservoirs"]:
        rid = entry["reservoir_id"]; folder = out / rid; (folder / "raw").mkdir(parents=True)
        parts, provenance = {}, {}
        for variable, column in (("storage", "storage_m3"), ("release", "release_m3s")):
            if variable not in entry: continue
            spec = entry[variable]; source = source_path(cfg["archive"], spec["file"])
            expected = native.sha(source); original[str(source)] = expected
            raw = folder / "raw" / source.name; shutil.copyfile(source, raw)
            if native.sha(raw) != expected: raise ValueError("Observation changed during archive copy")
            values, quality, receipt = read_observations(raw, spec)
            parts[column] = values
            quality.to_csv(folder / (variable + "_daily_quality.csv"), index_label="date", float_format="%.17g")
            for name in spec["statistics"]:
                stat = source_path(cfg["archive"], name); original[str(stat)] = native.sha(stat)
                shutil.copyfile(stat, folder / "raw" / stat.name)
            provenance[variable] = {**receipt, **spec, "source_file": str(source), "source_sha256": expected,
                "source_units_evidence": "Etienne reader multiplies storage by 1e6; archive statistics declare SI output. Flow field/units are explicitly selected in this import.",
                "missing_value_interpretation_certified": False, "time_basis_verified": False,
                "storage_definition": "unresolved_total_active_or_datum_based" if variable == "storage" else None}
        index = pd.date_range(min(v.index[0] for v in parts.values()), max(v.index[-1] for v in parts.values()), freq="D", name="date")
        frame = pd.DataFrame({"inflow_m3s": np.nan, "storage_m3": np.nan, "release_m3s": np.nan}, index=index)
        for column, values in parts.items(): frame[column] = values.reindex(index)
        frame.to_csv(folder / "observations_si.csv", index_label="date", float_format="%.17g")
        targets, counts = target_components(frame, cfg["quantile_method"])
        targets.to_csv(folder / "monthly_targets_si.csv", float_format="%.17g")
        counts.to_csv(folder / "monthly_sample_counts.csv")
        candidates = catalog_candidates(grand, entry["catalog_terms"])
        storage_ready = bool(targets.iloc[:, :3].notna().all().all())
        release_ready = bool(targets.iloc[:, 3:].notna().all().all())
        route = "observed_monthly" if storage_ready and release_ready else "storage_only" if storage_ready else "flow_only" if release_ready else None
        gates = ["needs_time_varying_inflow", "needs_adopted_physical_model_metadata", "observation_timing_requires_review", "descriptive_targets_must_be_refitted_on_frozen_calibration_years"]
        if "storage" in provenance: gates.append("storage_coordinate_requires_review")
        if any(p["uninterpreted_negative_rows"] for p in provenance.values()): gates.append("uninterpreted_negative_observations")
        if any(p["duplicate_policy"] != "reject" for p in provenance.values()): gates.append("untimed_within_day_storage_mean_proxy")
        if not storage_ready: gates.append("storage_targets_require_capacity_assumption")
        if not release_ready: gates.append("release_targets_require_explicit_prior")
        notes = []
        if "storage" in provenance and min(counts.storage_samples) < 5 * 28: notes.append("short_storage_record_in_at_least_one_month")
        for candidate in candidates:
            if storage_ready and candidate.get("CAP_MCM"):
                cap = float(candidate["CAP_MCM"]) * 1e6
                if cap > 0:
                    candidate["observed_storage_max_over_candidate_capacity"] = float(frame.storage_m3.max() / cap)
        channel = float(np.percentile(frame.release_m3s.dropna(), 99, method=cfg["quantile_method"])) if release_ready else None
        record = {"reservoir_id": rid, "name": entry["name"], "GRAND_ID": None, "is_paper_reservoir": False,
            "collection_status": "new_observations_and_descriptive_targets", "method_from_response_availability": route,
            "storage_target_component_available": storage_ready, "release_target_component_available": release_ready,
            "all_72_target_estimates_available": storage_ready and release_ready,
            "parameters_ready": False, "simulation_ready": False, "calibration_ready": False,
            "inflow_available": False, "storage_valid_days": int(frame.storage_m3.notna().sum()), "release_valid_days": int(frame.release_m3s.notna().sum()),
            "simultaneous_storage_release_days": int(frame[["storage_m3", "release_m3s"]].notna().all(axis=1).sum()),
            "first_date": index[0].date().isoformat(), "last_date": index[-1].date().isoformat(), "rows": len(frame),
            "fitting_scope": "all_available_records_descriptive_no_assessment", "quantile_method": cfg["quantile_method"],
            "valid_zeros_retained": True, "channel_p99_proxy_m3s": channel, "capacity_m3": None,
            "source_gates": gates, "quality_notes": notes, "catalog_candidates": candidates,
            "GRAND_mapping_verified": False, "source_certified": False,
            "observations_file": (folder / "observations_si.csv").relative_to(ROOT).as_posix(),
            "observations_sha256": native.sha(folder / "observations_si.csv"),
            "targets_file": (folder / "monthly_targets_si.csv").relative_to(ROOT).as_posix()}
        write(folder / "observation_provenance.json", provenance)
        write(folder / "parameterization.json", record)
        registry.append(record)
        row = {k: record[k] for k in ("reservoir_id", "name", "GRAND_ID", "method_from_response_availability", "storage_target_component_available", "release_target_component_available", "all_72_target_estimates_available", "parameters_ready", "simulation_ready", "calibration_ready", "fitting_scope", "quantile_method", "channel_p99_proxy_m3s")}
        row["source_gates"] = "; ".join(gates)
        row["candidate_GRAND_IDs"] = ",".join(str(c["GRAND_ID"]) for c in candidates)
        for month, values in targets.iterrows():
            for column, value in values.items(): row[f"{column}_month_{month:02d}"] = value
        parameter_rows.append(row)
    modern = {r["reservoir_id"]: r for r in catalog["reservoirs"]}
    inventory = []
    for old in base["reservoirs"]:
        current = modern.get(old["reservoir_id"], {})
        inventory.append({"reservoir_id": old["reservoir_id"], "name": old["name"], "GRAND_ID": current.get("GRAND_ID"),
            "collection_status": "existing_preserved", "is_paper_reservoir": old["reservoir_id"] in modern,
            "monthly_calibrated_profile_available": current.get("monthly_profile_available"),
            "first_date": old["first_date"], "last_date": old["last_date"], "rows": old["rows"], "observations_file": old["observations_file"], "observations_sha256": old["observations_sha256"],
            "existing_profile_status": current.get("status"), "legacy_config_file": old["config_file"]})
    inventory += [{k: r[k] for k in ("reservoir_id", "name", "GRAND_ID", "collection_status", "is_paper_reservoir", "first_date", "last_date", "rows", "observations_file", "observations_sha256", "storage_target_component_available", "release_target_component_available", "all_72_target_estimates_available", "parameters_ready", "simulation_ready", "calibration_ready")} for r in registry]
    pd.DataFrame(parameter_rows).to_csv(out / "new_reservoir_parameterization_by_GRAND_ID.csv", index=False, float_format="%.17g")
    pd.DataFrame(inventory).to_csv(out / "reservoir_inventory_all.csv", index=False)
    write(out / "parameterization_registry.json", {"kind": "descriptive_dztr_target_registry_v1", "profiles": registry,
        "optimizer_evaluations": 0, "simulation_runs": 0, "all_record_targets_are_calibration_ready": False})
    write(out / "combined_inventory.json", {"kind": "reservoir_collection_index_v2", "base_collection": str(Path(cfg["base_collection"]).relative_to(ROOT)),
        "base_collection_sha256": native.sha(cfg["base_collection"]), "excluded_reservoirs": base.get("excluded_reservoirs", []),
        "existing_aliases": cfg["existing_aliases"], "reservoirs": inventory})
    for name, expected in original.items():
        if native.sha(name) != expected: raise ValueError("Source changed during import: " + name)
    proof = {"kind": "etienne_reservoir_collection_extension_v1", "schema_version": 1, "config_signature": digest(cfg),
        "existing_reservoirs": len(base["reservoirs"]), "added_reservoirs": len(registry), "combined_reservoirs": len(inventory),
        "new_storage_components": sum(r["storage_target_component_available"] for r in registry),
        "new_release_components": sum(r["release_target_component_available"] for r in registry),
        "new_complete_target_tables": sum(r["all_72_target_estimates_available"] for r in registry),
        "new_calibration_ready_reservoirs": 0, "optimizer_evaluations": 0, "new_verified_GRAND_IDs": 0,
        "input_sha256": original, "source_sha256": sources(), "artifact_sha256": {
            p.relative_to(out).as_posix(): native.sha(p) for p in out.rglob("*") if p.is_file()}}
    write(out / "manifest.json", proof)
    return write_report(out, report)


def write_report(collection, output):
    out, proof = verify(collection)
    report = Path(output).resolve()
    if report.exists():
        saved = native.read(report / "manifest.json")
        if saved["collection_manifest_sha256"] != native.sha(out / "manifest.json"): raise ValueError("Report belongs to another collection")
        for name, expected in saved["artifact_sha256"].items():
            if native.sha(report / name) != expected: raise ValueError("Report changed")
        return report
    report = native.fresh_output(report)
    rows = native.read(out / "parameterization_registry.json")["profiles"]
    for name in ("new_reservoir_parameterization_by_GRAND_ID.csv", "reservoir_inventory_all.csv"):
        shutil.copyfile(out / name, report / name)
    lines = ["# Reservoir collection expanded from Etienne's archive", "",
        f"Added **{proof['added_reservoirs']} reservoirs** to the database index: **{proof['existing_reservoirs']} existing + {proof['added_reservoirs']} new = {proof['combined_reservoirs']}**. Existing data, calibrated profiles and the 37-reservoir paper inventory were preserved. Tuyen Quang remains excluded.", "",
        f"Prepared **{proof['new_storage_components']} storage-target components**, **{proof['new_release_components']} release-target components**, and **{proof['new_complete_target_tables']} complete 12-by-6 target tables**. These are descriptive all-record Hazen percentile estimates, not optimized or calibration-ready profiles. Valid zero observations remain in the estimates. Missing targets remain blank.", "",
        "Every new record currently lacks an identified inflow time series and adopted physical/initial-state metadata. Storage coordinates and observation timing also require reconciliation. No inflow reconstruction, new simulation, optimizer search, capacity substitution or verified GRAND-ID assignment was performed. Catalog matches are explicitly search candidates.", "",
        "| Reservoir | Suggested response-data route | Storage days | Release days | Paired days | Complete targets | Notes |",
        "|---|---|---:|---:|---:|---|---|"]
    for r in rows:
        notes = r["quality_notes"][:]
        if "untimed_within_day_storage_mean_proxy" in r["source_gates"]: notes.append("daily storage averaging proxy")
        lines.append(f"| {r['name']} | {r['method_from_response_availability']} | {r['storage_valid_days']} | {r['release_valid_days']} | {r['simultaneous_storage_release_days']} | {'yes' if r['all_72_target_estimates_available'] else 'partial'} | {'; '.join(notes)} |")
    lines += ["", "The archive's storage reader explicitly multiplies its input by one million; imported storage is therefore normalized from million m³ to m³ with that source evidence retained. Declared -1/-99.99 placeholders become missing values, with counts and raw files preserved. Their interpretation remains an explicit import assumption. Lake Chelan uses the already converted m³/s column, rather than its adjacent cfs column. Hungry Horse retains the observed input values; the archive's 0.993 experiment multiplier is not a unit conversion and is not applied.", "",
        "Wrightsville contains repeated dates without within-day times. Its explicitly requested arithmetic daily storage mean is labeled a proxy, with daily sample counts and ranges saved. It is not an end-of-day or exact time-weighted mean record. Its original statistics overstated coverage by counting within-day rows as days; this import reports actual distinct dates.", "",
        "For each reservoir, observations_si.csv, monthly_targets_si.csv, monthly_sample_counts.csv, raw source files, normalization provenance and parameterization.json are archived in the permanent collection. Missing inflow is NaN. Each component retains its own observed date support; unequal storage/release periods and their actual overlap are reported. Existing names such as American Falls and Williston/Bennett are aliases of existing records and are not added again.", "",
        "Next admission steps: identify independent inflow or explicitly authorize and validate a conditional reconstruction; establish physical metadata/storage definitions and any GRAND mapping; resolve observation timing; then use the existing run manager to select mixed years with at least 65% calibration, refit targets/bounds on those years, and assess the held-out responses. The full-record descriptive profiles must not initialize a held-out calibration benchmark without that refitting.", "",
        f"Permanent collection: `{out.relative_to(ROOT).as_posix()}`.", "",
        "[All-reservoir inventory](reservoir_inventory_all.csv) · [New target estimates](new_reservoir_parameterization_by_GRAND_ID.csv)", ""]
    (report / "README.md").write_text("\n".join(lines), encoding="utf-8")
    write(report / "manifest.json", {"collection_manifest_sha256": native.sha(out / "manifest.json"), "artifact_sha256": {
        p.name: native.sha(p) for p in report.iterdir() if p.is_file()}})
    return report
