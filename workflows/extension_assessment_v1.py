"""Batch source admission and explicitly conditional baseline probes.

No missing records are filled, no capacity/datum is inferred, and a baseline
probe never promotes a source-gated record to a calibration-ready profile.
"""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from dztr.data.intervals import DailyIntervalPolicy, prepare_daily_intervals
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters
from dztr.simulation.intervals import (
    IntervalSimulationError, IntervalSimulationSettings, simulate_interval_reservoir,
)
from workflows import etienne_collection_v1 as collection
from workflows import profile_application_v1 as application

ROOT = collection.ROOT
SECONDS = 86400.
ACRE_FOOT_M3 = 1233.48183754752
CFS_M3S = .028316846592
COLUMNS = ["inflow_m3s", "storage_m3", "release_m3s"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def frame_read(path):
    return pd.read_csv(path, index_col="date", parse_dates=True, float_precision="round_trip")


def daily(frame):
    index = frame.index
    if (not isinstance(index, pd.DatetimeIndex) or len(index) < 2 or index.hasnans
            or index.tz is not None or not index.equals(index.normalize())
            or not index.equals(pd.date_range(index[0], index[-1], name=index.name))):
        raise ValueError("Use a unique consecutive daily calendar; gaps must remain explicit rows")
    if np.isinf(frame[COLUMNS].to_numpy(float)).any():
        raise ValueError("Infinite observations are invalid")


def blocks(mask):
    """Half-open positions of true runs; never connect across a missing day."""
    values = np.asarray(mask, bool)
    starts = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    ends = np.flatnonzero(values & ~np.r_[values[1:], False]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def conditional_inflow(frame):
    """End-labelled interval mean Q + dS/dt; negative net flux is retained.

    Storage endpoints and current mean release must exist. This is an algebraic
    boundary flux, not an independently observed gross catchment inflow.
    """
    daily(frame)
    for column in ("storage_m3", "release_m3s"):
        if (frame[column].dropna() < 0).any():
            raise ValueError("Negative storage/release requires a source decision")
    result = frame.copy()
    result["storage_change_m3"] = frame.storage_m3.diff()
    result["inflow_m3s"] = frame.release_m3s + result.storage_change_m3 / SECONDS
    result["reconstruction_closure_m3"] = (
        result.storage_change_m3 - SECONDS * (result.inflow_m3s - frame.release_m3s))
    result["uses_current_storage_response"] = result.inflow_m3s.notna()
    result["uses_current_release_response"] = result.inflow_m3s.notna()
    return result


def source_budget(frame, capacity=None, threshold=.02):
    """Diagnostic flags only: no flagged record is removed or repaired."""
    daily(frame)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Declare a positive diagnostic capacity fraction")
    if capacity is not None and (not np.isfinite(capacity) or capacity <= 0):
        raise ValueError("Capacity reference must be positive")
    result = frame.copy()
    result["storage_change_m3"] = frame.storage_m3.diff()
    result["reported_net_flow_volume_m3"] = SECONDS * (frame.inflow_m3s - frame.release_m3s)
    result["source_budget_residual_m3"] = result.storage_change_m3 - result.reported_net_flow_volume_m3
    result["residual_capacity_fraction"] = result.source_budget_residual_m3 / capacity if capacity else np.nan
    result["large_budget_residual_flag"] = result.residual_capacity_fraction.abs() > threshold
    result["storage_outside_reference_flag"] = ((frame.storage_m3 < 0) | (frame.storage_m3 > capacity)
        if capacity else frame.storage_m3 < 0)
    return result


def read_horizon(path):
    raw = pd.read_csv(path, index_col="date", parse_dates=True)
    expected = {"s_af", "i_cfs", "r_cfs"}
    if set(raw.columns) != expected: raise ValueError("Unexpected HORIZON fields")
    frame = raw.rename(columns={"s_af": "storage_m3", "i_cfs": "inflow_m3s", "r_cfs": "release_m3s"})
    frame.storage_m3 *= ACRE_FOOT_M3
    frame[["inflow_m3s", "release_m3s"]] *= CFS_M3S
    frame = frame[COLUMNS]; daily(frame)
    return frame


def conditional_summary(frame):
    values = frame.inflow_m3s.dropna()
    spans = blocks(frame.inflow_m3s.notna())
    closure = frame.reconstruction_closure_m3.abs().max()
    return {"usable_intervals": len(values), "negative_net_intervals": int((values < 0).sum()),
        "net_inflow_min_m3s": float(values.min()) if len(values) else None,
        "net_inflow_max_m3s": float(values.max()) if len(values) else None,
        "longest_inflow_run_days": max((b-a for a, b in spans), default=0),
        "max_reconstruction_closure_m3": float(closure) if pd.notna(closure) else None,
        "assessment_responses_used_in_forcing": True,
        "independent_inflow": False,
        "clock_assumption": "current release is a mean over 86400 s ending at current storage endpoint",
        "loss_attribution": "none; computed signed net boundary flux"}


def baseline_probe(frame, capacity, semantics, *, tolerance=1e-4):
    """All-record descriptive P10/P45/P85 sensitivity; not calibration.

    Capacity is an explicitly labelled reference, dead fraction zero an
    exploratory assumption and Q99 a release-capacity proxy. A failed block
    stops the probe; there is no search for a more successful subperiod.
    """
    daily(frame)
    if semantics not in {"physical_inflow", "computed_net_inflow"}:
        raise ValueError("Declare forcing semantics")
    proof = {"status": "not_started", "capacity_reference_m3": float(capacity),
        "capacity_adopted_for_production": False, "dead_storage_fraction_assumption": 0.,
        "channel_capacity_method": "all-record Hazen Q99 proxy",
        "target_fitting_scope": "all-record descriptive; no independent assessment",
        "optimizer_evaluations": 0, "calibration_ready": False,
        "root_policy": "nearest_previous_storage", "mass_balance_tolerance_m3": tolerance,
        "blocks_completed": 0, "intervals_completed": 0}
    valid = frame[COLUMNS].notna().all(axis=1)
    valid &= (frame.storage_m3 >= 0) & (frame.release_m3s >= 0)
    if semantics == "physical_inflow": valid &= frame.inflow_m3s >= 0
    spans = [(a, b) for a, b in blocks(valid) if b-a >= 2]
    proof["candidate_blocks"] = len(spans)
    if not spans:
        return {**proof, "status": "blocked", "reason": "No complete consecutive interval block"}, None, None
    targets, _ = collection.target_components(frame, "hazen")
    first = frame.iloc[spans[0][0]]
    channel = float(np.percentile(frame.release_m3s.dropna(), 99, method="hazen"))
    proof["channel_capacity_proxy_m3s"] = channel
    try:
        parameters = ReservoirParameters(targets, capacity_m3=capacity,
            initial_storage_m3=float(first.storage_m3), initial_release_m3s=float(first.release_m3s),
            channel_capacity_m3s=channel, dead_storage_fraction=0., inflow_correction=1., flood_mode="storage")
    except ValueError as exc:
        return {**proof, "status": "blocked", "reason": str(exc)}, None, targets
    parts = []
    for number, (a, b) in enumerate(spans):
        section = frame.iloc[a:b][COLUMNS]
        prep = prepare_daily_intervals(section, policy=DailyIntervalPolicy(flow_semantics=semantics))
        failure = None
        try:
            result = simulate_interval_reservoir(prep, parameters,
                settings=IntervalSimulationSettings(root_policy="nearest_previous_storage"), atol_storage_m3=tolerance)
        except IntervalSimulationError as exc:
            result, failure = exc.partial_result, str(exc)
        sim = result.frame
        completed = sim.interval_completed
        # Independently recompute volume closure, including physical spill.
        residual = (sim.storage_m3.diff() - section.inflow_m3s * SECONDS
                    + sim.interval_mean_total_outflow_m3s * SECONDS)
        maximum = float(residual[completed].abs().max()) if completed.any() else 0.
        if maximum > tolerance: raise ValueError("Independent mass-balance verification failed")
        proof["max_independent_balance_residual_m3"] = max(proof.get("max_independent_balance_residual_m3", 0.), maximum)
        proof["intervals_completed"] += int(completed.sum())
        sim = sim.copy(); sim["probe_block"] = number; sim["independent_balance_residual_m3"] = residual
        parts.append(sim)
        if failure:
            proof.update(status="failed", reason=failure, failure_block_start=str(section.index[0].date()))
            break
        proof["blocks_completed"] += 1
    else: proof["status"] = "completed_descriptive_probe"
    return proof, pd.concat(parts), targets


def verify(output):
    output = Path(output).resolve(strict=True)
    manifest = read(output / "manifest.json")
    if manifest.get("kind") != "reservoir_extension_assessment_v1": raise ValueError("Wrong assessment kind")
    for name, expected in manifest["artifact_sha256"].items():
        path = (output / name).resolve(strict=True)
        if not path.is_relative_to(output) or sha(path) != expected:
            raise ValueError("Assessment artifact changed: " + name)
    for name, expected in manifest["input_sha256"].items():
        if sha(ROOT / name) != expected: raise ValueError("Assessment input changed: " + name)
    return manifest


def _relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def build(config_path, output):
    """One batch, immutable artifacts, reusable receipts, no optimizer jobs."""
    config_path = Path(config_path).resolve(strict=True); cfg = read(config_path)
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / "outputs") or output == ROOT / "outputs":
        raise ValueError("Use a fresh project output directory")
    if output.exists():
        manifest = verify(output)
        if manifest["config_sha256"] != sha(config_path): raise ValueError("Assessment decisions changed")
        return manifest
    if cfg.get("kind") != "reservoir_extension_assessment_v1": raise ValueError("Wrong configuration kind")
    package = (ROOT / cfg["collection"]).resolve(strict=True)
    _, import_proof = collection.verify(package)
    index = read(package / "combined_inventory.json")
    new = [r for r in index["reservoirs"] if r["reservoir_id"].startswith("etienne_")]
    decisions = cfg["decisions"]
    if set(decisions) != {r["reservoir_id"] for r in new}: raise ValueError("Every new reservoir requires a decision")
    # Check original collection inputs before and after this assessment.
    pins = {_relative(package / "manifest.json"): sha(package / "manifest.json"), _relative(config_path): sha(config_path)}
    base_proof = read(ROOT / "data/reservoirs/manifest.json")
    for row in base_proof["reservoirs"]:
        for name, expected in row.get("files_sha256", {}).items():
            path = ROOT / name
            if sha(path) != expected: raise ValueError("Original collection changed")
            pins[name] = expected
    original_count = len(pins) - 2
    catalog_path = ROOT / cfg["profile_catalog"]
    catalog = application.read_catalog(catalog_path)
    old_profiles = {r["reservoir_id"]: r for r in catalog["reservoirs"]}
    pins[_relative(catalog_path)] = sha(catalog_path)
    for entry in catalog["reservoirs"]:
        pins.update(entry["input_sha256"])
    output.mkdir(parents=True)
    sources, diagnostics, probe_rows, new_rows, target_rows = {}, [], [], {}, []
    for source in cfg["supplementary_sources"]:
        name = collection.safe_id(source["key"])
        raw = Path(source["file"]).resolve(strict=True)
        destination = output / "supplementary_sources" / name
        destination.mkdir(parents=True)
        shutil.copyfile(raw, destination / "raw.csv")
        frame = read_horizon(destination / "raw.csv")
        frame.to_csv(destination / "observations_si.csv")
        audited = source_budget(frame, source.get("capacity_reference_m3"), cfg["budget_flag_capacity_fraction"])
        audited.to_csv(destination / "daily_source_budget.csv")
        largest = audited.loc[audited.source_budget_residual_m3.abs().nlargest(30).index]
        largest.to_csv(destination / "largest_source_residuals.csv")
        proof = {**source, "original_sha256": sha(raw), "storage_factor_m3_per_af": ACRE_FOOT_M3,
            "flow_factor_m3s_per_cfs": CFS_M3S, "first_date": str(frame.index[0].date()),
            "last_date": str(frame.index[-1].date()), "finite_counts": frame.notna().sum().to_dict(),
            "negative_storage_days": int((frame.storage_m3 < 0).sum()),
            "negative_reported_inflow_days": int((frame.inflow_m3s < 0).sum()),
            "large_budget_residual_intervals": int(audited.large_budget_residual_flag.sum()),
            "outside_capacity_reference_days": int(audited.storage_outside_reference_flag.sum()),
            "max_absolute_budget_residual_m3": float(audited.source_budget_residual_m3.abs().max()),
            "provider_inflow_independence_verified": False, "calibration_ready": False}
        write(destination / "provenance.json", proof)
        sources[name] = (source, frame, proof)
        if source.get("baseline_probe"):
            result, simulation, targets = baseline_probe(frame, source["capacity_reference_m3"], "physical_inflow")
            write(destination / "baseline_probe.json", result)
            if simulation is not None: simulation.to_csv(destination / "baseline_probe.csv")
            if targets is not None: targets.to_csv(destination / "baseline_probe_targets.csv")
            probe_rows.append({"reservoir_id": source["reservoir_id"], "source": name, **result})
    for entry in new:
        rid = entry["reservoir_id"]; folder = output / rid; folder.mkdir()
        source_path = ROOT / entry["observations_file"]
        if sha(source_path) != entry["observations_sha256"]: raise ValueError("New observations changed")
        pins[_relative(source_path)] = sha(source_path)
        frame = frame_read(source_path); meta = read(package / rid / "parameterization.json")
        decision = decisions[rid]
        row = {**entry, "availability_route": meta["method_from_response_availability"],
            "parameter_status": "descriptive_components_only", "calibration_status": "source_gated",
            "next_action": decision["next_action"], "source_decision": decision["decision"],
            "parameters_ready": False, "simulation_ready": False, "calibration_ready": False,
            "has_conditional_inflow_diagnostic": False, "verified_GRAND_mapping": False,
            "monthly_targets_file": _relative(package / rid / "monthly_targets_si.csv"),
            "target_fitting_scope": "all-record descriptive; not calibration-only",
            "catalog_candidate_GRAND_IDs": ";".join(str(c["GRAND_ID"]) for c in meta["catalog_candidates"]),
            "catalog_candidate_capacities_m3": ";".join(str(c["CAP_MCM"]*1e6) for c in meta["catalog_candidates"] if pd.notna(c["CAP_MCM"])),
            "physical_capacity_adopted": False, "independent_forcing_verified": False,
            "storage_valid_days": int(frame.storage_m3.notna().sum()),
            "release_valid_days": int(frame.release_m3s.notna().sum()),
            "primary_inflow_valid_days": int(frame.inflow_m3s.notna().sum()),
            "observed_storage_min_m3": float(frame.storage_m3.min()) if frame.storage_m3.notna().any() else None,
            "observed_storage_max_m3": float(frame.storage_m3.max()) if frame.storage_m3.notna().any() else None}
        if frame.storage_m3.notna().any() and frame.release_m3s.notna().any():
            reconstructed = conditional_inflow(frame)
            reconstructed.to_csv(folder / "conditional_inflow_si.csv")
            summary = conditional_summary(reconstructed)
            summary.update(reservoir_id=rid, source_decision=decision["decision"])
            write(folder / "conditional_inflow_provenance.json", summary)
            diagnostics.append(summary); row["has_conditional_inflow_diagnostic"] = True
            row["conditional_inflow_file"] = _relative(folder / "conditional_inflow_si.csv")
            row["negative_conditional_inflow_intervals"] = summary["negative_net_intervals"]
            if "probe_capacity_reference_m3" in decision:
                result, simulation, targets = baseline_probe(reconstructed, decision["probe_capacity_reference_m3"], "computed_net_inflow")
                write(folder / "baseline_probe.json", result)
                if simulation is not None: simulation.to_csv(folder / "baseline_probe.csv")
                if targets is not None: targets.to_csv(folder / "baseline_probe_targets.csv")
                probe_rows.append({"reservoir_id": rid, "source": "response_derived_net_flux", **result})
        additions = [(key, source, proof) for key, (source, _, proof) in sources.items() if source["reservoir_id"] == rid]
        if additions:
            row["supplementary_source_keys"] = ";".join(key for key, _, _ in additions)
        for key, source, proof in additions:
            if source.get("verified_GRAND_ID"):
                # Identity requires a declared local ID crosswalk AND external position agreement.
                crosswalk = Path(cfg["identity_crosswalk"]).resolve(strict=True)
                table = pd.read_excel(crosswalk, sheet_name="Sean_etalID")
                found = table.loc[table.iloc[:, 0] == source["crosswalk_station_id"], table.columns[1]]
                grand = pd.read_csv(ROOT / "data/catalogs/grand_v1_1_reference.csv")
                candidate = grand.loc[grand.GRAND_ID == source["verified_GRAND_ID"]]
                if (len(found) != 1 or int(found.iloc[0]) != source["verified_GRAND_ID"] or len(candidate) != 1
                        or abs(candidate.LONG_DD.iloc[0] - source["official_lon"]) > .01
                        or abs(candidate.LAT_DD.iloc[0] - source["official_lat"]) > .01):
                    raise ValueError("GRAND identity evidence disagrees")
                evidence = {"station_id": source["crosswalk_station_id"], "GRAND_ID": int(found.iloc[0]),
                    "crosswalk_file": str(crosswalk), "crosswalk_sha256": sha(crosswalk),
                    "official_lon": source["official_lon"], "official_lat": source["official_lat"],
                    "official_position_source": source["position_source"], "catalog_record": candidate.iloc[0].where(candidate.iloc[0].notna(), None).to_dict()}
                write(folder / "identity_evidence.json", evidence)
                row.update(GRAND_ID=int(found.iloc[0]), verified_GRAND_mapping=True)
                row["availability_route"] = "observed_monthly_supplementary_source_pending_review"
        write(folder / "admission.json", {**row, "inherited_source_gates": meta["source_gates"],
            "physical_decision": decision, "supplementary_source_provenance": [proof for _, _, proof in additions]})
        new_rows[rid] = row
    # One reservoir row and one 12-month component table, preserving each profile's status.
    rows = []
    for original in index["reservoirs"]:
        rid = original["reservoir_id"]
        if rid in new_rows:
            row = new_rows[rid]
            targets = pd.read_csv(ROOT / row["monthly_targets_file"], index_col="month")
        else:
            old = old_profiles.get(rid)
            row = {**original, "availability_route": "existing_preserved", "parameter_status": original.get("existing_profile_status"),
                "calibration_status": old["status"] if old else "outside_paper_catalog_preserved", "next_action": old["next_action"] if old else "Retain existing record and its original workflow status",
                "target_fitting_scope": "original frozen profile provenance", "verified_GRAND_mapping": pd.notna(original.get("GRAND_ID"))}
            targets = pd.DataFrame(np.nan, index=range(1, 13), columns=MONTHLY_TARGET_COLUMNS)
            if old and old["monthly_profile_available"]:
                _, _, parameters = application.load_profile(catalog_path, reservoir_id=rid)
                targets = parameters.monthly_targets
                row.update(capacity_m3=parameters.capacity_m3, channel_capacity_m3s=parameters.channel_capacity_m3s,
                    monthly_targets_file=old["registry"], parameter_status="frozen_calibrated_profile")
        for month in range(1, 13):
            for column in MONTHLY_TARGET_COLUMNS: row[f"m{month:02d}_{column}"] = targets.loc[month, column]
            target_rows.append({"reservoir_id": rid, "GRAND_ID": row.get("GRAND_ID"), "month": month,
                "parameter_status": row["parameter_status"], **targets.loc[month].to_dict()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "reservoir_parameterization_all_62_by_GRAND_ID.csv", index=False)
    pd.DataFrame(target_rows).to_csv(output / "monthly_components_all_62.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "conditional_inflow_summary.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(output / "baseline_probe_summary.csv", index=False)
    for name, expected in pins.items():
        if sha(ROOT / name) != expected: raise ValueError("Input changed during assessment")
    summary = {"kind": "reservoir_extension_assessment_v1", "combined_reservoirs": len(rows), "assessed_additions": len(new_rows),
        "conditional_inflow_diagnostics": len(diagnostics), "supplementary_flow_sources": len(sources),
        "new_verified_GRAND_mappings": sum(r["verified_GRAND_mapping"] for r in new_rows.values()),
        "baseline_probes": probe_rows, "new_calibration_ready_reservoirs": 0, "optimizer_evaluations": 0,
        "calibration_decision": "No additions admitted: source definitions, clock and/or forcing still unresolved. No optimizer can resolve those source decisions.",
        "future_calibration_protocol": {"optimizer": "dds", "minimum_calibration_fraction": .65,
            "year_selection": "whole mixed dry/normal/wet years; calibration-only target/bounds fitting",
            "selection": "calibration evidence only", "assessment": "response-derived forcing must remain explicitly conditional"},
        "config_sha256": sha(config_path), "input_sha256": pins,
        "original_collection_files_verified": original_count}
    write(output / "summary.json", summary)
    _report(output, summary, rows, sources)
    artifacts = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()}
    manifest = {**summary, "artifact_sha256": artifacts,
        "source_sha256": {**collection.native.source_hashes(), **{name: sha(ROOT / name) for name in
            ("workflows/extension_assessment_v1.py", "tools/assess_reservoir_extension_v1.py")}}}
    write(output / "manifest.json", manifest)
    return manifest


def _report(output, summary, rows, sources):
    new = [r for r in rows if r["reservoir_id"].startswith("etienne_")]
    lines = ["# Reservoir expansion: source assessment and conditional probes", "",
        "All 21 additions were assessed. The consolidated table retains all 62 records and existing calibrated profiles. No new reservoir has been promoted to calibration-ready status.", "",
        "Seven paired records now have auditable signed net-inflow series, computed as `I_net(t) = Q_mean(t) + [S(t)-S(t-1)]/86400`. Missing days remain missing. Negative values are retained. Storage and release responses, including held-out responses if later selected, enter this forcing: algebraic closure is not independent validation. A level/datum offset does not affect storage differences, but total/active definitions still affect DZTR thresholds and initial states.", "",
        "The local HORIZON archive supplies additional I/S/Q series for Hungry Horse and Minidoka. Raw CSV files and exact SI conversions are packaged here; they do not replace the Etienne source. Provider inflow independence and observation clocks remain unverified. No spline filling, nonpositive-value filtering or legacy inflow multiplier was applied.", "",
        "Source budget flags use an explicit 2% reference-capacity screening threshold. This is a diagnostic threshold, not a statistical confidence limit or permission to remove data. Differences can reflect record errors, storage coordinates, unrepresented outlets/losses, or timing.", "",
        "## Supplementary sources", "", "| Source | Negative storage days | Large budget residuals | Maximum absolute residual (million m³) |",
        "|---|---:|---:|---:|"]
    for key, (_, _, proof) in sources.items():
        lines.append(f"| {key} | {proof['negative_storage_days']} | {proof['large_budget_residual_intervals']} | {proof['max_absolute_budget_residual_m3']/1e6:.2f} |")
    lines += ["", "Hungry Horse is linked to GRAND 299 using the archived station crosswalk and agreement with the official dam location. This verifies identity, not storage capacity or record quality. Reclamation reports total capacity of 3,468,000 acre-feet; its relation to observed values above that reference remains unresolved. [Official project](https://www.usbr.gov/projects/index.php?id=337), [official storage-series metadata](https://catalog.data.gov/dataset/hungry-horse-reservoir-dam-and-powerplant-hgh-daily-instant-lake-reservoir-storage-time-se).", "",
        "Minidoka requires an active/total storage decision and clarification of river-only versus canal-inclusive discharge. Official documents give conflicting capacity descriptions, so no capacity offset or negative-value clipping is applied. [Dam overview](https://www.usbr.gov/projects/index.php?id=181), [historical project description](https://usbr.gov/projects/pdf.php?id=137), [canal-inclusive release description](https://www.usbr.gov/pn/hydromet/uppersnake/01_BOR%20Upper%20Snake%20Update_12.15.2020.pdf).", "",
        "Hugh Keenleyside's operator reports 8.8 billion m³ of live storage. That must not be treated as interchangeable with the archived absolute storage series or a catalog total capacity. [BC Hydro](https://www.bchydro.com/community/recreation_areas/hugh_keenleyside_dam.html).", "",
        "## Baseline probes", "", "These all-record descriptive probes test the existing volume-conserving daily-interval solver; they are not fitted or validated profiles. Catalog capacity references for Hugh Keenleyside, Seton and Revelstoke are sensitivity inputs, not adopted physical metadata. Dead storage is provisionally zero and channel capacity is all-record Q99. Each gap starts a separately observed, unscored anchor; a failed block stops the probe without selecting a more successful period. The mean total modeled outflow includes spill. Capacity and model equations are never changed to force completion.", "",
        "| Reservoir | Result | Completed intervals | Reason |", "|---|---|---:|---|"]
    for p in summary["baseline_probes"]:
        lines.append(f"| {p['reservoir_id']} | {p['status']} | {p['intervals_completed']} | {p.get('reason','Completed descriptive blocks; source gates remain')} |")
    lines += ["", "## Decisions for all additions", "", "| Reservoir | Route from available responses | Next action |", "|---|---|---|"]
    for row in new: lines.append(f"| {row['name']} | {row['availability_route']} | {row['next_action']} |")
    lines += ["", "## Files and continuation", "",
        "- [One row per reservoir, including target values and status](reservoir_parameterization_all_62_by_GRAND_ID.csv).",
        "- [All 62 monthly component tables](monthly_components_all_62.csv). Descriptive components and frozen calibrated profiles carry different labels; unavailable values remain blank.",
        "- [Seven conditional-inflow diagnostics](conditional_inflow_summary.csv). Per-reservoir folders retain the daily reconstructions and provenance.",
        "- [Baseline probe receipts](baseline_probe_summary.csv). Failed probes preserve their completed prefix and dated reason.",
        "- `supplementary_sources/*/largest_source_residuals.csv`: first source records to reconcile, with raw I/S/Q preserved alongside.",
        "", "No DDS evaluations were run because no addition passed source admission. Once a source decision is frozen, use the existing DDS workflow with at least 65% whole mixed-year calibration, targets and bounds fitted on calibration data only, and calibration-based parameter selection. Revelstoke has fewer than two storage years, so it requires a separately labelled short-record strategy; it cannot support the normal multi-regime held-out protocol.", "",
        "The next concrete task is reconciling Hungry Horse's flagged storage/inflow dates against the provider, and confirming Minidoka's storage coordinate and all outlet components. The other reservoir-specific dependencies are listed above. Existing results and original collections remain unchanged.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
