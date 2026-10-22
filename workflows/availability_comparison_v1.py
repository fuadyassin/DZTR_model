"""Assess four data-availability routes on identical frozen calendars and responses."""
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.no_observations_v1 import recommend
from strategies.representative_years import YearSelection
from workflows import no_observations_v1 as ungauged
from workflows import profile_application_v1 as app
from workflows import reservoir_v2 as native
from workflows.atomic_io_v1 import write
from workflows.flow_calibration_v1 import checked, scores

ROOT = native.ROOT
METHODS = ("observed_monthly", "flow_only", "storage_only", "ungauged_mean_flow")


def read_simulation(path):
    """Read numerical trajectories while allowing archived phase/regime annotations."""
    import csv
    with Path(path).open(encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream))
    if len(header) != len(set(header)): raise ValueError("Duplicate trajectory columns")
    columns = ["date", "inflow_m3s", "storage_m3", "release_m3s", "step_seconds", "balance_residual_m3"]
    frame = pd.read_csv(path, usecols=columns, index_col="date", parse_dates=True, float_precision="round_trip")
    ungauged.availability.validate_index(frame.index)
    for column in frame: frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame.to_numpy()).all(): raise ValueError("Nonfinite cached trajectory")
    return frame


def sources():
    return {**ungauged.sources(), "workflows/availability_comparison_v1.py": native.sha(__file__)}


def config(path):
    path = Path(path).resolve(strict=True); cfg = native.read(path)
    required = {"kind", "schema_version", "no_observations_campaign", "observed_catalog", "diagnostic_cache", "flow_campaign", "storage_campaign", "flow_objective", "budget"}
    if set(cfg) != required or (cfg["kind"], cfg["schema_version"]) != ("dztr_availability_comparison_v1", 1):
        raise ValueError("Declare the complete availability comparison configuration")
    if cfg["flow_objective"] not in {"release_nse", "sqrt_release_nse"}:
        raise ValueError("Predeclare the release-only comparison objective")
    if isinstance(cfg["budget"], bool) or not isinstance(cfg["budget"], int) or cfg["budget"] < 1:
        raise ValueError("Declare an existing positive comparison checkpoint")
    for key in ("no_observations_campaign", "observed_catalog", "diagnostic_cache", "flow_campaign", "storage_campaign"):
        if cfg[key] is not None: cfg[key] = str((path.parent / cfg[key]).resolve(strict=True))
    return path, cfg


def report(path, output):
    path, cfg = config(path)
    out = Path(output).resolve()
    if out.exists() or not out.is_relative_to(ROOT / "outputs"): raise ValueError("Use a fresh report under project outputs")
    inputs = {str(path): native.sha(path)}
    def pin(value):
        value = Path(value).resolve(strict=True); inputs[str(value)] = native.sha(value); return value
    base, _ = ungauged.verify(cfg["no_observations_campaign"]); checked(base / "release")
    for name in ("protocol.json", "preparation.json", "release/manifest.json"): pin(base / name)
    inventory = native.read(base / "preparation.json")["reservoirs"]
    requests = {r["reservoir_id"] for r in native.read(base / "config.json")["profiles"]}
    registries = {"ungauged_mean_flow": base / "release/registry.json"}
    campaigns = {}; catalog = {}
    if cfg["observed_catalog"]:
        cat = app.read_catalog(cfg["observed_catalog"]); pin(cfg["observed_catalog"])
        catalog = {r["reservoir_id"]: r for r in cat["reservoirs"]}
    for method, key, arm in (("flow_only", "flow_campaign", cfg["flow_objective"]), ("storage_only", "storage_campaign", "storage_nse")):
        if cfg[key]:
            from workflows import flow_calibration_v1, storage_calibration_v1
            module = flow_calibration_v1 if method == "flow_only" else storage_calibration_v1
            campaign, saved = module.verify(cfg[key]); release = campaign / "releases" / f"budget_{cfg['budget']}"
            checked(release); pin(campaign / "protocol.json"); pin(release / "manifest.json")
            campaigns[method] = campaign
            registries[method] = release / arm / "registry.json"
    verified = {}
    def load(registry, rid):
        registry = Path(registry).resolve(strict=True)
        if registry not in verified:
            verified[registry] = native.verify_registry(registry); pin(registry); pin(registry.parent / "manifest.json")
        rel = verified[registry]["profiles_by_reservoir_id"].get(rid)
        if rel is None: return None
        target = (registry.parent / rel).resolve(strict=True)
        if not target.is_relative_to(registry.parent): raise ValueError("Profile path escapes registry")
        metadata = native.read(pin(target))
        return metadata, pd.read_csv(pin(target.parent / metadata["monthly_targets_file"]), index_col="month", float_precision="round_trip")
    rows, parameter_rows, routing, checks, seasonal = [], [], [], [], []
    simulations = {}; references = {}
    for admission in inventory:
        rid = admission["reservoir_id"]; prepared = base / "prepared" / rid
        has_prepared = (prepared / "forcing_si.csv").exists()
        hidden = plan = forcing = None
        if has_prepared:
            hidden = ungauged.availability.read_frame(pin(prepared / "hidden_assessment_observations_si.csv"))
            forcing = ungauged.availability.read_frame(pin(prepared / "forcing_si.csv"))
            if list(forcing.columns) != ["inflow_m3s"]: raise ValueError("Responses entered the no-observation forcing frame")
            plan = YearSelection(**native.read(pin(prepared / "selection.json"))); references[rid] = hidden
        route = recommend(admission.get("capabilities", {}), inflow_usable=has_prepared,
                          release_prior_usable=(prepared / "release_prior/receipt.json").exists())
        if admission["status"] in {"deferred_source_route", "source_gated"}:
            route = {"method": None, "status": admission["status"], "reason": admission.get("reason")}
        route_row = {"reservoir_id": rid, "GRAND_ID": admission.get("GRAND_ID"), "recommended_method": route["method"],
                     "status": route["status"], "reason": route["reason"], "profiles": {}, "selection_uses_assessment": False}
        for method in METHODS:
            registry = registries.get(method); loaded = None; simulation = None
            status, reason = "profile_unavailable", admission.get("reason")
            selection_matches = False
            if method == "observed_monthly" and catalog.get(rid, {}).get("monthly_profile_available"):
                entry = catalog[rid]
                for rel, expected in entry["input_sha256"].items():
                    if native.sha(pin(app.reference(rel))) != expected: raise ValueError("Catalog input changed")
                registry = app.reference(entry["registry"])
            if registry is not None and registry.exists(): loaded = load(registry, rid)
            param_row = {"reservoir_id": rid, "GRAND_ID": admission.get("GRAND_ID"), "method": method,
                         "parameters_available": loaded is not None, "source_certified": False}
            if loaded:
                profile, targets = loaded
                route_row["profiles"][method] = {"registry": str(registry.resolve()), "registry_sha256": native.sha(registry),
                    "reservoir_id": rid, "profile_method": method}
                param_row.update(seed=profile.get("seed"), evaluation_budget=profile.get("evaluation_budget"),
                    initial_storage_m3=profile["physical_constants"]["initial_storage_m3"],
                    initial_release_m3s=profile["physical_constants"]["initial_release_m3s"],
                    numerical_route=profile.get("route", "standard_native_v1"),
                    parameterization_provenance_file=str((registry.parent / verified[registry.resolve()]["profiles_by_reservoir_id"][rid]).resolve()))
                for month, values in targets.iterrows():
                    for column, value in values.items(): param_row[f"{column}_month_{int(month):02d}"] = value
                if has_prepared:
                    # All comparisons must retain the exact original year roles and response values.
                    if method == "ungauged_mean_flow":
                        diagnostic = base / "jobs" / rid
                        checked(diagnostic); receipt = native.read(pin(diagnostic / "receipt.json"))
                        status, reason = receipt["status"], receipt.get("reason"); selection_matches = True
                        if status == "completed": simulation = ungauged.availability.read_frame(pin(diagnostic / "simulation.csv"))
                        prior = pd.read_csv(pin(prepared / "release_prior/monthly_release_prior_si.csv"), index_col="month", float_precision="round_trip")
                        np.testing.assert_array_equal(targets.iloc[:, 3:], prior)
                        model = native.read(prepared / "model_config.json")
                        expected = ungauged.parameters(forcing, model["physical_constants"], prior, storage_basis=admission["storage_basis"])
                        np.testing.assert_array_equal(targets, expected.monthly_targets)
                        if profile["physical_constants"]["initial_storage_m3"] != targets.iloc[0, 1]: raise ValueError("Initial storage used an observation")
                    elif method in campaigns:
                        other = campaigns[method] / "prepared" / rid
                        original = ungauged.availability.read_frame(pin(other / "hidden_assessment_observations_si.csv"))
                        other_plan = YearSelection(**native.read(pin(other / "selection.json")))
                        pd.testing.assert_frame_equal(hidden, original, check_freq=False)
                        selection_matches = other_plan.signature == plan.signature
                        arm = cfg["flow_objective"] if method == "flow_only" else "storage_nse"
                        diagnostic = campaigns[method] / "releases" / f"budget_{cfg['budget']}" / "diagnostics" / rid / arm
                        receipt = native.read(pin(diagnostic / "assessment.json"))
                        status, reason = receipt["status"], receipt.get("reason")
                        simulation = ungauged.availability.read_frame(pin(diagnostic / "simulation.csv"))
                    else:
                        from workflows.cohort_assessment_v1 import describe
                        description = describe(pd.Series({"profile_registry": str(registry.relative_to(ROOT)), "reservoir_id": rid}))
                        original = ungauged.availability.read_frame(pin(description["prepared"] / "observations_si.csv"))
                        other_plan = YearSelection(**native.read(pin(description["prepared"] / "selection.json")))
                        pd.testing.assert_frame_equal(hidden, original, check_freq=False)
                        selection_matches = other_plan.signature == plan.signature
                        archived = registry.parent / "diagnostics" / rid / ("assessment_simulation.csv" if description["route"] == "signed_net_native_v1" else "simulation.csv")
                        if archived.exists(): simulation = read_simulation(pin(archived))
                        scored = hidden.index[np.isin(plan.roles, ["calibration", "assessment"])]
                        if simulation is None or not scored.isin(simulation.index).all():
                            cache = Path(cfg["diagnostic_cache"]) / rid / "selected" if cfg["diagnostic_cache"] else None
                            if cache is not None and (cache / "receipt.json").exists():
                                receipt = native.read(pin(cache / "receipt.json")); identity = receipt["identity"]
                                if native.sha(pin(cache / "simulation.csv")) != receipt["simulation_sha256"]: raise ValueError("Cached joint trajectory changed")
                                if identity["source_sha256"] != native.sha(description["prepared"] / "observations_si.csv") or identity["targets"] != targets.to_numpy().tolist() or identity["constants"] != profile["physical_constants"]:
                                    raise ValueError("Joint trajectory identity does not match selected profile")
                                simulation = read_simulation(cache / "simulation.csv")
                        status = "completed" if simulation is not None and scored.isin(simulation.index).all() else "cached_support_incomplete"
                        reason = None if status == "completed" else "Retained cached coverage; no diagnostic model rerun"
                    if not selection_matches: raise ValueError("Year selections differ: " + rid + "/" + method)
                    if simulation is not None:
                        residual = native.independent_balance(simulation, profile["physical_constants"]["inflow_correction"])
                        if residual > profile["atol_storage_m3"] or not simulation.storage_m3.between(0, profile["physical_constants"]["capacity_m3"]).all():
                            raise ValueError("Cached physical check failed: " + rid + "/" + method)
                        checks.append({"reservoir_id": rid, "method": method, "maximum_balance_residual_m3": residual,
                                       "same_selection_and_observations": True, "new_optimizer_evaluations": 0})
                        simulations[rid, method] = simulation
            param_row.update(simulation_status=status, reason=reason); parameter_rows.append(param_row)
            for phase in ("calibration", "assessment"):
                metrics = {}; phase_status = status
                if hidden is not None and simulation is not None:
                    dates = hidden.index[plan.mask(hidden.index, phase)]
                    if dates.isin(simulation.index).all() and (phase == "calibration" or status == "completed"):
                        metrics = scores(hidden, simulation, plan, phase); phase_status = "completed"
                        for month in range(1, 13):
                            selected = dates[dates.month == month]
                            item = {"reservoir_id": rid, "method": method, "phase": phase, "month": month}
                            for column in ("storage_m3", "release_m3s"):
                                error = simulation.loc[selected, column] - hidden.loc[selected, column]
                                item[column + "_mean_bias"] = error.mean()
                            seasonal.append(item)
                rows.append({"reservoir_id": rid, "GRAND_ID": admission.get("GRAND_ID"), "method": method,
                    "phase": phase, "status": phase_status, "reason": reason, "selection_matches": selection_matches,
                    "requested_benchmark_reservoir": rid in requests, **metrics})
        preferred = route_row["profiles"].get(route_row["recommended_method"])
        route_row["recommended_profile_available"] = preferred is not None
        if preferred is None and route_row["status"] == "recommended": route_row["status"] = "recommended_profile_not_prepared"
        routing.append(route_row)
    out.mkdir(parents=True)
    frame = pd.DataFrame(rows)
    for column in ("storage_nse", "release_nse", "sqrt_release_nse"):
        if column not in frame: frame[column] = np.nan
    frame.to_csv(out / "all_route_phase_scores.csv", index=False, float_format="%.17g")
    pd.DataFrame(parameter_rows).to_csv(out / "reservoir_parameterization_all_by_GRAND_ID_and_method.csv", index=False, float_format="%.17g")
    pd.DataFrame([{k: v for k, v in r.items() if k != "profiles"} for r in routing]).to_csv(out / "reservoir_route_inventory_by_GRAND_ID.csv", index=False)
    write(out / "route_registry.json", {"kind": "availability_route_registry_v1", "selection_uses_assessment": False,
        "source_certified": False, "reservoirs": routing})
    summaries, paired = [], []
    selected = frame[(frame.phase == "assessment") & frame.requested_benchmark_reservoir]
    successful = selected[(selected.status == "completed") & selected.storage_nse.notna() & selected.release_nse.notna()]
    common = set.intersection(*(set(successful[successful.method == m].reservoir_id) for m in METHODS))
    for scope, subset in (("available_cases", selected), ("common_successful_cases", selected[selected.reservoir_id.isin(common)])):
        for method in METHODS:
            part = subset[subset.method == method]; valid = part[part.status == "completed"]
            summaries.append({"scope": scope, "method": method, "requested": len(part), "completed": len(valid),
                "storage_scored": int(valid.storage_nse.notna().sum()), "release_scored": int(valid.release_nse.notna().sum()),
                "median_storage_nse": valid.storage_nse.median(), "median_release_nse": valid.release_nse.median(),
                "both_nse_positive": int(((valid.storage_nse > 0) & (valid.release_nse > 0)).sum())})
    for other in METHODS[:-1]:
        paired_frame = selected[selected.method == "ungauged_mean_flow"].merge(selected[selected.method == other], on=["reservoir_id", "phase"], suffixes=("_ungauged", "_other"), validate="one_to_one")
        for _, row in paired_frame.iterrows():
            paired.append({"reservoir_id": row.reservoir_id, "compared_with": other,
                "storage_nse_difference": row.storage_nse_ungauged - row.storage_nse_other,
                "release_nse_difference": row.release_nse_ungauged - row.release_nse_other,
                "both_completed": row.status_ungauged == row.status_other == "completed"})
    summary = pd.DataFrame(summaries); summary.to_csv(out / "summary.csv", index=False, float_format="%.17g")
    pd.DataFrame(paired).to_csv(out / "paired_no_observation_differences.csv", index=False, float_format="%.17g")
    pd.DataFrame(checks).to_csv(out / "scientific_checks.csv", index=False, float_format="%.17g")
    pd.DataFrame(seasonal).to_csv(out / "seasonal_bias.csv", index=False, float_format="%.17g")
    plot(frame, simulations, references, out)
    lines = ["# Four-route data-availability benchmark", "",
        "The no-observation recipe uses fixed 20/30/97% storage fractions and a mean-flow-scaled external donor release prior. Both target responses are hidden during parameter estimation and initialization. All other profiles are reused from frozen calibration selections; assessment scores never select a route or profile.", "",
        "The mean-flow source is recorded per reservoir: verified GRAND DIS_AVG_LS, or the explicitly allowed calibration effective-inflow proxy. The donor pool excludes the target. This is an exploratory regional transfer, with conditional inflow/channel/capacity metadata; it is not an independent ungauged validation.", "",
        "Release-only and storage-only searches use the 64-evaluation DDS checkpoint; joint calibration uses 512-evaluation NSGA-II. The no-observation recipe runs no optimizer. Different data access, initialization and parameter dimensions prevent attribution of differences to the optimizer alone.", "",
        "| Assessment scope | Method | Requested | Completed | Median storage NSE | Median release NSE | Both positive |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['scope']} | {row['method']} | {row['requested']} | {row['completed']} | {row['median_storage_nse']:.3f} | {row['median_release_nse']:.3f} | {row['both_nse_positive']} |")
    lines += ["", "Available-case medians have different membership. The common-successful-case block uses identical reservoir membership across all four routes; failures remain in the full score table. All comparisons use the same source observations, native calendars and year masks.", "",
        "![Four-route performance](cohort_comparison.png)", "",
        "The route inventory retains every paper reservoir. Source-gated and interval cases keep their original status. The parameter table contains one row per reservoir and method; missing profiles have blank parameters. The automatic recommendation uses usable calibration observations and source dependencies, never held-out skill. It selects an already prepared profile and does not launch calibration implicitly.", "",
        "- [Route inventory](reservoir_route_inventory_by_GRAND_ID.csv)",
        "- [Parameters by GRAND ID and method](reservoir_parameterization_all_by_GRAND_ID_and_method.csv)",
        "- [All scores and failures](all_route_phase_scores.csv)",
        "- [Paired differences](paired_no_observation_differences.csv)",
        "- [Scientific checks](scientific_checks.csv)",
        "- [Seasonal bias](seasonal_bias.csv)", "",
        "No existing optimizer jobs were resumed or rerun. Scientific checks and this report reuse saved trajectories. Convergence, independent forcing validity and transferability remain unproven.", ""]
    for rid in ("ex14_Gardiner", "ex30_Big_horn", "ex5_Garrison"):
        if (out / (rid + ".png")).exists(): lines += [f"![{rid}]({rid}.png)", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    write(out / "config.json", cfg)
    for p, expected in inputs.items():
        if native.sha(p) != expected: raise ValueError("Comparison input changed during reporting")
    write(out / "manifest.json", {"code_sha256": sources(), "input_sha256": inputs,
        "artifact_sha256": {p.relative_to(out).as_posix(): native.sha(p) for p in out.rglob("*") if p.is_file()}})
    return out


def plot(scores_frame, simulations, references, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    held = scores_frame[(scores_frame.phase == "assessment") & scores_frame.requested_benchmark_reservoir]
    colors = {m: c for m, c in zip(METHODS, ("#267349", "#ca7537", "#267eae", "#8b568e"))}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for ax, method in zip(axes.flat, METHODS):
        part = held[held.method == method]
        ax.scatter(part.release_nse, part.storage_nse, color=colors[method], alpha=.8)
        ax.set_xscale("symlog", linthresh=1); ax.set_yscale("symlog", linthresh=1)
        ax.axhline(0, color="grey", lw=.7); ax.axvline(0, color="grey", lw=.7); ax.grid(alpha=.2)
        ax.set_title(f"{method}: {int((part.status == 'completed').sum())}/{len(part)} completed")
        ax.set_xlabel("Held-out release NSE"); ax.set_ylabel("Held-out storage NSE")
    fig.suptitle("Data-availability comparison; symmetric log axes, linear near zero")
    fig.tight_layout(); fig.savefig(out / "cohort_comparison.png", dpi=160); fig.savefig(out / "cohort_comparison.pdf"); plt.close(fig)
    for rid in ("ex14_Gardiner", "ex30_Big_horn", "ex5_Garrison"):
        if rid not in references: continue
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for ax, column, scale in zip(axes, ("storage_m3", "release_m3s"), (1e6, 1)):
            ax.plot(references[rid][column].resample("MS").mean() / scale, color="#222222", label="Observed", lw=1.4)
            for method in METHODS:
                if (rid, method) in simulations:
                    ax.plot(simulations[rid, method][column].resample("MS").mean() / scale, color=colors[method], label=method, alpha=.75)
            ax.set_ylabel("Storage (million m³)" if column == "storage_m3" else "Release (m³/s)"); ax.grid(alpha=.2)
        axes[0].legend(fontsize=7, ncol=3); fig.suptitle(rid + ": monthly means for display; metrics use native support")
        fig.tight_layout(); fig.savefig(out / (rid + ".png"), dpi=160); plt.close(fig)


def resolve(registry, *, identifier=None, grand_id=None, method=None):
    path = Path(registry).resolve(strict=True)
    if (identifier is None) == (grand_id is None): raise ValueError("Select one reservoir ID or GRAND ID")
    proof = native.read(path.parent / "manifest.json")
    for rel, expected in proof["code_sha256"].items():
        if native.sha(ROOT / rel) != expected: raise ValueError("Route-registry implementation changed")
    for rel, expected in proof["artifact_sha256"].items():
        target = (path.parent / rel).resolve(strict=True)
        if not target.is_relative_to(path.parent) or native.sha(target) != expected: raise ValueError("Route-registry artifact changed")
    data = native.read(path)
    if data.get("kind") != "availability_route_registry_v1" or data.get("selection_uses_assessment") is not False: raise ValueError("Unknown route registry")
    if grand_id is not None and (isinstance(grand_id, bool) or not isinstance(grand_id, int) or grand_id <= 0): raise ValueError("Use a positive GRAND ID")
    found = [r for r in data["reservoirs"] if r["reservoir_id"] == identifier] if identifier is not None else [r for r in data["reservoirs"] if r["GRAND_ID"] == grand_id]
    if len(found) != 1: raise ValueError("No unique verified reservoir mapping")
    row = found[0]
    if row["status"] in {"source_gated", "deferred_source_route", "needs_inflow", "needs_release_prior"}: raise ValueError(str(row["reason"]))
    selected = row["profiles"].get(method or row["recommended_method"])
    if selected is None: raise ValueError("Requested/recommended profile is unavailable; no automatic weaker-data fallback")
    if native.sha(selected["registry"]) != selected["registry_sha256"]: raise ValueError("Selected source registry changed")
    return selected
