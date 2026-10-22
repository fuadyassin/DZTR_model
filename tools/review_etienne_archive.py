"""Audit the Etienne archive and rescore cached release, without model reruns.

External programs are never imported or launched. Only two inspected, restricted
arithmetic release methods are extracted from ASTs for isolated numerical probes.
Run after tools/Set-ProjectCache.ps1; use a fresh output directory.
"""
from __future__ import annotations

import argparse
import ast
from calendar import monthrange
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dztr.model.release import dztr_release
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class Review:
    def __init__(self, archive: Path, cohort: Path):
        self.archive = archive.resolve(strict=True)
        self.cohort = cohort.resolve(strict=True)
        self.external: dict[Path, dict] = {}
        self.local: dict[Path, str] = {}

    def external_text(self, path: Path) -> str:
        path = path.resolve(strict=True)
        if not path.is_relative_to(self.archive):
            raise ValueError("External read escaped the requested archive")
        digest = sha(path)
        if path in self.external and self.external[path]["sha256"] != digest:
            raise ValueError(f"Archive changed while reading {path}")
        self.external[path] = {"path": path.relative_to(self.archive).as_posix(),
                               "sha256": digest, "bytes": path.stat().st_size}
        return path.read_text(encoding="utf-8-sig", errors="replace")

    def pin_local(self, path: Path) -> Path:
        path = path.resolve(strict=True)
        if not path.is_relative_to(ROOT):
            raise ValueError("Local input escaped the project")
        digest = sha(path)
        if path in self.local and self.local[path] != digest:
            raise ValueError(f"Local input changed while reading {path}")
        self.local[path] = digest
        return path

    def release_method(self, path: Path):
        tree = ast.parse(self.external_text(path).expandtabs(8))
        functions = [node for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef) and node.name == "sim_outflow"]
        if len(functions) != 1:
            raise ValueError("Expected exactly one sim_outflow method")
        fn = functions[0]
        if fn.decorator_list or fn.returns or fn.args.defaults:
            raise ValueError("Unexpected function metadata")
        # These inspected functions only read scalar parameters/arrays and write
        # Qsim[1]. No imports, file access, other methods, or top-level code run.
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and not (
                isinstance(node.func, ast.Name)
                and node.func.id in {"min", "max", "float", "monthrange"}
            ):
                raise ValueError("Unexpected function call in arithmetic probe")
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.With, ast.Try,
                                 ast.Lambda, ast.ClassDef)):
                raise ValueError("Unexpected statement in arithmetic probe")
            if isinstance(node, ast.Attribute) and not (
                isinstance(node.value, ast.Name) and node.value.id == "datenum"
                and node.attr in {"day", "month", "year"}
            ):
                raise ValueError("Unexpected attribute in arithmetic probe")
        namespace = {"__builtins__": {}, "min": min, "max": max,
                     "float": float, "monthrange": monthrange}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
        return namespace["sim_outflow"], namespace

    def inventory(self):
        rows = []
        for path in sorted(self.archive.rglob("*.py")):
            source = self.external_text(path)
            try:
                ast.parse(source)
                issue = ""
            except SyntaxError as exc:
                issue = f"{type(exc).__name__} line {exc.lineno}"
            ast.parse(source.expandtabs(8))  # normalization only in memory
            rows.append({**self.external[path.resolve()], "raw_parse_issue": issue,
                         "parse_after_in_memory_tab_expansion": True})
        return rows

    def configurations(self):
        rows, notes = [], []
        for case in sorted((self.archive / "python_model").glob("ost_*")):
            if not case.is_dir() or not (case / "ostIn.txt").exists():
                continue
            lines = self.external_text(case / "ostIn.txt").splitlines()
            block = lines[lines.index("BeginParams") + 1:lines.index("EndParams")]
            params = [s.split("#")[0].strip() for s in block]
            params = [s for s in params if s]
            def find(prefix):
                return next((s.strip() for s in lines if s.strip().startswith(prefix)), "")
            rows.append({"case": case.name, "active_parameters": len(params),
                         "algorithm": find("ProgramType"), "seed": find("RandomSeed"),
                         "budget_setting": find("MaxIterations"),
                         "weighted_objective": find("OBJ_F "),
                         "parameter_correction": "BeginParameterCorrection" in lines,
                         "model_launch": self.external_text(case / "model_launch.sh")
                         if (case / "model_launch.sh").exists() else ""})
            for filename in ("Res_params.tpl", "param_correction.py", "create_Ostin.py"):
                if (case / filename).exists():
                    self.external_text(case / filename)
            for path in sorted(case.glob("TRY*/README*")):
                notes.append({"path": path.relative_to(self.archive).as_posix(),
                              "note": self.external_text(path).strip()})
        return rows, notes

    def probes(self):
        main = self.archive / "python_model/ost_Gardiner/DZTR_model.py"
        fn, ns = self.release_method(main)
        targets = pd.DataFrame([
            [20e6 + month * 1e5, 50e6 + month * 1e5, 70e6 + month * 1e5,
             10., 40., 100.] for month in range(1, 13)
        ], columns=MONTHLY_TARGET_COLUMNS, index=range(1, 13))
        ns.update(month_stor=targets.iloc[:, :3].to_numpy().T,
                  month_flows=targets.iloc[:, 3:].to_numpy().T,
                  dead_stor_frac=0.1, res_maxcap=100e6, max_outflow=500.,
                  rand_error=0., Stosim=[0., 0.], Qsim=[0., 0.], Qin=[0., 0.])
        count, max_error = 0, 0.
        for mode, external_mode in (("storage", 1), ("inflow", 0)):
            parameters = ReservoirParameters(targets, 100e6, 50e6, 0., 500., flood_mode=mode)
            for month in range(1, 13):
                sc, sn, sf = targets.iloc[month - 1, :3]
                storage_points = [-1e6, 0., 10e6, 10e6 + 1, sc, sc + 1,
                                  (sc + sn) / 2, sn, sn + 1, sf, sf + 1, 110e6]
                for seconds in (3600., 21600., 86400.):
                    for inflow in (0., 250.):
                        for storage in storage_points:
                            ns.update(dt=seconds, res_type=external_mode)
                            ns["Stosim"][1] = storage
                            ns["Qin"][1] = inflow
                            fn(None, month)
                            actual = dztr_release(storage, inflow, month, parameters, seconds)
                            error = abs(actual - ns["Qsim"][1])
                            max_error = max(max_error, error)
                            if not np.isclose(actual, ns["Qsim"][1], rtol=1e-12, atol=1e-12):
                                raise AssertionError((mode, month, seconds, inflow, storage))
                            count += 1
        # Probe the separate daily-target variant using constant storage targets
        # and equal flow targets within each month to isolate interpolation.
        smooth, smooth_ns = self.release_method(
            self.archive / "python_model/ost_Gardiner/DZTR_model_w_graphs_new.py")
        smooth_ns.update({k: v for k, v in ns.items() if k not in {"sim_outflow", "__builtins__"}})
        smooth_ns["month_stor"] = np.tile([[20e6], [50e6], [70e6]], (1, 12))
        smooth_ns["month_flows"] = np.tile(np.arange(1., 13.) * 10., (3, 1))
        smooth_ns["Stosim"][1] = 35e6
        smooth(None, datetime(2019, 1, 31)); jan31 = smooth_ns["Qsim"][1]
        smooth(None, datetime(2019, 2, 1)); feb1 = smooth_ns["Qsim"][1]
        assert feb1 < jan31
        exact_feb1 = 10 + 10 * ((datetime(2019, 2, 1) - datetime(2019, 1, 15)).days /
                              (datetime(2019, 2, 15) - datetime(2019, 1, 15)).days)
        # Independent arithmetic, not a run of the archived integrator. If the
        # unconstrained trapezoid converged, clipping E and adding E/dt to its
        # endpoint outflow leaves -E/2 storage residual. 2E/dt closes the trapezoid.
        excess, seconds = 1e6, 21600.
        legacy_residual = -excess + 0.5 * seconds * (excess / seconds)
        corrected_residual = -excess + 0.5 * seconds * (2 * excess / seconds)
        assert legacy_residual == -500000. and corrected_residual == 0.
        daily_volume = 86400 * 30.
        archived_volume = daily_volume + seconds / 2 * (10. - 30.)
        assert archived_volume == 2376000.
        return {
            "release_law": {"comparisons": count, "maximum_absolute_error_m3s": max_error,
                            "random_error_setting": 0., "scope": "release rule only; no trajectories or root solver tested"},
            "interpolation": {"synthetic_monthly_targets": "Jan=10, Feb=20 m3/s; all three flow targets equal",
                              "archive_jan31_m3s": jan31, "archive_feb1_m3s": feb1,
                              "dated_midmonth_reference_feb1_m3s": exact_feb1,
                              "conclusion": "Current-month day count can reverse a rising curve across month boundary."},
            "spill_arithmetic": {"assumption": "converged unconstrained trapezoidal balance, excess=1e6 m3",
                                 "archive_endpoint_adjustment_residual_m3": legacy_residual,
                                 "conserving_endpoint_adjustment_residual_m3": corrected_residual},
            "daily_inflow_clock": {"substep_seconds": seconds, "previous_day_m3s": 10.,
                                   "current_day_m3s": 30., "daily_mean_interpretation_volume_m3": daily_volume,
                                   "archive_trapezoid_incoming_volume_m3": archived_volume,
                                   "difference_m3": archived_volume - daily_volume,
                                   "conclusion": "Use only with an explicit endpoint inflow convention; daily means need volume-preserving integration."},
            "percentiles": {"sample": [1, 2, 3, 4], "percentile": 10,
                            "numpy_default_linear": float(np.percentile([1, 2, 3, 4], 10, method="linear")),
                            "hazen": float(np.percentile([1, 2, 3, 4], 10, method="hazen"))},
            "pbias_convention": {"observed": [10, 20], "simulated": [5, 10],
                                 "archive_positive_underprediction_pct": 50.,
                                 "project_positive_overprediction_pct": -50.},
        }

    def corrected_parameters(self):
        case = self.archive / "python_model/ost_Ghost"
        def read(path):
            lines = self.external_text(path).splitlines()
            headers = [float(lines[i].split("#")[0].strip()) for i in (4, 5, 8, 10)]
            target_rows = []
            for i in range(14, 20):
                fields = lines[i].split(",")
                if len(fields) == 13 and not fields[-1].strip():
                    fields = fields[:-1]  # correction writer uses a trailing comma
                if len(fields) != 12:
                    raise ValueError("Expected exactly twelve monthly values")
                target_rows.append([float(v) for v in fields])
            targets = np.array(target_rows)
            return np.r_[headers, targets.ravel()]
        before = read(case / "Res_params_0.txt")
        after = read(case / "corr_Res_params.txt")
        labels = ["capacity_m3", "channel_capacity_m3s", "dead_storage_fraction", "initial_storage_m3"]
        labels += [f"{name}_month_{month}" for name in MONTHLY_TARGET_COLUMNS for month in range(1, 13)]
        return [{"parameter": label, "raw": a, "corrected": b, "difference": b-a}
                for label, a, b in zip(labels, before, after) if a != b]

    @staticmethod
    def nse(obs, sim):
        denominator = float(np.sum((obs - np.mean(obs)) ** 2))
        return float(1 - np.sum((sim - obs) ** 2) / denominator) if denominator > 0 else None

    def cached_release_diagnostics(self):
        catalog = json.loads(self.pin_local(self.cohort / "catalog.json").read_text())
        summary = pd.read_csv(self.pin_local(self.cohort / "cohort_summary.csv")).set_index("reservoir_id")
        # Verify the existing published snapshot without importing its simulation
        # entry points. This review cannot change production identities.
        for relative, digest in catalog["application_source_sha256"].items():
            if sha(self.pin_local(ROOT / relative)) != digest:
                raise AssertionError(f"Frozen code differs: {relative}")
        if sha(self.pin_local(ROOT / catalog["source_table"])) != catalog["source_table_sha256"]:
            raise AssertionError("Frozen table differs")
        rows = []
        for entry in catalog["reservoirs"]:
            if not entry.get("monthly_profile_available"):
                continue
            identifier = entry["reservoir_id"]
            for relative, digest in entry["input_sha256"].items():
                if sha(self.pin_local(ROOT / relative)) != digest:
                    raise AssertionError(f"Frozen input differs: {relative}")
            forcing_path = self.pin_local(ROOT / entry["prepared_forcing"])
            forcing = pd.read_csv(forcing_path, parse_dates=["date"]).set_index("date")
            selection = json.loads(self.pin_local(forcing_path.with_name("selection.json")).read_text())
            roles = pd.Series(selection["roles"], index=pd.to_datetime(selection["dates"]))
            if not forcing.index.is_unique or not roles.index.equals(forcing.index):
                raise AssertionError(f"Selection calendar differs: {identifier}")
            trajectory = ROOT / "outputs/cohort_diagnostic_cache_v1_20261017" / identifier / "selected/simulation.csv"
            if not trajectory.exists():
                filename = "assessment_simulation.csv" if entry["route"] == "signed_net_native_v1" else "simulation.csv"
                trajectory = (ROOT / entry["registry"]).parent / "diagnostics" / identifier / filename
            simulation = pd.read_csv(self.pin_local(trajectory), parse_dates=["date"]).set_index("date")
            if not simulation.index.is_unique:
                raise AssertionError(f"Duplicate simulation dates: {identifier}")
            simulated = simulation["release_m3s"].reindex(forcing.index)
            observed = forcing["release_m3s"]
            for phase in ("calibration", "assessment"):
                mask = roles.eq(phase) & observed.notna() & simulated.notna()
                obs, sim = observed[mask].to_numpy(), simulated[mask].to_numpy()
                if not np.isfinite(obs).all() or not np.isfinite(sim).all() or (obs < 0).any() or (sim < 0).any():
                    raise AssertionError(f"Nonfinite or signed scored release: {identifier}, {phase}")
                expected_pairs = int(summary.loc[identifier, phase + "_release_pairs"])
                if len(obs) != expected_pairs:
                    raise AssertionError(f"Scored coverage differs: {identifier}, {phase}")
                raw = self.nse(obs, sim)
                expected = summary.loc[identifier, phase + "_release_nse"]
                if raw is None or not np.isclose(raw, expected, rtol=0, atol=1e-10):
                    raise AssertionError(f"Raw score differs: {identifier}, {phase}: {raw}, {expected}")
                transformed = self.nse(np.sqrt(obs), np.sqrt(sim))
                rows.append({"reservoir_id": identifier, "name": entry["name"],
                             "GRAND_ID": entry["GRAND_ID"], "phase": phase, "pairs": len(obs),
                             "zero_observed_release_pairs": int(np.sum(obs == 0)),
                             "raw_release_nse": raw, "sqrt_release_nse": transformed,
                             "raw_nse_reproduction_absolute_error": abs(raw - expected),
                             "simulation_path": trajectory.relative_to(ROOT).as_posix(),
                             "simulation_sha256": sha(trajectory),
                             "diagnostic_only_no_parameter_reselection": True})
        if len(rows) != 64:
            raise AssertionError("Expected 32 reservoirs x 2 scored phases")
        return rows

    def storage_sources(self):
        rows = []
        for identifier, filename in (("ex26_Ghost", "05BE005_stor.txt"),
                                     ("ex15_Bennett", "Williston_storage.txt")):
            source = self.archive / filename
            data = []
            seen = set()
            for line in self.external_text(source).splitlines():
                fields = line.split()
                try:
                    date = datetime.strptime(fields[0], "%Y%m%d")
                    value = float(fields[1])
                except (ValueError, IndexError):
                    continue
                if date in seen:
                    raise AssertionError("Duplicate external storage date")
                seen.add(date)
                if np.isfinite(value) and value > 0:
                    data.append((date, value * 1e6))
            external = pd.Series(dict(data), name="external_storage_m3")
            canonical_path = ROOT / "outputs/paper_completion_v1_20261014_03/source_profiles" / identifier / "source_observations_si.csv"
            canonical = pd.read_csv(self.pin_local(canonical_path), parse_dates=["date"]).set_index("date")["storage_m3"]
            joint = pd.concat([canonical.rename("project_storage_m3"), external], axis=1, join="inner").dropna()
            joint = joint.loc[(joint > 0).all(axis=1)]
            # Descriptive differences only. Never fit/apply a datum conversion.
            differences = joint["external_storage_m3"] - joint["project_storage_m3"]
            rows.append({"reservoir_id": identifier, "external_source": filename,
                         "positive_overlap_pairs": len(joint), "first_overlap": joint.index.min().strftime("%Y-%m-%d"),
                         "last_overlap": joint.index.max().strftime("%Y-%m-%d"),
                         "mean_difference_m3": float(differences.mean()),
                         "difference_std_m3": float(differences.std(ddof=0)),
                         "minimum_difference_m3": float(differences.min()),
                         "maximum_difference_m3": float(differences.max()),
                         "conversion_applied": False,
                         "source_gate_resolved": False})
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cohort", type=Path, default=ROOT / "outputs/calibrated_paper_cohort_v1_20261017_final")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "outputs") or output.exists():
        raise ValueError("Choose a fresh directory inside project outputs")
    review = Review(args.archive, args.cohort)
    review.pin_local(Path(__file__))
    inventory = review.inventory()
    configurations, notes = review.configurations()
    probes = review.probes()
    corrected = review.corrected_parameters()
    diagnostics = review.cached_release_diagnostics()
    sources = review.storage_sources()
    for relative in ("compute_monthly_statistics.py", "crude_params/compute_monthly_statistics.py",
                     "observed_stor/reserv_list.txt", "python_model/Gardiner_poly.txt",
                     "python_model/Newton_Raphson_method.py"):
        review.external_text(review.archive / relative)
    for path, metadata in review.external.items():
        if sha(path) != metadata["sha256"]:
            raise AssertionError("An external input changed during review")
    for path, digest in review.local.items():
        if sha(path) != digest:
            raise AssertionError("A project input changed during review")
    output.mkdir(parents=True)
    for filename, rows in (("python_inventory.csv", inventory), ("optimizer_configurations.csv", configurations),
                           ("experiment_notes.csv", notes), ("ghost_parameter_corrections.csv", corrected),
                           ("cached_release_diagnostics.csv", diagnostics), ("storage_source_comparison.csv", sources)):
        pd.DataFrame(rows).to_csv(output / filename, index=False, float_format="%.17g")
    dump(output / "numerical_probes.json", probes)
    diagnostic_frame = pd.DataFrame(diagnostics)
    held = diagnostic_frame.loc[diagnostic_frame.phase == "assessment"]
    validation = {
        "python_files_reviewed": len(inventory),
        "distinct_python_byte_versions": len({r["sha256"] for r in inventory}),
        "raw_python_parse_issues": sum(bool(r["raw_parse_issue"]) for r in inventory),
        "all_parse_after_in_memory_tab_expansion": True,
        "optimizer_configurations": len(configurations),
        "experiment_notes": len(notes),
        "release_rule_comparisons": probes["release_law"]["comparisons"],
        "cached_release_diagnostic_rows": len(diagnostics),
        "maximum_raw_nse_reproduction_error": float(diagnostic_frame.raw_nse_reproduction_absolute_error.max()),
        "held_out_raw_release_nse_median": float(held.raw_release_nse.median()),
        "held_out_sqrt_release_nse_median": float(held.sqrt_release_nse.median()),
        "held_out_raw_release_nse_positive": int((held.raw_release_nse > 0).sum()),
        "held_out_sqrt_release_nse_positive": int((held.sqrt_release_nse > 0).sum()),
        "external_inputs_unchanged": True, "frozen_project_inputs_unchanged": True,
        "reservoir_simulations_run": 0, "optimization_evaluations_run": 0,
        "historical_experiments_reproduced": False, "source_gates_changed": False,
        "supplement_retrieved": False,
    }
    dump(output / "validation.json", validation)
    dump(output / "input_manifest.json", {
        "external_root": str(review.archive),
        "external_files": sorted(review.external.values(), key=lambda r: r["path"]),
        "project_inputs": {p.relative_to(ROOT).as_posix(): digest for p, digest in sorted(review.local.items())},
        "publication_doi": "10.1139/cjce-2021-0507",
        "publication_url": "https://cdnsciencepub.com/doi/full/10.1139/cjce-2021-0507",
        "accessible_full_text": "https://www.researchgate.net/publication/360979258_On_the_implementation_of_the_dynamically_zoned_target_release_reservoir_model_in_the_GEM-Hydro_streamflow_forecasting_system",
        "publication_access_note": "Publisher-hosted article/supplement requests failed; published full text read via ResearchGate and publisher syndicated search extract. Supplement S1-S7 not directly inspected.",
    })
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
