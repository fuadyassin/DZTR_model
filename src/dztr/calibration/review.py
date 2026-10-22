"""Prepare calibration bounds and verify objective evaluation before optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from dztr import __version__
from dztr.data import read_reservoir_observations
from dztr.integrations import read_matlab_coefficients, read_matlab_observations
from dztr.parameterization import estimate_generalized_parameters
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters

from .baseline import json_ready
from .bounds import estimate_calibration_bounds
from .candidates import parameters_to_vector
from .objectives import CalibrationObjective, CandidateEvaluation
from .periods import CalibrationValidationSplit, DatePeriod, select_period


def _describe_evaluation(result: CandidateEvaluation) -> dict:
    return {"feasible": result.feasible, "objectives": list(result.objectives), "metrics": result.metrics,
            "paired_counts": result.paired_counts, "reason": result.reason, "period": result.period.as_dict()}


def _load_mat(path: Path, variable: str):
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("Saved MATLAB reference checks require the project MATLAB extra: pip install '.[matlab]'.") from exc
    content = loadmat(path, simplify_cells=True, variable_names=[variable])
    if variable not in content:
        raise ValueError(f"MAT file {path.name} is missing {variable}.")
    return content[variable]


def run_calibration_review(
    config_path: str | Path, repo_root: str | Path, output_dir: str | Path,
    *, matlab_folder: str | Path | None = None,
) -> dict:
    """Export a bounded candidate space, baseline costs and optional saved parity.

    Historical candidates are verified using their original physical settings
    and saved evaluation outputs. They are not newly optimized candidates or
    evidence of independent held-out performance under the current split.
    """
    root, output, config_path = Path(repo_root), Path(output_dir), Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selection = config["periods"]
    split = CalibrationValidationSplit(DatePeriod(**selection["calibration"]), DatePeriod(**selection["validation"]),
                                       warmup_steps=selection["warmup_steps"])
    source = root / config["observations_file"]
    artifact_names = ("bounds_si.csv", "candidate_space.csv", "baseline_targets_si.csv", "candidate_vectors.csv",
                      "candidate_checks.csv", "reference_checks.csv", "summary.json", "resolved_config.json")
    if any((output / name).resolve() in {source.resolve(), config_path.resolve()} for name in artifact_names):
        raise ValueError("The output folder would overwrite source observations or configuration; choose another folder.")
    observed = read_reservoir_observations(source)
    physical = dict(config["model"])
    channel = physical.pop("channel_capacity_m3s", None)
    method = config.get("quantile_method", "hazen")
    estimates = estimate_generalized_parameters(select_period(observed, split.calibration),
        tuple(config.get("target_percentiles", (10, 45, 85))), channel_capacity_m3s=channel, quantile_method=method)
    template = ReservoirParameters.from_generalized(estimates, **physical)
    bounds = estimate_calibration_bounds(observed, split.calibration, template, quantile_method=method)
    settings = config.get("simulation", {})
    problem = CalibrationObjective(observed, split.calibration, template, warmup_steps=split.warmup_steps,
                                   bounds=bounds, **settings)
    baseline_vector = parameters_to_vector(template)
    calibration = problem.evaluate(baseline_vector)
    validation = problem.evaluate_validation(baseline_vector, split.validation)
    if not calibration.feasible or not validation.feasible:
        raise ValueError(f"Generalized baseline is not a valid starting candidate: {calibration.reason or validation.reason}")
    report = {
        "experiment": "calibration_objective_review",
        "objective_order": ["negative_release_nse", "negative_storage_nse"],
        "objective_direction": "minimize",
        "reservoir": config.get("reservoir"),
        "warmup_steps": split.warmup_steps,
        "bounds": bounds.summary(),
        "baseline": {"calibration": _describe_evaluation(calibration), "validation": _describe_evaluation(validation)},
        "model": {key: getattr(template, key) for key in (
            "capacity_m3", "initial_storage_m3", "initial_release_m3s", "channel_capacity_m3s",
            "dead_storage_fraction", "inflow_correction", "flood_mode",
        )},
        "solver": problem.solver, "overflow_mode": problem.overflow_mode,
        "reference_checks": [],
        "source_sha256": {"observations": hashlib.sha256(source.read_bytes()).hexdigest(),
                          "configuration": hashlib.sha256(config_path.read_bytes()).hexdigest()},
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "dztr": __version__},
        "notes": [
            "No optimizer was run. The baseline is a starting candidate, not an optimized solution.",
            "Bounds and targets use calibration data only; optimization costs do not use validation data.",
            "Raw percentile boxes include invalid physical combinations; candidates are rejected rather than repaired.",
            "Candidate rejection uses infinite costs plus an explicit reason; saved JSON represents infinity as null.",
            "Historical MATLAB candidates may lie outside the current calibration-derived bounds and are verification references only.",
        ],
    }
    space = pd.DataFrame({
        "vector_index": np.arange(72), "target_column": np.repeat(MONTHLY_TARGET_COLUMNS, 12),
        "month": np.tile(np.arange(1, 13), 6), "unit": ["fraction_of_capacity"] * 36 + ["m3/s"] * 36,
        "lower": bounds.lower_vector, "upper": bounds.upper_vector, "baseline": baseline_vector,
    })
    vectors = {"generalized_baseline": baseline_vector}
    candidate_checks = [{"candidate": "generalized_baseline", "role": "starting_candidate", "inside_percentile_box": True,
                         "physically_valid_within_bounds": True, "reason": None}]
    reference_rows = []
    if matlab_folder is not None:
        folder = Path(matlab_folder)
        compatibility_path = root / "configs/trinity_simulation.json"
        compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
        matlab_observed = read_matlab_observations(folder / compatibility["observations_file"])
        if not matlab_observed.frame.equals(observed.frame):
            raise ValueError("Historical Trinity references do not match the configured observation record.")
        original = read_matlab_coefficients(folder / compatibility["full_precision_coefficient_file"],
            **{key: compatibility[key] for key in ("inflow_correction", "dead_storage_fraction", "flood_mode")})
        ensemble = np.asarray(_load_mat(folder / "amalgam_out.mat", "X"), dtype=float)
        saved_storage = np.asarray(_load_mat(folder / "stoSIM_eval.mat", "stoSIM_eval"), dtype=float)
        saved_release = np.asarray(_load_mat(folder / "flowSIM_eval.mat", "flowSIM_eval"), dtype=float)
        saved_metrics = np.asarray(_load_mat(folder / "FA_eval.mat", "FA_eval"), dtype=float)
        if ensemble.ndim != 2 or ensemble.shape[1] != 72:
            raise ValueError("Historical ensemble must contain 72 values per candidate.")
        expected_shape = (len(matlab_observed.frame), len(ensemble))
        if (saved_storage.shape != expected_shape or saved_release.shape != expected_shape
                or saved_metrics.ndim != 2 or saved_metrics.shape[0] != len(ensemble) or saved_metrics.shape[1] < 11):
            raise ValueError("Saved simulation and metric dimensions do not match the candidate ensemble.")
        warmup = int(compatibility["warmup_steps"])
        historical_period = DatePeriod(matlab_observed.frame.index[warmup], matlab_observed.frame.index[-1])
        reference_problem = CalibrationObjective(matlab_observed, historical_period, original, warmup_steps=warmup)
        for member in compatibility["reference_member_indices"]:
            name = f"matlab_member_{member + 1}"
            vector = ensemble[member]
            print(f"Checking signed objectives for saved MATLAB member {member + 1}...", flush=True)
            evaluated = reference_problem.evaluate(vector)
            if not evaluated.feasible:
                raise ValueError(f"Saved reference {name} is infeasible: {evaluated.reason}")
            expected = -saved_metrics[member, [3, 10]]
            objective_error = np.abs(np.asarray(evaluated.objectives) - expected)
            storage_error = float(np.max(np.abs(evaluated.simulation.frame.storage_m3.to_numpy() - saved_storage[:, member])))
            release_error = float(np.max(np.abs(evaluated.simulation.frame.release_m3s.to_numpy() - saved_release[:, member])))
            check = {
                "matlab_member_number": member + 1,
                "evaluation_period": historical_period.as_dict(),
                "simulated_daily_rows": len(evaluated.simulation.frame),
                "objectives": list(evaluated.objectives), "saved_objectives": expected.tolist(),
                "max_objective_error": float(objective_error.max()),
                "max_storage_error_m3": storage_error, "max_release_error_m3s": release_error,
                "passed": bool(objective_error.max() <= 1e-10 and storage_error <= 1e-3 and release_error <= 1e-8),
            }
            report["reference_checks"].append(check)
            reference_rows.append({key: value for key, value in check.items() if key not in ("evaluation_period", "objectives", "saved_objectives")})
            vectors[name] = vector
            try:
                bounds.validate(vector, template)
                valid, reason = True, None
            except ValueError as exc:
                valid, reason = False, str(exc)
            candidate_checks.append({"candidate": name, "role": "historical_verification_reference",
                                     "inside_percentile_box": bounds.contains(vector),
                                     "physically_valid_within_bounds": valid, "reason": reason})
        for path in (compatibility_path, *(folder / name for name in (
            compatibility["observations_file"], compatibility["full_precision_coefficient_file"],
            "amalgam_out.mat", "stoSIM_eval.mat", "flowSIM_eval.mat", "FA_eval.mat",
        ))):
            report["source_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["notes"].extend([
            "Saved 2018 AMALGAM bounds use P5-25/P30-65/P70-95, distinct from the published P5-35/P35-75/P75-95 bounds used here.",
            "Signed objective parity uses actual-calendar evaluation outputs; calendar-shifted amalgam_out F/sim arrays are excluded.",
            "Reference candidates were optimized historically with a different protocol; these checks do not establish independent validation performance.",
        ])
    report["historical_reference_check_performed"] = matlab_folder is not None
    report["passed"] = calibration.feasible and validation.feasible and all(check["passed"] for check in report["reference_checks"])
    report["configuration"] = config
    report = json_ready(report)
    output.mkdir(parents=True, exist_ok=True)
    bounds.monthly_bounds.to_csv(output / "bounds_si.csv")
    space.to_csv(output / "candidate_space.csv", index=False)
    estimates.monthly_targets.to_csv(output / "baseline_targets_si.csv")
    pd.DataFrame.from_dict(vectors, orient="index", columns=[f"x{i:02d}" for i in range(72)]).rename_axis("candidate").to_csv(output / "candidate_vectors.csv")
    pd.DataFrame(candidate_checks).to_csv(output / "candidate_checks.csv", index=False)
    pd.DataFrame(reference_rows, columns=("matlab_member_number", "simulated_daily_rows", "max_objective_error",
        "max_storage_error_m3", "max_release_error_m3s", "passed")).to_csv(output / "reference_checks.csv", index=False)
    for name, value in (("summary.json", report), ("resolved_config.json", config)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/trinity_calibration"))
    parser.add_argument("--matlab-folder", type=Path, help="Optional original Trinity folder for full-record objective parity.")
    args = parser.parse_args()
    report = run_calibration_review(args.config or args.repo / "configs/trinity_baseline.json", args.repo,
                                    args.output, matlab_folder=args.matlab_folder)
    for period, result in report["baseline"].items():
        print(f"Baseline {period}: objectives={result['objectives']}")
    print(f"Calibration review passed: {report['passed']}")
    print(f"Artifacts: {args.output.resolve()}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
