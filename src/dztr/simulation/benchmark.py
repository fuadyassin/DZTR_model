"""Compare the Python port with Trinity MATLAB observations and saved outputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dztr.data import read_reservoir_observations
from dztr.evaluation import evaluate_simulation
from dztr.integrations import parameters_from_matlab_vector, read_matlab_coefficients, read_matlab_observations
from dztr.parameterization import estimate_generalized_parameters
from dztr.schemas import ReservoirParameters

from .runner import simulate_reservoir


def _load_mat(path: Path) -> dict:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("MATLAB reference files require scipy; install the project with .[matlab].") from exc
    return loadmat(path, simplify_cells=True)


def _legacy_climatology(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve the MATLAB first-365-rows policy only for artifact comparison."""
    groups = [group.iloc[:365][["storage_m3", "release_m3s"]].to_numpy()
              for year, group in frame.groupby(frame.index.year) if 1971 <= year <= 2000]
    if len(groups) != 30 or any(len(group) != 365 for group in groups):
        raise ValueError("Legacy annual comparison requires all years 1971 through 2000.")
    return pd.DataFrame(np.mean(np.stack(groups), axis=0), columns=["storage_m3", "release_m3s"])


def run_trinity_comparison(matlab_folder: str | Path, repo_root: str | Path, output_dir: str | Path) -> dict:
    """Run coefficient, saved ensemble and current parameterization comparisons."""
    folder, root, output = Path(matlab_folder), Path(repo_root), Path(output_dir)
    config = json.loads((root / "configs/trinity_simulation.json").read_text(encoding="utf-8"))
    kwargs = {key: config[key] for key in ("inflow_correction", "dead_storage_fraction", "flood_mode")}
    observations = read_matlab_observations(folder / config["observations_file"])
    repository_obs = read_reservoir_observations(root / "ResDataInflowStorageRelease/Trinity_131.xlsx")
    identical = observations.frame.equals(repository_obs.frame)
    if not identical:
        raise ValueError("MATLAB observations differ from the repository's Trinity dataset.")
    parameters = read_matlab_coefficients(folder / config["coefficient_file"], **kwargs)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "daily_records": len(observations.frame),
        "observations_match_repository": identical,
        "initial_conditions": {"capacity_m3": parameters.capacity_m3,
                               "initial_storage_m3": parameters.initial_storage_m3,
                               "initial_release_m3s": parameters.initial_release_m3s,
                               "channel_capacity_m3s": parameters.channel_capacity_m3s},
        "settings": kwargs,
        "warmup_steps": config["warmup_steps"],
        "baseline": {}, "ensemble_reference_comparisons": [],
        "source_sha256": {},
        "notes": [
            "Verification uses saved MATLAB outputs; MATLAB itself was not executed.",
            "Evaluation references use the actual 1970 calendar. Saved optimization outputs appear to use a leap-year-start calendar and are excluded from primary parity checks.",
            "Annual artifact comparison deliberately preserves MATLAB's first-365-rows policy, including its leap-year misalignment.",
        ],
    }
    print("Running imported Trinity coefficients with the legacy iteration scheme...", flush=True)
    legacy = simulate_reservoir(observations, parameters, solver="matlab_fixed_point", overflow_mode="matlab")
    implicit = simulate_reservoir(observations, parameters)
    implicit.frame.to_csv(output / "daily_imported_matlab.csv", index_label="date")
    report["baseline"]["legacy_vs_implicit_max_storage_error_m3"] = float(np.max(np.abs(legacy.frame.storage_m3 - implicit.frame.storage_m3)))
    report["baseline"]["legacy_vs_implicit_max_release_error_m3s"] = float(np.max(np.abs(legacy.frame.release_m3s - implicit.frame.release_m3s)))
    report["baseline"]["legacy_metrics"] = evaluate_simulation(observations, legacy, warmup_steps=config["warmup_steps"])
    report["baseline"]["implicit_metrics"] = evaluate_simulation(observations, implicit, warmup_steps=config["warmup_steps"])
    annual = _load_mat(folder / "anual.mat")["anual"]
    climatology = _legacy_climatology(legacy.frame)
    annual_storage = np.asarray(annual["storages"]).reshape(-1)
    annual_release = np.asarray(annual["outflows"]).reshape(-1)
    report["baseline"]["annual_max_storage_error_m3"] = float(np.max(np.abs(climatology.storage_m3 - annual_storage)))
    report["baseline"]["annual_max_release_error_m3s"] = float(np.max(np.abs(climatology.release_m3s - annual_release)))
    climatology.to_csv(output / "legacy_climatology_comparison.csv", index_label="legacy_day_index")
    for method in ("linear", "hazen"):
        estimates = estimate_generalized_parameters(observations,
                    channel_capacity_m3s=parameters.channel_capacity_m3s, quantile_method=method)
        current = ReservoirParameters.from_generalized(estimates,
                    capacity_m3=parameters.capacity_m3,
                    initial_storage_m3=parameters.initial_storage_m3,
                    initial_release_m3s=parameters.initial_release_m3s, **kwargs)
        result = simulate_reservoir(observations, current)
        result.frame.to_csv(output / f"daily_generalized_{method}.csv", index_label="date")
        estimates.monthly_targets.to_csv(output / f"monthly_targets_{method}.csv")
        report[f"generalized_{method}"] = {
            "metrics": evaluate_simulation(observations, result, warmup_steps=config["warmup_steps"]),
            "max_balance_residual_m3": float(result.frame.balance_residual_m3.abs().max()),
            "issues": list(result.issues),
        }
    full_precision = read_matlab_coefficients(folder / config["full_precision_coefficient_file"], **kwargs)
    full_precision_run = simulate_reservoir(observations, full_precision)
    metrics = evaluate_simulation(observations, full_precision_run, warmup_steps=config["warmup_steps"])
    fa = np.asarray(_load_mat(folder / "FA.mat")["FA"]).reshape(-1)
    report["full_precision_reference"] = {
        "metrics": metrics,
        "saved_release_nse": float(fa[3]), "saved_storage_nse": float(fa[10]),
        "release_nse_error": abs(metrics["release_nse"] - float(fa[3])),
        "storage_nse_error": abs(metrics["storage_nse"] - float(fa[10])),
    }
    ensemble = _load_mat(folder / "amalgam_out.mat")["X"]
    saved_storage = _load_mat(folder / "stoSIM_eval.mat")["stoSIM_eval"]
    saved_release = _load_mat(folder / "flowSIM_eval.mat")["flowSIM_eval"]
    saved_metrics = _load_mat(folder / "FA_eval.mat")["FA_eval"]
    for member in config["reference_member_indices"]:
        print(f"Comparing MATLAB evaluation member {member + 1} over the full record...", flush=True)
        candidate = parameters_from_matlab_vector(ensemble[member], full_precision)
        result = simulate_reservoir(observations, candidate)
        result_metrics = evaluate_simulation(observations, result, warmup_steps=config["warmup_steps"])
        storage_error = float(np.max(np.abs(result.frame.storage_m3.to_numpy() - saved_storage[:, member])))
        release_error = float(np.max(np.abs(result.frame.release_m3s.to_numpy() - saved_release[:, member])))
        report["ensemble_reference_comparisons"].append({
            "matlab_member_number": member + 1,
            "max_storage_error_m3": storage_error,
            "max_release_error_m3s": release_error,
            "max_balance_residual_m3": float(result.frame.balance_residual_m3.abs().max()),
            "metrics": result_metrics,
            "release_nse_error": abs(result_metrics["release_nse"] - float(saved_metrics[member, 3])),
            "storage_nse_error": abs(result_metrics["storage_nse"] - float(saved_metrics[member, 10])),
            "passed": storage_error <= 1e-3 and release_error <= 1e-8,
        })
    for name in (config["observations_file"], config["coefficient_file"], config["full_precision_coefficient_file"],
                 "DZTR.m", "SimRISM14.m", "anual.mat", "FA.mat", "FA_eval.mat",
                 "amalgam_out.mat", "stoSIM_eval.mat", "flowSIM_eval.mat"):
        report["source_sha256"][name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    report["passed"] = (
        report["baseline"]["annual_max_storage_error_m3"] <= 1e-3
        and report["baseline"]["annual_max_release_error_m3s"] <= 1e-8
        and report["full_precision_reference"]["release_nse_error"] <= 1e-10
        and report["full_precision_reference"]["storage_nse_error"] <= 1e-10
        and all(item["passed"] for item in report["ensemble_reference_comparisons"])
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab-folder", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("outputs/trinity_simulation"))
    args = parser.parse_args()
    report = run_trinity_comparison(args.matlab_folder, args.repo, args.output)
    print(f"Trinity MATLAB comparison passed: {report['passed']}")
    print(f"Artifacts: {args.output.resolve()}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
