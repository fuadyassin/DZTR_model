"""Fleet execution summary and descriptive convergence from completed archives."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from policy_fleet import checked, DEFAULT_OUT
from policy_fleet_job import ROOT, read, sha, write
from dztr.calibration.convergence import hypervolume_2d


def main():
    out = DEFAULT_OUT
    checked(out)
    verification = read(out / "scientific_verification.json")
    if verification["status"] != "passed" or verification["completed_jobs_audited"] != 60:
        raise ValueError("Summaries require complete scientific verification.")
    checks = pd.read_csv(out / "scientific_checks.csv").set_index(["reservoir_id", "seed"])
    rows, fronts, consumed = [], {}, {}
    for job in read(out / "execution_summary.json")["jobs"]:
        folder = out / "pilots" / job["reservoir_id"] / f"seed_{job['seed']}"
        result = read(folder / "result.json")
        role = next(r for r in result["representatives"] if r["role"] == "balanced")
        trial_path, front_path = folder / "trials.csv", folder / "pareto_front.csv"
        trial = pd.read_csv(trial_path, float_precision="round_trip")
        front = pd.read_csv(front_path, float_precision="round_trip")
        for path in (folder / "result.json", trial_path, front_path):
            consumed[path.relative_to(ROOT).as_posix()] = sha(path)
        feasible = trial[trial.feasible]
        costs = feasible[["release_cost", "storage_cost"]].to_numpy()
        prefix = feasible[feasible.evaluation_id < 256][["release_cost", "storage_cost"]].to_numpy()
        fronts.setdefault(job["reservoir_id"], {})[job["seed"]] = front[["release_cost", "storage_cost"]].to_numpy()
        row = {"reservoir_id": job["reservoir_id"], "seed": job["seed"], "root_policy": job["root_policy"],
               "search_evaluations": result["search_evaluations"], "balanced_evaluation_id": role["evaluation_id"],
               "feasible_search_trials": int(checks.loc[(job["reservoir_id"], job["seed"]), "feasible_trials"]),
               "baseline_calibration_release_nse": result["baseline_calibration"]["release_nse"],
               "baseline_calibration_storage_nse": result["baseline_calibration"]["storage_nse"],
               "baseline_validation_release_nse": result["baseline_validation"]["metrics"].get("release_nse"),
               "baseline_validation_storage_nse": result["baseline_validation"]["metrics"].get("storage_nse"),
               **{k: role[k] for k in ("calibration_release_nse", "calibration_storage_nse", "validation_feasible", "validation_release_nse", "validation_storage_nse")},
               "calibration_hypervolume_256": hypervolume_2d(prefix), "calibration_hypervolume_512": hypervolume_2d(costs),
               "front_points_outside_hypervolume_reference": int((front[["release_cost", "storage_cost"]].to_numpy() > 1).any(axis=1).sum()),
               "initialization_calls": result["initialization"]["simulation_calls"],
               "representative_calibration_replays": result["representative_calibration_replays"], "validation_attempts": result["validation_attempts"]}
        row["hypervolume_gain_256_to_512"] = row["calibration_hypervolume_512"] - row["calibration_hypervolume_256"]
        rows.append(row)
    jobs = pd.DataFrame(rows).sort_values(["reservoir_id", "seed"])
    jobs.to_csv(out / "job_metrics.csv", index=False, float_format="%.17g")
    records = []
    for source in read(out / "admission.json")["reservoirs"]:
        identifier = source["reservoir_id"]
        subset = jobs[jobs.reservoir_id == identifier]
        record = {"reservoir_id": identifier, "name": source["name"], "included_in_program": True,
                  "status": "calibration_completed" if len(subset) else source["status"],
                  "completed_jobs": len(subset), "next_action": source["next_action"] if not len(subset) else "Review held-out behavior and calibration seed variation; preserve source assumptions.",
                  "convergence_established": False}
        if len(subset):
            for metric in ("calibration_release_nse", "calibration_storage_nse", "validation_release_nse", "validation_storage_nse"):
                record["median_" + metric] = float(subset[metric].median())
            distances = []
            for a, b in combinations(fronts[identifier].values(), 2):
                distance = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
                distances.append(max(float(distance.min(axis=0).max()), float(distance.min(axis=1).max())))
            record["max_between_seed_front_distance_nse"] = max(distances)
            record["median_hypervolume_gain_256_to_512"] = float(subset.hypervolume_gain_256_to_512.median())
            record["held_out_negative_storage_seeds"] = int((subset.validation_storage_nse < 0).sum())
            record["held_out_negative_release_seeds"] = int((subset.validation_release_nse < 0).sum())
            record["front_points_outside_reference"] = int(subset.front_points_outside_hypervolume_reference.sum())
        records.append(record)
    reservoirs = pd.DataFrame(records)
    reservoirs.to_csv(out / "reservoir_results.csv", index=False, float_format="%.17g")
    completed = reservoirs[reservoirs.completed_jobs > 0]
    summary = {"status": "completed_and_scientifically_verified", "active_reservoirs": 41,
               "newly_calibrated_reservoirs": 20, "completed_jobs": 60,
               "total_search_evaluations": int(jobs.search_evaluations.sum()),
               "initialization_calls": int(jobs.initialization_calls.sum()),
               "selected_calibration_replays": int(jobs.representative_calibration_replays.sum()),
               "validation_attempts": int(jobs.validation_attempts.sum()),
               "failed_balanced_validation_jobs": int((~jobs.validation_feasible).sum()),
               "balanced_jobs_negative_held_out_release_nse": int((jobs.validation_release_nse < 0).sum()),
               "balanced_jobs_negative_held_out_storage_nse": int((jobs.validation_storage_nse < 0).sum()),
               "reservoirs_median_both_held_out_nse_positive": int(((completed.median_validation_release_nse > 0) & (completed.median_validation_storage_nse > 0)).sum()),
               "jobs_positive_hypervolume_gain_256_to_512": int((jobs.hypervolume_gain_256_to_512 > 1e-12).sum()),
               "max_balance_residual_m3": verification["max_balance_residual_m3"],
               "convergence_established": False, "new_summary_model_evaluations": 0,
               "reference": [1, 1], "reference_limitation": "Fronts with costs outside the fixed reference are partially or wholly excluded; zero hypervolume gain does not establish convergence.",
               "consumed_sha256": consumed}
    write(out / "execution_results.json", summary)
    text = "# Completed reservoir calibration batch\n\nAll 60 jobs completed across 20 reservoirs, using population 32, seeds42/43/44 and a512-evaluation search cap. Total search evaluations: **30,720**. All 41 active reservoirs remain in the program; Bull Lake retains its separate reviewed interval archives and 20 others retain their specific action queues.\n\n"
    text += f"Scientific checks passed for every completed job. Maximum numerical balance residual was **{verification['max_balance_residual_m3']:.8g} m3**, within the configured0.0001 m3 tolerance. Checks reconstructed trial/checkpoint equality, empirical bounds, physical targets, fronts, selections, scores, continuity and conservative spill from saved artifacts. No verification or reporting simulations were performed.\n\n"
    text += f"Outside the search budget, the jobs made {summary['initialization_calls']} initialization checks, {summary['selected_calibration_replays']} selected calibration replays and {summary['validation_attempts']} held-out reporting attempts. Search trials include rejected candidates; trial counts are not a count of successful full simulations.\n\n"
    text += f"All balanced held-out runs completed, but numerical success does not guarantee useful prediction: {summary['balanced_jobs_negative_held_out_release_nse']} seed runs have negative held-out release NSE and {summary['balanced_jobs_negative_held_out_storage_nse']} have negative held-out storage NSE. {summary['reservoirs_median_both_held_out_nse_positive']} of20 reservoirs have positive median held-out NSE for both variables. All seeds are retained; no validation winner is selected.\n\n"
    text += f"Calibration hypervolume improved between256 and512 evaluations for {summary['jobs_positive_hypervolume_gain_256_to_512']} of60 jobs under the fixed[1,1] negative-NSE reference. Between-seed front distances and reference-exclusion counts are reported. These are descriptive diagnostics, not proof of convergence; zero area/gain can reflect a front outside the reference. No further optimization was launched.\n\n"
    text += "[41-reservoir results/action table](reservoir_results.csv), [all seed metrics](job_metrics.csv), [scientific checks](scientific_checks.csv), [verification](scientific_verification.json), [execution totals](execution_results.json).\n\nCommon reports include trajectories, monthly targets, seed fronts and metrics:\n\n"
    if (out / "parameters/parameter_index.json").exists():
        text += "[Parameter export index](parameters/parameter_index.json): initialization and three calibration-selected roles, with monthly SI target tables and explicit model constants/policies for every seed.\n\n"
    text += "\n".join(f"- [{row['name']}](reports/reviewed_fleet/{row['reservoir_id']}/README.md)" for row in records if row["completed_jobs"]) + "\n\nSource definitions, inherited operating floors and monthly clocks remain provisional. Root-policy cases use a declared continuity heuristic, not independently identified operating intent. Poor held-out fits should be diagnosed before assuming a larger search budget will fix them.\n"
    (out / "EXECUTION_REPORT.md").write_text(text, encoding="utf-8")
    print({k: summary[k] for k in ("completed_jobs", "total_search_evaluations", "jobs_positive_hypervolume_gain_256_to_512", "reservoirs_median_both_held_out_nse_positive")})


if __name__ == "__main__":
    main()
