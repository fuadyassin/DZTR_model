"""Independently verify a completed Gardiner handoff without optimizer/model calls."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from dztr.calibration.checkpoint import load_optimizer_checkpoint
from strategies.representative_years import YearSelection
from workflows import prepared_calibration_v1 as adapter
from workflows import profile_application_v1 as application
from workflows import reservoir_v2 as native
from workflows import run_manager_v2 as manager


def nse(observed, simulated):
    observed, simulated = np.asarray(observed, dtype=float), np.asarray(simulated, dtype=float)
    valid = np.isfinite(observed)
    if not np.isfinite(simulated[valid]).all():
        raise ValueError("Invalid simulated values cannot remove scored observations")
    x, y = observed[valid], simulated[valid]
    denominator = np.sum((x - np.mean(x)) ** 2)
    return float(1 - np.sum((x - y) ** 2) / denominator) if denominator else np.nan


def verify(output, reference_archive=None):
    out = Path(output).resolve(strict=True)
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Verify a run inside project outputs")
    manager.verify_inputs(native.read(out / "plan.json"))
    status = native.read(out / "status.json")
    if status["status"] != "completed":
        raise ValueError("Managed workflow has not completed")
    stages = {r["id"]: r for r in status["stages"]}
    for stage in stages.values():
        receipt = native.read(out / "stages" / stage["id"] / "receipt.json")
        for path, expected in receipt["artifact_sha256"].items():
            if native.sha(path) != expected:
                raise ValueError("Managed stage artifact changed")
    calibration = Path(stages["calibration"]["outputs"]["campaign"])
    _, cfg = adapter.verify(calibration)
    registry_path = Path(stages["calibration"]["outputs"]["registry"])
    native.verify_registry(registry_path)
    profile, parameters = native.load_profile(registry_path, grand_id=253)
    folder = calibration / "prepared/ex14_Gardiner"
    observations = pd.read_csv(folder / "observations_si.csv", index_col="date", parse_dates=True, float_precision="round_trip")
    selection = YearSelection(**native.read(folder / "selection.json"))
    targets = pd.read_csv(folder / "initial_targets_si.csv", index_col="month", float_precision="round_trip")
    mask = selection.mask(observations.index, "calibration")
    expected = np.vstack([np.concatenate([
        np.percentile(observations.loc[mask & (observations.index.month == m), c], [10, 45, 85], method="hazen")
        for c in ("storage_m3", "release_m3s")]) for m in range(1, 13)])
    np.testing.assert_array_equal(targets, expected)
    source_cfg = native.read(calibration / "handoff_config.json")
    source_campaign = Path(source_cfg["campaign"])
    source_profile = native.read(folder / "source_profile.json")
    source_folder = source_campaign / source_profile["prepared_folder"]
    for name, source in (("initial_targets_si.csv", source_folder / "monthly_targets_si.csv"),
                         ("observations_si.csv", source_folder.parent / "observations_si.csv"),
                         ("selection.json", source_folder.parent / "selection.json")):
        assert native.sha(folder / name) == native.sha(source)
    assert profile["physical_constants"] == source_profile["identity"]["physical_constants"]
    assert profile["selection_sha256"] == selection.signature
    assert profile["selection_uses_assessment"] is False
    replay_folder = Path(stages["replay"]["outputs"]["campaign"])
    simulated = pd.read_csv(replay_folder / "simulation.csv", index_col="date", parse_dates=True, float_precision="round_trip")
    assert simulated.index.equals(observations.index)
    flux = parameters.inflow_correction * observations.inflow_m3s.to_numpy() - simulated.release_m3s.to_numpy()
    residual = np.diff(simulated.storage_m3.to_numpy()) - simulated.step_seconds.to_numpy()[1:] * .5 * (flux[1:] + flux[:-1])
    maximum = float(np.max(np.abs(residual)))
    assert maximum <= profile["atol_storage_m3"]
    assert simulated.storage_m3.between(0, parameters.capacity_m3).all()
    assert np.isfinite(simulated[["storage_m3", "release_m3s"]]).all().all()
    for phase in ("calibration", "assessment"):
        support = selection.mask(observations.index, phase)
        recorded = profile[phase]["metrics"] if phase == "assessment" else profile[phase]
        for variable, column in (("storage", "storage_m3"), ("release", "release_m3s")):
            independent = nse(observations.loc[support, column], simulated.loc[support, column])
            np.testing.assert_allclose(independent, recorded[variable + "_nse"], rtol=0, atol=1e-12)
    budget = cfg["optimization"]["evaluation_budget"]
    histories, archive_comparison = {}, {}
    for seed in cfg["optimization"]["seeds"]:
        job = calibration / "jobs/ex14_Gardiner" / f"seed_{seed}"
        state = load_optimizer_checkpoint(job / "checkpoint.npz", native.read(job / "identity.json"))
        assert len(state.evaluated) == budget
        trials = pd.read_csv(job / f"budget_{budget}/trials.csv", float_precision="round_trip")
        vectors = pd.read_csv(job / f"budget_{budget}/vectors.csv", float_precision="round_trip")
        previous = job / "budget_256/trials.csv"
        if previous.exists() and budget >= 256:
            prefix = pd.read_csv(previous, float_precision="round_trip")
            pd.testing.assert_frame_equal(trials.iloc[:len(prefix)].reset_index(drop=True), prefix)
        histories[str(seed)] = {"evaluations": len(trials), "feasible": int(trials.feasible.sum())}
        if reference_archive:
            prior = Path(reference_archive) / "jobs/ex14_Gardiner/mixed_pooled" / f"seed_{seed}"
            identical = True
            for name, current in (("trials.csv", trials), ("vectors.csv", vectors)):
                historical = pd.read_csv(prior / name, float_precision="round_trip")
                try: pd.testing.assert_frame_equal(current, historical)
                except AssertionError: identical = False
            archive_comparison[str(seed)] = {"entire_search_exactly_matches_prior_archive": identical}
    continuation = native.read(out / "checkpoint_256_receipt.json")
    for relative, expected_hash in continuation["immutable_snapshot_sha256"].items():
        assert native.sha(out / relative) == expected_hash
    assert native.sha(Path(continuation["registry"]).parent / "manifest.json") == continuation["published_manifest_sha256"]
    catalog = application.read_catalog(ROOT / "outputs/calibrated_paper_cohort_v1_20261017_final/catalog.json")
    originals = {**catalog["application_source_sha256"], catalog["source_table"]: catalog["source_table_sha256"]}
    for entry in catalog["reservoirs"]:
        originals.update(entry.get("input_sha256", {}))
    for relative, expected_hash in originals.items():
        assert native.sha(application.reference(relative)) == expected_hash
    comparisons = pd.read_csv(registry_path.parent / "baseline_comparison.csv")
    assert len(comparisons) == 2 * (len(cfg["optimization"]["seeds"]) + 1)
    assert not comparisons.selection_uses_assessment.any()
    for variable in ("storage", "release"):
        np.testing.assert_allclose(comparisons["improvement_" + variable + "_nse"],
            comparisons["calibrated_" + variable + "_nse"] - comparisons["baseline_" + variable + "_nse"])
    return {"kind": "prepared_gardiner_end_to_end_validation_v1", "selected_seed": profile["seed"],
            "selected_evaluation_id": profile["evaluation_id"], "selection_signature": selection.signature,
            "calibration_fraction": float(mask.mean()), "target_values_reproduced": 72,
            "exact_prepared_input_handoff": True, "physical_constants_unchanged": True,
            "independent_mass_balance_max_m3": maximum, "mass_balance_tolerance_m3": profile["atol_storage_m3"],
            "raw_nse_independently_recomputed": 4, "checkpoint_continuation": histories,
            "budget_256_snapshots_and_release_unchanged": True, "comparison_to_prior_archive": archive_comparison,
            "original_pinned_files_unchanged": len(originals), "verification_optimizer_evaluations": 0,
            "verification_model_simulations": 0, "baseline_comparison_rows": len(comparisons)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--reference-archive")
    args = parser.parse_args()
    result = verify(args.run, args.reference_archive)
    native.write(Path(args.run) / "scientific_verification.json", result)
    print(result)
