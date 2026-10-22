"""Verify the managed Gardiner geometry handoff using saved numerical outputs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from workflows import reservoir_v2 as native
from workflows import run_manager_v3 as manager
from workflows import geometry_calibration_v1 as calibration
from workflows.geometry_v1 import read_regeom
from strategies.representative_years import YearSelection
from tools.verify_level_storage_pilot_v1 import verify as verify_prior_pilot


def verify(output):
    out = Path(output).resolve(strict=True)
    state = native.read(out / "status.json"); design = native.read(out / "plan.json")
    assert state["status"] == "completed"
    manager.verify_inputs(design)
    for stage in state["stages"]:
        receipt = native.read(out / "stages" / stage["id"] / "receipt.json")
        for p, expected in receipt["artifact_sha256"].items(): assert native.sha(p) == expected, p
    targets = out / "stages/targets/campaign"
    campaign = out / "stages/calibration/campaign"
    calibration.verify(campaign)
    folder = campaign / "prepared/ex14_Gardiner"
    original = pd.read_csv(folder / "supplied_observations_si.csv", index_col="date", parse_dates=True, float_precision="round_trip")
    proxy = pd.read_csv(folder / "observations_si.csv", index_col="date", parse_dates=True, float_precision="round_trip")
    selection = YearSelection(**native.read(folder / "selection.json"))
    cal = selection.mask(original.index, "calibration"); held = selection.mask(original.index, "assessment")
    assert cal.sum() == 7669 and cal.mean() >= .65
    assert selection.signature == "2680f2925c7cce95717ec7b477f39fe2579e390f888eef825616dec89f5b0b60"
    source_frame = pd.read_csv(ROOT / "outputs/strategy_comparison_fleet_v2_20261012/prepared/ex14_Gardiner/mixed_pooled/observations_si.csv", index_col="date", parse_dates=True, float_precision="round_trip")
    pd.testing.assert_frame_equal(original, source_frame)
    pd.testing.assert_frame_equal(original[["inflow_m3s", "release_m3s"]], proxy[["inflow_m3s", "release_m3s"]])
    geometry = read_regeom(ROOT / "data/geometry_profiles_v1_20261017/253/source_regeom.csv", expected_grand_id=253)
    levels = pd.read_csv(ROOT / "data/gardiner_levels_v1_20261017/acquisition/levels_si.csv", float_precision="round_trip").level_m.to_numpy()
    offsets = levels[cal] - np.interp(original.storage_m3.to_numpy()[cal], geometry.storage_m3, geometry.depth_m)
    conversion = native.read(folder / "level_conversion.json")
    assert conversion["reference"]["zero_level_m"] == float(np.median(offsets))
    np.testing.assert_array_equal(conversion["uncertainty"]["effective_zero_bounds_m"], np.quantile(offsets, [.05, .95]))
    assert not conversion["uncertainty"]["confidence_interval"]
    initial = pd.read_csv(folder / "initial_targets_si.csv", index_col="month", float_precision="round_trip")
    prior = ROOT / "outputs/gardiner_level_storage_pilot_v1_20261017"
    previous = pd.read_csv(prior / "calibration_storage_alignment/monthly_targets_si.csv", index_col="month", float_precision="round_trip")
    pd.testing.assert_frame_equal(initial, previous)
    checkpoint = native.read(out / "checkpoint_8_receipt.json")
    for name, expected in checkpoint["snapshot_sha256"].items():
        assert native.sha(campaign / "jobs/ex14_Gardiner/seed_42/budget_8" / name) == expected
    for name in ("trials.csv", "vectors.csv"):
        early = pd.read_csv(campaign / "jobs/ex14_Gardiner/seed_42/budget_8" / name, float_precision="round_trip")
        continued = pd.read_csv(campaign / "jobs/ex14_Gardiner/seed_42/budget_16" / name, float_precision="round_trip")
        pd.testing.assert_frame_equal(early, continued.iloc[:len(early)].reset_index(drop=True))
        assert len(continued) == 16
    registry = campaign / "releases/budget_16/registry.json"
    profile, parameters = native.load_profile(registry, grand_id=253)
    simulation = pd.read_csv(registry.parent / "diagnostics/ex14_Gardiner/simulation.csv", index_col="date", parse_dates=True, float_precision="round_trip")
    replay = pd.read_csv(out / "stages/replay/campaign/simulation.csv", index_col="date", parse_dates=True, float_precision="round_trip")
    # Held-out diagnostics stop at the last assessment date; application replay
    # covers the full supplied calendar, including later calibration/context.
    assert simulation.index.isin(replay.index).all()
    np.testing.assert_array_equal(simulation.storage_m3, replay.loc[simulation.index, "storage_m3"])
    np.testing.assert_array_equal(simulation.release_m3s, replay.loc[simulation.index, "release_m3s"])
    diagnostic_rows = len(simulation)
    simulation = replay
    bands = pd.read_csv(folder / "storage_band_si.csv", float_precision="round_trip")
    s = simulation.storage_m3.to_numpy(); q = simulation.release_m3s.to_numpy()
    center = proxy.storage_m3.to_numpy()
    distance = np.maximum(np.maximum(bands.storage_lower_m3.to_numpy()[cal] - s[cal], s[cal] - bands.storage_upper_m3.to_numpy()[cal]), 0.)
    band_score = 1 - np.sum(distance**2) / np.sum((center[cal] - center[cal].mean())**2)
    np.testing.assert_allclose(-profile["storage_cost"], band_score, atol=1e-12)
    for phase, mask in (("calibration", cal), ("assessment", held)):
        for variable, values in (("storage", s), ("release", q)):
            observed = original.storage_m3.to_numpy() if variable == "storage" else original.release_m3s.to_numpy()
            nse = 1 - np.sum((observed[mask] - values[mask])**2) / np.sum((observed[mask] - observed[mask].mean())**2)
            recorded = profile["calibration_against_supplied_observations"] if phase == "calibration" else profile["assessment"]["metrics"]
            np.testing.assert_allclose(recorded[variable + "_nse"], nse, atol=1e-12)
    inflow = simulation.inflow_m3s.to_numpy() * parameters.inflow_correction
    dt = simulation.step_seconds.to_numpy()[1:]
    residual = np.diff(s) - dt * ((inflow[1:] + inflow[:-1]) - (q[1:] + q[:-1])) / 2
    maximum = float(np.abs(residual).max()); assert maximum <= 1e-4
    assert np.isfinite(s).all() and ((s >= 0) & (s <= parameters.capacity_m3)).all()
    table = pd.read_csv(registry.parent / "reservoir_parameterization_by_GRAND_ID.csv")
    assert table.GRAND_ID.iloc[0] == 253
    assert table.parameterization_method.iloc[0] == "level_geometry_monthly"
    assert table.level_reference_method.iloc[0] == "calibration_storage_alignment"
    assert table.storage_uncertainty_kind.iloc[0] == "calibration_offset_quantiles"
    old = verify_prior_pilot(prior)
    return {"status": "passed", "calibration_days": int(cal.sum()), "calibration_fraction": float(cal.mean()),
        "shared_initial_target_cells_equal_prior_pilot": int(initial.size), "reference_and_band_fitted_from_calibration_only": True,
        "supplied_observations_preserved": True, "continued_checkpoint_evaluations": 16, "earlier_8_evaluations_unchanged": True,
        "exported_profile_replay_exact_on_shared_timeline": True, "assessment_diagnostic_rows": diagnostic_rows,
        "application_replay_rows": len(replay), "maximum_independent_balance_residual_m3": maximum,
        "GRAND_ID": 253, "storage_objective": "reference_band_nse", "original_input_hashes_rechecked": old["original_inputs_rechecked"],
        "verification_new_model_calls": 0, "verification_new_optimizer_evaluations": 0,
        "scientific_convergence_established": False, "source_certified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = verify(args.output)
    calibration.write(Path(args.output) / "scientific_verification.json", result)
    print(json.dumps(result, indent=2))
