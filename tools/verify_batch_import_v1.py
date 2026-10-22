"""Check the batch against original profiles/reports without model or optimizer calls."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from workflows import profile_application_v1 as app
from workflows import reservoir_v2 as w
from workflows import verified_batch_v1 as batch
from dztr.schemas import MONTHLY_TARGET_COLUMNS


def verify(output):
    out = Path(output).resolve(strict=True)
    if not out.is_relative_to(ROOT / "outputs"): raise ValueError("Use a project output directory")
    design = w.read(out / "plan.json")
    batch.verify_source_hashes(design)
    state = w.read(out / "status.json")
    if state["status"] != "completed": raise ValueError("Batch has not completed")
    release = out / "release"
    registry = w.verify_registry(release / "registry.json")
    catalog = app.read_catalog(release / "catalog.json")
    original_catalog = app.read_catalog(ROOT / design["config"]["catalog"])
    original = pd.read_csv(app.reference(original_catalog["source_table"]), float_precision="round_trip").set_index("reservoir_id")
    current = pd.read_csv(release / registry["table"], float_precision="round_trip").set_index("reservoir_id")
    inventory = pd.read_csv(release / "report/reservoir_status.csv").set_index("reservoir_id")
    expected_ids = {r["reservoir_id"] for r in design["config"]["reservoirs"]}
    assert set(current.index) == set(inventory.index) == expected_ids
    assert {r["reservoir_id"] for r in catalog["reservoirs"]} == expected_ids
    pd.testing.assert_series_equal(current.GRAND_ID.astype("Int64"), original.loc[current.index].GRAND_ID.astype("Int64"))
    target_columns = [f"{c}_month_{m:02d}" for m in range(1, 13) for c in MONTHLY_TARGET_COLUMNS]
    imported = set(registry["profiles_by_reservoir_id"])
    copies, searches, routes = 0, 0, {}
    for identifier in expected_ids:
        receipt = w.read(out / "jobs" / identifier / "receipt.json")
        assert receipt["plan_signature"] == design["signature"]
        for relative, expected in receipt["artifact_sha256"].items(): assert w.sha(out / relative) == expected
        result = receipt["result"]
        assert result["new_optimizer_evaluations"] == result["new_simulations"] == 0
        if identifier in imported:
            source_registry = Path(result["original_registry"])
            old_registry = w.verify_registry(source_registry)
            old_params = source_registry.parent / old_registry["profiles_by_reservoir_id"][identifier]
            new_params = release / registry["profiles_by_reservoir_id"][identifier]
            assert w.sha(old_params) == w.sha(new_params)
            metadata = w.read(new_params)
            old_targets = old_params.parent / metadata["monthly_targets_file"]
            new_targets = new_params.parent / metadata["monthly_targets_file"]
            assert w.sha(old_targets) == w.sha(new_targets)
            np.testing.assert_array_equal(current.loc[identifier, target_columns].astype(float), original.loc[identifier, target_columns].astype(float))
            assert metadata["selection_uses_assessment"] is False
            routes[result["route"]] = routes.get(result["route"], 0) + 1
            for record in result["checkpoint_records"]:
                identity = Path(record["original_identity_file"])
                copied = release / "identities" / identifier / f"seed_{record['seed']}.json"
                assert w.sha(identity) == w.sha(copied) == record["original_identity_sha256"]
                assert w.sha(identity.parent / "checkpoint.npz") == record["checkpoint_sha256"]
                assert not record["checkpoint_copied_or_reidentified"]
                searches += 1
            copies += 72
        else:
            assert current.loc[identifier, target_columns].isna().all()
            assert not current.loc[identifier, "parameter_values_available"]
    for column in ("capacity_m3", "initial_storage_m3", "initial_release_m3s", "dead_storage_fraction", "inflow_correction",
                   "channel_capacity_m3s", "inflow_preprocessing_factor", "inflow_preprocessing_offset_m3s"):
        np.testing.assert_allclose(current[column], original.loc[current.index, column], rtol=0, atol=0, equal_nan=True)
    assessment = ROOT / design["config"]["assessment"]
    for name in batch.REPORT_FILES:
        before = pd.read_csv(assessment / name, float_precision="round_trip")
        expected = before.loc[before.reservoir_id.isin(imported)].reset_index(drop=True)
        after = pd.read_csv(release / "report" / name, float_precision="round_trip")
        pd.testing.assert_frame_equal(after, expected)
    comparisons = pd.read_csv(release / "report/baseline_comparison_all_seeds_and_selected.csv", float_precision="round_trip")
    assert len(comparisons) == len(imported) * 8
    assert not comparisons.selection_uses_assessment.any()
    for variable in ("storage", "release"):
        np.testing.assert_allclose(comparisons[variable + "_nse_gain"], comparisons[variable + "_nse"] - comparisons["baseline_" + variable + "_nse"], equal_nan=True)
    return {"kind": "verified_batch_independent_check_v1", "inventory_rows": len(expected_ids),
        "imported_profiles": len(imported), "retained_or_incompatible_rows": len(expected_ids) - len(imported),
        "routes_preserved": routes, "monthly_values_exact": copies, "original_seed_identities_exact": searches,
        "cached_assessment_reports_exact": len(batch.REPORT_FILES), "all_GRAND_mappings_preserved": True,
        "physical_constants_and_preprocessing_preserved": True, "verification_optimizer_evaluations": 0,
        "verification_model_simulations": 0, "original_source_hashes_rechecked": len(design["input_sha256"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", required=True)
    args = parser.parse_args(); result = verify(args.run)
    w.write(Path(args.run) / "scientific_verification.json", result)
    print(result)
