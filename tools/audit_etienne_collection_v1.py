"""Audit packaged observations and target estimates without simulation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import pandas as pd
from workflows import etienne_collection_v1 as w


def audit(collection, output):
    out, proof = w.verify(collection)
    registry = w.native.read(out / "parameterization_registry.json")
    inventory = w.native.read(out / "combined_inventory.json")
    ids = [r["reservoir_id"] for r in inventory["reservoirs"]]
    assert len(ids) == len(set(ids)) == proof["combined_reservoirs"]
    assert not any("Tuyen" in rid for rid in ids)
    base = w.native.read(ROOT / inventory["base_collection"])
    old_files = 0
    for entry in base["reservoirs"]:
        for relative, expected in entry["files_sha256"].items():
            assert w.native.sha(ROOT / relative) == expected, relative
            old_files += 1
    checked_quantiles, raw_sources, proxy_dates = 0, 0, 0
    for record in registry["profiles"]:
        folder = out / record["reservoir_id"]
        frame = pd.read_csv(folder / "observations_si.csv", index_col="date", parse_dates=True, float_precision="round_trip")
        assert frame.index.equals(pd.date_range(frame.index[0], frame.index[-1], name="date"))
        assert frame.index.is_unique and frame.inflow_m3s.isna().all()
        assert record["GRAND_ID"] is None and not record["GRAND_mapping_verified"]
        assert not record["calibration_ready"] and not record["simulation_ready"] and not record["parameters_ready"]
        assert record["fitting_scope"] == "all_available_records_descriptive_no_assessment"
        target = pd.read_csv(folder / "monthly_targets_si.csv", index_col="month", float_precision="round_trip")
        provenance = w.native.read(folder / "observation_provenance.json")
        for variable, column, columns in (("storage", "storage_m3", target.columns[:3]), ("release", "release_m3s", target.columns[3:])):
            if variable not in provenance:
                assert frame[column].isna().all() and target[list(columns)].isna().all().all()
                continue
            source = provenance[variable]
            assert w.native.sha(folder / "raw" / source["file"]) == source["source_sha256"]
            raw_sources += 1
            quality = pd.read_csv(folder / (variable + "_daily_quality.csv"), index_col="date", parse_dates=True)
            assert len(quality) == source["calendar_days"]
            assert int(quality.source_rows.sum()) == source["source_rows"]
            assert int((quality.valid_samples > 0).sum()) == source["valid_days"]
            assert int(frame[column].eq(0).sum()) == source["zero_days"]
            assert int(frame[column].notna().sum()) == source["valid_days"]
            if source["duplicate_policy"] != "reject":
                assert source["observation_basis"].endswith("proxy")
                proxy_dates += len(quality)
            for month in range(1, 13):
                sample = frame.loc[frame.index.month == month, column].dropna().to_numpy()
                if len(sample):
                    expected = np.percentile(sample, [10, 45, 85], method=record["quantile_method"])
                    np.testing.assert_array_equal(target.loc[month, list(columns)].to_numpy(), expected)
                    checked_quantiles += 3
        if record["reservoir_id"] == "etienne_Lake_Chelan":
            raw = next((folder / "raw/LakeChelan_outflow.txt").open()).split()
            assert frame.release_m3s.iloc[0] == float(raw[2]) == 62.579640
        if record["reservoir_id"] == "etienne_Hungry_Horse":
            assert frame.release_m3s.iloc[0] == 25.99
        if record["reservoir_id"] == "etienne_Hugh_Keenleyside":
            assert provenance["storage"]["declared_missing_text_rows"] == 87
    before = ROOT / "outputs/managed_full_data_ga_dds_gardiner_v1_20261020_final/completion_manifest.json"
    unchanged = 0
    for relative, expected in w.native.read(before)["files_sha256"].items():
        assert w.native.sha(ROOT / relative) == expected, relative
        unchanged += 1
    result = {"status": "passed", "existing_reservoirs": proof["existing_reservoirs"], "new_reservoirs": len(registry["profiles"]),
        "combined_reservoirs": len(ids), "raw_observation_sources_verified": raw_sources, "monthly_quantiles_recomputed": checked_quantiles,
        "existing_collection_files_unchanged": old_files, "prior_full_data_completion_files_unchanged": unchanged,
        "untimed_proxy_days_counted": proxy_dates, "new_verified_GRAND_IDs": 0,
        "simulation_runs": 0, "optimizer_evaluations": 0, "descriptive_targets_not_promoted_to_calibration": True}
    w.native.write(Path(output), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--collection", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); print(audit(args.collection, args.output))
