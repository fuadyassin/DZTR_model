"""Independent arithmetic verification; no simulations or optimizer calls."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from workflows import level_storage_pilot_v1 as pilot
from workflows import reservoir_v2 as w
from strategies.representative_years import YearSelection


def verify(output):
    out = Path(output).resolve(strict=True)
    pilot.verify(out)
    design = w.read(out / "plan.json"); cfg = design["config"]
    paths = {k: ROOT / v for k, v in cfg["inputs"].items()}
    frame = pilot.read_frame(paths["observations"])
    levels = pilot.read_frame(paths["levels"]).level_m
    selection = YearSelection(**w.read(paths["selection"]))
    fit = selection.mask(frame.index, "calibration")
    roles = pd.Series(selection.roles, index=frame.index)
    # Decode the original archived ReGeom table independently of production reader.
    import io
    lines = paths["geometry"].read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Depth(m)"))
    geometry = np.loadtxt(io.StringIO("\n".join(lines[start + 1:])), delimiter=",")
    d, volume = geometry[:, 0], geometry[:, 2] * 1e6
    raw = []
    for page in (paths["acquisition_provenance"].parent / "raw").glob("*/page_*.json"):
        raw.extend(item["properties"] for item in json.loads(page.read_text())["features"])
    assert all(row["STATION_NUMBER"] == cfg["station_number"] for row in raw)
    provider = pd.Series([row["LEVEL"] for row in raw],
                         index=pd.DatetimeIndex([row["DATE"] for row in raw])).sort_index()
    assert provider.index.is_unique
    np.testing.assert_array_equal(provider.index.values, frame.index.values)
    np.testing.assert_array_equal(provider.to_numpy(), levels.to_numpy())
    inverse = np.interp(frame.storage_m3.to_numpy()[fit], volume, d)
    expected_zero = float(np.median(levels.to_numpy()[fit] - inverse))
    refs = w.read(out / "references.json")
    assert len(refs) == 4
    fitted = next(r for r in refs if r["method"] == "calibration_storage_alignment")
    assert fitted["zero_level_m"] == expected_zero
    assert fitted["fitting_pairs"] == int(fit.sum())
    assert fitted["fitting_signature"] == selection.signature
    assert not fitted["assessment_used_for_fitting"]
    target_cells, supported_rows, balances = 0, 0, []
    metrics = pd.read_csv(out / "conversion_metrics.csv")
    model = w.read(paths["model"])["model"]
    for method in ("provided_storage", *pilot.METHODS):
        if method == "provided_storage":
            source_storage = frame.storage_m3.to_numpy()
        else:
            ref = next(r for r in refs if r["method"] == method)
            depth = levels.to_numpy() + ref["level_to_reference_offset_m"] - ref["zero_level_m"]
            valid = np.isfinite(depth) & (depth >= d[0]) & (depth <= d[-1])
            expected = np.full(len(depth), np.nan)
            # Verify by explicit neighbouring-point weights, not np.interp.
            j = np.clip(np.searchsorted(d, depth[valid], side="right") - 1, 0, len(d) - 2)
            weights = (depth[valid] - d[j]) / (d[j + 1] - d[j])
            expected[valid] = volume[j] * (1 - weights) + volume[j + 1] * weights
            converted = pilot.read_frame(out / method / "level_conversion_si.csv")
            np.testing.assert_allclose(converted.depth_m, depth, rtol=0, atol=1e-12)
            np.testing.assert_allclose(converted.storage_m3, expected, rtol=0, atol=5e-6, equal_nan=True)
            source_storage = converted.storage_m3.to_numpy()
            supported_rows += int(valid.sum())
            for phase in ("calibration", "assessment"):
                support = roles.eq(phase).to_numpy() & np.isfinite(frame.storage_m3)
                supported = support & valid
                x, y = frame.storage_m3.to_numpy()[supported], expected[supported]
                errors = y - x
                row = metrics[(metrics.method == method) & (metrics.phase == phase)].iloc[0]
                assert row.converted_pairs == len(x)
                assert row.missing_estimates_on_fixed_support == int((support & ~valid).sum())
                np.testing.assert_allclose(row.supported_nse, 1 - (errors**2).sum() / ((x - x.mean())**2).sum(), atol=2e-12)
                np.testing.assert_allclose(row.bias_m3, errors.mean(), atol=5e-6)
        targets = pd.read_csv(out / method / "monthly_targets_si.csv", index_col="month", float_precision="round_trip")
        for month in range(1, 13):
            mask = fit & (frame.index.month == month)
            for values, columns in ((source_storage, targets.columns[:3]),
                                    (frame.release_m3s.to_numpy(), targets.columns[3:])):
                sample = np.sort(np.asarray(values)[mask & np.isfinite(values)])
                # Hazen inverse CDF: interpolation positions n*p - 0.5, endpoints bounded.
                positions = np.clip(len(sample) * np.array([.1, .45, .85]) - .5, 0, len(sample) - 1)
                lo, hi = np.floor(positions).astype(int), np.ceil(positions).astype(int)
                estimated = sample[lo] + (positions - lo) * (sample[hi] - sample[lo])
                np.testing.assert_allclose(targets.loc[month, columns], estimated, rtol=2e-15, atol=5e-6)
                target_cells += 3
        receipt = w.read(out / method / "simulation_receipt.json")
        if receipt["status"] == "completed":
            sim = pilot.read_frame(out / method / "simulation.csv")
            step = sim.step_seconds.to_numpy()[1:]
            inflow = sim.inflow_m3s.to_numpy() * model["inflow_correction"]
            release = sim.release_m3s.to_numpy()
            residual = np.diff(sim.storage_m3.to_numpy()) - step * ((inflow[1:] + inflow[:-1]) - (release[1:] + release[:-1])) / 2
            maximum = float(np.max(np.abs(residual)))
            assert maximum <= 1e-4
            assert sim.storage_m3.between(0, model["capacity_m3"]).all()
            assert np.isfinite(sim[["storage_m3", "release_m3s", "inflow_m3s", "step_seconds"]].to_numpy()).all()
            balances.append({"method": method, "maximum_balance_residual_m3": maximum})
    # Earlier source exports, profiles and completed release remain verifiable.
    batch = ROOT / "outputs/managed_paper_batch_v1_20261017_final"
    completion = w.read(batch / "completion_manifest.json")
    for name, expected in completion["artifact_sha256"].items():
        assert w.sha(batch / name) == expected, name
    for name, expected in completion["implementation_source_sha256"].items():
        assert w.sha(ROOT / name) == expected, name
    old_release = w.read(batch / "release/manifest.json")
    for name, expected in old_release["artifact_sha256"].items():
        assert w.sha(batch / "release" / name) == expected, name
    old_plan = w.read(batch / "plan.json")
    for name, expected in old_plan["input_sha256"].items():
        assert w.sha(name) == expected, name
    return {"status": "passed", "provider_days_verified": len(provider),
        "independently_verified_conversion_rows": supported_rows, "reference_fitting_pairs": int(fit.sum()),
        "independently_verified_target_cells": target_cells, "independent_balance_checks": balances,
        "original_inputs_rechecked": len(old_plan["input_sha256"]),
        "existing_completed_release_unchanged": True, "new_simulation_calls": 0, "new_optimizer_evaluations": 0,
        "source_certified": False, "reference_is_surveyed_bed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify(args.output)
    w.write(Path(args.output) / "scientific_verification.json", result)
    print(json.dumps(result, indent=2))
