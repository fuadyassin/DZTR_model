"""Read-only audit of the original level-to-storage bridge and local ReGeom.

Writes a new review directory; never fetches observations, rewrites geometry,
assigns new reservoir IDs, or invokes a reservoir simulation/optimization.
"""
from __future__ import annotations

import argparse
from io import StringIO
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from dztr_param.res_data_est_from_hydrometric import get_area_volume_from_csv


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(geometry_root: Path, output: Path):
    geometry_root = geometry_root.resolve(strict=True)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("Use a fresh directory under project outputs")
    consumed = {}

    def pin(path):
        path = Path(path).resolve(strict=True)
        digest = sha(path)
        if str(path) in consumed and consumed[str(path)] != digest:
            raise AssertionError(f"Input changed while reading: {path}")
        consumed[str(path)] = digest
        return path

    pin(Path(__file__))
    for name in ("__init__.py", "columns.py", "file_reader.py", "percentiles.py",
                 "data_updater.py", "csv_creator.py", "csv_creator_hylakes.py",
                 "txt_creator_hylakes.py", "res_data_est_from_hydrometric.py"):
        pin(ROOT / "dztr_param" / name)
    catalog_path = pin(geometry_root / "ReGeomData_WOW_V1.xlsx")
    pin(geometry_root / "Readme.txt")
    catalog = pd.read_excel(catalog_path, sheet_name="ReGeomData")
    if catalog.GRAND_ID.duplicated().any():
        raise ValueError("Ambiguous ReGeom ID")
    catalog = catalog.set_index("GRAND_ID")
    legacy = pd.read_excel(pin(ROOT / "GRanD_Dams_v1_1_DZTR.xlsx"))
    legacy = legacy.set_index("GRAND_ID")
    if not legacy.index.is_unique:
        raise ValueError("Ambiguous legacy GRanD ID")
    cohort_path = pin(ROOT / "outputs/calibrated_paper_cohort_v1_20261017_final/catalog.json")
    cohort = json.loads(cohort_path.read_text())
    verified = {int(e["GRAND_ID"]): e for e in cohort["reservoirs"] if e.get("GRAND_ID") is not None}
    ids = sorted(set(int(i) for i in legacy.index) | set(verified))
    rows, curves, lookup_failures = [], {}, []
    lookup_passed = 0
    for identifier in ids:
        path = geometry_root / "Area_Strg_Dp" / f"{identifier}.csv"
        entry = verified.get(identifier, {})
        base = {"GRAND_ID": identifier, "reservoir_id": entry.get("reservoir_id", ""),
                "current_paper_id_verified": bool(entry), "in_legacy_65_workbook": identifier in legacy.index,
                "geometry_file_available": path.exists(), "catalog_row_available": identifier in catalog.index}
        if not path.exists() or identifier not in catalog.index:
            rows.append(base)
            continue
        lines = pin(path).read_text().splitlines()
        header = {}
        for line in lines:
            if "=" in line:
                name, value = line.split("=", 1)
                header[name.strip()] = value.strip()
        assert int(header["GRanD ID"]) == identifier
        start = next(i for i, line in enumerate(lines) if line.lstrip().startswith("Depth(m)"))
        table = pd.read_csv(StringIO("\n".join(lines[start + 1:])), header=None,
                            names=["depth_m", "area_km2", "storage_mcm"])
        values = table.to_numpy(dtype=float)
        finite = bool(np.isfinite(values).all())
        increasing_depth = bool((np.diff(values[:, 0]) > 0).all())
        nonnegative = bool((values >= 0).all())
        monotone_area = bool((np.diff(values[:, 1]) >= 0).all())
        monotone_storage = bool((np.diff(values[:, 2]) >= 0).all())
        if not (finite and increasing_depth and nonnegative and monotone_area and monotone_storage):
            raise AssertionError(f"Geometry monotonicity/finite check failed: {identifier}")
        top_depth, top_area, top_storage = values[-1]
        geom_height = float(header["Dam Height (m)"])
        cat = catalog.loc[identifier]
        area_integral = float(np.trapz(values[:, 1], values[:, 0]))
        delta_storage = float(values[-1, 2] - values[0, 2])
        # km2 * m == million m3. This checks the distributed paired columns,
        # without inferring the source algorithm or modifying either column.
        closure_pct = 100 * (delta_storage / area_integral - 1) if area_integral else None
        # Exercise the existing read-only lookup against an independent lookup
        # at three actual grid points and one interpolated coordinate per curve.
        for z in (values[0, 0], values[len(values) // 2, 0], top_depth,
                  (values[len(values) // 2, 0] + values[len(values) // 2 + 1, 0]) / 2):
            try:
                actual = get_area_volume_from_csv(identifier, geom_height - z, path.parent)
            except ValueError as exc:
                lookup_failures.append({"GRAND_ID": identifier, "depth_m": float(z),
                                        "curve_header_height_m": geom_height, "error": str(exc)})
                continue
            expected = (np.interp(z, values[:, 0], values[:, 1]), np.interp(z, values[:, 0], values[:, 2]))
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)
            lookup_passed += 1
        row = {**base, "geometry_name": header["Dam/Reservoir Name"],
               "shape": header["Approximated Geometry"], "curve_rows": len(table),
               "finite_nonnegative_monotone": True, "catalog_dam_height_m": float(cat.DAM_H_m),
               "curve_header_height_m": geom_height, "last_depth_m": top_depth,
               "header_minus_last_depth_m": geom_height - top_depth,
               "curve_top_storage_mcm": top_storage, "curve_top_area_km2": top_area,
               "catalog_estimated_storage_mcm": float(cat.V_est_mcm),
               "catalog_grand_storage_mcm": float(cat.V_GRanD_mcm),
               "curve_top_minus_catalog_estimated_storage_pct": 100 * (top_storage / cat.V_est_mcm - 1),
               "curve_top_minus_catalog_grand_storage_pct": 100 * (top_storage / cat.V_GRanD_mcm - 1),
               "area_integral_mcm": area_integral, "tabulated_storage_change_mcm": delta_storage,
               "storage_change_minus_area_integral_pct": closure_pct,
               "curve_sha256": sha(path)}
        rows.append(row)
        curves[identifier] = (geom_height, table)
    # Explicit synthetic level-reference example using the actual Gardiner curve.
    # No gauge data are fetched or interpreted as measurements in this example.
    height, frame = curves[253]
    estimates = []
    for reference_level in (100., 101.):
        depth = height - (reference_level - 100.)
        storage = float(np.interp(depth, frame.depth_m, frame.storage_mcm))
        estimates.append({"observed_level_m": 100., "assumed_full_level_m": reference_level,
                          "converted_depth_m": depth, "storage_mcm": storage})
    example = {"kind": "synthetic gauge levels, actual Gardiner geometry",
               "scenario": "calibration maximum=100 m; adding an assessment observation at 101 m changes a full-record maximum anchor",
               "estimates": estimates,
               "change_to_same_day_storage_mcm": estimates[1]["storage_mcm"] - estimates[0]["storage_mcm"],
               "source_conversion": "z=H-(max_level-level); S=table(z)",
               "recommended_conversion": "z=z_ref+(level-level_ref), with an explicitly frozen independently justified reference pair"}
    for path, digest in consumed.items():
        assert sha(path) == digest, f"Input changed: {path}"
    output.mkdir(parents=True)
    audit = pd.DataFrame(rows)
    audit.to_csv(output / "geometry_audit.csv", index=False, float_format="%.17g")
    available = audit[audit.geometry_file_available & audit.catalog_row_available]
    summary = {"global_catalog_reservoirs": len(catalog),
               "local_curve_files": len(list((geometry_root / "Area_Strg_Dp").glob("*.csv"))),
               "legacy_workbook_reservoirs": len(legacy),
               "verified_paper_grand_ids": len(verified), "audited_ids": len(audit),
               "available_audited_curves": len(available),
               "monotone_nonnegative_curves": int(available.finite_nonnegative_monotone.sum()),
               "legacy_lookup_attempts": 4 * len(available),
               "independent_legacy_lookup_comparisons_passed": lookup_passed,
               "legacy_lookup_rejections": len(lookup_failures),
               "curve_header_exceeds_last_grid_point": int((available.header_minus_last_depth_m > 1e-8).sum()),
               "last_grid_point_exceeds_curve_header": int((available.header_minus_last_depth_m < -1e-8).sum()),
               "top_storage_differs_catalog_estimate_over_1pct": int((available.curve_top_minus_catalog_estimated_storage_pct.abs() > 1).sum()),
               "area_storage_integral_differs_over_5pct": int((available.storage_change_minus_area_integral_pct.abs() > 5).sum()),
               "geometry_is_surveyed_bathymetry": False,
               "datum_validated": False, "new_grand_assignments": 0,
               "observations_downloaded": 0, "reservoir_simulations": 0, "optimization_evaluations": 0,
               "existing_code_data_and_experiments_unchanged": True}
    for name, payload in (("summary.json", summary), ("level_reference_example.json", example),
                          ("legacy_lookup_rejections.json", lookup_failures),
                          ("input_manifest.json", {"consumed_sha256": consumed,
                            "regeom_dataset_doi": "10.5281/zenodo.1322884",
                            "regeom_paper_doi": "10.1029/2017WR022040"})):
        (output / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspect(args.geometry_root, args.output)
