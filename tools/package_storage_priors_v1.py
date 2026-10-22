"""Package GRAND mean discharge and Etienne naturalized donor shapes on D:."""
import argparse
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from workflows import reservoir_v2 as native
from workflows.release_prior_v1 import read_source
from workflows.atomic_io_v1 import write

MAPPING = {
    "ex30_Big_horn": ("bighorn", "05DC010"), "ex14_Gardiner": ("gardiner", "05HG001"),
    "ex28_Dickson": ("dickson", "05CB007"), "ex29_EB_Campbell": ("ebcampbell", "05KD003"),
    "ex24_Oldman": ("oldman", "05AA024"), "ex20_St_marry": ("stmary", "05AE006"), "ex23_Waterton": ("waterton", "05AD028")}


def dbf_frame(path):
    with Path(path).open("rb") as stream:
        header = stream.read(32); count = struct.unpack("<I", header[4:8])[0]; length, record_length = struct.unpack("<HH", header[8:12])
        fields = []
        while True:
            lead = stream.read(1)
            if lead == b"\r": break
            if not lead: raise ValueError("Truncated DBF descriptors")
            descriptor = lead + stream.read(31)
            fields.append((descriptor[:11].split(b"\0")[0].decode("ascii"), chr(descriptor[11]), descriptor[16]))
        stream.seek(length); rows = []
        for _ in range(count):
            raw = stream.read(record_length)
            if len(raw) != record_length: raise ValueError("Truncated DBF record")
            if raw[0:1] == b"*": continue
            row = {}; offset = 1
            for name, kind, size in fields:
                token = raw[offset:offset + size].decode("latin1").strip(); offset += size
                row[name] = float(token) if kind in {"N", "F"} and token and set(token) != {"*"} else token or None
            rows.append(row)
    return pd.DataFrame(rows)


def package(archive, dbf, catalog, output):
    archive, dbf, catalog = [Path(p).resolve(strict=True) for p in (archive, dbf, catalog)]
    out = Path(output).resolve()
    if not out.is_relative_to(ROOT / "data") or out == ROOT / "data" or out.exists(): raise ValueError("Use a fresh project data package")
    out.mkdir(parents=True); inputs = {str(p): native.sha(p) for p in (dbf, catalog, Path(__file__))}
    grand = dbf_frame(dbf)[["GRAND_ID", "DAM_NAME", "RES_NAME", "COUNTRY", "LONG_DD", "LAT_DD", "DIS_AVG_LS"]]
    grand["GRAND_ID"] = grand.GRAND_ID.astype(int)
    grand["mean_discharge_m3s"] = pd.to_numeric(grand.DIS_AVG_LS).where(grand.DIS_AVG_LS > 0) / 1000.
    if grand.GRAND_ID.duplicated().any(): raise ValueError("Duplicate GRAND IDs")
    reference = ROOT / "data/catalogs/grand_v1_1_reference.csv"; inputs[str(reference)] = native.sha(reference)
    previous = pd.read_csv(reference); verified = native.read(catalog)["reservoirs"]
    for row in verified:
        gid = row.get("GRAND_ID")
        if gid is None: continue
        actual = grand[grand.GRAND_ID == gid]; known = previous[previous.GRAND_ID == gid]
        if len(actual) != 1 or len(known) != 1: raise ValueError("Catalog GRAND ID absent in source/reference")
        for coordinate in ("LONG_DD", "LAT_DD"):
            if abs(float(actual.iloc[0][coordinate]) - float(known.iloc[0][coordinate])) > 1e-4: raise ValueError("GRAND coordinate mismatch")
    grand.to_csv(out / "grand_mean_discharge.csv", index=False, float_format="%.17g")
    coefficient = archive / "coeff_reserv_crude.txt"; inputs[str(coefficient)] = native.sha(coefficient)
    lines = [line for line in coefficient.read_text().splitlines() if line.strip()]
    n = int(lines[0].split("#")[0]); blocks = {}
    if len(lines) != 1 + n * 15: raise ValueError("Unexpected archived CR1 coefficient layout")
    for i in range(n):
        block = lines[1 + i * 15:1 + (i + 1) * 15]
        label = re.sub("[^a-z0-9]", "", block[0].split("#", 1)[1].lower().replace("reservoir id", ""))
        blocks[label] = [float(block[j].split(",")[0]) for j in (12, 13, 14)]
    donors, mappings = [], []
    by_id = {row["reservoir_id"]: row for row in verified}
    for identifier, (label, station) in MAPPING.items():
        raw_path = archive / "crude_params" / (station + ".txt")
        stats_path = archive / "crude_params" / (station + "_stats.txt")
        for path in (raw_path, stats_path): inputs[str(path)] = native.sha(path)
        stats = stats_path.read_text()
        annual = [float(re.search(r"Unique " + word + r" Flow:\s*([0-9.eE+-]+)", stats).group(1)) for word in ("Crit", "Norm", "Flood")]
        np.testing.assert_allclose(annual, blocks[label], rtol=0, atol=1e-10)
        values = read_source(raw_path, "etienne_crude")
        values.to_frame().to_csv(out / (identifier + "_naturalized.csv"), index_label="date", float_format="%.17g")
        mean = float(values.mean())
        if mean <= 0 or values.isna().any(): raise ValueError("Invalid donor reference flow")
        for method in ("hazen", "linear"):
            ratios = np.percentile(values, [5, 30, 97, 99], method=method) / mean
            donors.append({"reservoir_id": identifier, "GRAND_ID": by_id[identifier].get("GRAND_ID"), "station_id": station,
                "quantile_method": method, **dict(zip(("p05_ratio", "p30_ratio", "p97_ratio", "p99_ratio"), ratios)),
                "donor_mean_m3s": mean, "reference_samples": len(values), "reference_years": ",".join(map(str, sorted(set(values.index.year)))),
                "reference_start": str(values.index[0].date()), "reference_end": str(values.index[-1].date()), "source_sha256": native.sha(raw_path)})
        mappings.append({"reservoir_id": identifier, "station_id": station, "source": str(raw_path),
            "mapping_evidence": "Station statistics annual P5/P30/P97 match the named archived CR1 coefficient block", "mapping_check_passed": True})
    pd.DataFrame(donors).to_csv(out / "naturalized_donor_shapes.csv", index=False, float_format="%.17g")
    write(out / "source_mappings.json", mappings)
    write(out / "manifest.json", {"kind": "storage_release_prior_package_v1", "input_sha256": inputs,
        "GRAND_source_field": "DIS_AVG_LS", "source_units": "L/s", "conversion_to_m3s": .001,
        "GRAND_ID_mappings_changed": False, "donor_reservoirs": len(MAPPING), "observed_flow_column_used": False,
        "naturalized_independence_verified": False, "source_certified": False,
        "artifact_sha256": {p.name: native.sha(p) for p in out.iterdir() if p.is_file()}})
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--archive", required=True); p.add_argument("--dbf", required=True)
    p.add_argument("--catalog", required=True); p.add_argument("--output", required=True); a = p.parse_args()
    print(package(a.archive, a.dbf, a.catalog, a.output))
