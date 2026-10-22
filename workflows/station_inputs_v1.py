"""Auditable Canadian daily-mean station inputs for the run manager.

Flow components are added only when all selected stations are present. A level
station is selected separately; gauge heights are never added or auto-aligned.
"""
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import pandas as pd
import requests

from workflows import reservoir_v2 as w
from workflows.geometry_v1 import digest

ENDPOINT = "https://api.weather.gc.ca/collections/hydrometric-daily-mean/items"


def validate(spec):
    allowed = {"start", "end", "inflow_stations", "outflow_stations", "level_stations", "evidence"}
    if set(spec) - allowed or not spec.get("evidence"):
        raise ValueError("Station configuration requires evidence and known fields")
    for key in ("start", "end"):
        value = spec.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Station dates must be explicit YYYY-MM-DD dates")
        pd.Timestamp(value)
    if spec["start"] > spec["end"]:
        raise ValueError("Station start must precede end")
    for key in ("inflow_stations", "outflow_stations", "level_stations"):
        ids = spec.get(key, [])
        if not isinstance(ids, list) or any(not isinstance(i, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids):
            raise ValueError("Station groups must be lists of station IDs")
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate stations would double count flow")
    if len(spec.get("level_stations", [])) > 1:
        raise ValueError("Select one level gauge; multiple level datums need an explicit alignment adapter")
    if set(spec.get("inflow_stations", [])) & set(spec.get("outflow_stations", [])):
        raise ValueError("A station cannot be both an inflow and an outflow component")
    if not any(spec.get(k) for k in ("inflow_stations", "outflow_stations", "level_stations")):
        raise ValueError("Declare at least one station")


def _download(station, start, end, folder):
    """Cache immutable raw response pages, with a completion receipt per query."""
    query = {"STATION_NUMBER": station, "datetime": f"{start}/{end}", "limit": 1000,
             "f": "json", "sortby": "DATE"}
    cache = folder / digest({"endpoint": ENDPOINT, "query": query})
    cache.mkdir(parents=True, exist_ok=True)
    receipt = cache / "receipt.json"
    if receipt.exists():
        proof = w.read(receipt)
        for name, expected in proof["page_sha256"].items():
            if w.sha(cache / name) != expected:
                raise ValueError("Cached hydrometric response changed")
        return [cache / name for name in proof["page_sha256"]], proof
    pages, offset = {}, 0
    # Stable offset pagination follows ECCC's documented DATE ordering.
    while True:
        params = dict(query, offset=offset)
        response = requests.get(ENDPOINT, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload.get("features"), list):
            raise ValueError("Unexpected hydrometric API response")
        name = f"page_{offset:09d}.json"
        w.write(cache / name, payload)
        pages[name] = w.sha(cache / name)
        count = len(payload["features"])
        if count < query["limit"]:
            break
        offset += count
        if offset > 1000000:
            raise ValueError("Unexpected station pagination length")
    proof = {"endpoint": ENDPOINT, "query": query, "page_sha256": pages,
             "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    w.write(receipt, proof)
    return [cache / name for name in pages], proof


def _series(pages, station, property_name, index):
    result = pd.Series(np.nan, index=index, name=station)
    seen = set()
    for page in pages:
        for item in w.read(page)["features"]:
            props = item["properties"]
            if props.get("STATION_NUMBER") != station:
                raise ValueError("Hydrometric response contains a different station")
            # DATE labels are daily dates, not UTC-to-local instant conversions.
            day = pd.Timestamp(str(props["DATE"])[:10])
            if day not in index:
                continue
            value = props.get(property_name)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value):
                raise ValueError("Station values must be finite or null")
            if property_name == "DISCHARGE" and value < 0:
                raise ValueError("Signed gauge discharge needs an explicit interpretation")
            if day in seen and value != result.loc[day]:
                raise ValueError(f"Conflicting station/day {station} {day.date()}")
            result.loc[day] = value
            seen.add(day)
    return result


def acquire(spec, output):
    validate(spec)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    index = pd.date_range(spec["start"], spec["end"], name="date")
    frame = pd.DataFrame(index=index, columns=["inflow_m3s", "release_m3s", "storage_m3"], dtype=float)
    station_table = pd.DataFrame(index=index)
    stations = sorted(set(sum((spec.get(k, []) for k in ("inflow_stations", "outflow_stations", "level_stations")), [])))
    downloaded = {s: _download(s, spec["start"], spec["end"], out / "raw") for s in stations}
    for group, column, prop in (("inflow_stations", "inflow_m3s", "DISCHARGE"),
                                 ("outflow_stations", "release_m3s", "DISCHARGE"),
                                 ("level_stations", "level_m", "LEVEL")):
        selected = spec.get(group, [])
        values = []
        for station in selected:
            series = _series(downloaded[station][0], station, prop, index)
            station_table[f"{group}:{station}"] = series
            values.append(series)
        if not values:
            continue
        if prop == "LEVEL":
            values[0].to_frame("level_m").to_csv(out / "levels_si.csv", index_label="date", float_format="%.17g")
        else:
            frame[column] = pd.concat(values, axis=1).sum(axis=1, min_count=len(values))
    frame.to_csv(out / "observations_si.csv", index_label="date", float_format="%.17g")
    station_table.to_csv(out / "station_components_si.csv", index_label="date", float_format="%.17g")
    proof = {"configuration": spec, "sources": {s: p for s, (_, p) in downloaded.items()},
             "flow_units": "m3/s", "level_units": "m", "time_basis": "daily_mean",
             "flow_sum_requires_all_components": True, "levels_summed": False,
             "storage_derived": False, "ungauged_lateral_inflow_inferred": False,
             "station_topology_verified": False,
             "available_days": {c: int(frame[c].notna().sum()) for c in frame}}
    w.write(out / "provenance.json", proof)
    outputs = {"observations_csv": str(out / "observations_si.csv"), "provenance": str(out / "provenance.json")}
    if spec.get("level_stations"):
        outputs["levels_csv"] = str(out / "levels_si.csv")
    return outputs
