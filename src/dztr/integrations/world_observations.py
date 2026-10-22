"""Reconcile legacy world-reservoir measurements with their source calendar.

MAT measurements are retained only after comparison with the headerless source
workbook and its documented conversion. Monthly 30.5-day MATLAB clocks remain
provenance; they do not determine the calendar month of an observation.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from dztr.schemas import DataQuality, ReservoirObservations


_FIELDS = ("stoobs", "inflow", "outflow")
_COLUMNS = ("storage_m3", "inflow_m3s", "release_m3s")


class WorldObservationConflict(ValueError):
    """A source conflict, with a JSON-compatible account for cohort reporting."""

    def __init__(self, message: str, provenance: dict | None = None):
        super().__init__(message)
        self.provenance = provenance if provenance is not None else {}
        self.provenance["status"] = "conflict"
        self.provenance["error"] = message


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _guard(hashes: dict, required: tuple[Path, ...], provenance: dict) -> None:
    if not isinstance(hashes, dict) or not hashes:
        raise WorldObservationConflict("Source SHA256 manifest is required.", provenance)
    normalized = {}
    for file, digest in hashes.items():
        path = Path(file)
        if not path.is_absolute() or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise WorldObservationConflict("Source SHA256 manifest contains an invalid path or digest.", provenance)
        normalized[str(path.resolve())] = digest.lower()
    if any(str(path.resolve()) not in normalized for path in required):
        raise WorldObservationConflict("Workbook, MAT file, and writer must all be pinned by source SHA256.", provenance)
    for file, digest in normalized.items():
        path = Path(file)
        if not path.is_file() or _sha256(path) != digest:
            raise WorldObservationConflict(f"Source SHA256 changed or file missing: {path}", provenance)
    provenance["source_sha256"] = normalized


def _read_mat(path: Path) -> dict[str, np.ndarray]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("World MAT importing requires pip install '.[matlab]'.") from exc
    info = loadmat(path, simplify_cells=True).get("res_info_out")
    if not isinstance(info, dict) or {"timed", *_FIELDS} - set(info):
        raise WorldObservationConflict("MAT file requires res_info_out timed/stoobs/inflow/outflow vectors.")
    arrays = {}
    for field in ("timed", *_FIELDS):
        raw = np.asarray(info[field])
        if np.iscomplexobj(raw) or raw.ndim > 2 or (raw.ndim == 2 and min(raw.shape) > 1):
            raise WorldObservationConflict(f"MAT field {field} must be a real numeric vector.")
        try:
            arrays[field] = raw.astype(float).reshape(-1)
        except (ValueError, TypeError) as exc:
            raise WorldObservationConflict(f"MAT field {field} must be numeric.") from exc
    if len({len(value) for value in arrays.values()}) != 1 or not len(arrays["timed"]):
        raise WorldObservationConflict("MAT observation vectors must be nonempty and have equal lengths.")
    return arrays


def _date(value: object) -> pd.Timestamp:
    text = str(value)
    return pd.Timestamp(pd.to_datetime(text, format="%Y%m%d" if re.fullmatch(r"\d{8}", text) else None))


def _legacy_origin(writer: str, mapping: dict, fallback: object) -> pd.Timestamp:
    value = mapping.get("workbook_legacy_start")
    if value is None:
        match = re.search(r"\bstrdate2\s*=\s*['\"]([^'\"]+)['\"]", writer, re.I)
        value = match.group(1) if match else fallback
    if value is None:
        raise WorldObservationConflict("Writer-generated row calendar requires an explicit workbook start date.")
    return _date(value)


def _calendar(book: pd.DataFrame, mapping: dict, origin: pd.Timestamp, monthly: bool) -> tuple[pd.DatetimeIndex, dict]:
    order = mapping.get("date_column_order")
    if isinstance(order, str):
        order = [part.strip() for part in order.split(",")]
    if order is not None and (len(order) != 3 or set(order) != {"year", "month", "day"}):
        raise WorldObservationConflict("date_column_order must name year, month, and day exactly once.")
    candidates = []
    for candidate in ([order] if order is not None else [("year", "month", "day"), ("month", "day", "year")]):
        parts = {part: book.iloc[:, index].to_numpy(dtype=float).copy() for index, part in enumerate(candidate)}
        offset = mapping.get("year_offset")
        finite_years = parts["year"][np.isfinite(parts["year"])]
        if offset is None and len(finite_years) and np.all((finite_years >= 0) & (finite_years < 100)):
            offset = (origin.year // 100) * 100
        offset = 0 if offset is None else offset
        parts["year"] += offset
        if monthly:
            parts["day"] = np.ones(len(book))
        # pandas' numeric assembler can normalize overflowing float components;
        # a month such as 2003 must not make an MDY alternative look valid.
        if any(not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all() for values in parts.values()):
            continue
        if not ((parts["month"] >= 1) & (parts["month"] <= 12)).all() or not ((parts["day"] >= 1) & (parts["day"] <= 31)).all():
            continue
        try:
            dates = pd.DatetimeIndex(pd.to_datetime(pd.DataFrame(parts), errors="raise"))
        except (ValueError, TypeError, OverflowError):
            continue
        if dates.hasnans:
            continue
        candidates.append((dates, {"date_column_order": list(candidate), "year_offset": int(offset)}))
    if len(candidates) != 1:
        raise WorldObservationConflict("Workbook calendar layout is invalid or ambiguous; supply evidenced column order.")
    return candidates[0]


def _selection(spec: dict, dates: pd.DatetimeIndex) -> np.ndarray:
    selection = spec.get("selection") or {}
    first, last = selection.get("t1"), selection.get("t2")
    if first is not None or last is not None:
        first = 1 if first is None else first
        last = len(dates) if last is None else last
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in (first, last)):
            raise WorldObservationConflict("MATLAB t1/t2 must be resolved 1-based integer row indices.")
        if not 1 <= first <= last <= len(dates):
            raise WorldObservationConflict("MATLAB t1/t2 lie outside the observation vectors.")
        positions = np.arange(first - 1, last)
    else:
        keep = np.ones(len(dates), dtype=bool)
        for key, comparison in (("start", np.greater_equal), ("end", np.less_equal)):
            if selection.get(key) is not None:
                keep &= comparison(dates, _date(selection[key]))
        positions = np.flatnonzero(keep)
    if not len(positions):
        raise WorldObservationConflict("Source simulation selection contains no observations.")
    chosen = dates[positions]
    if selection.get("start") is not None and chosen[0] < _date(selection["start"]):
        raise WorldObservationConflict("Resolved t1 begins before the source simulation period.")
    if selection.get("end") is not None and chosen[-1] > _date(selection["end"]):
        raise WorldObservationConflict("Resolved t2 ends after the source simulation period.")
    return positions


def _stats(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {
        "missing_count": int(np.isnan(values).sum()), "infinite_count": int(np.isinf(values).sum()),
        "negative_count": int((values < 0).sum()), "zero_count": int((values == 0).sum()),
        "minimum": float(finite.min()) if len(finite) else None,
        "maximum": float(finite.max()) if len(finite) else None,
        "first": float(values[0]) if len(values) and np.isfinite(values[0]) else None,
    }


def read_world_observations(spec: dict) -> tuple[ReservoirObservations, dict]:
    """Read a hash-pinned discovered source without clipping or filling data.

    ``observation_mapping.storage_factor`` must come from the source writer.
    The spec may resolve MATLAB 1-based ``selection.t1/t2``; otherwise its
    calendar start/end select the observations. Flow preprocessing belongs to
    simulation and is deliberately reported without altering raw observations.
    """
    if not isinstance(spec, dict):
        raise TypeError("spec must be a discovered world-reservoir source dictionary.")
    folder = Path(spec["folder"])
    workbook = Path(spec.get("workbook_file", folder / "res_info.xlsx"))
    mat_file = Path(spec.get("observation_file", folder / "res_info_out.mat"))
    writer_paths = [Path(spec["writer_file"])] if spec.get("writer_file") else [Path(path) for path in spec.get("writer_candidates", [])]
    if not writer_paths:
        raise WorldObservationConflict("No source writer establishes observation units and calendar mapping.", {"reservoir_id": spec.get("reservoir_id")})
    writer_file = writer_paths[0]
    provenance = {
        "schema_version": 1, "reservoir_id": spec.get("reservoir_id"),
        "workbook_file": str(workbook), "observation_file": str(mat_file), "writer_file": spec.get("writer_file"),
        "observation_writer_candidates": [str(path) for path in writer_paths],
        "selection": spec.get("selection") or {}, "issues": [],
        "inflow_preprocessing": spec.get("inflow_preprocessing", {"factor": 1.0, "offset_m3s": 0.0}),
        "inflow_preprocessing_applied": False,
    }
    hashes = spec.get("source_sha256")
    required = (workbook, mat_file, *writer_paths)
    _guard(hashes, required, provenance)
    try:
        if len(writer_paths) > 1:
            signatures = []
            for path in writer_paths:
                text = path.read_text(encoding="utf-8-sig", errors="replace")
                relevant = re.findall(r"(?:res_info_out\.(?:timed|stoobs|inflow|outflow)|strdate2?|enddate2?|timed2?)\s*=\s*[^;\n]+|[^\n]*mean\(resdata\([^\n]+", text)
                signatures.append(tuple(re.sub(r"\s+", "", line) for line in relevant))
            if not signatures[0] or any(signature != signatures[0] for signature in signatures[1:]):
                raise WorldObservationConflict("Multiple writers disagree on observation conversion or calendar; no authoritative mapping.", provenance)
            provenance["observation_writer_consensus"] = True
        return _read_verified(spec, workbook, mat_file, writer_file, provenance)
    except WorldObservationConflict as exc:
        provenance.update(exc.provenance)
        exc.provenance = provenance
        raise
    finally:
        _guard(hashes, required, provenance)


def _read_verified(spec: dict, workbook: Path, mat_file: Path, writer_file: Path, provenance: dict):
    mapping = spec.get("observation_mapping")
    if not isinstance(mapping, dict):
        raise WorldObservationConflict("No unambiguous source observation mapping is available.", provenance)
    factor = mapping.get("storage_factor")
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not np.isfinite(factor) or factor <= 0:
        raise WorldObservationConflict("Storage conversion requires a positive, finite factor evidenced by the writer.", provenance)
    if mapping.get("measurement_columns", [4, 5, 6]) != [4, 5, 6]:
        raise WorldObservationConflict("Unsupported workbook measurement columns; expected documented MATLAB columns 4/5/6.", provenance)
    writer = writer_file.read_text(encoding="utf-8-sig", errors="replace")
    book = pd.read_excel(workbook, sheet_name=0, header=None)
    if book.shape[1] < 6 or not len(book):
        raise WorldObservationConflict("Headerless workbook requires at least six numeric columns and one row.", provenance)
    try:
        numeric = book.iloc[:, :6].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise WorldObservationConflict("Workbook's first six columns must be numeric; no header row is discarded.", provenance) from exc
    arrays = _read_mat(mat_file)
    serial = arrays["timed"]
    if not np.isfinite(serial).all() or (np.diff(serial) <= 0).any():
        raise WorldObservationConflict("MAT timestamps must be finite, unique, and increasing.", provenance)
    try:
        legacy = pd.DatetimeIndex(pd.to_datetime(serial - 719529, unit="D", origin="1970-01-01"))
    except (ValueError, OverflowError) as exc:
        raise WorldObservationConflict("MAT timestamps are outside the supported date range.", provenance) from exc
    sampling = spec.get("sampling") or {}
    frequency = sampling.get("frequency")
    if frequency not in ("daily", "monthly"):
        raise WorldObservationConflict("World observations require an explicitly discovered daily or monthly sampling frequency.", provenance)
    monthly = frequency == "monthly"
    origin = _legacy_origin(writer, mapping, sampling.get("legacy_start", str(legacy[0].date())))
    dates, calendar_mapping = _calendar(numeric, mapping, origin, monthly)
    expected = numeric.iloc[:, 3:6].to_numpy(dtype=float)
    expected[:, 0] *= float(factor)
    aggregation = mapping.get("aggregation", "none")
    if isinstance(aggregation, dict):
        aggregation = aggregation.get("method")
    if aggregation in ("unweighted_monthly_mean", "monthly_mean", "mean_by_year_month"):
        if not monthly:
            raise WorldObservationConflict("Monthly source aggregation conflicts with daily sampling.", provenance)
        groups = pd.DataFrame(expected, index=dates)
        # MATLAB mean propagates missing measurements rather than ignoring NaN.
        grouped = groups.groupby(level=0, sort=True).agg(lambda values: np.mean(values.to_numpy()))
        counts = groups.groupby(level=0, sort=True).size()
        provenance["aggregation_group_counts"] = {date.strftime("%Y-%m"): int(count) for date, count in counts.items()}
        dates, expected = pd.DatetimeIndex(grouped.index), grouped.to_numpy()
    elif aggregation not in (None, "none"):
        raise WorldObservationConflict(f"Unsupported source aggregation {aggregation!r}.", provenance)
    n = len(serial)
    calendar_source = mapping.get("calendar_source", "writer_generated_row_order" if not monthly else "workbook_month_labels")
    if monthly:
        timestep = sampling.get("timestep_days")
        if timestep is None or not np.isfinite(timestep) or timestep <= 0 or (n > 1 and not np.allclose(np.diff(serial), timestep, rtol=0, atol=1e-9)):
            raise WorldObservationConflict("MAT monthly clock disagrees with the discovered fixed timestep.", provenance)
        if len(dates) < n:
            raise WorldObservationConflict("Workbook month labels cannot account for all MAT rows; missing or stale source months.", provenance)
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise WorldObservationConflict("Monthly labels must be unique and increasing after documented aggregation.", provenance)
        if dates[0].to_period("M") != legacy[0].to_period("M"):
            raise WorldObservationConflict("First workbook month does not match the source MAT clock origin.", provenance)
        canonical, expected = dates[:n], expected[:n]
    elif calendar_source == "writer_generated_row_order":
        if not legacy.equals(legacy.normalize()) or (n > 1 and not np.equal(np.diff(serial), 1).all()):
            raise WorldObservationConflict("Source-generated daily MAT clock is not a regular midnight daily sequence.", provenance)
        offsets = (legacy - origin).days.to_numpy()
        if (offsets < 0).any() or (offsets >= len(expected)).any():
            raise WorldObservationConflict("Writer-generated calendar cannot map MAT rows to workbook row ordinals.", provenance)
        expected = expected[offsets]
        canonical = legacy
        mismatches = np.flatnonzero(dates[offsets] != canonical)
        provenance["workbook_calendar_label_discrepancies"] = [
            {"workbook_row_1based": int(offsets[index] + 1), "worksheet_date": dates[offsets[index]].isoformat(), "source_date": canonical[index].isoformat()}
            for index in mismatches
        ]
        if len(mismatches):
            provenance["issues"].append(f"{len(mismatches)} worksheet date labels differ from the writer-generated row calendar; original row order retained.")
    elif calendar_source in ("workbook", "workbook_calendar"):
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise WorldObservationConflict("Workbook daily calendar must be unique and increasing.", provenance)
        offsets = dates.get_indexer(legacy)
        if (offsets < 0).any() or not legacy.equals(legacy.normalize()):
            raise WorldObservationConflict("MAT daily dates do not map exactly to workbook dates.", provenance)
        canonical, expected = legacy, expected[offsets]
    else:
        raise WorldObservationConflict(f"Unsupported calendar source {calendar_source!r}.", provenance)
    positions = _selection(spec, canonical)
    chosen = canonical[positions].rename("date")
    actual = np.column_stack([arrays[field][positions] for field in _FIELDS])
    expected = expected[positions]
    comparison = {}
    for column, saved, converted in zip(_COLUMNS, actual.T, expected.T):
        aligned = np.isclose(saved, converted, rtol=1e-12, atol=1e-7, equal_nan=True)
        finite_pairs = np.isfinite(saved) & np.isfinite(converted)
        comparison[column] = {
            "matching": bool(aligned.all()), "mismatch_count": int((~aligned).sum()),
            "maximum_absolute_difference": float(np.max(np.abs(saved[finite_pairs] - converted[finite_pairs]))) if finite_pairs.any() else None,
        }
    provenance.update({
        "workbook_row_count": len(book), "mat_row_count": n, "selected_row_count": len(positions),
        "calendar_mapping": calendar_mapping, "calendar_source": calendar_source, "aggregation": aggregation,
        "units": {"storage_m3": "m3", "inflow_m3s": "m3/s", "release_m3s": "m3/s", "workbook_storage_factor": float(factor)},
        "unit_evidence": {"storage": "Explicit source-writer storage factor; selected converted values match saved MAT.", "flows": "Source writer retains columns 5/6 unchanged; base balance uses rates with timestep in seconds."},
        "monthly_calendar_convention": "Actual source year/month labels at month start; worksheet day placeholders are not dates." if monthly else None,
        "monthly_measurement_convention": "Unweighted arithmetic mean of actual source rows in each year/month, including storage; missing values propagate like MATLAB mean." if aggregation == "unweighted_monthly_mean" else "Native workbook measurements retained; no duration weighting or flow-volume conversion." if monthly else None,
        "workbook_mat_comparison": comparison,
        "sampling": {"frequency": frequency, "timestep_days": sampling.get("timestep_days")},
        "legacy_clock": {"first": legacy[0].isoformat(), "last": legacy[-1].isoformat(), "nonmidnight_count": int((legacy != legacy.normalize()).sum()), "step_days": sorted({float(value) for value in np.diff(serial)})},
        "canonical_clock": {"first": chosen[0].isoformat(), "last": chosen[-1].isoformat()},
        "selected_mat_rows_1based": {"first": int(positions[0] + 1), "last": int(positions[-1] + 1)},
    })
    if any(not item["matching"] for item in comparison.values()):
        raise WorldObservationConflict("Selected MAT measurements conflict with the workbook after the writer's documented conversion; no inferred scale or hypsometry is applied.", provenance)
    frame = pd.DataFrame(actual, columns=_COLUMNS, index=chosen)
    fields = {column: _stats(frame[column].to_numpy()) for column in _COLUMNS}
    issues = provenance["issues"]
    for column, stats in fields.items():
        for key, label in (("negative_count", "negative"), ("missing_count", "missing"), ("infinite_count", "infinite")):
            if stats[key]:
                issues.append(f"{column} contains {stats[key]} {label} observations; values retained for review.")
        if stats["zero_count"] == len(frame):
            issues.append(f"{column} is entirely zero; confirm forcing or measurement availability.")
    if monthly:
        periods = chosen.to_period("M").asi8
        missing_periods = int(np.sum(np.maximum(np.diff(periods) - 1, 0)))
        missing_days = 0
    else:
        missing_days = int((chosen[-1] - chosen[0]).days + 1 - len(chosen))
        missing_periods = missing_days
    if missing_periods:
        issues.append(f"{missing_periods} calendar {'months' if monthly else 'days'} are missing; no observations filled.")
    placeholder = len(frame) >= 2 and np.equal(actual, 0).all()
    unavailable = np.isnan(actual).all()
    status = "placeholder" if placeholder else "unavailable" if unavailable else "available"
    quality = DataQuality(status=status, row_count=len(frame), missing_counts={column: fields[column]["missing_count"] for column in _COLUMNS}, missing_days=missing_days, issues=tuple(issues))
    provenance.update({"status": "review_required" if issues else "verified", "fields": fields, "missing_observation_periods": missing_periods, "values_source": "hash-pinned MAT, verified against converted source workbook"})
    return ReservoirObservations(frame=frame, quality=quality, source=str(mat_file)), provenance
