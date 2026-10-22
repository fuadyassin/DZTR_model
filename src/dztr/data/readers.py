"""File adapters for local reservoir observation records."""

from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd

from dztr.schemas import ReservoirObservations

from .observations import normalize_observations


_SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xlsm"}


def read_reservoir_observations(
    path: str | Path,
    *,
    storage_unit: str | None = None,
    flow_unit: str | None = None,
    source: str | None = None,
    read_options: dict[str, Any] | None = None,
) -> ReservoirObservations:
    """Read one CSV/Excel table and normalize it to daily SI observations.

    ``read_options`` are passed to pandas (for example ``sheet_name`` or
    ``skiprows``). Excel reading defaults to the first sheet.
    """
    file_path = Path(path)
    suffix = file_path.suffix.casefold()
    options = dict(read_options or {})
    if suffix == ".csv":
        data = pd.read_csv(file_path, **options)
    elif suffix in _SUPPORTED_SUFFIXES:
        data = pd.read_excel(file_path, **options)
    else:
        raise ValueError(f"Unsupported observation file type {suffix!r}; use .csv, .xlsx or .xlsm.")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Read a single observation table; select one Excel sheet_name.")
    return normalize_observations(
        data,
        storage_unit=storage_unit,
        flow_unit=flow_unit,
        source=str(file_path) if source is None else source,
    )


def find_reservoir_file(folder: str | Path, grand_id: int | str) -> Path:
    """Find a single table whose filename stem ends with exactly ``_<ID>``."""
    if isinstance(grand_id, bool):
        raise ValueError("grand_id must be a positive integer.")
    if isinstance(grand_id, Integral):
        identifier = int(grand_id)
    elif isinstance(grand_id, Real) and float(grand_id).is_integer():
        identifier = int(grand_id)
    elif isinstance(grand_id, str) and grand_id.strip().isdigit():
        identifier = int(grand_id.strip())
    else:
        raise ValueError("grand_id must be a positive integer.")
    if identifier <= 0:
        raise ValueError("grand_id must be a positive integer.")
    directory = Path(folder)
    matches = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.casefold() in _SUPPORTED_SUFFIXES
        and path.stem.endswith(f"_{identifier}")
        and not path.name.startswith("~$")
    )
    if not matches:
        raise FileNotFoundError(f"No observation file for GRAND_ID {identifier} in {directory}.")
    if len(matches) > 1:
        raise ValueError(f"Multiple observation files for GRAND_ID {identifier}: {', '.join(path.name for path in matches)}")
    return matches[0]
