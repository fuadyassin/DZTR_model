"""Import the original MATLAB DZTR tables into the package's SI contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from dztr.data import normalize_observations
from dztr.calibration.candidates import parameters_from_vector
from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirObservations, ReservoirParameters


def _coefficient_table(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse four constants followed by exactly nineteen twelve-month rows."""
    tokens = Path(path).read_text(encoding="utf-8-sig").split()
    if len(tokens) != 232:
        raise ValueError("MATLAB coefficients require exactly 232 values: four constants and a 19 by 12 table.")
    try:
        values = np.asarray([float(token) for token in tokens], dtype=float)
    except ValueError as exc:
        raise ValueError("MATLAB coefficients must contain only numeric values.") from exc
    if not np.isfinite(values).all():
        raise ValueError("MATLAB coefficients must contain finite values.")
    table = values[4:].reshape(19, 12)
    if not np.array_equal(table[0], np.arange(1, 13)):
        raise ValueError("MATLAB coefficient month header must be exactly 1 through 12.")
    return values[:4], table


def _monthly_targets(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        values.T,
        columns=MONTHLY_TARGET_COLUMNS,
        index=pd.Index(range(1, 13), name="month"),
    )


def read_matlab_coefficients(
    path: str | Path,
    *,
    dead_storage_fraction: float = 0.1,
    inflow_correction: float = 1.0,
    flood_mode: str = "storage",
) -> ReservoirParameters:
    """Read the coefficient format written by ``trinityRES_fit.m``.

    Constants are capacity, initial release, initial storage and channel
    capacity. The next row contains months; the following six contain storage
    targets in m3 and release targets in m3/s. Remaining rows contain bounds.
    Inflow correction and flood-zone behavior are explicit caller settings;
    neither setting is encoded in the file.
    """
    constants, table = _coefficient_table(path)
    capacity, initial_release, initial_storage, channel = constants
    return ReservoirParameters(
        monthly_targets=_monthly_targets(table[1:7]),
        capacity_m3=float(capacity),
        initial_storage_m3=float(initial_storage),
        initial_release_m3s=float(initial_release),
        channel_capacity_m3s=float(channel),
        dead_storage_fraction=dead_storage_fraction,
        inflow_correction=inflow_correction,
        flood_mode=flood_mode,
    )


def read_matlab_bounds(path: str | Path) -> pd.DataFrame:
    """Read six lower and six upper monthly calibration bounds in SI units.

    Columns use ``lower_`` or ``upper_`` followed by the corresponding monthly
    target column name. MATLAB stores release bounds before storage bounds.
    """
    _, table = _coefficient_table(path)
    lower = np.vstack((table[13:16], table[7:10]))
    upper = np.vstack((table[16:19], table[10:13]))
    if (lower < 0).any() or (upper < lower).any():
        raise ValueError("MATLAB bounds must be nonnegative with lower <= upper.")
    frame = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
    for prefix, values in (("lower", lower), ("upper", upper)):
        for column, row in zip(MONTHLY_TARGET_COLUMNS, values):
            frame[f"{prefix}_{column}"] = row
    return frame


def _matlab_vector(value: object, field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MATLAB observation field {field!r} must be numeric.") from exc
    if array.ndim > 2 or (array.ndim == 2 and min(array.shape) > 1):
        raise ValueError(f"MATLAB observation field {field!r} must be a vector.")
    return array.reshape(-1)


def read_matlab_observations(path: str | Path) -> ReservoirObservations:
    """Read headerless ``res_info.xlsx`` or the ``res_info_out.mat`` struct.

    Workbook storage is million m3; MAT storage is already m3. MATLAB serial
    dates are converted to their actual calendar dates. Extra workbook columns
    contain configuration and are excluded from the observation table.
    SciPy is needed only when importing MAT files.
    """
    file_path = Path(path)
    if file_path.suffix.casefold() in (".xlsx", ".xlsm"):
        frame = pd.read_excel(file_path, header=None)
        if frame.shape[1] < 6:
            raise ValueError("MATLAB observation workbooks require at least six columns.")
        frame = frame.iloc[:, :6].copy()
        frame.columns = ("year", "month", "day", "Storage(mil_m3)", "Inflow(cms)", "Outflow(cms)")
    elif file_path.suffix.casefold() == ".mat":
        try:
            from scipy.io import loadmat
        except ImportError as exc:
            raise ImportError("MAT-file importing requires SciPy; install the MATLAB extra with pip install '.[matlab]'.") from exc
        content = loadmat(file_path, simplify_cells=True)
        info = content.get("res_info_out")
        fields = ("timed", "stoobs", "inflow", "outflow")
        if not isinstance(info, dict) or set(fields) - set(info):
            raise ValueError("MAT files require a res_info_out struct with timed, stoobs, inflow and outflow fields.")
        arrays = {field: _matlab_vector(info[field], field) for field in fields}
        if len({array.size for array in arrays.values()}) != 1:
            raise ValueError("MATLAB observation fields must have equal lengths.")
        serial_dates = arrays["timed"]
        if not np.isfinite(serial_dates).all() or not np.equal(serial_dates, np.floor(serial_dates)).all():
            raise ValueError("MATLAB observation dates must be finite daily serial dates.")
        try:
            dates = pd.to_datetime(serial_dates - 719529, unit="D", origin="1970-01-01", errors="raise")
        except (ValueError, OverflowError) as exc:
            raise ValueError("MATLAB serial dates are outside the supported calendar range.") from exc
        frame = pd.DataFrame({
            "date": dates,
            "storage_m3": arrays["stoobs"],
            "inflow_m3s": arrays["inflow"],
            "release_m3s": arrays["outflow"],
        })
    else:
        raise ValueError("MATLAB observations require a .xlsx, .xlsm or .mat file.")
    return normalize_observations(frame, source=str(file_path))


def parameters_from_matlab_vector(
    vector: Iterable[float], template: ReservoirParameters,
) -> ReservoirParameters:
    """Convert one 72-value AMALGAM candidate into monthly SI targets.

    MATLAB's vector groups twelve months for each target: critical, normal and
    flood storage fractions, then critical, normal and flood release in m3/s.
    The template supplies physical configuration and initial conditions.
    """
    if not isinstance(template, ReservoirParameters):
        raise TypeError("template must be ReservoirParameters.")
    try:
        raw = np.asarray(list(vector))
        if np.iscomplexobj(raw):
            raise ValueError("MATLAB parameter vectors must contain real values.")
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("MATLAB parameter vectors must contain 72 real numeric values.") from exc
    if values.shape != (72,) or not np.isfinite(values).all():
        raise ValueError("MATLAB parameter vectors must contain exactly 72 finite values.")
    return parameters_from_vector(values, template)
