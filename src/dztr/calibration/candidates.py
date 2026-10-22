"""The optimizer-independent 72-value monthly DZTR candidate representation."""

from dataclasses import replace

import numpy as np
import pandas as pd

from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters


def _validated_vector(vector) -> np.ndarray:
    """Return a detached finite real vector, rejecting shape reinterpretation."""
    try:
        raw = np.asarray(vector)
        if np.iscomplexobj(raw):
            raise ValueError("Candidate values must be real numbers.")
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Candidate vector must contain 72 finite real values.") from exc
    if values.shape != (72,) or not np.isfinite(values).all():
        raise ValueError("Candidate vector must have shape (72,) and contain only finite values.")
    return values.copy()


def parameters_to_vector(parameters: ReservoirParameters) -> np.ndarray:
    """Pack six 12-month blocks: Sc/C, Sn/C, Sf/C, Qc, Qn, Qf.

    Storage targets are fractions of physical capacity; release targets remain
    m3/s. The order matches MATLAB's 72-variable calibration vectors. Neither
    the parameter table nor the caller's physical configuration is modified.
    """
    if not isinstance(parameters, ReservoirParameters):
        raise TypeError("parameters must be ReservoirParameters.")
    # Revalidate because a frozen dataclass still contains a mutable DataFrame.
    checked = replace(parameters)
    values = checked.monthly_targets.loc[list(range(1, 13)), list(MONTHLY_TARGET_COLUMNS)].to_numpy(copy=True)
    values[:, :3] /= checked.capacity_m3
    return values.T.reshape(72).copy()


def parameters_from_vector(vector, template: ReservoirParameters) -> ReservoirParameters:
    """Decode a native candidate and validate its physical target ordering.

    All non-target settings are inherited from the template. Invalid candidates
    are rejected; values are never repaired, sorted, or clipped.
    """
    if not isinstance(template, ReservoirParameters):
        raise TypeError("template must be ReservoirParameters.")
    values = _validated_vector(vector).reshape(6, 12).T.copy()
    values[:, :3] *= template.capacity_m3
    monthly = pd.DataFrame(values, index=pd.Index(range(1, 13), name="month"), columns=MONTHLY_TARGET_COLUMNS)
    return replace(template, monthly_targets=monthly)
