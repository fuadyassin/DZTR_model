"""Compatibility adapter for percentile tables expressed in legacy units."""

import pandas as pd

from dztr.data import normalize_observations
from dztr.parameterization import DEFAULT_PERCENTILES, MONTH_NAMES, estimate_percentiles
from dztr.schemas import ReservoirObservations


def calculate_percentiles(resdata, percentiles=DEFAULT_PERCENTILES):
    """Estimate quantiles using the new core and return the historical wide table.

    Storage is returned in million cubic metres for compatibility with existing
    Excel and MESH exports. The core calculations use cubic metres. Input dates
    and missing observations are validated by ``normalize_observations``.
    """
    observations = (
        resdata if isinstance(resdata, ReservoirObservations)
        else normalize_observations(resdata)
    )
    estimates = estimate_percentiles(observations, percentiles=percentiles).frame
    quantiles = sorted(estimates["percentile"].unique())
    values = estimates.set_index(["variable", "month", "percentile"])["value"]
    result = {}
    for variable, label, factor in (
        ("storage_m3", "Storage(mil_m3)", 1e-6),
        ("inflow_m3s", "Inflow(cms)", 1.0),
        ("release_m3s", "Outflow(cms)", 1.0),
    ):
        for month, suffix in enumerate(("all", *MONTH_NAMES)):
            result[f"{label}_{suffix}"] = [
                values.get((variable, month, percentile), float("nan")) * factor
                for percentile in quantiles
            ]
    return pd.DataFrame(result, index=[f"P{p:g}" for p in quantiles])
