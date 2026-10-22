"""Update legacy parameter workbooks without changing their storage units."""

from numbers import Integral
import warnings

import pandas as pd

from dztr.data import normalize_observations
from dztr.parameterization import DEFAULT_PERCENTILES, MONTH_NAMES
from .columns import create_extended_columns
from .file_reader import check_and_read_files
from .percentiles import calculate_percentiles


def update_extended_data(extended_data, grand_ids_to_update, folder="ResDataInflowStorageRelease"):
    """Return a copy with computed parameters; preserve unrelated fields and names.

    Placeholder and unavailable datasets are skipped with a warning. Invalid
    observations raise an error instead of producing parameters. No daily dates
    are invented, so gaps in otherwise valid observations remain valid inputs.
    """
    if "GRAND_ID" not in extended_data and not extended_data.empty:
        raise ValueError("The catalog must contain a GRAND_ID column.")
    if "GRAND_ID" in extended_data and extended_data["GRAND_ID"].duplicated().any():
        raise ValueError("The catalog contains duplicate GRAND_ID values.")

    columns = list(extended_data.columns)
    columns.extend(column for column in create_extended_columns() if column not in extended_data)
    result = extended_data.reindex(columns=columns).copy()
    if not result.index.is_unique:
        result = result.reset_index(drop=True)

    for entry in grand_ids_to_update:
        if isinstance(entry, tuple) and len(entry) == 3 and isinstance(entry[0], Integral):
            grand_id, res_name, dam_name = entry
        elif isinstance(entry, Integral) and not isinstance(entry, bool):
            grand_id, res_name, dam_name = entry, None, None
        else:
            raise ValueError("Each entry must be an integer ID or (integer ID, reservoir name, dam name).")

        raw = check_and_read_files(grand_id, folder)
        if raw is None:
            continue
        observations = normalize_observations(raw)
        if observations.is_placeholder or observations.quality.status == "unavailable":
            warnings.warn(
                f"Skipping GRAND_ID {grand_id}: observations are {observations.quality.status}.",
                UserWarning,
                stacklevel=2,
            )
            continue
        percentiles = calculate_percentiles(observations)
        update = {}
        for metric, variable in (
            ("Q", "Outflow(cms)"), ("S", "Storage(mil_m3)"), ("I", "Inflow(cms)"),
        ):
            for percentile in DEFAULT_PERCENTILES:
                label = f"{metric}{percentile:g}"
                row = f"P{percentile:g}"
                update[label] = percentiles.at[row, f"{variable}_all"]
                for month in MONTH_NAMES:
                    update[f"{label}_{month}"] = percentiles.at[row, f"{variable}_{month}"]

        matching = result.index[result["GRAND_ID"] == grand_id]
        if matching.empty:
            result = pd.concat(
                [result, pd.DataFrame([{"GRAND_ID": grand_id}], columns=result.columns)],
                ignore_index=True,
            )
            index = result.index[-1]
        else:
            index = matching[0]
        if res_name is not None:
            result.loc[index, "RES_NAME"] = res_name
        if dam_name is not None:
            result.loc[index, "DAM_NAME"] = dam_name
        result.loc[index, list(update)] = list(update.values())
    return result
