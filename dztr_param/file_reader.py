"""Raw file compatibility reader; validated readers live in ``dztr.data``."""

import warnings

import pandas as pd

from dztr.data import find_reservoir_file


def check_and_read_files(grand_id, folder="ResDataInflowStorageRelease"):
    """Read one matching CSV/workbook, warning if missing or ambiguous."""
    try:
        path = find_reservoir_file(folder, grand_id)
    except (FileNotFoundError, ValueError) as error:
        warnings.warn(str(error), UserWarning, stacklevel=2)
        return None
    return pd.read_csv(path) if path.suffix.casefold() == ".csv" else pd.read_excel(path)
