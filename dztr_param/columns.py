"""Column layout for the historical GRanD parameter workbooks."""

from dztr.parameterization import DEFAULT_PERCENTILES, MONTH_NAMES


def create_extended_columns(percentiles=DEFAULT_PERCENTILES):
    """Return the legacy schema, including P65, P99, and P100 consistently."""
    columns = ["GRAND_ID", "RES_NAME", "DAM_NAME"]
    for metric in ("Q", "S", "I"):
        for percentile in percentiles:
            label = f"{metric}{percentile:g}"
            columns.append(label)
            columns.extend(f"{label}_{month}" for month in MONTH_NAMES)
    return columns
