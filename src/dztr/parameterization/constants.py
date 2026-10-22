"""One percentile definition shared by calculation and legacy workbook schemas."""

DEFAULT_PERCENTILES = tuple(sorted({*range(5, 101, 5), 99}))
TARGET_PERCENTILES = (10, 45, 85)
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
OBSERVATION_VARIABLES = ("storage_m3", "inflow_m3s", "release_m3s")
