"""Fixed DZTR targets without target storage or release responses."""
import numpy as np
import pandas as pd

from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters


def parameters(forcing, physical, release_targets, *, storage_basis):
    """Only inflow, adopted physical constants and an explicit prior enter here."""
    if storage_basis not in {"total_capacity", "active_capacity"}:
        raise ValueError("Declare total_capacity or active_capacity")
    rates = forcing["inflow_m3s"].to_numpy(float)
    if not len(rates) or not np.isfinite(rates).all():
        raise ValueError("Finite inflow forcing is required")
    if list(release_targets.columns) != list(MONTHLY_TARGET_COLUMNS[3:]) or list(release_targets.index) != list(range(1, 13)):
        raise ValueError("Release prior must contain the twelve monthly SI target rows")
    adopted = dict(physical)
    capacity = adopted["capacity_m3"]
    dead = capacity * adopted["dead_storage_fraction"]
    fractions = np.array([.20, .30, .97])
    storage = fractions * capacity if storage_basis == "total_capacity" else dead + fractions * (capacity - dead)
    adopted["initial_storage_m3"] = float(storage[1])
    adopted["initial_release_m3s"] = float(np.clip(rates[0] * adopted["inflow_correction"], 0, adopted["channel_capacity_m3s"]))
    targets = pd.DataFrame(np.column_stack([np.tile(storage, (12, 1)), release_targets.to_numpy(float)]),
        columns=MONTHLY_TARGET_COLUMNS, index=pd.Index(range(1, 13), name="month"))
    return ReservoirParameters(targets, **adopted)


def recommend(capabilities, *, inflow_usable, release_prior_usable):
    """Recommend by usable calibration data, without reading assessment scores."""
    gates = capabilities.get("source_gates", [])
    if gates:
        return {"method": None, "status": "source_gated", "reason": "; ".join(gates)}
    if not inflow_usable:
        return {"method": None, "status": "needs_inflow", "reason": "Usable inflow forcing is required by every numerical route"}
    storage = capabilities.get("storage_monthly_available", False)
    release = capabilities.get("release_monthly_available", False)
    if storage and release:
        method = "level_geometry_monthly" if capabilities.get("storage_source_kind") == "level_geometry" else "observed_monthly"
    elif release:
        method = "flow_only"
    elif storage:
        method = "storage_only"
    else:
        method = "ungauged_mean_flow"
    prior_needed = method in {"storage_only", "ungauged_mean_flow"}
    return {"method": method, "status": "recommended" if release_prior_usable or not prior_needed else "needs_release_prior",
            "reason": "Selected from usable calibration observations and declared dependencies; assessment scores are excluded"}
