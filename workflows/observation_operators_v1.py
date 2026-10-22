"""Fixed-support assessment of declared native observations or explicit proxies.

No interval-mean quantity is fabricated from native endpoint output. Those
operators require a separately versioned solver supplying the correct integral.
"""
import numpy as np
import pandas as pd

from dztr.evaluation.metrics import objective_metric


def score_native(frame, simulation, roles, *, observation_basis="native_archive"):
    if observation_basis not in {"native_archive", "point_storage_at_native_timestamp", "mean_level_volume_proxy"}:
        raise ValueError("Unsupported observation operator; interval means need matching model output")
    if not frame.index.equals(simulation.index) or not frame.index.equals(roles.index):
        raise ValueError("Assessment requires identical observation, simulation and role calendars")
    rows = []
    for phase in ("calibration", "assessment"):
        row = {"phase": phase, "observation_basis": observation_basis,
               "exact_interval_mean_operator": False,
               "storage_proxy_comparison": observation_basis == "mean_level_volume_proxy"}
        for label, column in (("storage", "storage_m3"), ("release", "release_m3s")):
            observed = frame[column].to_numpy(dtype=float)
            simulated = simulation[column].to_numpy(dtype=float)
            if np.isinf(observed).any() or (observed[np.isfinite(observed)] < 0).any():
                raise ValueError("Observed responses must be nonnegative or missing")
            # The mask depends only on declared observation support, never on a
            # candidate's validity. Invalid simulated values fail assessment.
            mask = roles.eq(phase).to_numpy() & np.isfinite(observed)
            obs, sim = observed[mask], simulated[mask]
            if not np.isfinite(sim).all() or (sim < 0).any():
                raise ValueError("Invalid simulated response on fixed scored support")
            row[label + "_pairs"] = len(obs)
            for metric in ("nse", "pbias", "rmse"):
                row[label + "_" + metric] = objective_metric(obs, sim, metric)
            if label == "release":
                row["release_sqrt_nse"] = objective_metric(np.sqrt(obs), np.sqrt(sim), "nse")
        rows.append(row)
    return rows
