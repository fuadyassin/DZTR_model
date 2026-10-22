"""Release-only fitting: fixed storage assumptions and no storage responses.

Etienne-inspired options are explicit: CR2-style capacity fractions, sqrt-Q
NSE and annual release-bound envelopes. The conservative solver is unchanged.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters
from strategies.data_availability_v1 import empirical_targets
from workflows import profile_application_v1 as application


def nse(observed, simulated, *, square_root=False):
    o, s = np.asarray(observed, float), np.asarray(simulated, float)
    if o.shape != s.shape: raise ValueError("NSE calendars must align")
    keep = np.isfinite(o)
    o, s = o[keep], s[keep]
    if len(o) < 2 or not np.isfinite(s).all() or (o < 0).any() or (s < 0).any():
        raise ValueError("NSE needs finite nonnegative simulations on fixed observation support")
    if square_root: o, s = np.sqrt(o), np.sqrt(s)
    denominator = np.sum((o - o.mean()) ** 2)
    if denominator <= 0: raise ValueError("NSE is undefined for constant observations")
    return float(1 - np.sum((o - s) ** 2) / denominator)


def fit(frame, selection, physical, *, quantile_method, storage_basis, bounds_policy):
    """Fit 36 release coordinates; never inspect the supplied storage column."""
    mask = pd.Series(selection.mask(frame.index, "calibration"), index=frame.index)
    if mask.mean() + 1e-12 < .65: raise ValueError("At least 65% calibration is required")
    if storage_basis not in {"total_capacity", "active_capacity"}: raise ValueError("Declare storage fraction basis")
    if bounds_policy not in {"monthly", "etienne_annual_envelope"}: raise ValueError("Unknown release bounds policy")
    if not np.isfinite(frame.inflow_m3s.to_numpy()).all(): raise ValueError("Missing model forcing")
    physical = dict(physical)
    capacity = physical["capacity_m3"]; dead = physical["dead_storage_fraction"] * capacity
    storage = np.array([.20, .30, .97]) * (capacity if storage_basis == "total_capacity" else capacity - dead)
    if storage_basis == "active_capacity": storage += dead
    # An explicitly assumed cold start; observed storage never initializes a run.
    physical["initial_storage_m3"] = float(storage[1])
    physical["initial_release_m3s"] = float(np.clip(frame.inflow_m3s.iloc[0] * physical["inflow_correction"],
                                                   0, physical["channel_capacity_m3s"]))
    release, _ = empirical_targets(frame.release_m3s, mask, monthly=True, percentiles=(10, 45, 85), method=quantile_method)
    low_p = (5, 35, 75) if bounds_policy == "monthly" else (5, 20, 60)
    low, _ = empirical_targets(frame.release_m3s, mask, monthly=True, percentiles=low_p, method=quantile_method)
    high, _ = empirical_targets(frame.release_m3s, mask, monthly=True, percentiles=(35, 75, 95), method=quantile_method)
    if bounds_policy == "etienne_annual_envelope":
        low, high = np.tile(low.min(axis=0), (12, 1)), np.tile(high.max(axis=0), (12, 1))
        high[:, 2] = max(physical["channel_capacity_m3s"], high[0, 2])
    if not all(np.isfinite(v).all() for v in (release, low, high)):
        raise ValueError("Release observations are required in every calibration month")
    targets = pd.DataFrame(np.column_stack([np.tile(storage, (12, 1)), release]),
                           columns=MONTHLY_TARGET_COLUMNS, index=pd.Index(range(1, 13), name="month"))
    template = ReservoirParameters(targets, **physical)
    return template, low.ravel(), high.ravel()


@dataclass
class Evaluation:
    feasible: bool
    cost: float
    reason: str | None = None
    simulation: object = None
    parameters: object = None
    balance: float | None = None


class FlowObjective:
    def __init__(self, frame, selection, template, lower, upper, metadata, preprocessing, *, objective, scoring):
        if objective not in {"release_nse", "sqrt_release_nse"}: raise ValueError("Unknown release objective")
        if scoring not in {"pooled", "equal_regime"}: raise ValueError("Unknown regime weighting")
        # Store only the allowed responses. Hidden storage cannot affect bounds,
        # feasibility, optimization, initialization or the calibration choice.
        self.frame = frame[["inflow_m3s", "release_m3s"]].copy()
        self.selection, self.template = selection, template
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.metadata, self.preprocessing = metadata, preprocessing
        self.objective, self.scoring = objective, scoring
        self.mask = selection.mask(frame.index, "calibration")
        self.labels = selection.table().regime.to_numpy()
        for group in self.groups(self.mask):
            o = self.frame.release_m3s.to_numpy()[group]
            nse(o, o, square_root=objective == "sqrt_release_nse")

    def groups(self, mask):
        return [mask] if self.scoring == "pooled" else [mask & (self.labels == label) for label in sorted(set(self.labels[mask]))]

    def parameters(self, vector):
        vector = np.asarray(vector, float)
        if vector.shape != (36,) or not np.isfinite(vector).all() or (vector < self.lower).any() or (vector > self.upper).any():
            raise ValueError("Release vector violates its fixed 36-dimensional bounds")
        targets = self.template.monthly_targets.copy(); targets.iloc[:, 3:] = vector.reshape(12, 3)
        from dataclasses import replace
        return replace(self.template, monthly_targets=targets)

    def evaluate(self, vector):
        try:
            parameters = self.parameters(vector)
            end = self.frame.index[self.mask][-1]
            simulation, balance = application.run_frame(self.frame.loc[:end], self.metadata, parameters,
                                                         forcing_stage="prepared", preprocessing=self.preprocessing)
            mask = self.mask[:len(simulation.frame)]
            observed = self.frame.release_m3s.to_numpy()[:len(mask)]
            simulated = simulation.frame.release_m3s.to_numpy()
            scores = [nse(observed[g[:len(mask)]], simulated[g[:len(mask)]], square_root=self.objective == "sqrt_release_nse")
                      for g in self.groups(self.mask)]
            return Evaluation(True, 1 - float(np.mean(scores)), simulation=simulation,
                              parameters=parameters, balance=balance)
        except (ValueError, RuntimeError) as exc:
            return Evaluation(False, float("inf"), str(exc))

    def __call__(self, vector): return self.evaluate(vector)
