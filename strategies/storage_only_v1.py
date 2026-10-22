"""Storage-only DDS objective with a fixed, documented naturalized release prior."""
from dataclasses import replace

import numpy as np
import pandas as pd

from dztr.schemas import MONTHLY_TARGET_COLUMNS, ReservoirParameters
from strategies.data_availability_v1 import empirical_targets
from strategies.flow_only_v1 import Evaluation, nse
from workflows import profile_application_v1 as application


def fit(frame, selection, physical, release_prior, *, quantile_method, bounds_policy):
    mask = pd.Series(selection.mask(frame.index, "calibration"), index=frame.index)
    if mask.mean() + 1e-12 < .65: raise ValueError("At least 65% calibration is required")
    if bounds_policy not in {"monthly", "etienne_annual_envelope"}: raise ValueError("Unknown storage-bound policy")
    if not np.isfinite(frame.inflow_m3s.to_numpy()).all(): raise ValueError("Missing model forcing")
    storage, counts = empirical_targets(frame.storage_m3, mask, monthly=True, percentiles=(10, 45, 85), method=quantile_method)
    low, _ = empirical_targets(frame.storage_m3, mask, monthly=True,
        percentiles=(5, 35, 75) if bounds_policy == "monthly" else (5, 20, 60), method=quantile_method)
    high, _ = empirical_targets(frame.storage_m3, mask, monthly=True, percentiles=(35, 75, 95), method=quantile_method)
    if not all(np.isfinite(v).all() for v in (storage, low, high)): raise ValueError("Storage observations are required in every calibration month")
    if bounds_policy == "etienne_annual_envelope": low, high = np.tile(low.min(axis=0), (12, 1)), np.tile(high.max(axis=0), (12, 1))
    physical = dict(physical); capacity = physical["capacity_m3"]; dead = capacity * physical["dead_storage_fraction"]
    gap = max(capacity * 1e-10, 1e-6)
    raw_low, raw_high = low.copy(), high.copy()
    for col in range(3): low[:, col] = np.maximum(low[:, col], dead + gap if col == 0 else low[:, col - 1] + gap)
    for col in (2, 1, 0): high[:, col] = np.minimum(high[:, col], capacity if col == 2 else high[:, col + 1] - gap)
    if (low > high).any(): raise ValueError("No physically feasible intersection of storage percentile bounds with adopted dead storage/capacity")
    initial = np.clip(storage, low, high)
    for col in (1, 2): initial[:, col] = np.maximum(initial[:, col], initial[:, col - 1] + gap)
    if (initial > high).any():
        initial = low + (high - low) / 2
        for col in (1, 2): initial[:, col] = np.maximum(initial[:, col], initial[:, col - 1] + gap)
    original_initial = frame.storage_m3.iloc[0]
    observed_initial = selection.roles[0] != "assessment" and np.isfinite(original_initial) and 0 <= original_initial <= capacity
    physical["initial_storage_m3"] = float(original_initial if observed_initial else initial[frame.index[0].month - 1, 1])
    physical["initial_release_m3s"] = float(np.clip(frame.inflow_m3s.iloc[0] * physical["inflow_correction"], 0, physical["channel_capacity_m3s"]))
    if list(release_prior.columns) != list(MONTHLY_TARGET_COLUMNS[3:]) or list(release_prior.index) != list(range(1, 13)):
        raise ValueError("Release prior must contain 12 ordered months and the three SI release targets")
    targets = pd.DataFrame(np.column_stack([initial, release_prior.to_numpy()]), index=pd.Index(range(1, 13), name="month"), columns=MONTHLY_TARGET_COLUMNS)
    template = ReservoirParameters(targets, **physical)
    receipt = {"storage_estimation": "calibration_monthly_p10_p45_p85", "storage_sample_counts": counts,
        "storage_bound_policy": bounds_policy, "physical_bound_intersection": True,
        "bound_cells_changed": int((low != raw_low).sum() + (high != raw_high).sum()),
        "initializer_cells_changed": int((initial != storage).sum()), "strict_order_gap_m3": gap,
        "initial_storage_rule": "first_nonassessment_storage_observation" if observed_initial else "calibration_normal_target",
        "initial_release_rule": "first effective inflow clipped to [0, channel]", "release_targets_fixed": True}
    return template, low.ravel(), high.ravel(), receipt


class StorageObjective:
    def __init__(self, frame, selection, template, lower, upper, metadata, preprocessing, *, objective, scoring):
        if objective != "storage_nse": raise ValueError("The storage-only objective is 1 - NSE(storage)")
        if scoring not in {"pooled", "equal_regime"}: raise ValueError("Unknown regime weighting")
        self.frame = frame[["inflow_m3s", "storage_m3"]].copy()
        self.selection, self.template = selection, template
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.metadata, self.preprocessing = metadata, preprocessing
        self.objective, self.scoring = objective, scoring
        self.mask = selection.mask(frame.index, "calibration"); self.labels = selection.table().regime.to_numpy()
        for g in self.groups(): nse(self.frame.storage_m3.to_numpy()[g], self.frame.storage_m3.to_numpy()[g])

    def groups(self):
        return [self.mask] if self.scoring == "pooled" else [self.mask & (self.labels == label) for label in sorted(set(self.labels[self.mask]))]

    def parameters(self, vector):
        vector = np.asarray(vector, float)
        if vector.shape != (36,) or not np.isfinite(vector).all() or (vector < self.lower).any() or (vector > self.upper).any(): raise ValueError("Storage vector violates its fixed 36-dimensional bounds")
        targets = self.template.monthly_targets.copy(); targets.iloc[:, :3] = vector.reshape(12, 3)
        return replace(self.template, monthly_targets=targets)

    def evaluate(self, vector):
        try:
            parameters = self.parameters(vector); end = self.frame.index[self.mask][-1]
            simulation, balance = application.run_frame(self.frame.loc[:end], self.metadata, parameters, forcing_stage="prepared", preprocessing=self.preprocessing)
            observed = self.frame.storage_m3.to_numpy()[:len(simulation.frame)]; simulated = simulation.frame.storage_m3.to_numpy()
            cost = 1 - float(np.mean([nse(observed[g[:len(observed)]], simulated[g[:len(observed)]]) for g in self.groups()]))
            return Evaluation(True, cost, simulation=simulation, parameters=parameters, balance=balance)
        except (ValueError, RuntimeError) as exc: return Evaluation(False, float("inf"), str(exc))

    def __call__(self, vector): return self.evaluate(vector)
