"""Scalar NSE calibration with all 72 observed-monthly DZTR targets free.

Storage remains available for empirical initialization/bounds even when its
objective weight is zero. This is distinct from the storage-hidden CR2 route.
"""
from numbers import Real

import numpy as np

from strategies.flow_only_v1 import Evaluation, nse
from workflows import profile_application_v1 as application


def weights(values):
    if not isinstance(values, dict) or set(values) != {"storage", "release"}:
        raise ValueError("Declare storage and release objective weights")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("Objective weights must be finite nonnegative numbers")
    scale = max(values.values())
    if scale == 0: raise ValueError("At least one objective weight must be positive")
    scaled = {k: float(v / scale) for k, v in values.items()}
    total = sum(scaled.values())
    return {k: v / total for k, v in scaled.items()}


class FullDataObjective:
    def __init__(self, frame, selection, template, bounds, metadata, preprocessing, *, objective_weights, scoring="pooled"):
        if scoring not in {"pooled", "equal_regime"}: raise ValueError("Unknown regime scoring")
        if bounds.selection_signature != selection.signature: raise ValueError("Bounds and year selection disagree")
        self.mask = selection.mask(frame.index, "calibration")
        if self.mask.mean() + 1e-12 < .65: raise ValueError("At least 65% calibration data required")
        self.frame = frame[["inflow_m3s", "storage_m3", "release_m3s"]].copy()
        # Assessment responses are never held by an optimization objective.
        self.frame.loc[~self.mask, ["storage_m3", "release_m3s"]] = np.nan
        self.selection, self.template, self.bounds = selection, template, bounds
        self.metadata, self.preprocessing, self.scoring = metadata, preprocessing, scoring
        self.weights = weights(objective_weights)
        self.lower, self.upper = bounds.lower_vector, bounds.upper_vector
        labels = selection.table().regime.to_numpy()
        self.groups = [self.mask] if scoring == "pooled" else [self.mask & (labels == g) for g in sorted(set(labels[self.mask]))]
        for variable, weight in self.weights.items():
            if weight:
                for group in self.groups:
                    o = self.frame[self.column(variable)].to_numpy()[group]
                    nse(o, o)

    @staticmethod
    def column(variable): return "storage_m3" if variable == "storage" else "release_m3s"

    def parameters(self, vector): return self.bounds.validate(vector, self.template)

    def cost(self, simulation):
        expected = self.frame.index[:len(simulation)]
        if not simulation.index.equals(expected) or len(simulation) <= np.flatnonzero(self.mask)[-1]:
            raise ValueError("Simulation must preserve the complete continuous calibration support")
        loss = 0.
        for variable, weight in self.weights.items():
            if not weight: continue
            column = self.column(variable)
            observed = self.frame.loc[simulation.index, column].to_numpy()
            predicted = simulation[column].to_numpy()
            scores = [nse(observed[g[:len(simulation)]], predicted[g[:len(simulation)]]) for g in self.groups]
            loss += weight * (1 - float(np.mean(scores)))
        if not np.isfinite(loss): raise ValueError("Nonfinite weighted objective")
        return float(loss)

    def evaluate(self, vector):
        try:
            parameters = self.parameters(vector)
            end = self.frame.index[self.mask][-1]
            simulation, balance = application.run_frame(self.frame.loc[:end], self.metadata, parameters,
                forcing_stage="prepared", preprocessing=self.preprocessing)
            return Evaluation(True, self.cost(simulation.frame), simulation=simulation, parameters=parameters, balance=balance)
        except (ValueError, RuntimeError) as exc:
            return Evaluation(False, float("inf"), str(exc))

    def __call__(self, vector): return self.evaluate(vector)
