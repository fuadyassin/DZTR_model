"""An explicit proxy/band objective around the unchanged continuous DZTR model."""
from dataclasses import replace

import numpy as np
import pandas as pd

from strategies.representative_years import RepresentativeYearObjective
from workflows.geometry_v1 import digest


def band_nse(center, lower, upper, simulated):
    center, lower, upper, simulated = [np.asarray(v, dtype=float) for v in (center, lower, upper, simulated)]
    if not (center.shape == lower.shape == upper.shape == simulated.shape):
        raise ValueError("Storage band arrays must align")
    mask = np.isfinite(center)
    c, lo, hi, s = [v[mask] for v in (center, lower, upper, simulated)]
    if not np.isfinite(np.column_stack([lo, hi, s])).all() or (lo > c).any() or (c > hi).any():
        raise ValueError("Missing or unordered storage band on fixed proxy support")
    if not len(c) or np.sum((c - c.mean()) ** 2) == 0:
        return float("nan")
    distance = np.maximum(np.maximum(lo - s, s - hi), 0.)
    return float(1 - np.sum(distance ** 2) / np.sum((c - c.mean()) ** 2))


class GeometryObjective(RepresentativeYearObjective):
    """Two costs: release NSE and explicitly chosen point-proxy or band NSE.

    Band NSE is a compatibility score: only distance outside a declared volume
    band is penalized. It is not a likelihood or proof of observation accuracy.
    """
    def __init__(self, *args, storage_band, storage_objective, provenance, **kwargs):
        super().__init__(*args, **kwargs)
        if storage_objective not in {"proxy_nse", "reference_band_nse"}:
            raise ValueError("Choose proxy_nse or reference_band_nse")
        if not storage_band.index.equals(self._source.frame.index) or list(storage_band.columns) != ["storage_lower_m3", "storage_upper_m3"]:
            raise ValueError("Storage band needs the exact proxy calendar and SI columns")
        self.band = storage_band.copy()
        self.storage_objective = storage_objective
        self.identity = {**self.identity, "strategy": "geometry_objective_v1", "storage_objective": storage_objective,
                         "geometry_provenance_sha256": digest(provenance)}
        # Validate fixed calibration support before preparing any optimizer job.
        mask = self.selection.mask(self._source.frame.index, "calibration")
        c = self._source.frame.loc[mask, "storage_m3"]
        band_nse(c, self.band.loc[mask].iloc[:, 0], self.band.loc[mask].iloc[:, 1], c)

    def _evaluate(self, vector, role):
        evaluated = super()._evaluate(vector, role)
        if not evaluated.feasible: return evaluated
        dates = self._source.frame.index[self.selection.mask(self._source.frame.index, role)]
        center = self._source.frame.loc[dates, "storage_m3"]
        sim = evaluated.simulation.frame.loc[dates, "storage_m3"]
        band = self.band.loc[dates]
        labels = self.selection.table().loc[dates, "regime"]
        metrics = dict(evaluated.metrics)
        scores = []
        try:
            metrics["storage_band_nse"] = band_nse(center, band.iloc[:, 0], band.iloc[:, 1], sim)
            for regime in sorted(set(labels)):
                mask = labels.eq(regime)
                value = band_nse(center.loc[mask], band.loc[mask].iloc[:, 0], band.loc[mask].iloc[:, 1], sim.loc[mask])
                metrics[regime + "_storage_band_nse"] = value; scores.append(value)
            metrics["regime_balanced_storage_band_nse"] = float(np.mean(scores))
            key = "regime_balanced_" if self.cost_mode == "regime_balanced_nse" else ""
            storage_cost = -metrics[key + ("storage_band_nse" if self.storage_objective == "reference_band_nse" else "storage_nse")]
            if not np.isfinite(storage_cost): raise ValueError("Undefined storage objective")
            return replace(evaluated, objectives=(evaluated.objectives[0], storage_cost), metrics=metrics)
        except ValueError as exc:
            return replace(evaluated, feasible=False, objectives=(float("inf"), float("inf")), reason=str(exc), metrics={})
