"""Versioned exploratory signed-net forcing with the native endpoint clock.

Net flux is integrated literally. Its positive/negative parts are diagnostic
components, not measured gross inflow and evaporation. An inflow-driven flood
rule requires an explicit net-as-rule-proxy hypothesis. Existing standard
solvers stay unchanged.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from dztr.model.release import _monthly_values, _release_for_values, _zone
from dztr.schemas import ReservoirObservations, ReservoirParameters, SimulationResult
from .root_policy import POLICIES, _resolve
from .sampled import SamplingSettings, validate_sampling_index


@dataclass(frozen=True)
class SignedNetPolicy:
    interpretation: str = "exploratory_net_boundary_flux"
    clock: str = "native_endpoint_trapezoid"
    independent_forcing_verified: bool = False
    flood_rule_driver: str = "storage_only"

    def __post_init__(self):
        if self.interpretation != "exploratory_net_boundary_flux" or self.clock != "native_endpoint_trapezoid":
            raise ValueError("Unsupported signed forcing interpretation or clock")
        if self.independent_forcing_verified is not False:
            raise ValueError("This exploratory adapter does not certify independent forcing")
        if self.flood_rule_driver not in ("storage_only", "signed_net_proxy"):
            raise ValueError("Unknown flood-rule forcing interpretation")


def simulate_signed_net_reservoir(observations, parameters, *, forcing_policy,
                                  sampling=None, policy="nearest_previous_storage",
                                  atol_storage_m3=1e-4):
    """Conserve signed endpoint net flux without inventing roots or clipping it."""
    if not isinstance(forcing_policy, SignedNetPolicy):
        raise TypeError("Declare a SignedNetPolicy explicitly")
    if not isinstance(observations, ReservoirObservations) or observations.quality.status != "available":
        raise ValueError("Available normalized observations required")
    if not isinstance(parameters, ReservoirParameters):
        raise TypeError("Physical reservoir parameters required")
    if parameters.flood_mode == "inflow" and forcing_policy.flood_rule_driver != "signed_net_proxy":
        raise ValueError("Inflow-driven flood rule requires the explicit signed-net proxy hypothesis")
    if parameters.inflow_correction != 1:
        raise ValueError("Net losses must not be rescaled by an inflow correction")
    sampling = SamplingSettings() if sampling is None else sampling
    if not isinstance(sampling, SamplingSettings) or sampling.month_clock != "observed":
        raise ValueError("Use explicit observed-calendar sampling")
    if policy not in POLICIES:
        raise ValueError("Unknown root policy")
    if isinstance(atol_storage_m3, bool) or not np.isfinite(atol_storage_m3) or atol_storage_m3 <= 0:
        raise ValueError("Storage tolerance must be finite and positive")
    index = observations.frame.index
    validate_sampling_index(index, sampling.frequency)
    net = observations.frame.inflow_m3s.to_numpy(dtype=float, copy=True)
    if not np.isfinite(net).all():
        raise ValueError("Missing net forcing requires a separate quality/block adapter")
    dt = (np.r_[np.nan, np.diff(index.asi8) / 1e9] if sampling.fixed_step_days is None
          else np.full(len(index), sampling.fixed_step_days * 86400.))
    s, q = np.empty(len(index)), np.empty(len(index))
    spill, residual = np.zeros(len(index)), np.zeros(len(index))
    counts, applied, ties = np.zeros(len(index), int), np.zeros(len(index), bool), np.zeros(len(index), bool)
    rule_residual = np.full(len(index), np.nan)
    s[0], q[0] = parameters.initial_storage_m3, parameters.initial_release_m3s
    monthly = {int(m): _monthly_values(int(m), parameters) for m in index.month.unique()}
    for k in range(1, len(index)):
        step = float(dt[k]); values = monthly[index[k].month]
        predictor = s[k-1] + step/2 * (net[k] + net[k-1] - q[k-1])
        try:
            if not np.isfinite(predictor):
                raise ValueError("Nonfinite balance predictor")
            root = _resolve(predictor, net[k], step, values, atol_storage_m3, s[k-1], policy)
        except ValueError as error:
            raise ValueError(f"Signed-net simulation failed on {index[k].date()}: {error}") from error
        s[k], q[k] = root.storage_m3, root.release_m3s
        counts[k], applied[k], ties[k] = root.root_count, root.policy_applied, root.tie_break_applied
        rule_residual[k] = q[k] - _release_for_values(s[k], net[k], values, step)
        spill[k] = max(s[k] - parameters.capacity_m3, 0.)
        if spill[k]:
            s[k] = parameters.capacity_m3
            q[k] += 2*spill[k]/step  # Same conservative endpoint overflow as native solver.
        residual[k] = s[k] - s[k-1] - step/2*(net[k]+net[k-1]-q[k]-q[k-1])
    if (not np.isfinite(s).all() or not np.isfinite(q).all() or (s < 0).any()
            or (s > parameters.capacity_m3).any() or (q < 0).any()
            or np.max(np.abs(residual[1:])) > atol_storage_m3):
        raise ValueError("Signed-net physical or mass-balance gate failed")
    frame = pd.DataFrame({"storage_m3":s,"release_m3s":q,"inflow_m3s":net,
        "effective_inflow_m3s":net,"positive_net_component_m3s":np.maximum(net,0.),
        "negative_net_component_m3s":np.maximum(-net,0.),"step_seconds":dt,
        "balance_residual_m3":residual,"overflow_volume_m3":spill,
        "release_rule_residual_m3s":rule_residual,"root_count":counts,
        "root_policy_applied":applied,"root_policy_tie_break":ties,"rule_month":index.month,
        "zone":[_zone(float(v),monthly[d.month]) for v,d in zip(s,index)]},index=index.copy())
    return SimulationResult(frame,parameters,f"signed_net_v1:{policy}","conservative",
        ("Exploratory signed net flux; source derivation and endpoint clock are not certified.",))
