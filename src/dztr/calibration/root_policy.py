"""Calibration and separate validation for an explicitly declared root policy."""

from dataclasses import replace

import numpy as np

from dztr.calibration.candidates import parameters_from_vector
from dztr.calibration.objectives import CandidateEvaluation
from dztr.calibration.sampled import SampledCalibrationObjective
from dztr.evaluation import evaluate_simulation
from dztr.simulation.root_policy import POLICIES, simulate_root_policy_reservoir


class RootPolicyCalibrationObjective(SampledCalibrationObjective):
    """Retain calibration-only bounds/objectives and continuous held-out state.

    The numerical convention is fixed before the search and never inferred from
    validation results. Existing sampled and daily objectives remain unchanged.
    """

    def __init__(self, observations, split, template, *, sampling, policy="reject",
                 bounds=None, atol_storage_m3=1e-4):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}.")
        self.root_policy = policy
        super().__init__(observations, split, template, sampling=sampling,
                         bounds=bounds, atol_storage_m3=atol_storage_m3)

    def _evaluate(self, vector, continuous, observed, period):
        counts = {label: int(observed.frame[column].notna().sum())
                  for label, column in (("release", "release_m3s"), ("storage", "storage_m3"))}
        parameters = simulation = None
        try:
            parameters = (parameters_from_vector(vector, self.template) if self.bounds is None
                          else self.bounds.validate(vector, self.template))
            simulation = simulate_root_policy_reservoir(continuous, parameters, sampling=self.sampling,
                policy=self.root_policy, atol_storage_m3=self.atol_storage_m3)
            if simulation.frame.storage_m3.lt(0).any():
                raise ValueError("Candidate produced negative storage.")
            if simulation.frame.balance_residual_m3.abs().max() > self.atol_storage_m3:
                raise ValueError("Candidate water-balance residual exceeds the configured tolerance.")
            scored = replace(simulation, frame=simulation.frame.loc[period.start:period.end].copy())
            metrics = evaluate_simulation(observed, scored)
            costs = (-metrics["release_nse"], -metrics["storage_nse"])
            if not np.isfinite(costs).all():
                raise ValueError("Candidate release/storage NSE is undefined.")
        except (TypeError, ValueError) as error:
            return CandidateEvaluation(False, (float("inf"), float("inf")), {}, counts,
                                       str(error), parameters, simulation, period)
        return CandidateEvaluation(True, costs, metrics, counts, None, parameters, simulation, period)
