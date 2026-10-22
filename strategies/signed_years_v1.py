"""Mixed-year objectives for the separately declared signed-net solver."""
from dataclasses import asdict, replace
import numpy as np
import pandas as pd
from dztr.calibration.candidates import parameters_from_vector
from dztr.calibration.objectives import CandidateEvaluation
from dztr.calibration.periods import DatePeriod
from dztr.calibration.sampled import _observations
from dztr.evaluation import evaluate_simulation
from dztr.simulation.signed_net import SignedNetPolicy, simulate_signed_net_reservoir
from strategies.representative_years import RepresentativeYearObjective
from strategies.fleet_comparison_v2 import comparison_selections
from dztr.simulation.sampled import validate_sampling_index


def select_signed_years(observations, sampling):
    """Use signed annual net-flux means, with the existing matched year selector.

    Daily records carry equal weights; monthly records carry calendar-day weights,
    exactly as the reference selector. No forcing is clipped or translated.
    """
    frame = observations.frame
    validate_sampling_index(frame.index,sampling.frequency)
    if not np.isfinite(frame.inflow_m3s.to_numpy()).all():
        raise ValueError("Annual net-flux classification requires complete forcing")
    rows = []
    for year, part in frame.groupby(frame.index.year):
        expected = pd.date_range(f"{year}-01-01",f"{year}-12-31",freq="D" if sampling.frequency == "daily" else "MS")
        weights = np.ones(len(part)) if sampling.frequency == "daily" else part.index.days_in_month.to_numpy()
        rows.append({"water_year":int(year),"complete":part.index.equals(expected.rename(part.index.name)),
            "rows":len(part),"inflow_mean_m3s":float(np.average(part.inflow_m3s,weights=weights)),
            "inflow_p10_m3s":float(np.percentile(part.inflow_m3s,10)),
            "storage_pairs":int(part.storage_m3.notna().sum()),"release_pairs":int(part.release_m3s.notna().sum())})
    annual = pd.DataFrame(rows).set_index("water_year")
    annual["antecedent_inflow_m3s"] = annual.inflow_mean_m3s.shift(1).rolling(2,min_periods=1).mean().fillna(annual.inflow_mean_m3s)
    complete = annual.index[annual.complete].tolist()
    eligible = complete[1:]
    if len(eligible) < 6: raise ValueError("At least six complete post-warmup years required")
    q1,q2 = np.quantile(annual.loc[eligible,"inflow_mean_m3s"],[1/3,2/3])
    annual["regime"] = "unclassified"
    annual.loc[eligible,"regime"] = np.where(annual.loc[eligible,"inflow_mean_m3s"] < q1,"dry",
        np.where(annual.loc[eligible,"inflow_mean_m3s"] > q2,"wet","normal"))
    annual["role"] = "context"
    annual.loc[complete[0],"role"] = "warmup"
    annual.loc[eligible,"role"] = "assessment"
    plans,held = comparison_selections(frame,annual)
    return plans["mixed_pooled"],annual,held


class SignedYearObjective(RepresentativeYearObjective):
    def __init__(self, *args, forcing_policy, **kwargs):
        if not isinstance(forcing_policy, SignedNetPolicy):
            raise TypeError("Declare SignedNetPolicy")
        super().__init__(*args, **kwargs)
        self.forcing_policy = forcing_policy
        self.identity = dict(self.identity, strategy="signed_years_v1", forcing_policy=asdict(forcing_policy))

    def _evaluate(self, vector, role):
        frame = self._source.frame; dates = frame.index[self.selection.mask(frame.index, role)]
        period = DatePeriod(dates[0], dates[-1])
        observed = _observations(frame.loc[dates])
        counts = {v:int(observed.frame[c].notna().sum()) for v,c in
                  (("storage","storage_m3"),("release","release_m3s"))}
        parameters = simulation = None
        try:
            parameters = (parameters_from_vector(vector,self.template) if self.bounds is None
                          else self.bounds.validate(vector,self.template))
            end = dates[-1] if role == "calibration" else frame.index[-1]
            simulation = simulate_signed_net_reservoir(_observations(frame.loc[:end]), parameters,
                forcing_policy=self.forcing_policy, sampling=self.sampling,
                policy=self.root_policy, atol_storage_m3=self.atol_storage_m3)
            scored = replace(simulation, frame=simulation.frame.loc[dates])
            metrics = evaluate_simulation(observed, scored)
            labels = self.selection.table().loc[dates,"regime"]
            groups = {v:[] for v in ("release","storage")}
            for regime in sorted(set(labels)):
                sub = labels == regime
                scores = evaluate_simulation(_observations(observed.frame.loc[sub]),
                                             replace(scored,frame=scored.frame.loc[sub]))
                for v in groups:
                    metrics[regime+"_"+v+"_nse"] = scores[v+"_nse"]
                    groups[v].append(scores[v+"_nse"])
            for v, values in groups.items():
                metrics["regime_balanced_"+v+"_nse"] = float(np.mean(values))
            prefix = "regime_balanced_" if self.cost_mode == "regime_balanced_nse" else ""
            costs = tuple(-metrics[prefix+v+"_nse"] for v in ("release","storage"))
            if not np.isfinite(costs).all():
                raise ValueError("NSE is undefined on active scoring support")
        except (TypeError,ValueError) as error:
            return CandidateEvaluation(False,(float("inf"),float("inf")),{},counts,str(error),parameters,simulation,period)
        return CandidateEvaluation(True,costs,metrics,counts,None,parameters,simulation,period)
