"""Mixed-year scalar calibration with mean-flow volumes and frozen gap starts.

The array path calls the established interval root solver literally. It avoids
building the full diagnostic DataFrame at every optimizer evaluation; exported
representatives are independently replayed through the canonical simulator.
"""
from math import ceil, isfinite
from dataclasses import replace
import numpy as np
import pandas as pd

from dztr.calibration.sampled import _observations
from dztr.model.release import _monthly_values, _release_for_values, _zone
from dztr.simulation.intervals import _solve_interval_step
from strategies.flow_only_v1 import Evaluation, nse
from strategies.full_data_scalar_v1 import weights
from strategies.representative_years import YearSelection, fit_selected_years
from workflows.extension_assessment_v1 import blocks


def simulation_support(frame, *, restart_warmup_days):
    if isinstance(restart_warmup_days, bool) or not isinstance(restart_warmup_days, int) or restart_warmup_days < 1:
        raise ValueError("Declare a positive restart warmup")
    index = frame.index
    if not index.equals(pd.date_range(index[0], index[-1], name=index.name)):
        raise ValueError("Keep a consecutive daily calendar")
    inflow = frame.inflow_m3s.to_numpy(float)
    if np.isinf(inflow).any() or (inflow[np.isfinite(inflow)] < 0).any():
        raise ValueError("Physical inflow must be nonnegative or explicitly missing")
    anchors = np.zeros(len(frame), bool); eligible = anchors.copy()
    for start, stop in blocks(np.isfinite(inflow)):
        anchors[start] = True
        eligible[min(start + restart_warmup_days, stop):stop] = True
    return anchors, eligible


def select_years(frame, score_eligible, *, minimum_fraction=.65, seed=42, minimum_inflow_coverage=.95):
    """Whole years stratified by observed annual mean, without filling flows.

    Missing-day annual means are labelled available-day summaries. Selection
    uses inflow features and response counts, never response performance.
    Calibration support must cover at least 65% of the entire calendar as well
    as 65% of available response samples after all fixed exclusions.
    """
    if not .65 <= minimum_fraction < 1: raise ValueError("At least 65% calibration required")
    eligible_mask = np.asarray(score_eligible, bool)
    if eligible_mask.shape != (len(frame),): raise ValueError("Support must align")
    records = []
    for year, part in frame.groupby(frame.index.year):
        expected = pd.date_range(f"{year}-01-01", f"{year}-12-31", name=frame.index.name)
        supported = eligible_mask[frame.index.year == year]
        records.append(dict(year=int(year), complete=part.index.equals(expected), rows=len(part),
            inflow_coverage=float(part.inflow_m3s.notna().mean()),
            inflow_mean_m3s=float(part.inflow_m3s.mean()),
            storage_pairs=int((part.storage_m3.notna().to_numpy() & supported).sum()),
            release_pairs=int((part.release_m3s.notna().to_numpy() & supported).sum()),
            fit_rows=int(supported.sum())))
    annual = pd.DataFrame(records).set_index("year")
    valid = annual.index[annual.complete & (annual.inflow_coverage >= minimum_inflow_coverage)].tolist()
    if len(valid) < 7: raise ValueError("Need a warmup and six eligible complete years")
    warmup, pool = valid[0], valid[1:]
    q1, q2 = np.quantile(annual.loc[pool, "inflow_mean_m3s"], [1/3, 2/3])
    annual["regime"] = "unclassified"
    annual.loc[pool, "regime"] = np.where(annual.loc[pool, "inflow_mean_m3s"] < q1, "dry",
        np.where(annual.loc[pool, "inflow_mean_m3s"] > q2, "wet", "normal"))
    selected = set(); rng = np.random.default_rng(seed)
    groups = {g: sorted(y for y in pool if annual.loc[y, "regime"] == g) for g in ("dry", "normal", "wet")}
    def choose(candidates):
        # Fill broad temporal/flow coverage within a regime; tie breaking is seeded.
        candidates = sorted(candidates); scores = []
        for year in candidates:
            peers = [y for y in selected if annual.loc[y, "regime"] == annual.loc[year, "regime"]]
            width = max(float(annual.loc[pool, "inflow_mean_m3s"].max()-annual.loc[pool, "inflow_mean_m3s"].min()), 1.)
            scores.append(min(((year-y)/max(len(pool)-1, 1))**2 +
                ((annual.loc[year, "inflow_mean_m3s"]-annual.loc[y, "inflow_mean_m3s"])/width)**2 for y in peers) if peers else 10.)
        return candidates[int(rng.choice(np.flatnonzero(np.isclose(scores, max(scores), rtol=0, atol=1e-12))))]
    for group in groups.values():
        if not group: continue
        quota = min(ceil(minimum_fraction*len(group)), len(group)-1 if len(group)>1 else 1)
        while len(selected.intersection(group)) < quota: selected.add(choose(set(group)-selected))
    total_pairs = {v: int(frame[c].notna().sum()) for v, c in (("storage", "storage_m3"), ("release", "release_m3s"))}
    def sufficient():
        part = annual.loc[sorted(selected)]
        return (part.fit_rows.sum() >= ceil(minimum_fraction*len(frame)) and
            all(part[v+"_pairs"].sum() >= ceil(minimum_fraction*n) for v, n in total_pairs.items()))
    while not sufficient():
        remaining = set(pool)-selected
        preserving = {y for y in remaining if len(set(groups[annual.loc[y, "regime"]])-selected) > 1}
        if not preserving: raise ValueError("Insufficient calibration coverage with held-out regimes retained")
        selected.add(choose(preserving))
    annual["role"] = "context"; annual.loc[warmup, "role"] = "warmup"
    annual.loc[pool, "role"] = "assessment"; annual.loc[sorted(selected), "role"] = "calibration"
    selection = YearSelection(tuple(str(d.date()) for d in frame.index),
        tuple(annual.loc[int(y), "role"] for y in frame.index.year), tuple(int(y) for y in frame.index.year),
        tuple(annual.loc[int(y), "regime"] for y in frame.index.year), minimum_fraction, seed, 1, 1)
    return selection, annual


def fit(frame, selection, score_eligible, physical):
    view = frame.copy()
    view.loc[~np.asarray(score_eligible, bool), ["storage_m3", "release_m3s"]] = np.nan
    fit_mask = selection.mask(frame.index, "calibration") & score_eligible
    if fit_mask.mean() < .65: raise ValueError("Actual fitting support must cover 65% of the full calendar")
    template, bounds = fit_selected_years(_observations(view, "frozen daily-interval fitting observations"),
        selection, physical, quantile_method="hazen")
    return template, bounds


class ArrayRunner:
    """Forcing and fixed initial states only; never retains response records."""
    def __init__(self, inflow, monthly_initial_storage, *, tolerance=1e-4, policy="nearest_previous_storage"):
        if not isinstance(inflow, pd.Series): raise ValueError("Supply one labelled forcing series")
        if policy not in {"reject", "minimum_storage", "maximum_storage", "nearest_previous_storage"}: raise ValueError("Unknown root policy")
        if not np.isfinite(tolerance) or tolerance <= 0: raise ValueError("Positive balance tolerance required")
        initials = np.asarray(monthly_initial_storage, float)
        if initials.shape != (12,) or not np.isfinite(initials).all(): raise ValueError("Twelve fixed initial storage values required")
        self.index = inflow.index.copy(); self.inflow = inflow.to_numpy(float, copy=True)
        self.months = self.index.month.to_numpy()-1
        self.initials = initials.copy(); self.tolerance = tolerance; self.policy = policy
        simulation_support(pd.DataFrame({"inflow_m3s": inflow}), restart_warmup_days=1)
        for array in (self.inflow, self.initials, self.months): array.setflags(write=False)

    def run(self, parameters, *, end_position=None):
        parameters = replace(parameters)
        count = len(self.index) if end_position is None else int(end_position)+1
        if not 1 <= count <= len(self.index): raise ValueError("Invalid simulation prefix")
        result = {c: np.full(len(self.index), np.nan) for c in ("storage_m3", "release_endpoint_m3s",
            "interval_mean_total_outflow_m3s", "spill_volume_m3", "balance_residual_m3", "zone")}
        result["interval_completed"] = np.zeros(len(self.index), bool)
        result["segment_initialization"] = np.zeros(len(self.index), bool)
        monthly = [_monthly_values(m, parameters) for m in range(1, 13)]
        previous = release = None
        for i in range(count):
            mean = float(self.inflow[i])
            if not isfinite(mean): previous = release = None; continue
            values = monthly[self.months[i]]
            try:
                if previous is None:
                    previous = float(self.initials[self.months[i]])
                    if not 0 <= previous <= parameters.capacity_m3: raise ValueError("Gap initial storage outside capacity")
                    release = _release_for_values(previous, mean, values, 86400.)
                    result["storage_m3"][i], result["release_endpoint_m3s"][i] = previous, release
                    result["zone"][i] = _zone(previous, values)
                    result["segment_initialization"][i] = True
                    continue
                volume = mean*86400.*parameters.inflow_correction
                predictor = previous+volume-43200.*release
                (storage, next_release, spill), roots, _, _ = _solve_interval_step(predictor, mean, 86400.,
                    values, parameters.capacity_m3, self.tolerance, previous, self.policy)
                outflow = 43200.*(release+next_release)+spill
                residual = storage-previous-volume+outflow
                rule_residual = next_release-_release_for_values(storage, mean, values, 86400.)
                if (not all(isfinite(v) for v in (storage, next_release, spill, residual, outflow, rule_residual))
                        or not 0 <= storage <= parameters.capacity_m3 or next_release < 0 or spill < 0
                        or abs(residual) > self.tolerance
                        or 43200.*abs(rule_residual) > roots["root_admissibility_tolerance_m3"]):
                    raise ValueError("Interval physical, release-rule or mass-balance check failed")
                for column, value in (("storage_m3", storage), ("release_endpoint_m3s", next_release),
                    ("interval_mean_total_outflow_m3s", outflow/86400.), ("spill_volume_m3", spill),
                    ("balance_residual_m3", residual), ("zone", _zone(storage, values))): result[column][i] = value
                result["interval_completed"][i] = True
                previous, release = storage, next_release
            except ValueError as exc:
                raise ValueError(f"{self.index[i].date()}: {exc}") from exc
        return result


class Objective:
    def __init__(self, frame, selection, score_eligible, runner, template, bounds, *, objective_weights):
        if bounds.selection_signature != selection.signature: raise ValueError("Bounds and fitting years disagree")
        self.mask = selection.mask(frame.index, "calibration") & score_eligible
        if self.mask.mean() < .65: raise ValueError("Require 65% actual calibration support")
        self.end = int(np.flatnonzero(self.mask)[-1])
        self.observed = frame[["storage_m3", "release_m3s"]].copy()
        self.observed.loc[~self.mask] = np.nan
        self.runner, self.template, self.bounds = runner, template, bounds
        self.weights = weights(objective_weights)
        for variable, c in (("storage", "storage_m3"), ("release", "release_m3s")):
            if self.weights[variable]: nse(self.observed[c].to_numpy()[self.mask], self.observed[c].to_numpy()[self.mask])

    def __call__(self, vector):
        try:
            parameters = self.bounds.validate(vector, self.template)
            result = self.runner.run(parameters, end_position=self.end)
            cost = 0.
            for variable, observed, predicted in (("storage", "storage_m3", "storage_m3"),
                    ("release", "release_m3s", "interval_mean_total_outflow_m3s")):
                if self.weights[variable]: cost += self.weights[variable]*(1-nse(
                    self.observed[observed].to_numpy()[self.mask], result[predicted][self.mask]))
            return Evaluation(True, float(cost), parameters=parameters, balance=float(np.nanmax(np.abs(result["balance_residual_m3"]))))
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            return Evaluation(False, float("inf"), str(exc))
