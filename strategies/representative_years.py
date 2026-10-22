"""Forcing-stratified whole-year fitting masks and continuous DZTR calibration."""
from dataclasses import dataclass, asdict, replace
import hashlib
import json
from math import ceil
from numbers import Integral, Real

import numpy as np
import pandas as pd

from dztr.calibration.bounds import CalibrationBounds, DEFAULT_PERCENTILE_BOUNDS
from dztr.calibration.candidates import parameters_from_vector
from dztr.calibration.objectives import CandidateEvaluation
from dztr.calibration.periods import DatePeriod
from dztr.calibration.sampled import _observations, target_observations
from dztr.evaluation import evaluate_simulation
from dztr.parameterization import estimate_generalized_parameters, estimate_percentiles
from dztr.schemas import ReservoirParameters
from dztr.simulation.root_policy import simulate_root_policy_reservoir
from dztr.simulation.sampled import validate_sampling_index


@dataclass(frozen=True)
class YearSelection:
    dates: tuple[str, ...]
    roles: tuple[str, ...]
    years: tuple[int, ...]
    regimes: tuple[str, ...]
    minimum_fraction: float
    seed: int
    water_year_start_month: int
    warmup_years: int

    def __post_init__(self):
        for name in ('dates','roles','years','regimes'):object.__setattr__(self,name,tuple(getattr(self,name)))
        if isinstance(self.minimum_fraction,bool) or not .65<=self.minimum_fraction<1:raise ValueError('Minimum calibration fraction must be at least 0.65 and below 1')
        if not (len(self.dates)==len(self.roles)==len(self.years)==len(self.regimes)):
            raise ValueError('Selection arrays must align')
        dates=pd.DatetimeIndex(pd.to_datetime(self.dates))
        if not dates.is_unique or not dates.is_monotonic_increasing:raise ValueError('Selection dates must be unique and sorted')
        if set(self.roles)-{'warmup','context','calibration','assessment'}:raise ValueError('Unknown selection role')
        if set(r for r,role in zip(self.regimes,self.roles) if role in ('calibration','assessment'))-{'dry','normal','wet'}:raise ValueError('Scored years require an identified inflow regime')
        if not {'calibration','assessment'}<=set(self.roles):raise ValueError('Separate calibration and assessment rows are required')
        if sum(r=='calibration' for r in self.roles)/len(self.roles)+1e-12 < self.minimum_fraction:
            raise ValueError('Calibration rows do not meet the minimum fraction of the entire record')
        for year in set(self.years):
            if len({r for y,r in zip(self.years,self.roles) if y==year})!=1:raise ValueError('Whole years cannot be split between roles')

    @property
    def signature(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def table(self):
        return pd.DataFrame({'role':self.roles,'water_year':self.years,'regime':self.regimes},index=pd.DatetimeIndex(pd.to_datetime(self.dates),name='date'))

    def mask(self,index,role):
        table=self.table()
        if not index.equals(table.index.rename(index.name)):raise ValueError('Selection requires its exact original timeline')
        return (table.role==role).to_numpy()

    def as_dict(self):return asdict(self)


def select_representative_years(observations,sampling,*,minimum_fraction=.65,seed=42,water_year_start_month=1,warmup_years=1):
    if isinstance(minimum_fraction,bool) or not isinstance(minimum_fraction,Real) or not .65<=minimum_fraction<1:
        raise ValueError('minimum_fraction must be at least 0.65 and below 1')
    for value,name,lo,hi in ((seed,'seed',0,None),(water_year_start_month,'water_year_start_month',1,12),(warmup_years,'warmup_years',1,None)):
        if isinstance(value,bool) or not isinstance(value,Integral) or value<lo or (hi is not None and value>hi):raise ValueError(f'Invalid {name}')
    frame=observations.frame
    validate_sampling_index(frame.index,sampling.frequency)
    inflow=frame.inflow_m3s.to_numpy()
    if not np.isfinite(inflow).all() or (inflow<0).any():raise ValueError('Year classification requires finite nonnegative adjusted inflows')
    labels=frame.index.year if water_year_start_month==1 else frame.index.year+(frame.index.month>=water_year_start_month)
    annual=[]
    for label in sorted(set(labels)):
        part=frame.loc[labels==label]
        start=pd.Timestamp(int(label) if water_year_start_month==1 else int(label)-1,water_year_start_month,1)
        end=start+pd.DateOffset(years=1)-pd.Timedelta(days=1)
        expected=pd.date_range(start,end,freq='D' if sampling.frequency=='daily' else 'MS')
        complete=part.index.equals(expected.rename(part.index.name))
        weights=np.ones(len(part)) if sampling.frequency=='daily' else part.index.days_in_month.to_numpy()
        annual.append({'water_year':int(label),'complete':complete,'rows':len(part),'inflow_mean_m3s':float(np.average(part.inflow_m3s,weights=weights)),
                       'inflow_p10_m3s':float(np.percentile(part.inflow_m3s,10)), 'storage_pairs':int(part.storage_m3.notna().sum()),'release_pairs':int(part.release_m3s.notna().sum())})
    annual=pd.DataFrame(annual).set_index('water_year')
    annual['antecedent_inflow_m3s']=annual.inflow_mean_m3s.shift(1).rolling(2,min_periods=1).mean().fillna(annual.inflow_mean_m3s)
    complete=annual.index[annual.complete].tolist()
    warmup=set(complete[:warmup_years]);eligible=complete[warmup_years:]
    if len(eligible)<6:raise ValueError('At least six complete post-warmup years are required')
    pool=annual.loc[eligible]
    q1,q2=np.quantile(pool.inflow_mean_m3s,[1/3,2/3])
    annual['regime']='unclassified'
    annual.loc[eligible,'regime']=np.where(pool.inflow_mean_m3s<q1,'dry',np.where(pool.inflow_mean_m3s>q2,'wet','normal'))
    rng=np.random.default_rng(seed);selected=set()
    features=annual[['inflow_mean_m3s','antecedent_inflow_m3s','inflow_p10_m3s']].copy()
    features['time']=features.index.to_numpy(dtype=float)
    for col in features:
        width=float(features.loc[eligible,col].max()-features.loc[eligible,col].min())
        features[col]=(features[col]-features.loc[eligible,col].min())/width if width else 0.

    def choose(candidates):
        ids=sorted(candidates)
        scores=[]
        for year in ids:
            same=[y for y in selected if annual.loc[y,'regime']==annual.loc[year,'regime']]
            scores.append(min(float(np.linalg.norm(features.loc[year]-features.loc[y])) for y in same) if same else 10.)
        best=np.flatnonzero(np.isclose(scores,max(scores),rtol=0,atol=1e-12))
        return ids[int(rng.choice(best))]

    for regime in ('dry','normal','wet'):
        group=pool.index[annual.loc[eligible,'regime']==regime].tolist()
        if not group:continue
        if regime=='dry':first=min(group,key=lambda y:(annual.loc[y,'inflow_mean_m3s'],y))
        elif regime=='wet':first=max(group,key=lambda y:(annual.loc[y,'inflow_mean_m3s'],-y))
        else:first=min(group,key=lambda y:(abs(annual.loc[y,'inflow_mean_m3s']-pool.inflow_mean_m3s.median()),y))
        selected.add(first)
        quota=min(ceil(minimum_fraction*len(group)),len(group)-1 if len(group)>1 else 1)
        while len(selected.intersection(group))<quota:selected.add(choose(set(group)-selected))
    total_rows=len(frame);total_pairs={v:int(frame[v+'_m3' if v=='storage' else 'release_m3s'].notna().sum()) for v in ('storage','release')}

    def sufficient():
        part=annual.loc[sorted(selected)]
        return (len(selected)>=ceil(minimum_fraction*len(eligible)) and part.rows.sum()>=ceil(minimum_fraction*total_rows)
                and all(part[v+'_pairs'].sum()>=ceil(minimum_fraction*n) for v,n in total_pairs.items()))
    while not sufficient():
        remaining=set(eligible)-selected
        if len(remaining)<=1:raise ValueError('Cannot reach 65% coverage while retaining assessment; check short/incomplete records or missing observations')
        # Keep an assessment year in every represented regime where feasible.
        preferred={y for y in remaining if sum(annual.loc[z,'regime']==annual.loc[y,'regime'] for z in remaining)>1}
        selected.add(choose(preferred or remaining))
    if len(selected)==len(eligible):raise ValueError('Independent assessment would be empty')
    annual['role']='context';annual.loc[list(warmup),'role']='warmup'
    annual.loc[eligible,'role']='assessment';annual.loc[sorted(selected),'role']='calibration'
    selection=YearSelection(tuple(d.date().isoformat() for d in frame.index),tuple(annual.loc[int(y),'role'] for y in labels),
                            tuple(int(y) for y in labels),tuple(annual.loc[int(y),'regime'] for y in labels),float(minimum_fraction),int(seed),int(water_year_start_month),int(warmup_years))
    warnings=[]
    for regime in ('dry','normal','wet'):
        for role in ('calibration','assessment'):
            if not ((annual.regime==regime)&(annual.role==role)).any():warnings.append(f'{role} has no {regime} year; no artificial regime labels were created')
    return selection,annual,warnings


@dataclass(frozen=True)
class SelectedYearBounds:
    empirical: CalibrationBounds
    selection_signature: str
    initialization_kind: str = 'empirical_generalized'
    generalized_targets: object = None
    @property
    def lower_vector(self):return self.empirical.lower_vector
    @property
    def upper_vector(self):return self.empirical.upper_vector
    def validate(self,vector,template):return self.empirical.validate(vector,template)


def fit_selected_years(observations,selection,physical_constants,*,quantile_method='hazen',allow_feasible_initialization=False):
    mask=selection.mask(observations.frame.index,'calibration')
    fitting=target_observations(_observations(observations.frame.loc[mask],observations.source))
    estimates=estimate_generalized_parameters(fitting,channel_capacity_m3s=physical_constants['channel_capacity_m3s'],quantile_method=quantile_method)
    initialization_kind='empirical_generalized'
    try:template=ReservoirParameters(estimates.monthly_targets,**physical_constants)
    except ValueError:
        if not allow_feasible_initialization:raise
        # A temporary carrier validates constants only; it is never simulated.
        capacity=physical_constants['capacity_m3'];dead=capacity*physical_constants['dead_storage_fraction']
        carrier=np.tile([dead+(capacity-dead)*.25,dead+(capacity-dead)*.5,dead+(capacity-dead)*.75,0.,0.,0.],(12,1))
        template=ReservoirParameters(pd.DataFrame(carrier,index=pd.Index(range(1,13),name='month'),columns=estimates.monthly_targets.columns),**physical_constants)
        initialization_kind='feasible_bounds_midpoint; empirical_generalized_targets_invalid'
    probabilities=sorted({p for pair in DEFAULT_PERCENTILE_BOUNDS for p in pair})
    quantiles=estimate_percentiles(fitting,probabilities,quantile_method=quantile_method).frame.set_index(['variable','month','percentile'])
    monthly=pd.DataFrame(index=pd.Index(range(1,13),name='month'))
    for variable,stem,suffix in (('storage_m3','storage','m3'),('release_m3s','release','m3s')):
        for level,pair in zip(('critical','normal','flood'),DEFAULT_PERCENTILE_BOUNDS):
            for side,p in zip(('lower','upper'),pair):monthly[f'{side}_{stem}_{level}_{suffix}']=[float(quantiles.loc[(variable,m,p),'value']) for m in range(1,13)]
    envelope=DatePeriod(fitting.frame.index[0],fitting.frame.index[-1])
    empirical=CalibrationBounds(monthly,template.capacity_m3,template.dead_storage_fraction,envelope,quantile_method,source=fitting.source)
    if initialization_kind!='empirical_generalized':
        lower=empirical.monthly_bounds.iloc[:,:6].to_numpy();upper=empirical.monthly_bounds.iloc[:,6:].to_numpy()
        values=lower+(upper-lower)/2
        for row in range(12):
            preceding=template.capacity_m3*template.dead_storage_fraction
            for col in range(3):
                lo=max(lower[row,col],np.nextafter(preceding,np.inf));hi=min(upper[row,col],template.capacity_m3)
                if lo>hi:raise ValueError('No feasible initialization in empirical storage bounds')
                values[row,col]=lo+(hi-lo)/2;preceding=values[row,col]
        template=ReservoirParameters(pd.DataFrame(values,index=pd.Index(range(1,13),name='month'),columns=estimates.monthly_targets.columns),**physical_constants)
    return template,SelectedYearBounds(empirical,selection.signature,initialization_kind,estimates.monthly_targets.copy())


class RepresentativeYearObjective:
    """Continuous causal state; sparse year masks and equal-regime NSE costs."""
    def __init__(self,observations,selection,template,*,sampling,bounds=None,policy='reject',atol_storage_m3=1e-4,cost_mode='regime_balanced_nse'):
        if cost_mode not in ('regime_balanced_nse','pooled_nse'):raise ValueError('Unknown cost mode')
        validate_sampling_index(observations.frame.index,sampling.frequency)
        selection.mask(observations.frame.index,'calibration')
        if bounds is not None and bounds.selection_signature!=selection.signature:raise ValueError('Bounds belong to a different year selection')
        self._source=_observations(observations.frame,observations.source)
        self.selection,self.template,self.sampling,self.bounds=selection,replace(template),sampling,bounds
        self.root_policy,self.atol_storage_m3,self.cost_mode=policy,atol_storage_m3,cost_mode
        mask=selection.mask(self._source.frame.index,'calibration')
        self.calibration_period=DatePeriod(self._source.frame.index[mask][0],self._source.frame.index[mask][-1])
        self.identity={'strategy':'representative_years_v1','selection_sha256':selection.signature,'cost_mode':cost_mode,'root_policy':policy,'atol_storage_m3':atol_storage_m3}

    def __call__(self,vector):return np.array(self.evaluate(vector).objectives)
    def evaluate(self,vector):return self._evaluate(vector,'calibration')
    def evaluate_validation(self,vector):return self._evaluate(vector,'assessment')

    def _evaluate(self,vector,role):
        frame=self._source.frame;mask=self.selection.mask(frame.index,role)
        dates=frame.index[mask];period=DatePeriod(dates[0],dates[-1])
        observed=_observations(frame.loc[mask])
        counts={v:int(observed.frame[col].notna().sum()) for v,col in (('storage','storage_m3'),('release','release_m3s'))}
        parameters=simulation=None
        try:
            parameters=parameters_from_vector(vector,self.template) if self.bounds is None else self.bounds.validate(vector,self.template)
            # Future assessment forcing cannot filter calibration candidates.
            end=dates[-1] if role=='calibration' else frame.index[-1]
            continuous=_observations(frame.loc[:end])
            simulation=simulate_root_policy_reservoir(continuous,parameters,sampling=self.sampling,policy=self.root_policy,atol_storage_m3=self.atol_storage_m3)
            if simulation.frame.storage_m3.lt(0).any() or simulation.frame.balance_residual_m3.abs().max()>self.atol_storage_m3:raise ValueError('Physical or mass-balance gate failed')
            scored=replace(simulation,frame=simulation.frame.loc[dates])
            metrics=evaluate_simulation(observed,scored)
            labels=self.selection.table().loc[dates,'regime']
            group_scores={v:[] for v in ('release','storage')}
            for regime in sorted(set(labels)):
                sub=labels==regime
                values=evaluate_simulation(_observations(observed.frame.loc[sub]),replace(scored,frame=scored.frame.loc[sub]))
                for v in group_scores:
                    value=values[v+'_nse'];metrics[regime+'_'+v+'_nse']=value;group_scores[v].append(value)
            for v,values in group_scores.items():metrics['regime_balanced_'+v+'_nse']=float(np.mean(values))
            key='regime_balanced_' if self.cost_mode=='regime_balanced_nse' else ''
            costs=tuple(-metrics[key+v+'_nse'] for v in ('release','storage'))
            if not np.isfinite(costs).all():raise ValueError('NSE is undefined in an active scoring regime; inspect observations')
        except (TypeError,ValueError) as error:
            return CandidateEvaluation(False,(float('inf'),float('inf')),{},counts,str(error),parameters,simulation,period)
        return CandidateEvaluation(True,costs,metrics,counts,None,parameters,simulation,period)
