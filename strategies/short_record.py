"""Explicit six-calendar-year protocol; no response-fit or timing changes."""
import numpy as np
import pandas as pd
from strategies.representative_years import YearSelection
from dztr.simulation.sampled import validate_sampling_index


def select_short_record(observations, sampling, year_selection='mixed', seed=42):
    if year_selection not in ('mixed','chronological'):raise ValueError('Choose mixed or chronological year selection')
    frame=observations.frame;validate_sampling_index(frame.index,sampling.frequency)
    if not np.isfinite(frame.inflow_m3s).all() or frame.inflow_m3s.lt(0).any():raise ValueError('Short-record classification requires nonnegative finite model inflows')
    years=sorted(set(frame.index.year))
    if len(years)!=6:raise ValueError('short_record_v1 requires exactly six complete calendar years')
    rows=[]
    for year in years:
        part=frame[frame.index.year==year]
        expected=pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='D' if sampling.frequency=='daily' else 'MS')
        if not part.index.equals(expected.rename(part.index.name)):raise ValueError('Short-record years must be complete; no filling or partial-year scoring')
        weights=np.ones(len(part)) if sampling.frequency=='daily' else part.index.days_in_month
        rows.append({'year':year,'rows':len(part),'inflow_mean_m3s':float(np.average(part.inflow_m3s,weights=weights))})
    annual=pd.DataFrame(rows).set_index('year');eligible=years[1:]
    lower,upper=np.quantile(annual.loc[eligible,'inflow_mean_m3s'],[1/3,2/3])
    annual['regime']='unclassified'
    annual.loc[eligible,'regime']=np.where(annual.loc[eligible,'inflow_mean_m3s']<lower,'dry',np.where(annual.loc[eligible,'inflow_mean_m3s']>upper,'wet','normal'))
    if set(annual.loc[eligible,'regime'])!={'dry','normal','wet'}:raise ValueError('Short-record protocol requires three genuine inflow regimes; no fabricated categories')
    candidates=[years[-1]] if year_selection=='chronological' else list(reversed(eligible))
    held=None
    for candidate in candidates:
        selected=[y for y in eligible if y!=candidate]
        if set(annual.loc[selected,'regime'])!={'dry','normal','wet'}:continue
        mask=frame.index.year.isin(selected)
        if mask.mean()<.65:continue
        if any(frame.loc[mask,col].notna().sum()<np.ceil(.65*frame[col].notna().sum()) for col in ('storage_m3','release_m3s')):continue
        held=candidate;break
    if held is None:raise ValueError('Cannot retain all calibration regimes, 65% support and a separate whole assessment year')
    roles=tuple('warmup' if date.year==years[0] else 'assessment' if date.year==held else 'calibration' for date in frame.index)
    plan=YearSelection(tuple(d.date().isoformat() for d in frame.index),roles,tuple(int(y) for y in frame.index.year),tuple(annual.loc[d.year,'regime'] for d in frame.index),.65,seed,1,1)
    selected_mask=plan.mask(frame.index,'calibration')
    labels=np.asarray(plan.regimes)
    undefined=[]
    for regime in ('dry','normal','wet'):
        for variable,column in [('storage','storage_m3'),('release','release_m3s')]:
            values=frame.loc[selected_mask & (labels==regime),column].dropna().to_numpy()
            if len(values)<2 or np.ptp(values)==0:undefined.append(regime+'_'+variable)
    constant_years={variable:[int(y) for y in eligible if frame.loc[frame.index.year==y,column].dropna().nunique()==1]
                    for variable,column in [('storage','storage_m3'),('release','release_m3s')]}
    metadata={'kind':'short_record_v1','assessment_years':[held],'calibration_years':[y for y in eligible if y!=held],
        'calibration_fraction':roles.count('calibration')/len(roles),'missing_assessment_regimes':sorted({'dry','normal','wet'}-{annual.loc[held,'regime']}),
        'interpretation':'retrospective sparse-year assessment; not matched chronological/mixed comparison or prospective forecast validation',
        'selection_uses_response_values':False,'selection_rule':'latest eligible whole year preserving all three calibration inflow regimes and >=65% rows/pairs',
        'undefined_calibration_regime_nse':undefined,'constant_observation_years':constant_years,
        'source_review_required':any(constant_years.values()),'diagnostics_change_year_selection':False}
    return plan,metadata
