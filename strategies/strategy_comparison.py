"""Forcing-only paired assessment design for the two monthly pilot reservoirs."""
from math import ceil
import numpy as np
import pandas as pd
from .representative_years import YearSelection


def comparison_selections(frame,annual,seed=42):
    if not annual.complete.all() or not annual.rows.eq(12).all():
        raise ValueError('This controlled pilot requires complete native monthly years')
    eligible=annual.index[annual.role.isin(['calibration','assessment'])].tolist()
    count=ceil(.65*len(frame)/12)
    chronology=eligible[:count]
    later=annual.loc[[y for y in eligible if y>chronology[-1]]]
    dry=later.index[later.regime=='dry'].tolist();normal=later.index[later.regime=='normal'].tolist();wet=later.index[later.regime=='wet'].tolist()
    if len(dry)<2 or not normal or not wet:raise ValueError('Later record cannot supply two dry, one normal and one wet common assessment year')
    d1=min(dry,key=lambda y:(annual.loc[y,'inflow_mean_m3s'],y))
    d2=max([y for y in dry if y!=d1],key=lambda y:(annual.loc[y,'inflow_mean_m3s'],-y))
    median=float(annual.loc[normal,'inflow_mean_m3s'].median())
    n=min(normal,key=lambda y:(round(abs(annual.loc[y,'inflow_mean_m3s']-median),9),y))
    w=max(wet,key=lambda y:(annual.loc[y,'inflow_mean_m3s'],-y))
    held=sorted([d1,d2,n,w]);available=[y for y in eligible if y not in held]
    regimes=['dry','normal','wet']
    desired={r:count*sum(annual.loc[y,'regime']==r for y in eligible)/len(eligible) for r in regimes}
    quota={r:min(int(np.floor(desired[r])),sum(annual.loc[y,'regime']==r for y in available)) for r in regimes}
    while sum(quota.values())<count:
        choices=[r for r in regimes if quota[r]<sum(annual.loc[y,'regime']==r for y in available)]
        if not choices:raise ValueError('Insufficient calibration years after common holdout')
        r=max(choices,key=lambda r:(desired[r]-quota[r],-regimes.index(r)));quota[r]+=1
    features=annual.loc[available,['inflow_mean_m3s','inflow_p10_m3s','antecedent_inflow_m3s']].copy()
    features['time']=np.array(available,dtype=float)
    features=(features-features.min()).divide((features.max()-features.min()).replace(0,1))
    rng=np.random.default_rng(seed);mixed=[]
    for r in regimes:
        group=[y for y in available if annual.loc[y,'regime']==r]
        if r=='dry':first=min(group,key=lambda y:(annual.loc[y,'inflow_mean_m3s'],y))
        elif r=='wet':first=max(group,key=lambda y:(annual.loc[y,'inflow_mean_m3s'],-y))
        else:first=min(group,key=lambda y:(round(abs(annual.loc[y,'inflow_mean_m3s']-annual.loc[available,'inflow_mean_m3s'].median()),9),y))
        chosen=[first]
        while len(chosen)<quota[r]:
            remaining=sorted(set(group)-set(chosen))
            scores=[min(float(np.linalg.norm(features.loc[y]-features.loc[z])) for z in chosen) for y in remaining]
            candidates=np.flatnonzero(np.isclose(scores,max(scores),rtol=0,atol=1e-12))
            chosen.append(remaining[int(rng.choice(candidates))])
        mixed.extend(chosen)
    warm=set(annual.index[annual.role=='warmup']);plans={}
    for arm,years in [('chronological_pooled',chronology),('mixed_pooled',mixed),('mixed_balanced',mixed)]:
        roles=tuple('warmup' if d.year in warm else 'calibration' if d.year in years else 'assessment' if d.year in held else 'context' for d in frame.index)
        plans[arm]=YearSelection(tuple(d.date().isoformat() for d in frame.index),roles,tuple(int(y) for y in frame.index.year),tuple(str(annual.loc[d.year,'regime']) for d in frame.index),.65,seed,1,1)
        assert not set(years)&set(held) and len(set(years))==count
    return plans,held


def pooled_choice(parts):
    """Choose a pooled calibration compromise, without assessment columns."""
    table=pd.concat(parts,ignore_index=True)
    costs=table[['release_cost','storage_cost']].to_numpy()
    keep=np.array([not np.any(np.all(costs<=x,axis=1)&np.any(costs<x,axis=1)) for x in costs])
    front=table.loc[keep].copy();values=front[['release_cost','storage_cost']].to_numpy()
    width=np.ptp(values,axis=0)
    norm=np.divide(values-values.min(axis=0),width,out=np.zeros_like(values),where=width!=0)
    distance=np.linalg.norm(norm,axis=1)
    idx=min(range(len(front)),key=lambda i:(distance[i],*values[i],int(front.iloc[i].seed),int(front.iloc[i].evaluation_id)))
    return front.iloc[idx],front
