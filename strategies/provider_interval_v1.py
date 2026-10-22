"""Signed provider net flux and endpoint-release calibration semantics."""
from dataclasses import replace
from math import isfinite
import numpy as np
import pandas as pd
from strategies import interval_scalar_v1 as old
from strategies.flow_only_v1 import Evaluation,nse


class SignedNetRunner(old.ArrayRunner):
    """Literal existing interval arithmetic with signed flux retained unchanged."""
    def __init__(self,inflow,monthly_initial_storage,*,tolerance=1e-4,policy='nearest_previous_storage'):
        if not isinstance(inflow,pd.Series) or not len(inflow):raise ValueError('Supply a labelled nonempty forcing series')
        index=inflow.index
        if not index.equals(pd.date_range(index[0],index[-1],name=index.name)):raise ValueError('Keep the complete daily calendar')
        initials=np.asarray(monthly_initial_storage,float)
        if initials.shape!=(12,) or not np.isfinite(initials).all():raise ValueError('Twelve fixed storage starts required')
        if not isfinite(tolerance) or tolerance<=0 or policy not in {'reject','minimum_storage','maximum_storage','nearest_previous_storage'}:raise ValueError('Invalid solver settings')
        self.index=index.copy();self.inflow=inflow.to_numpy(float,copy=True);self.months=index.month.to_numpy()-1
        self.initials=initials.copy();self.tolerance=tolerance;self.policy=policy
        if np.isinf(self.inflow).any():raise ValueError('Infinite forcing is invalid')
        for array in (self.inflow,self.months,self.initials):array.setflags(write=False)
    def run(self,parameters,**kwargs):
        if parameters.flood_mode!='storage' or parameters.inflow_correction!=1.:raise ValueError('Signed provider net flux requires storage flood mode and unit correction')
        return super().run(parameters,**kwargs)


def support(frame,warmup_days=30):
    from workflows.extension_assessment_v1 import blocks
    if isinstance(warmup_days,bool) or not isinstance(warmup_days,int) or warmup_days<1:raise ValueError('Positive warmup required')
    runner=SignedNetRunner(frame.inflow_m3s,np.ones(12));anchors=np.zeros(len(frame),bool);eligible=anchors.copy()
    for a,b in blocks(np.isfinite(runner.inflow)):anchors[a]=True;eligible[min(a+warmup_days,b):b]=True
    return anchors,eligible


class EndpointObjective(old.Objective):
    """Daily sampled QT is scored against modeled endpoint operating release.

    Interval volume still uses the simulator's trapezoid and separate spill.
    QT is never silently reinterpreted as an observed daily mean.
    """
    def __call__(self,vector):
        try:
            parameters=self.bounds.validate(vector,self.template);sim=self.runner.run(parameters,end_position=self.end)
            if self.weights['release'] and (sim['spill_volume_m3'][self.mask]>self.runner.tolerance).any():
                raise ValueError('Instant total release cannot identify instantaneous spill from interval spill volume')
            cost=0.
            for variable,observed,predicted in [('storage','storage_m3','storage_m3'),('release','release_m3s','release_endpoint_m3s')]:
                if self.weights[variable]:cost+=self.weights[variable]*(1-nse(self.observed[observed].to_numpy()[self.mask],sim[predicted][self.mask]))
            return Evaluation(True,float(cost),parameters=parameters,balance=float(np.nanmax(np.abs(sim['balance_residual_m3']))))
        except (ValueError,OverflowError,ZeroDivisionError) as exc:return Evaluation(False,float('inf'),str(exc))
