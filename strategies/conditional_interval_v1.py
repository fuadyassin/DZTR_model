"""Explicit conditional net-flux and short-record experiment contracts."""
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd

from strategies.interval_scalar_v1 import ArrayRunner
from workflows.geometry_v1 import digest


class NonnegativeNetRunner(ArrayRunner):
    """Reuse literal interval arithmetic where the computed net flux is positive.

    These are response-derived fluxes, never an independent physical inflow.
    With storage-only flood mode the release law ignores its inflow driver, so
    using the existing array path is identical to the canonical net policy.
    Signed cases require their existing signed adapter and a separate admission.
    """
    def run(self, parameters, **kwargs):
        if parameters.flood_mode != 'storage' or parameters.inflow_correction != 1.:
            raise ValueError('Computed net forcing requires storage flood mode and unit correction')
        return super().run(parameters, **kwargs)


@dataclass(frozen=True)
class ShortRecordSelection:
    dates: tuple[str, ...]
    roles: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self,'dates',tuple(self.dates));object.__setattr__(self,'roles',tuple(self.roles))
        index=pd.DatetimeIndex(pd.to_datetime(self.dates))
        if not len(index) or len(index)!=len(self.roles) or not index.equals(pd.date_range(index[0],index[-1])):
            raise ValueError('Short-record selection requires its complete daily calendar')
        if set(self.roles)-{'warmup','calibration','assessment'} or not {'calibration','assessment'}<=set(self.roles):
            raise ValueError('Declare separate calibration and assessment roles')
        if self.roles.count('calibration')/len(index)<.65:
            raise ValueError('Short-record fitting must cover at least 65% of the entire calendar')

    @property
    def signature(self):return digest(self.as_dict())

    def as_dict(self):return {'kind':'short_record_chronological_v1',**asdict(self)}

    def table(self):
        index=pd.DatetimeIndex(pd.to_datetime(self.dates),name='date')
        return pd.DataFrame({'role':self.roles,'water_year':index.year,'regime':'unclassified_short_record'},index=index)

    def mask(self,index,role):
        if not index.equals(self.table().index.rename(index.name)):raise ValueError('Selection calendar changed')
        return np.array(self.roles)==role


def short_record_selection(index, support, minimum_fraction=.65):
    """Chronological fit with all 12 months; no invented wet/dry year classes."""
    if not .65<=minimum_fraction<1:raise ValueError('Require at least 65% fitting')
    support=np.asarray(support,bool)
    if support.shape!=(len(index),):raise ValueError('Support must align')
    positions=np.flatnonzero(support);required=int(np.ceil(minimum_fraction*len(index)))
    if len(positions)-required<30:raise ValueError('Insufficient separate short-record assessment')
    roles=np.full(len(index),'warmup',dtype=object)
    roles[positions[:required]]='calibration';roles[positions[required:]]='assessment'
    if set(index[roles=='calibration'].month)!=set(range(1,13)):
        raise ValueError('Every calibration month needs support; do not invent seasonal targets')
    return ShortRecordSelection(tuple(str(d.date()) for d in index),tuple(roles))
