"""Explicit provider parsing and a reversible historical storage coordinate."""
from html.parser import HTMLParser
from pathlib import Path
import numpy as np
import pandas as pd
from workflows import extension_assessment_v1 as d


class _Table(HTMLParser):
    def __init__(self):super().__init__();self.rows=[];self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='tr':self.row=[]
        if tag in ('th','td'):self.cell=[]
    def handle_data(self,value):
        if self.cell is not None:self.cell.append(value)
    def handle_endtag(self,tag):
        if tag in ('th','td') and self.cell is not None:self.row.append(''.join(self.cell).strip());self.cell=None
        if tag=='tr' and self.row:self.rows.append(self.row);self.row=None


def read_provider(path):
    parser=_Table();parser.feed(Path(path).read_text(encoding='utf-8-sig'))
    if not parser.rows or parser.rows[0]!=['DateTime','min_af','min_id','min_qt','min_fb']:raise ValueError('Unexpected provider fields')
    frame=pd.DataFrame(parser.rows[1:],columns=parser.rows[0]).set_index('DateTime')
    frame.index=pd.DatetimeIndex(pd.to_datetime(frame.index),name='date')
    for col in frame:frame[col]=pd.to_numeric(frame[col].replace('',np.nan),errors='raise')
    if not frame.index.equals(pd.date_range(frame.index[0],frame.index[-1],name='date')) or np.isinf(frame.to_numpy()).any():raise ValueError('Invalid provider calendar/data')
    return frame


def normalize(frame,offset_af=115000.):
    """Keep signed computed ID, missing values and original discharge unchanged.

    The offset is external historical metadata, never inferred from extrema or
    assessment performance. It changes the storage coordinate, not dS/dt.
    """
    if not np.isfinite(offset_af) or offset_af<0:raise ValueError('Declare a finite nonnegative reference offset')
    result=pd.DataFrame({'inflow_m3s':frame.min_id*d.CFS_M3S,
        'storage_m3':(frame.min_af+offset_af)*d.ACRE_FOOT_M3,
        'release_m3s':frame.min_qt*d.CFS_M3S},index=frame.index)
    if (result.storage_m3.dropna()<0).any() or (result.release_m3s.dropna()<0).any():raise ValueError('Gross storage and release require physical source review')
    return result
