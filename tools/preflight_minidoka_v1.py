"""Check the declared scenario before any optimizer jobs are admitted."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows import extension_assessment_v1 as d
from strategies import interval_scalar_v1 as old,provider_interval_v1 as s
from strategies.representative_years import fit_selected_years
from dztr.calibration.sampled import _observations
from dztr.calibration.candidates import parameters_to_vector

cfg=d.read(ROOT/'configs/minidoka_provider_interval_v1_20261021.json')
frame=d.frame_read(ROOT/cfg['observations_csv']).loc[cfg['start']:cfg['end']]
anchors,eligible=s.support(frame);plan,annual=old.select_years(frame,eligible)
mask=plan.mask(frame.index,'calibration')&eligible;physical=cfg['physical_constants'].copy()
physical['channel_capacity_m3s']=float(frame.release_m3s[mask].max())
view=frame.copy();view.loc[~eligible,['storage_m3','release_m3s']]=np.nan
params,bounds=fit_selected_years(_observations(view,'provider fitting-only view'),plan,physical,quantile_method='hazen',allow_feasible_initialization=True)
runner=s.SignedNetRunner(frame.inflow_m3s,params.monthly_targets.storage_normal_m3.to_numpy())
obj=s.EndpointObjective(frame,plan,eligible,runner,params,bounds,objective_weights=cfg['objective_weights'])
result=obj(parameters_to_vector(params));report={'baseline_feasible':result.feasible,'baseline_cost':result.cost if result.feasible else None,
    'baseline_reason':result.reason,'fitting_fraction':float(mask.mean()),'initialization_kind':bounds.initialization_kind,'channel_capacity_m3s':physical['channel_capacity_m3s']}
try:
    sim=runner.run(params);report.update(full_record_feasible=True,spill_days=int((sim['spill_volume_m3']>1e-4).sum()),calibration_spill_days=int(((sim['spill_volume_m3']>1e-4)&mask).sum()))
except ValueError as exc:report.update(full_record_feasible=False,full_record_reason=str(exc))
out=ROOT/'outputs/minidoka_source_reconciliation_v1_20261021';d.write(out/'preflight_calibration_max.json',report);annual.to_csv(out/'preflight_annual_selection.csv');print(report)
