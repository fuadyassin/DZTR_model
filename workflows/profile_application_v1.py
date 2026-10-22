"""Load frozen paper profiles and dispatch forcing to their approved solver.

Raw canonical and prepared forcing are explicitly different input stages.
Source-gated and conditional interval records cannot become free-running native
profiles through this interface. Archived equations and registries stay intact.
"""
from dataclasses import replace
from numbers import Integral
from pathlib import Path
import numpy as np
import pandas as pd

from workflows import reservoir_v2 as w
from dztr.calibration.sampled import _observations
from dztr.simulation.sampled import SamplingSettings, validate_sampling_index
from dztr.simulation.root_policy import simulate_root_policy_reservoir
from dztr.simulation.signed_net import SignedNetPolicy, simulate_signed_net_reservoir

ROOT=w.ROOT


def reference(value):
    path=(ROOT/str(value).replace("\\","/")).resolve(strict=True)
    if not path.is_relative_to(ROOT): raise ValueError("Catalog reference escapes repository")
    return path


def application_sources():
    return {**w.source_hashes(),**{name:w.sha(ROOT/name) for name in (
        "workflows/profile_application_v1.py","examples/simulate_profile.py")}}


def read_catalog(path):
    path=Path(path).resolve(strict=True)
    catalog=w.read(path)
    manifest=w.read(path.parent/"manifest.json")
    if manifest["artifact_sha256"].get(path.name)!=w.sha(path):
        raise ValueError("Catalog checksum changed")
    if catalog.get("kind")!="paper_profile_application_v1": raise ValueError("Unknown catalog schema")
    for rel,digest in catalog["application_source_sha256"].items():
        if w.sha(reference(rel))!=digest: raise ValueError("Application/numerical source changed: "+rel)
    return catalog


def resolve_entry(catalog,*,reservoir_id=None,grand_id=None):
    if (reservoir_id is None)==(grand_id is None): raise ValueError("Select exactly one reservoir ID or GRAND ID")
    rows=catalog["reservoirs"]
    if grand_id is not None:
        if isinstance(grand_id,bool) or not isinstance(grand_id,Integral) or grand_id<=0:
            raise ValueError("GRAND ID must be a positive integer")
        found=[r for r in rows if r.get("GRAND_ID")==int(grand_id)]
    else:
        found=[r for r in rows if r["reservoir_id"]==reservoir_id]
    if len(found)!=1: raise ValueError("No unique verified ID mapping in this paper catalog")
    return found[0]


def load_profile(catalog_path,*,reservoir_id=None,grand_id=None):
    catalog=read_catalog(catalog_path)
    entry=resolve_entry(catalog,reservoir_id=reservoir_id,grand_id=grand_id)
    if not entry["monthly_profile_available"]:
        raise ValueError(entry["reservoir_id"]+": "+entry["status"]+"; "+entry["next_action"])
    for rel,digest in entry["input_sha256"].items():
        if w.sha(reference(rel))!=digest: raise ValueError("Profile input changed: "+rel)
    metadata,parameters=w.load_profile(reference(entry["registry"]),identifier=entry["reservoir_id"])
    if metadata.get("route",entry["route"])!=entry["route"] and entry["route"]!="standard_native_v1":
        raise ValueError("Catalog route disagrees with frozen profile")
    return entry,metadata,parameters


def normalize_forcing(frame,*,stage,factor,offset):
    """Return a copy; the caller declares whether preprocessing is already done."""
    if stage not in ("canonical","prepared"): raise ValueError("Declare canonical or prepared forcing")
    if not isinstance(frame,pd.DataFrame) or "inflow_m3s" not in frame or frame.columns.has_duplicates:
        raise ValueError("Forcing needs one inflow_m3s column in SI")
    if not np.isfinite([factor,offset]).all() or factor<=0: raise ValueError("Invalid preprocessing")
    copied=frame.copy(deep=True)
    values=copied.inflow_m3s.to_numpy(dtype=float,copy=True)
    if not np.isfinite(values).all(): raise ValueError("Forcing must be finite; missing values are not filled")
    if stage=="canonical": values=values*factor+offset
    copied["inflow_m3s"]=values
    return copied


def run_frame(frame,metadata,parameters,*,forcing_stage,preprocessing,initial=None):
    frame=normalize_forcing(frame,stage=forcing_stage,factor=preprocessing["factor"],offset=preprocessing["offset_m3s"])
    sampling=SamplingSettings(**metadata["sampling"])
    validate_sampling_index(frame.index,sampling.frequency)
    if initial is None:
        if frame.index[0].date().isoformat()!=metadata["initial_date"]:
            raise ValueError("A new forcing start requires explicit initial storage_m3 and release_m3s")
    else:
        if set(initial)!={"storage_m3","release_m3s"} or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in initial.values()):
            raise ValueError("Declare finite SI initial storage_m3 and release_m3s")
        parameters=replace(parameters,initial_storage_m3=initial["storage_m3"],initial_release_m3s=initial["release_m3s"])
    if sampling.month_clock=="legacy" and frame.index[0].date().isoformat()!=metadata["initial_date"]:
        raise ValueError("A different legacy clock requires a separate profile")
    # Only forcing is needed for application. Responses cannot initialize state.
    observations=_observations(pd.DataFrame({"inflow_m3s":frame.inflow_m3s,
        "storage_m3":np.nan,"release_m3s":np.nan},index=frame.index))
    options=dict(sampling=sampling,policy=metadata["root_policy"],atol_storage_m3=metadata["atol_storage_m3"])
    if metadata.get("route")=="signed_net_native_v1":
        result=simulate_signed_net_reservoir(observations,parameters,
            forcing_policy=SignedNetPolicy(**metadata["forcing_policy"]),**options)
    else:
        result=simulate_root_policy_reservoir(observations,parameters,**options)
    residual=w.independent_balance(result.frame,parameters.inflow_correction)
    if residual>metadata["atol_storage_m3"] or not result.frame.storage_m3.between(0,parameters.capacity_m3).all():
        raise ValueError("Independent physical/balance check failed")
    return result,residual


def simulate(catalog_path,output,*,reservoir_id=None,grand_id=None,forcing_csv=None,forcing_stage=None,initial=None):
    entry,metadata,parameters=load_profile(catalog_path,reservoir_id=reservoir_id,grand_id=grand_id)
    if forcing_csv is None:
        if forcing_stage not in (None,"prepared"): raise ValueError("Archived default forcing is already prepared")
        path=reference(entry["prepared_forcing"]); stage="prepared"
    else:
        if forcing_stage is None: raise ValueError("Explicit forcing requires --forcing-stage canonical or prepared")
        path=Path(forcing_csv).resolve(strict=True); stage=forcing_stage
    frame=pd.read_csv(path,index_col="date",parse_dates=True,float_precision="round_trip")
    result,residual=run_frame(frame,metadata,parameters,forcing_stage=stage,preprocessing=entry["inflow_preprocessing"],initial=initial)
    out=w.fresh_output(output)
    result.frame.to_csv(out/"simulation.csv",index_label="date",float_format="%.17g")
    result.parameters.monthly_targets.to_csv(out/"monthly_targets_si.csv",index_label="month",float_format="%.17g")
    receipt={"status":"passed","reservoir_id":entry["reservoir_id"],"GRAND_ID":entry.get("GRAND_ID"),
        "route":entry["route"],"registry":entry["registry"],"forcing_file":str(path),"forcing_sha256":w.sha(path),
        "forcing_stage":stage,"source_transform_applied_by_this_call":stage=="canonical",
        "source_inflow_preprocessing":entry["inflow_preprocessing"],"physical_inflow_correction":parameters.inflow_correction,
        "explicit_initial_state":initial,"observations_used_for_initialization":False,
        "maximum_independent_balance_residual_m3":residual,"new_optimizer_evaluations":0,
        "source_certified":False,"catalog_sha256":w.sha(catalog_path)}
    w.write(out/"simulation_receipt.json",receipt)
    w.write(out/"manifest.json",{"artifact_sha256":{p.name:w.sha(p) for p in out.iterdir() if p.is_file()}})
    return out
