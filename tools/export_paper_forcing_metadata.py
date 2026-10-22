"""Add explicit source inflow preprocessing to a versioned paper parameter export.

Copies metadata only. Never transforms observations, changes calibrated targets,
or admits an unresolved source profile to calibration.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))
from workflows import reservoir_v2 as w


def export(source,output):
    source=Path(source).resolve(strict=True)
    table=pd.read_csv(source,float_precision="round_trip")
    if len(table)!=37 or not table.reservoir_id.is_unique:
        raise ValueError("Expected one row for each of the 37 paper reservoirs")
    original=table.copy(deep=True); inputs={str(source.relative_to(ROOT)):w.sha(source)}
    additions={"inflow_preprocessing_factor","inflow_preprocessing_offset_m3s",
        "inflow_preprocessing_config","inflow_preprocessing_note"}
    if additions.intersection(table.columns):
        raise ValueError("Preprocessing metadata already exists; avoid duplicate application")
    for index,row in table.iterrows():
        path=ROOT/"configs/reservoirs"/(row.reservoir_id+".json")
        cfg=w.read(path); inputs[str(path.relative_to(ROOT))]=w.sha(path)
        spec=cfg["inflow_preprocessing"]
        factor,offset=spec["factor"],spec["offset_m3s"]
        if not np.isfinite([factor,offset]).all() or factor<=0:
            raise ValueError("Invalid declared source preprocessing")
        table.loc[index,"inflow_preprocessing_factor"]=factor
        table.loc[index,"inflow_preprocessing_offset_m3s"]=offset
        table.loc[index,"inflow_preprocessing_config"]=path.relative_to(ROOT).as_posix()
        table.loc[index,"inflow_preprocessing_note"]=(
            "Inherited MATLAB multiplier; physical justification unverified; Bennett remains source-gated."
            if row.reservoir_id=="ex15_Bennett" else
            "Copied source-config metadata; distinct from physical inflow_correction; do not reapply to prepared forcing.")
    pd.testing.assert_frame_equal(table[original.columns],original)
    out=w.fresh_output(output)
    target=out/"reservoir_parameterization_all_37_by_GRAND_ID.csv"
    table.to_csv(target,index=False,float_format="%.17g")
    restored=pd.read_csv(target,float_precision="round_trip")
    pd.testing.assert_frame_equal(restored[original.columns],original)
    row=restored.loc[restored.reservoir_id.eq("ex15_Bennett")].iloc[0]
    if row.inflow_preprocessing_factor!=1.45 or row.parameter_values_available:
        raise ValueError("Bennett assumption or admission status changed unexpectedly")
    monthly=[name for name in original if "_month_" in name]
    receipt={"status":"passed","paper_reservoirs":37,
        "monthly_profiles_available":int(restored.parameter_values_available.sum()),
        "existing_export_columns_exactly_preserved":True,
        "monthly_target_columns_preserved":len(monthly),
        "Bennett_inflow_preprocessing_factor":1.45,
        "Bennett_preprocessing_physical_justification_verified":False,
        "Bennett_calibration_admission_changed":False,
        "physical_inflow_correction_changed":False,
        "observations_transformed":False,"new_model_calls":0,"new_optimizer_evaluations":0,
        "input_sha256":inputs,"export_script_sha256":w.sha(Path(__file__))}
    w.write(out/"verification.json",receipt)
    (out/"README.md").write_text(
        "# Paper parameter export with source inflow preprocessing\n\n"
        "[All-37 parameterization table](reservoir_parameterization_all_37_by_GRAND_ID.csv) "
        "retains the 32 completed monthly profiles, Bull Lake's separate interval route and four source-gated cases.\n\n"
        "Bennett's original 1.45 inflow multiplier is now explicit in `inflow_preprocessing_factor`, "
        "with zero additive offset and its source configuration. This records a historical assumption; "
        "its physical justification remains unverified. Bennett retains its source gate and blank calibrated targets.\n\n"
        "The source preprocessing is separate from the physical `inflow_correction` parameter. "
        "Apply preprocessing once when creating model inputs; prepared forcing already includes its declared transform. "
        "This export copies metadata only and does not transform any observations or change parameters.\n\n"
        "Every existing export column was compared exactly after CSV round-trip. "
        "No simulation, optimization or numerical-code change was needed.\n",encoding="utf-8")
    w.write(out/"manifest.json",{"artifact_sha256":{p.name:w.sha(p) for p in out.iterdir() if p.is_file()}})
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-export",required=True); parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(export(args.source_export,args.output))
