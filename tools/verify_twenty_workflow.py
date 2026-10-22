"""Independent continuation, selection, score, balance and catalog checks."""
import numpy as np
import pandas as pd

from twenty_workflow import OUT,PARENT,checked
from policy_fleet_job import ROOT,read,sha,write
from verify_policy_fleet_results import balance,nse,nondominated_ids
from dztr.calibration.checkpoint import load_optimizer_checkpoint


def main():
    checked()
    jobs=read(OUT/"execution_summary.json")["jobs"]
    assert len(jobs)==60 and all(j["status"]=="completed" for j in jobs)
    maximum=0.;rows=0
    for job in jobs:
        identifier,seed=job["reservoir_id"],job["seed"]
        folder=OUT/"pilots"/identifier/f"seed_{seed}"
        for name,digest in read(folder/"fleet_completion.json")["artifact_sha256"].items(): assert sha(folder/name)==digest
        state=load_optimizer_checkpoint(folder/"checkpoint.npz",read(folder/"checkpoint_identity.json"))
        source=PARENT/"pilots"/identifier/f"seed_{seed}"
        parent=load_optimizer_checkpoint(source/"checkpoint.npz",read(source/"checkpoint_identity.json"))
        assert len(state.evaluated)==1024
        np.testing.assert_array_equal([p.vector for p in state.evaluated[:512]],[p.vector for p in parent.evaluated])
        np.testing.assert_array_equal([p.objectives for p in state.evaluated[:512]],[p.objectives for p in parent.evaluated])
        trials=pd.read_csv(folder/"trials.csv",index_col="evaluation_id",float_precision="round_trip")
        vectors=pd.read_csv(folder/"vectors.csv",index_col="evaluation_id",float_precision="round_trip")
        np.testing.assert_array_equal(vectors,[p.vector for p in state.evaluated])
        np.testing.assert_array_equal(trials[["release_cost","storage_cost"]],[p.objectives for p in state.evaluated])
        feasible=trials[trials.feasible]
        front=pd.read_csv(folder/"pareto_front.csv",index_col="evaluation_id",float_precision="round_trip")
        assert set(front.index)==nondominated_ids(feasible[["release_cost","storage_cost"]].to_numpy(),feasible.index)
        config,result=read(folder/"resolved_config.json"),read(folder/"result.json")
        for role in result["representatives"]:
            frame=pd.read_csv(folder/f"selected_{role['evaluation_id']}_simulation.csv",index_col="date",parse_dates=True,float_precision="round_trip")
            error=balance(frame,config["model"]["inflow_correction"])
            np.testing.assert_array_equal(error,frame.balance_residual_m3.to_numpy()[1:])
            assert np.abs(error).max()<=config["atol_storage_m3"]
            maximum=max(maximum,float(np.abs(error).max()));rows+=len(frame)
            for period in ("calibration","validation"):
                if period=="validation" and not role["validation_feasible"]: continue
                dates=config["periods"][period];data=frame.loc[dates["start"]:dates["end"]]
                for var,column in (("storage","storage_m3"),("release","release_m3s")):
                    np.testing.assert_allclose(nse(data["observed_"+column],data[column]),role[period+"_"+var+"_nse"],atol=1e-12,rtol=0)
    selection=read(OUT/"pooled_selection.json")
    assert selection["validation_used"] is False and selection["written_before_pooled_validation"]
    for item in selection["reservoirs"]:
        parts=[]
        for seed in (42,43,44):
            part=pd.read_csv(OUT/"pilots"/item["reservoir_id"]/f"seed_{seed}"/"pareto_front.csv",float_precision="round_trip")
            part["seed"]=seed;parts.append(part)
        candidates=pd.concat(parts,ignore_index=True)
        costs=candidates[["release_cost","storage_cost"]].to_numpy()
        indices=sorted(nondominated_ids(costs,candidates.index))
        front=candidates.loc[indices]
        values=front[["release_cost","storage_cost"]].to_numpy()
        width=np.ptp(values,axis=0)
        distances=np.linalg.norm(np.divide(values-values.min(axis=0),width,out=np.zeros_like(values),where=width!=0),axis=1)
        winner=min(range(len(front)),key=lambda i:(distances[i],values[i,0],values[i,1],int(front.iloc[i].seed),int(front.iloc[i].evaluation_id)))
        assert (int(front.iloc[winner].seed),int(front.iloc[winner].evaluation_id))==(item["seed"],item["evaluation_id"])
        folder=OUT/"selected"/item["reservoir_id"]
        metadata=read(folder/"parameters.json")
        frame=pd.read_csv(folder/"simulation.csv",index_col="date",parse_dates=True,float_precision="round_trip")
        error=balance(frame,metadata["physical_constants"]["inflow_correction"])
        maximum=max(maximum,float(np.abs(error).max()))
        assert np.abs(error).max()<=1e-4
        source=OUT/"pilots"/item["reservoir_id"]/f"seed_{item['seed']}"
        vector=pd.read_csv(source/"vectors.csv",index_col="evaluation_id",float_precision="round_trip").loc[item["evaluation_id"]].to_numpy().reshape(6,12).T.copy()
        vector[:,:3]*=metadata["physical_constants"]["capacity_m3"]
        table=pd.read_csv(folder/"monthly_targets_si.csv",index_col="month",float_precision="round_trip")
        np.testing.assert_array_equal(table,vector)
        for label in ("calibration","validation"):
            if label=="validation" and not metadata["validation"]["feasible"]: continue
            dates=metadata["periods"][label];data=frame.loc[dates["start"]:dates["end"]]
            for var,column in (("storage","storage_m3"),("release","release_m3s")):
                expected=item["calibration_"+var+"_nse"] if label=="calibration" else metadata["validation"]["metrics"][var+"_nse"]
                np.testing.assert_allclose(nse(data["observed_"+column],data[column]),expected,rtol=0,atol=1e-12)
    catalog=pd.read_csv(OUT/"reservoir_parameterization_by_GRAND_ID.csv",float_precision="round_trip")
    assert len(catalog)==20 and catalog.GRAND_ID.notna().sum()==19 and catalog.GRAND_ID.dropna().is_unique
    assert pd.isna(catalog.set_index("reservoir_id").loc["ex4_Fort_randall","GRAND_ID"])
    reference=pd.read_excel(ROOT/"Extended_GRanD_Dams_v1_1_DZTR.xlsx")
    original=pd.read_excel(OUT/"GRanD_DZTR_parameters_calibrated.xlsx",sheet_name="Original_reference")
    pd.testing.assert_frame_equal(reference,original)
    selected=pd.read_excel(OUT/"GRanD_DZTR_parameters_calibrated.xlsx",sheet_name="Twenty_selected")
    assert len(selected)==20 and set(selected.reservoir_id)==set(catalog.reservoir_id)
    write(OUT/"scientific_verification.json",{"status":"passed","continuations_verified":60,"prefix_trials_preserved":30720,"new_search_trials":30720,"total_archived_trials":61440,"selected_reservoirs":20,"role_trajectory_rows_checked":rows,"maximum_balance_residual_m3":maximum,"GRAND_ID_verified":19,"GRAND_ID_unresolved":1,"historical_reference_sheet_unchanged":True,"verification_model_evaluations":0,"verifier_sha256":sha(__import__('pathlib').Path(__file__))})
    print(f"Verified60 continuations and20 selected profiles; max balance residual {maximum} m3;19 IDs verified, Fort Randall unresolved.")


if __name__=="__main__": main()
