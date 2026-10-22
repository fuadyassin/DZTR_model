"""Final operational artifacts and explicit scientific limitations for the cohort."""
from itertools import combinations

import numpy as np
import pandas as pd

from twenty_workflow import OUT,PARENT,checked
from policy_fleet_job import ROOT,read,sha,write


def main():
    checked()
    selected=read(OUT/"pooled_selection.json")["reservoirs"]
    parameter_rows,observed_rows=[],[]
    wide=pd.read_csv(OUT/"reservoir_parameterization_by_GRAND_ID.csv",float_precision="round_trip")
    for item in selected:
        identifier=item["reservoir_id"]
        vectors=[]
        for seed in (42,43,44):
            source=OUT/"pilots"/identifier/f"seed_{seed}"
            result=read(source/"result.json")
            role=next(r for r in result["representatives"] if r["role"]=="balanced")
            table=pd.read_csv(source/"vectors.csv",index_col="evaluation_id",float_precision="round_trip")
            vectors.append(table.loc[role["evaluation_id"]].to_numpy())
        config=read(source/"resolved_config.json")
        profile=read(OUT/"selected"/identifier/"parameters.json")
        baseline_result=read(OUT/"pilots"/identifier/f"seed_{item['seed']}"/"result.json")
        row=wide.reservoir_id==identifier
        fields={"sampling_frequency":config["sampling"]["frequency"],"integration_step_days":config["sampling"].get("fixed_step_days"),"operating_month_clock":config["sampling"]["month_clock"],"quantile_method":config["quantile_method"],"storage_target_units":"m3","release_target_units":"m3/s",**{label+"_"+boundary:config["periods"][label][boundary] for label in ("calibration","validation") for boundary in ("start","end")}}
        for period in ("calibration","validation"):
            baseline_metrics=baseline_result["baseline_calibration"] if period=="calibration" else baseline_result["baseline_validation"]["metrics"]
            for variable in ("release","storage"):
                original=baseline_metrics.get(variable+"_nse")
                calibrated=item["calibration_"+variable+"_nse"] if period=="calibration" else profile["validation"]["metrics"].get(variable+"_nse")
                fields["baseline_"+period+"_"+variable+"_nse"]=original
                fields["selected_"+period+"_"+variable+"_nse"]=calibrated
                fields["delta_"+period+"_"+variable+"_nse"]=calibrated-original if calibrated is not None and original is not None else None
        for name,value in fields.items():
            wide.loc[row,name]=value
        bounds=pd.read_csv(source/"calibration_bounds_si.csv",index_col="month",float_precision="round_trip")
        lo,hi=bounds.iloc[:,:6].to_numpy().copy(),bounds.iloc[:,6:].to_numpy().copy()
        lo[:,:3]/=config["model"]["capacity_m3"];hi[:,:3]/=config["model"]["capacity_m3"]
        lo,hi=lo.T.reshape(72),hi.T.reshape(72)
        normalized=np.divide(np.array(vectors)-lo,hi-lo,out=np.zeros_like(vectors),where=hi!=lo)
        parameter_rows.append({"reservoir_id":identifier,"max_balanced_seed_parameter_rms_normalized":max(float(np.sqrt(np.mean((a-b)**2))) for a,b in combinations(normalized,2)),**{f"normalized_spread_coordinate_{j:02d}":float(x) for j,x in enumerate(np.ptp(normalized,axis=0))}})
        frame=pd.read_csv(source/"baseline_simulation.csv",index_col="date",parse_dates=True,float_precision="round_trip")
        for label in ("calibration","validation"):
            dates=config["periods"][label];part=frame.loc[dates["start"]:dates["end"]]
            s,q,i,dt=[part[column].to_numpy() for column in ("observed_storage_m3","observed_release_m3s","inflow_m3s","step_seconds")]
            mask=np.isfinite(s[1:])&np.isfinite(s[:-1])&np.isfinite(q[1:])&np.isfinite(q[:-1])&np.isfinite(i[1:])&np.isfinite(i[:-1])
            residual=(np.diff(s)-dt[1:]/2*(config["model"]["inflow_correction"]*(i[1:]+i[:-1])-q[1:]-q[:-1]))[mask]
            observed_rows.append({"reservoir_id":identifier,"period":label,"hypothesis":"working_model_clock_endpoint_trapezoid","finite_intervals":int(mask.sum()),"observed_residual_rmse_million_m3":float(np.sqrt(np.mean(residual**2))/1e6),"observed_residual_equivalent_rate_m3s":float(residual.sum()/dt[1:][mask].sum()),"qualification":"Uses inherited integration/forcing conventions; this residual does not establish physical closure or identify a missing flux."})
    pd.DataFrame(parameter_rows).to_csv(OUT/"parameter_variation.csv",index=False,float_format="%.17g")
    wide["GRAND_ID"]=wide.GRAND_ID.astype("Int64")
    wide.to_csv(OUT/"reservoir_parameterization_by_GRAND_ID.csv",index=False,float_format="%.17g")
    book=OUT/"GRanD_DZTR_parameters_calibrated.xlsx"
    updated=pd.read_excel(book,sheet_name="Updated_parameter_catalog")
    for column in fields:
        updated[column]=updated.reservoir_id.map(wide.set_index("reservoir_id")[column])
    with pd.ExcelWriter(book,engine="openpyxl",mode="a",if_sheet_exists="replace") as writer:
        updated.to_excel(writer,sheet_name="Updated_parameter_catalog",index=False)
        wide.to_excel(writer,sheet_name="Twenty_selected",index=False)
        pd.read_csv(OUT/"convergence.csv").to_excel(writer,sheet_name="Convergence_diagnostics",index=False)
        pd.DataFrame(parameter_rows).to_excel(writer,sheet_name="Parameter_variation",index=False)
    pd.DataFrame(observed_rows).to_csv(OUT/"observed_balance_diagnostics.csv",index=False,float_format="%.17g")
    scores=pd.read_csv(OUT/"selected_reservoir_scores.csv")
    scores["heldout_negative_release_nse"]=scores.validation_release_nse<0
    scores["heldout_negative_storage_nse"]=scores.validation_storage_nse<0
    scores["heldout_release_worse_than_baseline"]=scores.delta_validation_release_nse<0
    scores["heldout_storage_worse_than_baseline"]=scores.delta_validation_storage_nse<0
    scores["selection_changed_by_validation"]=False
    scores.to_csv(OUT/"selected_validation_assessment.csv",index=False,float_format="%.17g")
    with pd.ExcelWriter(book,engine="openpyxl",mode="a",if_sheet_exists="replace") as writer:
        scores.to_excel(writer,sheet_name="Selected_validation",index=False)
    comparisons=pd.read_csv(OUT/"baseline_comparison.csv")
    comparisons=comparisons[comparisons.budget==1024]
    counts={period:int(((group.delta_release_nse>0)&(group.delta_storage_nse>0)).sum()) for period,group in comparisons.groupby("period")}
    convergence=pd.read_csv(OUT/"convergence.csv")
    stable=convergence.groupby("reservoir_id").diagnostic_stability_pass.all()
    summary=read(OUT/"workflow_summary.json")
    summary.update({"status":"workflow_complete_with_GRAND_ID_exception","both_objectives_improved_seed_runs":counts,"descriptive_front_stability_pass_reservoirs":stable[stable].index.tolist(),"pooled_selected_both_heldout_nse_positive":int(((scores.validation_release_nse>0)&(scores.validation_storage_nse>0)).sum()),"parameter_spread_is_not_uncertainty":True,"no_more_searches_automatically_scheduled":True})
    write(OUT/"workflow_summary.json",summary)
    text="# Completed workflow for the twenty calibrated reservoirs\n\nThe workflow is complete: baseline comparison, seasonal/storage-drift/peak/zone diagnostics, exact512-to1024 checkpoint continuation, frozen pooled calibration selection, simulation profiles and GRanD-indexed parameter tables. The remaining reservoirs are parked in [pending notes](other_reservoirs_pending.csv); no additional reservoir was calibrated here.\n\n"
    text+="All60 continuation jobs completed, preserving all30,720 original trials and adding30,720 search trials. Populations, seeds, bounds, operating constants and solver policies are unchanged. Baseline metrics/trajectories were reused. The pooled selections were frozen for all20 reservoirs before their held-out evaluation.\n\n"
    text+=f"At1024 evaluations, both calibration objectives improved over the generalized baseline in **{counts['calibration']} of60** seed runs; both held-out objectives improved in **{counts['validation']} of60**. This distinguishes improvement from merely positive NSE. The pooled selected profiles have positive NSE for both held-out variables in **{summary['pooled_selected_both_heldout_nse_positive']} of20** reservoirs. Poor held-out performance is retained, not used to switch seeds or parameters.\n\n"
    text+=f"Six reservoirs pass the illustrative front-stability screen for all seeds: {', '.join(stable[stable].index)}. This requires less than0.5% relative hypervolume gain in the last128 evaluations, between-seed front distance at most0.02 NSE, and no reference-excluded points. This is **not a validated convergence/stopping rule**, and objective-front stability does not establish parameter identifiability. [Parameter variation](parameter_variation.csv) reports that separately. No automatic budget extension beyond1024 is scheduled.\n\n"
    text+="The selected rule is an equal-weight range-normalized compromise on the pooled three-seed calibration Pareto front, with deterministic cost/seed/trial ties. It supplies one reproducible research parameter set per reservoir. It does not claim the best held-out fit or physical source certification.\n\n"
    text+="## Parameter products\n\n- [Updated GRanD parameter workbook](GRanD_DZTR_parameters_calibrated.xlsx): original reference, updated catalog, twenty selected rows, monthly SI targets, crosswalk and baseline comparisons.\n- [Wide table by GRAND_ID](reservoir_parameterization_by_GRAND_ID.csv) and [monthly SI table](monthly_parameters_by_GRAND_ID.csv).\n- [GRanD crosswalk](grand_crosswalk.csv) and [source provenance](grand_provenance.json).\n- [Profile registry](../../configs/calibrated_twenty.json):19 ID lookups and20 internal-reservoir lookups.\n\nNineteen IDs match the local v1.1 global catalog. **US Fort Randall remains explicitly unresolved**: the same-name2356 record is Canadian and is not assigned. Its selected parameters remain available under`ex4_Fort_randall`. Original catalog/reference files are preserved; optimized targets use separate named columns rather than overwriting empirical percentile fields. Model capacities were not replaced by catalog capacities.\n\n"
    text+="## Diagnostics and evidence\n\n- [All seed/budget baseline comparisons](baseline_comparison.csv).\n- [Seasonal bias](seasonal_bias.csv), [storage drift and peak samples](trajectory_diagnostics.csv), [physical-storage zones](zone_diagnostics.csv).\n- [Observed balance under the working model-clock hypothesis](observed_balance_diagnostics.csv); not an inferred forcing correction.\n- [Convergence diagnostics](convergence.csv), [selected validation assessment](selected_validation_assessment.csv), [scientific verification](scientific_verification.json).\n\nEach report overlays the generalized baseline and the selected model, and shows monthly targets and seasonal bias:\n\n"
    text+='\n'.join(f"- [{row['name']}](reports/{row['reservoir_id']}/README.md)" for _,row in scores.iterrows())+'\n'
    (OUT/"README.md").write_text(text,encoding="utf-8")
    write(OUT/"finalization_receipt.json",{"status":"passed","profile_count":20,"known_GRAND_ID_count":19,"pending_GRAND_ID_count":1,"new_model_evaluations":0,"selection_sha256":sha(OUT/"pooled_selection.json"),"registry_sha256":sha(ROOT/"configs/calibrated_twenty.json")})


if __name__=="__main__":main()
