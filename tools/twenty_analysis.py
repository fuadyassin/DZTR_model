"""Complete calibration comparisons, diagnostics, pooled selection and ID tables."""
import argparse
from itertools import combinations
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from twenty_workflow import OUT, PARENT, objective_for, checked
from policy_fleet_job import ROOT, read, sha, write, safe_validation, trajectory
from verify_policy_fleet_results import nondominated_ids, nse, balance
from dztr.calibration.checkpoint import load_optimizer_checkpoint
from dztr.calibration.convergence import hypervolume_2d
from dztr.schemas import MONTHLY_TARGET_COLUMNS

GRAND_SOURCE = ROOT/"data/catalogs/grand_v1_1_reference.csv"
GRAND_IDS = read(ROOT/"configs/calibrated_twenty_grand_ids.json")["ids"]
GRAND_NAMES = {597:"Glen Canyon",310:"Grand Coulee",753:"Garrison Dam",870:"Oahe Dam",1307:"International Amistad Dam",253:"Gardiner",355:"Yellowtail",5146:"Bhumibol",4478:"High Aswan Dam",5140:"Sirikit",281:"St. Mary",270:"Travers",284:"Waterton",278:"Oldman River Dam",251:"Dickson Dam",710:"E.B. Campbell",148:"Oroville",4683:"Andizhan",4692:"Nurek"}


def pooled_selection(points):
    """Choose one balanced point using calibration costs only, across seeds."""
    frame = points.sort_values(["seed", "evaluation_id"]).reset_index(drop=True)
    costs = frame[["release_cost", "storage_cost"]].to_numpy()
    keep = sorted(nondominated_ids(costs, frame.index))
    front = frame.loc[keep].copy()
    values = front[["release_cost", "storage_cost"]].to_numpy()
    width = np.ptp(values, axis=0)
    distances = np.linalg.norm(np.divide(values-values.min(axis=0), width, out=np.zeros_like(values), where=width!=0), axis=1)
    index = min(range(len(front)), key=lambda i:(distances[i], values[i,0], values[i,1], int(front.iloc[i].seed), int(front.iloc[i].evaluation_id)))
    return front.iloc[index], front


def select_all():
    checked()
    if (OUT / "pooled_selection.json").exists():
        raise ValueError("Selection is already frozen.")
    summary = read(OUT / "execution_summary.json")
    if summary["status"] != "completed" or len(summary["jobs"]) != 60:
        raise ValueError("All sixty continuations must complete before selection.")
    chosen = []
    for identifier in sorted(GRAND_IDS):
        parts = []
        for seed in (42,43,44):
            folder = OUT / "pilots" / identifier / f"seed_{seed}"
            frame = pd.read_csv(folder / "pareto_front.csv", float_precision="round_trip")
            frame["seed"] = seed
            parts.append(frame)
        point, front = pooled_selection(pd.concat(parts, ignore_index=True))
        folder = OUT / "pooled_fronts"
        folder.mkdir(exist_ok=True)
        front.to_csv(folder / f"{identifier}.csv", index=False, float_format="%.17g")
        chosen.append({"reservoir_id":identifier,"seed":int(point.seed),"evaluation_id":int(point.evaluation_id),"calibration_release_nse":float(-point.release_cost),"calibration_storage_nse":float(-point.storage_cost)})
    # Freeze every reservoir's identity before any pooled-choice validation.
    write(OUT / "pooled_selection.json", {"rule_file_sha256":sha(OUT / "selection_rule.json"),"validation_used":False,"written_before_pooled_validation":True,"reservoirs":chosen})


def assess_selected():
    selected = read(OUT / "pooled_selection.json")
    if (OUT / "selected").exists():
        raise ValueError("Existing selected simulations are preserved.")
    (OUT / "selected").mkdir()
    for item in selected["reservoirs"]:
        source = OUT / "pilots" / item["reservoir_id"] / f"seed_{item['seed']}"
        state = load_optimizer_checkpoint(source / "checkpoint.npz", read(source / "checkpoint_identity.json"))
        objective, bounds, observations, split = objective_for(source, state)
        vector = state.evaluated[item["evaluation_id"]].vector
        calibration = objective.evaluate(vector)
        assert calibration.feasible
        np.testing.assert_array_equal(calibration.objectives, (-item["calibration_release_nse"],-item["calibration_storage_nse"]))
        validation, report = safe_validation(objective, vector)
        folder = OUT / "selected" / item["reservoir_id"]
        folder.mkdir()
        trajectory(folder / "simulation.csv", validation if validation is not None and validation.feasible else calibration, observations, split)
        targets = vector.reshape(6,12).T.copy()
        config = read(source / "resolved_config.json")
        targets[:,:3] *= config["model"]["capacity_m3"]
        pd.DataFrame(targets, columns=MONTHLY_TARGET_COLUMNS,index=pd.Index(range(1,13),name="month")).to_csv(folder / "monthly_targets_si.csv",float_format="%.17g")
        write(folder / "parameters.json", {**item,"grand_id":GRAND_IDS[item["reservoir_id"]],"physical_constants":config["model"],"sampling":config["sampling"],"periods":config["periods"],"root_policy":config["declared_root_policy"],"monthly_targets_file":"monthly_targets_si.csv","validation":report,"selection_uses_validation":False,"convergence_established":False,"physical_source_certified":False})
    write(OUT / "pooled_validation_receipt.json", {"selection_sha256":sha(OUT / "pooled_selection.json"),"calibration_replays":20,"validation_attempts":20})


def pair_metrics(frame, period):
    scored = frame.loc[period["start"]:period["end"]]
    answer = {}
    for label,column in (("storage","storage_m3"),("release","release_m3s")):
        observed, simulated = scored["observed_"+column],scored[column]
        mask = observed.notna() & simulated.notna()
        errors = simulated[mask]-observed[mask]
        answer[label+"_nse"] = nse(observed,simulated) if int(mask.sum())>=2 and np.ptp(observed[mask].to_numpy())>0 else None
        answer[label+"_mean_bias"] = float(errors.mean())
        answer[label+"_rmse"] = float(np.sqrt(np.mean(errors**2)))
    return answer


def diagnostic_rows(identifier, seed, budget, kind, frame, config, peak_threshold):
    seasonal, zones, whole = [], [], []
    for period, dates in config["periods"].items():
        if period not in ("calibration","validation"):
            continue
        data = frame.loc[dates["start"]:dates["end"]]
        if not len(data):
            continue
        common = {"reservoir_id":identifier,"seed":seed,"budget":budget,"parameter_set":kind,"period":period}
        for month, rows in data.groupby(data.index.month):
            seasonal.append({**common,"month":month,"samples":len(rows),**pair_metrics(rows,{"start":rows.index.min(),"end":rows.index.max()})})
        for zone, rows in data.groupby("zone"):
            zones.append({**common,"physical_storage_zone":int(zone),"samples":len(rows),**pair_metrics(rows,{"start":rows.index.min(),"end":rows.index.max()})})
        errors = data.storage_m3-data.observed_storage_m3
        valid = errors.notna()
        years = (data.index[valid]-data.index[valid][0]).total_seconds()/86400/365.25
        drift = float(np.polyfit(years,errors[valid].to_numpy(),1)[0]/1e6) if len(years)>1 else None
        observed_peak, model_peak = float(data.observed_release_m3s.max()),float(data.release_m3s.max())
        high = data.observed_release_m3s >= peak_threshold
        whole.append({**common,"storage_error_trend_million_m3_per_year":drift,"storage_error_endpoint_change_million_m3":float((errors[valid].iloc[-1]-errors[valid].iloc[0])/1e6),"observed_peak_sample_release_m3s":observed_peak,"simulated_peak_sample_release_m3s":model_peak,"peak_sample_ratio":model_peak/observed_peak if observed_peak else None,"calibration_p95_release_threshold_m3s":peak_threshold,"high_release_samples":int(high.sum()),"high_release_mean_bias_m3s":float((data.release_m3s-data.observed_release_m3s)[high].mean()) if high.any() else None,"peak_sample_date_offset_days":int((data.release_m3s.idxmax()-data.observed_release_m3s.idxmax()).days),**pair_metrics(data,dates)})
    return seasonal,zones,whole


def finish():
    checked()
    selected = read(OUT / "pooled_selection.json")["reservoirs"]
    comparisons, seasons, zones, diagnostics, convergence = [],[],[],[],[]
    for identifier in sorted(GRAND_IDS):
        fronts = []
        for seed in (42,43,44):
            for budget, experiment in ((512,PARENT),(1024,OUT)):
                folder = experiment / "pilots" / identifier / f"seed_{seed}"
                result,config = read(folder/"result.json"),read(folder/"resolved_config.json")
                balanced = next(r for r in result["representatives"] if r["role"]=="balanced")
                base = pd.read_csv(folder/"baseline_simulation.csv",index_col="date",parse_dates=True,float_precision="round_trip")
                cal = config["periods"]["calibration"]
                threshold = float(np.percentile(base.loc[cal["start"]:cal["end"],"observed_release_m3s"].dropna(),95,method=config["quantile_method"]))
                frame = pd.read_csv(folder/f"selected_{balanced['evaluation_id']}_simulation.csv",index_col="date",parse_dates=True,float_precision="round_trip")
                for period in ("calibration","validation"):
                    baseline = result["baseline_calibration"] if period=="calibration" else result["baseline_validation"]["metrics"]
                    comparisons.append({"reservoir_id":identifier,"seed":seed,"budget":budget,"period":period,**{f"baseline_{v}_nse":baseline.get(v+"_nse") for v in ("release","storage")},**{f"calibrated_{v}_nse":balanced.get(period+"_"+v+"_nse") for v in ("release","storage")},**{f"delta_{v}_nse":balanced[period+"_"+v+"_nse"]-baseline[v+"_nse"] if balanced.get(period+"_"+v+"_nse") is not None else None for v in ("release","storage")}})
                for kind, table in (("generalized_baseline",base),("seed_balanced",frame)):
                    a,b,c = diagnostic_rows(identifier,seed,budget,kind,table,config,threshold)
                    seasons.extend(a);zones.extend(b);diagnostics.extend(c)
            trials = pd.read_csv(folder/"trials.csv",float_precision="round_trip")
            feasible = trials[trials.feasible]
            values = feasible[["release_cost","storage_cost"]].to_numpy()
            front = pd.read_csv(folder/"pareto_front.csv",float_precision="round_trip")
            fronts.append(front[["release_cost","storage_cost"]].to_numpy())
            hv = {n:hypervolume_2d(feasible[feasible.evaluation_id<n][["release_cost","storage_cost"]].to_numpy()) for n in (512,896,1024)}
            convergence.append({"reservoir_id":identifier,"seed":seed,"hv512":hv[512],"hv896":hv[896],"hv1024":hv[1024],"gain512_to1024":hv[1024]-hv[512],"relative_gain_last128":(hv[1024]-hv[896])/max(abs(hv[1024]),1e-12),"reference_excluded_front_points":int((front[["release_cost","storage_cost"]]>1).any(axis=1).sum())})
        distance = max(max(float(np.linalg.norm(a[:,None]-b[None,:],axis=2).min(axis=0).max()),float(np.linalg.norm(a[:,None]-b[None,:],axis=2).min(axis=1).max())) for a,b in combinations(fronts,2))
        for row in convergence:
            if row["reservoir_id"]==identifier:
                row["max_between_seed_front_distance_nse"]=distance
                row["diagnostic_stability_pass"] = row["relative_gain_last128"]<.005 and distance<=.02 and row["reference_excluded_front_points"]==0
        pooled = next(r for r in selected if r["reservoir_id"]==identifier)
        chosen = pd.read_csv(OUT/"selected"/identifier/"simulation.csv",index_col="date",parse_dates=True,float_precision="round_trip")
        a,b,c = diagnostic_rows(identifier,pooled["seed"],1024,"pooled_selected",chosen,config,threshold)
        seasons.extend(a);zones.extend(b);diagnostics.extend(c)
    for name, rows in (("baseline_comparison",comparisons),("seasonal_bias",seasons),("zone_diagnostics",zones),("trajectory_diagnostics",diagnostics),("convergence",convergence)):
        pd.DataFrame(rows).to_csv(OUT/(name+".csv"),index=False,float_format="%.17g")
    catalog = pd.read_csv(GRAND_SOURCE,float_precision="round_trip")
    mapping,monthly,wide = [],[],[]
    for item in selected:
        identifier = item["reservoir_id"]
        grand = GRAND_IDS[identifier]
        folder=OUT/"selected"/identifier
        parameters=read(folder/"parameters.json")
        table=pd.read_csv(folder/"monthly_targets_si.csv",index_col="month",float_precision="round_trip")
        name=read(OUT/"pilots"/identifier/f"seed_{item['seed']}"/"resolved_config.json")["reservoir"]["name"]
        if grand is not None:
            matches=catalog[catalog.GRAND_ID==grand]
            assert len(matches)==1 and matches.iloc[0].DAM_NAME==GRAND_NAMES[grand]
            metadata=matches.iloc[0][["DAM_NAME","RES_NAME","COUNTRY","LONG_DD","LAT_DD","CAP_MCM"]].to_dict()
        else:
            metadata={"DAM_NAME":name,"COUNTRY":"United States"}
        mapping.append({"reservoir_id":identifier,"GRAND_ID":grand,"match_status":"verified_local_v1_1_named_record" if grand is not None else "unresolved_country_conflict_same_name_2356",**metadata})
        row={"GRAND_ID":grand,"reservoir_id":identifier,"reservoir_name":name,"selected_seed":item["seed"],"selected_evaluation_id":item["evaluation_id"],"root_policy":parameters["root_policy"],"parameter_status":"exploratory_calibration_selected","grand_id_status":mapping[-1]["match_status"],**parameters["physical_constants"]}
        for month, values in table.iterrows():
            monthly.append({"GRAND_ID":grand,"reservoir_id":identifier,"month":month,**values.to_dict()})
            for column,value in values.items():
                row[f"{column}_month{month:02d}"]=value
        wide.append(row)
    cross=pd.DataFrame(mapping);cross["GRAND_ID"]=cross.GRAND_ID.astype("Int64");cross.to_csv(OUT/"grand_crosswalk.csv",index=False)
    wide=pd.DataFrame(wide);wide["GRAND_ID"]=wide.GRAND_ID.astype("Int64");wide.to_csv(OUT/"reservoir_parameterization_by_GRAND_ID.csv",index=False,float_format="%.17g")
    monthly=pd.DataFrame(monthly);monthly["GRAND_ID"]=monthly.GRAND_ID.astype("Int64");monthly.to_csv(OUT/"monthly_parameters_by_GRAND_ID.csv",index=False,float_format="%.17g")
    reference=pd.read_excel(ROOT/"Extended_GRanD_Dams_v1_1_DZTR.xlsx")
    updated=reference.copy()
    # Optimized targets are separate from empirical Q10/S10 etc. Preserve their semantics.
    missing=[column for column in wide.columns if column not in updated]
    updated=pd.concat([updated,pd.DataFrame(None,index=updated.index,columns=missing)],axis=1)
    for _,row in wide.iterrows():
        matches=updated.index[updated.GRAND_ID==row.GRAND_ID] if pd.notna(row.GRAND_ID) else []
        if len(matches)==1:
            for column in wide.columns:
                if column!="GRAND_ID": updated.loc[matches[0],column]=row[column]
        elif len(matches)==0:
            updated=pd.concat([updated,pd.DataFrame([row])],ignore_index=True)
        else: raise ValueError("Nonunique reference GRanD ID.")
    with pd.ExcelWriter(OUT/"GRanD_DZTR_parameters_calibrated.xlsx",engine="openpyxl") as writer:
        updated.to_excel(writer,sheet_name="Updated_parameter_catalog",index=False)
        wide.to_excel(writer,sheet_name="Twenty_selected",index=False)
        monthly.to_excel(writer,sheet_name="Monthly_targets_SI",index=False)
        cross.to_excel(writer,sheet_name="GRanD_crosswalk",index=False)
        pd.DataFrame(comparisons).to_excel(writer,sheet_name="Baseline_comparison",index=False)
        reference.to_excel(writer,sheet_name="Original_reference",index=False)
    pending=pd.read_csv(PARENT/"reservoir_results.csv");pending=pending[pending.completed_jobs==0]
    pending.to_csv(OUT/"other_reservoirs_pending.csv",index=False)
    write(OUT/"grand_provenance.json",{"source":str(GRAND_SOURCE),"source_sha256":sha(GRAND_SOURCE),"reference_sha256":sha(ROOT/"Extended_GRanD_Dams_v1_1_DZTR.xlsx"),"verified_ids":19,"unresolved_reservoir":"ex4_Fort_randall","note":"Canadian GRAND_ID2356 is not assigned to the US Fort Randall. No model capacity/units were changed to agree with catalog values."})
    write(OUT/"workflow_summary.json",{"status":"analysis_complete","reservoirs":20,"jobs":60,"total_budget_per_job":1024,"new_search_evaluations":30720,"selection_uses_validation":False,"pooled_calibration_replays":20,"pooled_validation_attempts":20,"grand_ids_verified":19,"grand_ids_unresolved":1,"diagnostic_stability_rule":{"relative_hv_gain_last128_below":.005,"between_seed_front_distance_at_most":.02,"reference_exclusions_required":0,"qualification":"Illustrative descriptive screen, not a validated stopping rule or physical/parameter convergence certificate."},"diagnostic_stability_pass_jobs":sum(r["diagnostic_stability_pass"] for r in convergence)})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("select","assess","finish"))
    stage=parser.parse_args().stage
    {"select":select_all,"assess":assess_selected,"finish":finish}[stage]()
