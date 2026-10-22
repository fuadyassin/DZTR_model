"""Selected-parameter reports from saved trajectories, with baseline overlays."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from twenty_workflow import OUT,PARENT
from policy_fleet_job import ROOT,read,sha,write


def main():
    out=OUT/"reports"
    out.mkdir(exist_ok=False)
    scores=[]
    for item in read(OUT/"pooled_selection.json")["reservoirs"]:
        identifier=item["reservoir_id"]
        selected=OUT/"selected"/identifier
        parameters=read(selected/"parameters.json")
        source=OUT/"pilots"/identifier/f"seed_{item['seed']}"
        config=read(source/"resolved_config.json")
        name=config["reservoir"]["name"]
        frame=pd.read_csv(selected/"simulation.csv",index_col="date",parse_dates=True,float_precision="round_trip")
        baseline=pd.read_csv(source/"baseline_simulation.csv",index_col="date",parse_dates=True,float_precision="round_trip")
        targets=pd.read_csv(selected/"monthly_targets_si.csv",index_col="month",float_precision="round_trip")
        folder=out/identifier;folder.mkdir()
        fig,axes=plt.subplots(3,2,figsize=(13,10),layout="constrained")
        for ax,column,scale,label in ((axes[0,0],"storage_m3",1e6,"Storage (million m³)"),(axes[0,1],"release_m3s",1.,"Release (m³/s)")):
            ax.plot(frame.index,frame["observed_"+column]/scale,color=".6",lw=.6,label="Observed")
            ax.plot(baseline.index,baseline[column]/scale,color="#E69F00",lw=.6,label="Generalized baseline")
            ax.plot(frame.index,frame[column]/scale,color="#0072B2",lw=.65,label="Calibration-selected")
            ax.axvspan(pd.Timestamp(config["periods"]["validation"]["start"]),frame.index.max(),color=".7",alpha=.15,label="Held-out period")
            ax.set_ylabel(label);ax.legend(fontsize=7)
        for j,label in enumerate(("Critical","Normal","Flood")):
            axes[1,0].plot(targets.index,targets.iloc[:,j]/1e6,marker="o",label=label)
            axes[1,1].plot(targets.index,targets.iloc[:,j+3],marker="o",label=label)
        for ax,label in ((axes[1,0],"Storage targets (million m³)"),(axes[1,1],"Release targets (m³/s)")):
            ax.set(xlabel="Operating month",ylabel=label,xticks=range(1,13));ax.legend(fontsize=8)
        for ax,column,scale,label in ((axes[2,0],"storage_m3",1e6,"Monthly storage bias (million m³)"),(axes[2,1],"release_m3s",1.,"Monthly release bias (m³/s)")):
            for period,style in (("calibration","-"),("validation","--")):
                dates=config["periods"][period]
                for table,color,role in ((baseline,"#E69F00","Baseline"),(frame,"#0072B2","Selected")):
                    part=table.loc[dates["start"]:dates["end"]]
                    bias=(part[column]-part["observed_"+column]).groupby(part.index.month).mean()/scale
                    ax.plot(bias.index,bias.values,style,color=color,label=f"{role} {period}")
            ax.axhline(0,color=".6",lw=.5);ax.set(xlabel="Calendar month",ylabel=label,xticks=range(1,13));ax.legend(fontsize=7)
        for ax in axes.flat: ax.grid(alpha=.2)
        fig.suptitle(f"{name}: pooled calibration selection, seed{item['seed']}, trial{item['evaluation_id']}")
        for ext in ("png","pdf"):fig.savefig(folder/f"selected_diagnostics.{ext}",dpi=140,bbox_inches="tight")
        plt.close(fig)
        val=parameters["validation"]["metrics"]
        base=read(source/"result.json")
        scores.append({"reservoir_id":identifier,"GRAND_ID":parameters["grand_id"],"name":name,"selected_seed":item["seed"],"selected_evaluation_id":item["evaluation_id"],**{f"calibration_{v}_nse":item[f"calibration_{v}_nse"] for v in ("release","storage")},**{f"validation_{v}_nse":val.get(v+"_nse") for v in ("release","storage")},**{f"delta_validation_{v}_nse":val[v+"_nse"]-base["baseline_validation"]["metrics"][v+"_nse"] if v+"_nse" in val else None for v in ("release","storage")}})
        text=f"# {name}\n\nSelected by the frozen equal-weight pooled calibration rule: seed{item['seed']}, evaluation{item['evaluation_id']}. Validation was not used for this choice. Root policy: `{parameters['root_policy']}`.\n\nGRAND_ID: {parameters['grand_id'] if parameters['grand_id'] is not None else 'unresolved; US Fort Randall is not assigned the Canadian same-name entry'}.\n\nThe grey band denotes held-out dates. Solid seasonal-bias lines are calibration; dashed lines are validation. Peak diagnostics refer to native sample releases; monthly values are not instantaneous peaks. Source definitions remain provisional.\n\n![Selected results and baseline](selected_diagnostics.png)\n\n[PDF](selected_diagnostics.pdf), [monthly SI parameters](../../selected/{identifier}/monthly_targets_si.csv), [full parameter profile](../../selected/{identifier}/parameters.json).\n"
        (folder/"README.md").write_text(text,encoding="utf-8")
    pd.DataFrame(scores).to_csv(OUT/"selected_reservoir_scores.csv",index=False,float_format="%.17g")
    write(out/"manifest.json",{"status":"generated","reservoirs":20,"new_model_evaluations":0,"selection_sha256":sha(OUT/"pooled_selection.json"),"artifact_sha256":{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}})
    mapping={str(r["grand_id"]):str((OUT/"selected"/r["reservoir_id"]/'parameters.json').relative_to(ROOT)) for r in [read(OUT/"selected"/i["reservoir_id"]/'parameters.json') for i in read(OUT/'pooled_selection.json')["reservoirs"]] if r["grand_id"] is not None}
    write(ROOT/"configs/calibrated_twenty.json",{"kind":"calibration_selected_twenty","parameter_status":"exploratory","selection_rule":str((OUT/'selection_rule.json').relative_to(ROOT)),"profiles_by_GRAND_ID":mapping,"profiles_by_reservoir_id":{i['reservoir_id']:str((OUT/'selected'/i['reservoir_id']/'parameters.json').relative_to(ROOT)) for i in read(OUT/'pooled_selection.json')["reservoirs"]},"table":str((OUT/'reservoir_parameterization_by_GRAND_ID.csv').relative_to(ROOT)),"unresolved_GRAND_ID":["ex4_Fort_randall"]})


if __name__=="__main__":main()
