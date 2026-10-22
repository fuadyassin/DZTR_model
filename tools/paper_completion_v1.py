"""Prepare, verify, run or report the paper-completion source routes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import pandas as pd
from workflows import reservoir_v2 as w
from workflows import paper_completion_v1 as p
from dztr.calibration.candidates import parameters_to_vector
from dztr.calibration.optimizer import NSGA2Settings, optimize_nsga2
from dztr.calibration.checkpoint import load_optimizer_checkpoint, save_optimizer_checkpoint
from strategies.strategy_comparison import pooled_choice


def calibrate(output, identifier, seed, budget=512):
    out,proof = p.check_protocol(output)
    if seed not in (42,43,44) or isinstance(budget,bool) or budget < 512 or budget % 32:
        raise ValueError("Use declared seeds and full populations at budgets >= 512")
    objective = p.signed_objective(out,identifier)
    folder = out/"jobs"/identifier/f"seed_{seed}"; folder.mkdir(parents=True,exist_ok=True)
    lock = folder/"calibration.lock"
    with lock.open("x",encoding="utf-8") as stream: stream.write(str(os.getpid()))
    try:
        identity = {"kind":"paper_completion_signed_v1","protocol_sha256":w.sha(out/"protocol.json"),
            "reservoir_id":identifier,"seed":seed,"population":32,"objective":objective.identity,
            "runtime":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__}}
        checkpoint = folder/"checkpoint.npz"; snapshot = folder/f"budget_{budget}"
        state = None
        if checkpoint.exists():
            if w.read(folder/"identity.json") != identity: raise ValueError("Checkpoint identity changed")
            state = load_optimizer_checkpoint(checkpoint,identity)
            if budget < len(state.evaluated): raise ValueError("Budget cannot shrink")
        if (snapshot/"manifest.json").exists():
            p.verified_artifacts(snapshot,w.read(snapshot/"manifest.json")["artifact_sha256"])
            if state is None or len(state.evaluated) < budget: raise ValueError("Missing durable checkpoint")
            return snapshot
        w.write(folder/"identity.json",identity)
        def save(current):
            save_optimizer_checkpoint(checkpoint,current,identity)
            w.write(folder/"progress.json",{"evaluations":len(current.evaluated),"generation":current.generation})
            print(f"{identifier} seed {seed}: {len(current.evaluated)} evaluations",flush=True)
        settings = NSGA2Settings(seed=seed,population_size=32,evaluation_budget=budget)
        result = optimize_nsga2(objective,parameters_to_vector(objective.template),settings,
                               bounds=objective.bounds,resume_state=state,checkpoint_callback=save)
        if result.evaluations != budget: raise ValueError("Incomplete fixed budget")
        snapshot.mkdir(exist_ok=True)
        pd.DataFrame([{ "evaluation_id":v.evaluation_id,"generation":v.generation,"feasible":v.feasible,
            "release_cost":v.objectives[0],"storage_cost":v.objectives[1],"reason":v.reason}
            for v in result.evaluated]).to_csv(snapshot/"trials.csv",index=False,float_format="%.17g")
        pd.DataFrame([v.vector for v in result.evaluated],index=[v.evaluation_id for v in result.evaluated]).to_csv(
            snapshot/"vectors.csv",index_label="evaluation_id",float_format="%.17g")
        pd.DataFrame([{"evaluation_id":v.evaluation_id,"release_cost":v.objectives[0],"storage_cost":v.objectives[1]}
            for v in result.front],columns=["evaluation_id","release_cost","storage_cost"]).to_csv(snapshot/"front.csv",index=False,float_format="%.17g")
        pd.DataFrame([asdict(v) for v in result.history]).to_csv(snapshot/"history.csv",index=False,float_format="%.17g")
        w.write(snapshot/"result.json",{"status":"completed","evaluations":result.evaluations,"selection_uses_assessment":False})
        w.write(snapshot/"manifest.json",{"artifact_sha256":{f.name:w.sha(f) for f in snapshot.iterdir() if f.is_file()}})
        return snapshot
    finally:
        lock.unlink(missing_ok=True)


def run(output,budget=512):
    out,_ = p.check_protocol(output)
    ready = [v["reservoir_id"] for v in w.read(out/"preparation.json")["signed_profiles"] if v["calibration_ready"]]
    (out/"logs").mkdir(exist_ok=True)
    lock = out/"study.lock"
    with lock.open("x",encoding="utf-8") as stream: stream.write(str(os.getpid()))
    try:
        def job(identifier,seed):
            log = out/"logs"/f"{identifier}_seed_{seed}_budget_{budget}.log"
            with log.open("a",encoding="utf-8") as stream:
                subprocess.run([sys.executable,"-B",str(Path(__file__)),"calibrate","--output",str(out),
                    "--reservoir",identifier,"--seed",str(seed),"--budget",str(budget)],cwd=ROOT,
                    stdout=stream,stderr=subprocess.STDOUT,check=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        for identifier in ready:
            w.write(out/"execution.json",{"status":"running","reservoir_id":identifier,"budget":budget})
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(job,identifier,seed) for seed in (42,43,44)]
                for future in futures: future.result()
        w.write(out/"execution.json",{"status":"searches_completed","budget":budget,"reservoirs":ready})
        publish(out,budget)
    except Exception as error:
        w.write(out/"execution.json",{"status":"failed","reason":str(error),"budget":budget})
        raise
    finally:
        lock.unlink(missing_ok=True)


def publish(output,budget=512):
    """Freeze calibration selection before any held-out call; reuse final releases."""
    out,_ = p.check_protocol(output); release = out/"releases"/f"budget_{budget}"
    if (release/"manifest.json").exists():
        w.verify_registry(release/"registry.json"); return release
    ready = [v["reservoir_id"] for v in w.read(out/"preparation.json")["signed_profiles"] if v["calibration_ready"]]
    choices = []
    for identifier in ready:
        parts = []
        for seed in (42,43,44):
            folder = out/"jobs"/identifier/f"seed_{seed}"/f"budget_{budget}"
            p.verified_artifacts(folder,w.read(folder/"manifest.json")["artifact_sha256"])
            state = load_optimizer_checkpoint(folder.parent/"checkpoint.npz",w.read(folder.parent/"identity.json"))
            trials = pd.read_csv(folder/"trials.csv",float_precision="round_trip")
            vectors = pd.read_csv(folder/"vectors.csv",index_col="evaluation_id",float_precision="round_trip")
            if len(trials) != budget or len(state.evaluated) < budget: raise ValueError("Incomplete snapshot")
            np.testing.assert_array_equal(vectors.to_numpy(),np.array([v.vector for v in state.evaluated[:budget]]))
            np.testing.assert_array_equal(trials[["release_cost","storage_cost"]].to_numpy(),np.array([v.objectives for v in state.evaluated[:budget]]))
            front = pd.read_csv(folder/"front.csv",float_precision="round_trip"); front["seed"] = seed; parts.append(front)
        point,_ = pooled_choice(parts)
        choices.append({"reservoir_id":identifier,"seed":int(point.seed),"evaluation_id":int(point.evaluation_id),
            "release_cost":float(point.release_cost),"storage_cost":float(point.storage_cost)})
    release.mkdir(parents=True,exist_ok=True)
    selection_path = release/"selection.json"
    if selection_path.exists() and w.read(selection_path) != choices: raise ValueError("Frozen choice changed")
    w.write(selection_path,choices); selection_sha = w.sha(selection_path)
    profiles = []; scores = []; seed_scores = []
    for choice in choices:
        identifier = choice["reservoir_id"]; objective = p.signed_objective(out,identifier)
        table = pd.read_csv(out/"jobs"/identifier/f'seed_{choice["seed"]}'/f"budget_{budget}"/"vectors.csv",index_col="evaluation_id",float_precision="round_trip")
        vector = table.loc[choice["evaluation_id"]].to_numpy()
        cal = objective.evaluate(vector)
        if not cal.feasible: raise ValueError("Frozen calibration replay failed")
        np.testing.assert_allclose(cal.objectives,[choice["release_cost"],choice["storage_cost"]],rtol=0,atol=1e-10)
        held = objective.evaluate_validation(vector)
        baseline = objective.evaluate(parameters_to_vector(objective.template))
        baseline_held = objective.evaluate_validation(parameters_to_vector(objective.template))
        assessment = {"feasible":held.feasible,"metrics":held.metrics,"reason":held.reason}
        model = w.read(out/"signed_profiles"/identifier/"model.json")
        profile = {**choice,"grand_id":None,"year_selection":"mixed","scoring":"pooled",
            **model,"calibration":cal.metrics,"assessment":assessment,"evaluation_budget":budget,
            "route":"signed_net_native_v1","source_kind":"exploratory_signed_net_calibration",
            "physical_source_certified":False,"initial_date":objective._source.frame.index[0].date().isoformat(),
            "selection_sha256":objective.selection.signature}
        profiles.append((profile,cal.parameters.monthly_targets))
        scores.append({"reservoir_id":identifier,"assessment_feasible":held.feasible,
            "calibration":cal.metrics,"assessment":assessment,"fixed_initializer_calibration":baseline.metrics,
            "fixed_initializer_assessment":{"feasible":baseline_held.feasible,"metrics":baseline_held.metrics,"reason":baseline_held.reason},
            "independent_calibration_balance_m3":w.independent_balance(cal.simulation.frame,1.),
            "independent_assessment_balance_m3":w.independent_balance(held.simulation.frame,1.) if held.feasible else None})
        if held.feasible:
            folder = release/"diagnostics"/identifier; folder.mkdir(parents=True,exist_ok=True)
            held.simulation.frame.to_csv(folder/"assessment_simulation.csv",index_label="date",float_format="%.17g")
        for seed in (42,43,44):
            folder = out/"jobs"/identifier/f"seed_{seed}"/f"budget_{budget}"
            front = pd.read_csv(folder/"front.csv",float_precision="round_trip"); front["seed"] = seed
            if front.empty:
                seed_scores.append({"reservoir_id":identifier,"seed":seed,"status":"no_feasible_front"})
                continue
            representative,_ = pooled_choice([front]); evaluation_id = int(representative.evaluation_id)
            if seed == choice["seed"] and evaluation_id == choice["evaluation_id"]:
                seed_cal,seed_held = cal,held
            else:
                vectors = pd.read_csv(folder/"vectors.csv",index_col="evaluation_id",float_precision="round_trip")
                candidate = vectors.loc[evaluation_id].to_numpy()
                seed_cal,seed_held = objective.evaluate(candidate),objective.evaluate_validation(candidate)
                if not seed_cal.feasible: raise ValueError("Seed representative replay failed")
                np.testing.assert_allclose(seed_cal.objectives,[representative.release_cost,representative.storage_cost],rtol=0,atol=1e-10)
            seed_scores.append({"reservoir_id":identifier,"seed":seed,"evaluation_id":evaluation_id,
                "selection_uses_assessment":False,"assessment_feasible":seed_held.feasible,
                "assessment_reason":seed_held.reason,
                **{"calibration_"+k:v for k,v in seed_cal.metrics.items()},
                **{"assessment_"+k:v for k,v in seed_held.metrics.items()}})
    if w.sha(selection_path) != selection_sha: raise ValueError("Held-out assessment changed calibration selection")
    w.write(release/"scores.json",scores)
    pd.DataFrame(seed_scores).to_csv(release/"seed_scores.csv",index=False,float_format="%.17g")
    w.export_release(release,profiles,{"workflow":"paper_completion_signed_v1","protocol_sha256":w.sha(out/"protocol.json")})
    w.write(release/"manifest.json",{"code_sha256":p.code_hashes(),
        "artifact_sha256":{f.relative_to(release).as_posix():w.sha(f) for f in release.rglob("*") if f.is_file() and f.name != "manifest.json"}})
    w.verify_registry(release/"registry.json")
    report(out,budget)
    return release


def report(output,budget=512):
    out,_ = p.check_protocol(output); prep = w.read(out/"preparation.json")
    ready = [v for v in prep["signed_profiles"] if v["calibration_ready"]]
    lines = ["# Paper reservoir completion routes", "",
        "All 37 paper reservoirs remain represented. Existing 29 standard profiles and Bull Lake's separate interval study are preserved.", "",
        "## Existing results", "",
        f"{prep['completed_profiles_both_held_out_nse_positive']} of 29 standard profiles have positive held-out storage and release NSE. Positive scores alone do not certify source data or convergence.",
        "Bull Lake's three archived population-32 representatives are exposed through an explicit conditional interval route; there are zero held-out validation calls.", "",
        "## Signed-inflow admission", "",
        "| Reservoir | Negative records retained | Calibration support | Admission |", "| --- | ---: | ---: | --- |"]
    for v in prep["signed_profiles"]:
        lines.append(f"| {v['reservoir_id']} | {v['negative_inflow_records']} | {v['calibration_fraction']:.2%} | {v['status']} |")
    lines += ["", "The signed values enter the native endpoint balance literally. Diagnostic positive/negative parts are not measured gross inflow and loss. Flaming Gorge explicitly treats signed net forcing as a proxy in its inherited inflow-driven rule. The native clock and source derivation remain hypotheses.", "",
        "## Source profiles", ""]
    for v in prep["source_profiles"]:
        lines.append(f"- **{v['reservoir_id']}**: {v['status']}. Open gates: {'; '.join(v['gates'])}.")
    lines += ["", "Canyon Ferry's reconstructed calendar includes missing months as missing records. Release values are retained in an unverified source column, excluded from model inputs. No interpolation or simulation across the gap is performed.",
        "Trinity uses the pinned provider version and correct raw conversion. Invalid/missing storage is excluded only from scoring; original values, signs and quality flags remain available. No blanket rescaling of filled canonical storage is used.", "",
        "## Execution", "",
        f"Preparation used {prep['new_signed_preflight_calls']} fixed calibration preflight calls and zero optimizer/held-out calls."]
    release = out/"releases"/f"budget_{budget}"
    if (release/"manifest.json").exists():
        w.verify_registry(release/"registry.json")
        scores = w.read(release/"scores.json")
        lines += ["",f"Completed {len(scores)} signed-route calibrations, three seeds each, {budget} evaluations per seed.","",
            "| Reservoir | Calibration storage NSE | Calibration release NSE | Held-out storage NSE | Held-out release NSE |",
            "| --- | ---: | ---: | ---: | ---: |"]
        for v in scores:
            c,a = v["calibration"],v["assessment"]["metrics"]
            values = [c.get("storage_nse"),c.get("release_nse"),a.get("storage_nse"),a.get("release_nse")]
            lines.append("| "+v["reservoir_id"]+" | "+" | ".join("unavailable" if x is None else f"{x:.3f}" for x in values)+" |")
        inventory = pd.read_csv(out/"paper_reservoir_inventory_all_37.csv",float_precision="round_trip")
        targets = pd.read_csv(release/"reservoir_parameterization_by_GRAND_ID.csv",float_precision="round_trip").set_index("reservoir_id")
        for identifier,row in targets.iterrows():
            mask = inventory.reservoir_id.eq(identifier)
            for name,value in row.items():
                if "_month_" in name: inventory.loc[mask,name] = value
            inventory.loc[mask,"parameter_values_available"] = True
            inventory.loc[mask,"programme_status"] = "calibrated_signed_net_exploratory"
        inventory.to_csv(out/"paper_parameterization_with_signed_routes_all_37.csv",index=False,float_format="%.17g")
    elif (out/"execution.json").exists():
        lines += ["", "Job state: "+str(w.read(out/"execution.json"))]
    else:
        lines += ["",f"{len(ready)} signed reservoirs are prepared for checkpointed searches; no optimization has been started."]
    lines += ["", "[All-37 prepared inventory](paper_reservoir_inventory_all_37.csv), [29-profile performance report](standard_profile_performance.csv), [preparation and gates](preparation.json).", ""]
    (out/"README.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","check","calibrate","run","publish","report"])
    parser.add_argument("--output",required=True); parser.add_argument("--reservoir")
    parser.add_argument("--seed",type=int); parser.add_argument("--budget",type=int,default=512)
    args = parser.parse_args()
    if args.action == "prepare":
        out = p.prepare(args.output); report(out); print(out)
    elif args.action == "check":
        p.check_protocol(args.output); print("Pinned source/profile verification passed")
    elif args.action == "calibrate": calibrate(args.output,args.reservoir,args.seed,args.budget)
    elif args.action == "run": run(args.output,args.budget)
    elif args.action == "publish": publish(args.output,args.budget)
    else: report(args.output,args.budget)


if __name__ == "__main__": main()
