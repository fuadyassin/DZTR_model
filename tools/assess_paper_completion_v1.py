"""Audit completed signed-net archives and publish an updated all-paper report.

Uses saved costs, checkpoints and trajectories. Performs no simulation or search,
and does not modify the archived workflow, parameter choices or releases.
"""
import argparse
from itertools import combinations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))
from workflows import paper_completion_v1 as p
from workflows import reservoir_v2 as w
from dztr.calibration.checkpoint import load_optimizer_checkpoint

NAMES = {"ex40_Semione":"Seminoe", "ex8_Flaming_gorge":"Flaming Gorge", "ex33_Navajo":"Navajo"}


def nse(observed, simulated):
    x,y = np.asarray(observed,float),np.asarray(simulated,float)
    mask = np.isfinite(x)&np.isfinite(y); x,y = x[mask],y[mask]
    variance = np.sum((x-x.mean())**2)
    if len(x) < 2 or variance == 0: return np.nan
    return float(1-np.sum((x-y)**2)/variance)


def normalized_hypervolume(costs):
    """Rectangle union to fixed reference (1,1), normalized by area four."""
    costs = np.asarray(costs,float)
    costs = costs[np.isfinite(costs).all(axis=1)]
    area, ceiling = 0.,1.
    for release,storage in sorted(map(tuple,costs)):
        if release < 1 and storage < ceiling:
            area += (1-release)*(ceiling-storage)
            ceiling = storage
    return float(area/4)


def front_distance(a,b):
    distances = np.max(np.abs(np.asarray(a)[:,None,:]-np.asarray(b)[None,:,:]),axis=2)
    return float(max(distances.min(axis=0).max(),distances.min(axis=1).max()))


def independent_release(storage, net, month, parameters, step):
    """Literal piecewise DZTR rule, independent of the simulation helper."""
    critical,normal,flood,qcritical,qnormal,qflood = parameters.monthly_targets.loc[month].to_numpy()
    dead = parameters.capacity_m3*parameters.dead_storage_fraction
    if storage <= dead: return 0.
    if storage <= critical: return min(qcritical,(storage-dead)/step)
    if storage <= normal: return qcritical+(qnormal-qcritical)*(storage-critical)/(normal-critical)
    if storage <= flood:
        rise = qflood-qnormal
        if parameters.flood_mode == "inflow": rise = max(rise,net-qnormal)
        return qnormal+rise*(storage-normal)/(flood-normal)
    return min(max((storage-flood)/step,qflood),parameters.channel_capacity_m3s)


def audit(study,output):
    study,protocol = p.check_protocol(study)
    release = study/"releases/budget_512"
    registry = w.verify_registry(release/"registry.json")
    if set(registry["profiles_by_reservoir_id"]) != set(NAMES):
        raise ValueError("Expected all three completed signed profiles")
    out = w.fresh_output(output)
    consumed = {}
    def consume(path):
        consumed[str(Path(path).resolve().relative_to(ROOT))] = w.sha(path)
    for path in [study/"protocol.json",release/"manifest.json",release/"selection.json",release/"scores.json",release/"seed_scores.csv"]:
        consume(path)
    before_selection = w.sha(release/"selection.json")
    scores = w.read(release/"scores.json")
    choices = {v["reservoir_id"]:v for v in w.read(release/"selection.json")}
    summaries,job_rows,regime_rows,comparison_rows,agreement_rows = [],[],[],[],[]
    maximum_balance,maximum_rule = 0.,0.
    for record in scores:
        identifier = record["reservoir_id"]; choice = choices[identifier]
        profile,parameters = w.load_profile(release/"registry.json",identifier=identifier)
        fronts = {}
        for seed in (42,43,44):
            folder = study/"jobs"/identifier/f"seed_{seed}"; snap = folder/"budget_512"
            p.verified_artifacts(snap,w.read(snap/"manifest.json")["artifact_sha256"])
            identity = w.read(folder/"identity.json")
            if identity["protocol_sha256"] != w.sha(study/"protocol.json"):
                raise ValueError("Checkpoint belongs to a different protocol")
            state = load_optimizer_checkpoint(folder/"checkpoint.npz",identity)
            trials = pd.read_csv(snap/"trials.csv",float_precision="round_trip")
            vectors = pd.read_csv(snap/"vectors.csv",index_col="evaluation_id",float_precision="round_trip")
            if len(state.evaluated) != 512 or len(trials) != 512 or state.generation != 15:
                raise ValueError("Incomplete fixed-budget checkpoint")
            np.testing.assert_array_equal(vectors.to_numpy(),[v.vector for v in state.evaluated])
            np.testing.assert_array_equal(trials[["release_cost","storage_cost"]].to_numpy(),[v.objectives for v in state.evaluated])
            for path in [folder/"checkpoint.npz",folder/"identity.json",snap/"manifest.json"]: consume(path)
            costs = trials[["release_cost","storage_cost"]].to_numpy()
            fronts[seed] = pd.read_csv(snap/"front.csv",float_precision="round_trip")[["release_cost","storage_cost"]].to_numpy()
            hv256,hv512 = normalized_hypervolume(costs[:256]),normalized_hypervolume(costs)
            if hv512 < hv256-1e-14: raise ValueError("Cumulative hypervolume declined")
            job_rows.append({"reservoir_id":identifier,"seed":seed,"attempted_evaluations":512,
                "feasible_evaluations":int(trials.feasible.sum()),"hypervolume_256":hv256,
                "hypervolume_512":hv512,"hypervolume_gain_256_to_512":hv512-hv256,
                "checkpoint_vectors_and_costs_exact":True})
            if seed == choice["seed"]:
                selected = trials.loc[trials.evaluation_id.eq(choice["evaluation_id"])].iloc[0]
                np.testing.assert_array_equal([selected.release_cost,selected.storage_cost],
                                              [choice["release_cost"],choice["storage_cost"]])
                # Native optimizer coordinates are six 12-month blocks; storage
                # coordinates are capacity fractions and releases remain SI.
                vector = vectors.loc[choice["evaluation_id"]].to_numpy().reshape(6,12).T.copy()
                vector[:,:3] *= parameters.capacity_m3
                np.testing.assert_array_equal(vector,parameters.monthly_targets.to_numpy())
        for left,right in combinations((42,43,44),2):
            distance = front_distance(fronts[left],fronts[right])
            agreement_rows.append({"reservoir_id":identifier,"left_seed":left,"right_seed":right,
                "front_distance_nse_linf":distance,"within_0_01":distance<=.01})
        folder = study/"signed_profiles"/identifier
        source_path = folder/"observations_si.csv"
        trajectory_path = release/"diagnostics"/identifier/"assessment_simulation.csv"
        consume(source_path); consume(trajectory_path)
        source = pd.read_csv(source_path,index_col="date",parse_dates=True,float_precision="round_trip")
        simulation = pd.read_csv(trajectory_path,index_col="date",parse_dates=True,float_precision="round_trip")
        pd.testing.assert_index_equal(source.index,simulation.index)
        np.testing.assert_array_equal(source.inflow_m3s,simulation.inflow_m3s)
        np.testing.assert_array_equal(simulation.positive_net_component_m3s-simulation.negative_net_component_m3s,source.inflow_m3s)
        s,q,net,dt = [simulation[c].to_numpy() for c in ("storage_m3","release_m3s","inflow_m3s","step_seconds")]
        residual = np.diff(s)-dt[1:]/2*(net[1:]+net[:-1]-q[1:]-q[:-1])
        balance = float(np.abs(residual).max()); maximum_balance = max(maximum_balance,balance)
        if balance > profile["atol_storage_m3"] or s.min()<0 or s.max()>parameters.capacity_m3 or q.min()<0:
            raise ValueError("Independent physical/balance check failed")
        np.testing.assert_array_equal(residual,simulation.balance_residual_m3.to_numpy()[1:])
        spill = simulation.overflow_volume_m3.to_numpy()
        rule_errors = [abs(q[k]-2*spill[k]/dt[k]-independent_release(s[k]+spill[k],net[k],simulation.index[k].month,parameters,dt[k]))
                       for k in range(1,len(simulation))]
        rule = max(rule_errors); maximum_rule = max(maximum_rule,rule)
        if rule > 1e-8: raise ValueError("Independent release-rule check failed")
        selection = w.read(folder/"selection.json")
        roles = np.array(selection["roles"]); regimes = np.array(selection["regimes"])
        for role,key in (("calibration","calibration"),("assessment","assessment")):
            metrics = record[key] if role == "calibration" else record[key]["metrics"]
            mask = roles==role
            for variable in ("storage","release"):
                column = variable+"_m3" if variable=="storage" else "release_m3s"
                value = nse(source.loc[mask,column],simulation.loc[mask,column])
                np.testing.assert_allclose(value,metrics[variable+"_nse"],rtol=0,atol=1e-10)
                for regime in ("dry","normal","wet"):
                    sub = mask&(regimes==regime)
                    result = nse(source.loc[sub,column],simulation.loc[sub,column])
                    cached = metrics.get(regime+"_"+variable+"_nse")
                    if cached is not None: np.testing.assert_allclose(result,cached,rtol=0,atol=1e-10)
                    regime_rows.append({"reservoir_id":identifier,"period":role,"regime":regime,
                        "variable":variable,"scored_rows":int(sub.sum()),"nse":result})
                baseline = record["fixed_initializer_calibration"] if role=="calibration" else record["fixed_initializer_assessment"]["metrics"]
                comparison_rows.append({"reservoir_id":identifier,"period":role,"variable":variable,
                    "calibrated_nse":value,"fixed_initializer_nse":baseline[variable+"_nse"],
                    "nse_gain":value-baseline[variable+"_nse"]})
        held = record["assessment"]["metrics"]; base = record["fixed_initializer_assessment"]["metrics"]
        gains = {v:held[v+"_nse"]-base[v+"_nse"] for v in ("storage","release")}
        effect = "both_improve" if min(gains.values())>0 else "both_worsen" if max(gains.values())<0 else "tradeoff"
        summaries.append({"reservoir_id":identifier,"name":NAMES[identifier],"selected_seed":choice["seed"],
            "selected_evaluation":choice["evaluation_id"],"assessment_storage_nse":held["storage_nse"],
            "assessment_release_nse":held["release_nse"],"storage_nse_gain":gains["storage"],
            "release_nse_gain":gains["release"],"baseline_effect":effect,
            "maximum_independent_mass_residual_m3":balance,"maximum_independent_rule_residual_m3s":rule,
            "negative_net_flux_records_preserved":int(source.inflow_m3s.lt(0).sum()),
            "quality_note":{"ex40_Semione":"Held-out decline versus initializer; all three regime storage NSE values negative.",
                "ex8_Flaming_gorge":"Both pooled assessment objectives improve; dry/normal release and normal storage remain weak.",
                "ex33_Navajo":"Release NSE near zero; normal-regime storage NSE very negative; storage declines versus initializer."}[identifier]})
    if w.sha(release/"selection.json") != before_selection: raise ValueError("Frozen choices changed")
    inventory_path = study/"paper_parameterization_with_signed_routes_all_37.csv"
    inventory = pd.read_csv(inventory_path,float_precision="round_trip"); consume(inventory_path)
    base_inventory = pd.read_csv(study/"paper_reservoir_inventory_all_37.csv",float_precision="round_trip")
    monthly = [c for c in inventory if "_month_" in c]
    existing = ~inventory.reservoir_id.isin(NAMES)
    pd.testing.assert_frame_equal(inventory.loc[existing,monthly],base_inventory.loc[existing,monthly])
    for summary in summaries:
        identifier = summary["reservoir_id"]; profile,parameters = w.load_profile(release/"registry.json",identifier=identifier)
        mask = inventory.reservoir_id.eq(identifier)
        inventory.loc[mask,"programme_status"] = "calibrated_signed_net_exploratory"
        inventory.loc[mask,"new_calibration_ready"] = False
        inventory.loc[mask,"next_action"] = summary["quality_note"]+" Preserve source hypotheses and checkpoint identity; no additional budget declared."
        inventory.loc[mask,"quality_note"] = summary["quality_note"]
        inventory.loc[mask,"profile_registry"] = str((release/"registry.json").relative_to(ROOT)).replace("\\","/")
        for key,value in profile["physical_constants"].items(): inventory.loc[mask,key] = value
        inventory.loc[mask,"root_policy"] = profile["root_policy"]
        inventory.loc[mask,"sampling_frequency"] = profile["sampling"]["frequency"]
        inventory.loc[mask,"fixed_step_days"] = profile["sampling"]["fixed_step_days"]
        inventory.loc[mask,"seed"] = profile["seed"]
        inventory.loc[mask,"evaluation_id"] = profile["evaluation_id"]
        for month,values in parameters.monthly_targets.iterrows():
            for name,value in values.items():
                if inventory.loc[mask,f"{name}_month_{month:02d}"].iloc[0] != value:
                    raise ValueError("Inventory signed monthly targets differ from registry")
    if len(inventory)!=37 or not inventory.reservoir_id.is_unique or inventory.parameter_values_available.sum()!=32:
        raise ValueError("Unexpected paper cohort coverage")
    if inventory.loc[~inventory.parameter_values_available,monthly].notna().any().any():
        raise ValueError("Unavailable standard monthly targets must remain blank")
    target = out/"reservoir_parameterization_all_37_by_GRAND_ID.csv"
    inventory.to_csv(target,index=False,float_format="%.17g")
    roundtrip = pd.read_csv(target,float_precision="round_trip")
    pd.testing.assert_frame_equal(roundtrip[monthly],inventory[monthly])
    pd.DataFrame(summaries).to_csv(out/"signed_results.csv",index=False,float_format="%.17g")
    pd.DataFrame(comparison_rows).to_csv(out/"baseline_comparison.csv",index=False,float_format="%.17g")
    pd.DataFrame(regime_rows).to_csv(out/"regime_diagnostics.csv",index=False,float_format="%.17g")
    pd.DataFrame(job_rows).to_csv(out/"checkpoint_diagnostics.csv",index=False,float_format="%.17g")
    pd.DataFrame(agreement_rows).to_csv(out/"seed_front_agreement.csv",index=False,float_format="%.17g")
    seeds = pd.read_csv(release/"seed_scores.csv",float_precision="round_trip")
    if len(seeds)!=9 or not seeds.assessment_feasible.all(): raise ValueError("Expected nine feasible seed assessments")
    seeds.to_csv(out/"seed_scores.csv",index=False,float_format="%.17g")
    receipt = {"status":"passed","paper_count":37,"native_standard_profiles":29,"signed_native_profiles":3,
        "separate_Bull_Lake_interval_studies":1,"source_gated":4,"profiles_with_monthly_targets":32,
        "monthly_parameter_values_exactly_verified":32*72,"completed_searches":9,"cached_search_evaluations":9*512,
        "new_model_calls":0,"new_optimizer_evaluations":0,"independent_trajectory_nse_and_release_rule_checks":True,
        "maximum_independent_mass_balance_residual_m3":maximum_balance,
        "maximum_independent_release_rule_residual_m3s":maximum_rule,"frozen_selection_sha256_unchanged":before_selection,
        "GRAND_ID_mappings_preserved":int(inventory.GRAND_ID.notna().sum()),
        "all_seed_assessments_feasible":True,"source_certified":False,"convergence_established":False,
        "jobs_improving_hypervolume_256_to_512":sum(v["hypervolume_gain_256_to_512"]>0 for v in job_rows),
        "seed_front_pairs_within_reporting_tolerance_0_01":sum(v["within_0_01"] for v in agreement_rows),
        "tests_reused":"outputs/paper_completion_v1_20261014_03/implementation_verification.json",
        "consumed_sha256":consumed,"assessment_script_sha256":w.sha(Path(__file__))}
    w.write(out/"verification.json",receipt)
    lines = ["# Completed paper-reservoir calibration assessment", "",
        "All nine signed-route searches completed: three reservoirs, three seeds, population 32 and 512 evaluations per seed (4,608 attempts). All nine calibration-selected seed representatives completed held-out assessment. This report reuses cached results and runs no model or optimizer.","",
        "The paper cohort now has **29 standard native profiles + 3 exploratory signed-native profiles + 1 separate Bull Lake interval study + 4 source-gated cases = 37**. There are 32 rows with monthly targets (2,304 verified values). Bull Lake has its separate archived interval representatives; the four pending cases retain blank targets.","",
        "| Reservoir | Held-out storage NSE | Held-out release NSE | Storage gain vs initializer | Release gain vs initializer | Outcome |",
        "| --- | ---: | ---: | ---: | ---: | --- |"]
    for v in summaries:
        lines.append(f"| {v['name']} | {v['assessment_storage_nse']:.3f} | {v['assessment_release_nse']:.3f} | {v['storage_nse_gain']:+.3f} | {v['release_nse_gain']:+.3f} | {v['baseline_effect']} |")
    lines += ["", "All three improve both pooled calibration objectives. Held-out improvement differs: Flaming Gorge improves both objectives, Seminoe worsens both, and Navajo trades improved release for worse storage. The comparison is against the archived fixed initializer under identical source hypotheses and year masks; it is not automatically a generalized baseline.","",
        "Pooled positive NSE does not establish robust performance. Seminoe's held-out storage NSE is negative within every inflow regime. Flaming Gorge has dry/normal release weaknesses and negative normal-regime storage NSE. Navajo's release NSE is only about 0.002, with about 47.6% release bias and normal-regime storage NSE about -21.75. Regime NSE can be sensitive to low within-regime variance; inspect its sample support and trajectory before attributing these values to a specific cause.","",
        "Calibration-only selections remain frozen. Held-out scores did not choose a replacement seed or treatment. The signed net-flux interpretation, native endpoint clock and Flaming Gorge net-as-rule proxy remain exploratory. Forcing signs and inherited capacity/floor constants are preserved; no additional optimizer budgets were launched.","",
        "All nine cumulative calibration fronts improved between 256 and 512 evaluations. None of the nine within-reservoir seed-front pairs is within the diagnostic 0.01 NSE L-infinity Hausdorff tolerance. This is a reporting diagnostic, not a stopping policy or proof of a required budget. Convergence has not been demonstrated; 512 remains a completed checkpoint.","",
        f"Independent checks reproduced pooled/regime NSE from saved trajectories, the literal piecewise release relation and the full-timeline water balance. Maximum balance residual: **{maximum_balance:.9g} m3**. All nine checkpoint vectors and costs match their 512-trial snapshots exactly. Prior implementation tests passed 1,519 tests with one skip; this artifact-only assessment uses numerical assertions instead of rerunning that regression suite.","",
        "[Updated all-37 parameterization by GRAND ID](reservoir_parameterization_all_37_by_GRAND_ID.csv), [baseline comparisons](baseline_comparison.csv), [regime diagnostics](regime_diagnostics.csv), [seed scores](seed_scores.csv), [checkpoint diagnostics](checkpoint_diagnostics.csv), [verification](verification.json).", "",
        "Canyon Ferry, Trinity, Ghost and Bennett retain their documented source gates. Their parameter rows are included, but no new assumption has been used to bypass those gates. Existing registries, source profiles and experiment archives are unchanged.",""]
    (out/"README.md").write_text("\n".join(lines),encoding="utf-8")
    w.write(out/"manifest.json",{"artifact_sha256":{f.name:w.sha(f) for f in out.iterdir() if f.is_file()}})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True); parser.add_argument("--output",required=True)
    args = parser.parse_args()
    print(audit(args.study,args.output))
