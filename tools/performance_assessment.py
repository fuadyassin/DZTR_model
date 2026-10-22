"""Consolidate frozen twenty-reservoir evidence; no simulation or optimization."""
from pathlib import Path

import numpy as np
import pandas as pd

from simulate_calibrated_reservoirs import ROOT, ARCHIVE, read, write, sha

OUT = ROOT / "outputs/performance_assessment_20261012"


def recommendation(row):
    """Transparent descriptive triage, never a parameter-selection rule."""
    cal = [row[f"selected_calibration_{v}_nse"] for v in ("storage", "release")]
    val = [row[f"selected_validation_{v}_nse"] for v in ("storage", "release")]
    delta = [row[f"delta_validation_{v}_nse"] for v in ("storage", "release")]
    if not np.isfinite(cal + val + delta).all():
        return "review_missing_evidence", "Required NSE evidence is undefined; investigate coverage before assigning a performance action."
    if min(cal) <= 0:
        return "review_model_assumptions", "At least one calibration objective has nonpositive NSE. Review forcing/storage definitions and the fixed monthly operating-rule assumptions before attributing this to the optimizer."
    if min(val) <= 0:
        return "review_forcing_and_timing", "Both calibration NSE values are positive but at least one held-out objective is nonpositive. Check period-dependent forcing, timing, storage definitions and operating changes; this does not establish which caused the deterioration."
    if min(delta) < 0:
        return "review_calibration_tradeoff", "Both held-out NSE values are positive, but at least one is below the generalized baseline. Review the calibration-only storage/release compromise and period transfer; retain the baseline as a comparator."
    return "retain_exploratory", "Both held-out NSE values are positive and neither is below baseline. Retain the profile for exploratory use with existing physical and convergence limitations."


def largest_bias(data, column):
    valid = data.loc[np.isfinite(data[column])]
    if valid.empty:
        return None
    # idxmax preserves earliest sorted month on an exact tie.
    return valid.loc[valid[column].abs().idxmax()]


def fmt(value):
    return "undefined" if pd.isna(value) else f"{float(value):.4g}"


def main():
    if OUT.exists():
        raise ValueError("Use a fresh assessment output; earlier reports remain frozen")
    receipt = read(ARCHIVE / "final_verification.json")
    consumed = {}

    def load(name):
        path = ARCHIVE / (name + ".csv")
        expected = receipt["artifact_sha256"][path.name]
        if sha(path) != expected:
            raise ValueError(f"Archived evidence changed: {path.name}")
        consumed[path.relative_to(ROOT).as_posix()] = expected
        return pd.read_csv(path, float_precision="round_trip")

    catalog = load("reservoir_parameterization_by_GRAND_ID")
    scores = load("selected_reservoir_scores")
    seasonal = load("seasonal_bias")
    trajectory = load("trajectory_diagnostics")
    zones = load("zone_diagnostics")
    convergence = load("convergence")
    variation = load("parameter_variation").set_index("reservoir_id")
    observed = load("observed_balance_diagnostics")
    baseline_comparisons = load("baseline_comparison")
    if len(catalog) != 20 or not catalog.reservoir_id.is_unique or set(catalog.reservoir_id) != set(scores.reservoir_id):
        raise ValueError("Expected exactly twenty unique matched profiles")
    OUT.mkdir()
    reports = OUT / "reservoirs"
    reports.mkdir()
    summaries, comparisons, seed_rows = [], [], []
    for _, meta in catalog.sort_values("reservoir_id").iterrows():
        identifier = meta.reservoir_id
        score = scores.set_index("reservoir_id").loc[identifier]
        row = {"reservoir_id":identifier, "GRAND_ID":meta.GRAND_ID, "name":meta.reservoir_name if "reservoir_name" in meta else score["name"],
               "selected_seed":int(score.selected_seed), "selected_evaluation_id":int(score.selected_evaluation_id),
               "sampling_frequency":meta.sampling_frequency, "capacity_m3":meta.capacity_m3}
        for period in ("calibration", "validation"):
            for variable in ("storage", "release"):
                for prefix in ("baseline", "selected", "delta"):
                    key=f"{prefix}_{period}_{variable}_nse"
                    row[key] = float(meta[key])
                np.testing.assert_allclose(row[f"selected_{period}_{variable}_nse"],score[f"{period}_{variable}_nse"],rtol=0,atol=1e-12)
                np.testing.assert_allclose(row[f"delta_{period}_{variable}_nse"],row[f"selected_{period}_{variable}_nse"]-row[f"baseline_{period}_{variable}_nse"],rtol=0,atol=1e-12)
                comparisons.append({"reservoir_id":identifier,"GRAND_ID":meta.GRAND_ID,"period":period,"variable":variable,
                                     "baseline_nse":row[f"baseline_{period}_{variable}_nse"],"selected_nse":row[f"selected_{period}_{variable}_nse"],"delta_nse":row[f"delta_{period}_{variable}_nse"]})
        action, rationale = recommendation(row)
        row.update(primary_action=action, action_reason=rationale, physical_source_certified=False, convergence_established=False,
                   production_readiness="not_established", selection_changed_by_assessment=False)
        conv = convergence.loc[convergence.reservoir_id==identifier]
        assert set(conv.seed)=={42,43,44} and len(conv)==3
        row["descriptive_stability_all_seeds"] = bool(conv.diagnostic_stability_pass.all())
        row["maximum_relative_hv_gain_last128"] = float(conv.relative_gain_last128.max())
        row["maximum_between_seed_front_distance_nse"] = float(conv.max_between_seed_front_distance_nse.max())
        row["balanced_seed_parameter_rms_spread"] = float(variation.loc[identifier,"max_balanced_seed_parameter_rms_normalized"])
        row["further_calibration_action"] = ("defer_until_source_and_model_review; descriptive stability screen not passed" if not row["descriptive_stability_all_seeds"]
                                            else "not_justified_by_descriptive_stability_alone; this is not certified convergence")
        lines=[f"# {score['name']}", "", f"Reservoir ID: `{identifier}`; GRAND_ID: {int(meta.GRAND_ID) if pd.notna(meta.GRAND_ID) else 'unresolved'}. Native sampling: {meta.sampling_frequency}.", "",
               f"**Next action: {action.replace('_',' ')}.** {rationale}", "",
               "The frozen pooled calibration choice remains unchanged. Held-out evidence is used here to diagnose performance, not to choose a different seed or parameter set.", "",
               "| Period | Storage NSE baseline → selected (change) | Release NSE baseline → selected (change) |",
               "| --- | --- | --- |"]
        for period in ("calibration","validation"):
            cells=[]
            for variable in ("storage","release"):
                cells.append(f"{fmt(row[f'baseline_{period}_{variable}_nse'])} → {fmt(row[f'selected_{period}_{variable}_nse'])} ({row[f'delta_{period}_{variable}_nse']:+.4g})")
            lines.append(f"| {period} | {' | '.join(cells)} |")
        lines.extend(["", "Seasonal, drift and peak evidence (selected profile):", ""])
        for period in ("calibration","validation"):
            part=seasonal[(seasonal.reservoir_id==identifier)&(seasonal.parameter_set=="pooled_selected")&(seasonal.period==period)].sort_values("month")
            assert len(part)==12 and part.month.is_unique
            sb, qb = largest_bias(part,"storage_mean_bias"), largest_bias(part,"release_mean_bias")
            diag=trajectory[(trajectory.reservoir_id==identifier)&(trajectory.parameter_set=="pooled_selected")&(trajectory.period==period)]
            assert len(diag)==1
            diag=diag.iloc[0]
            for variable in ("storage","release"):
                np.testing.assert_allclose(diag[variable+"_nse"],row[f"selected_{period}_{variable}_nse"],rtol=0,atol=1e-12)
            baseline=trajectory[(trajectory.reservoir_id==identifier)&(trajectory.parameter_set=="generalized_baseline")&(trajectory.period==period)&(trajectory.budget==1024)]
            assert set(baseline.seed)=={42,43,44}
            for variable in ("storage","release"):
                np.testing.assert_allclose(baseline[variable+"_nse"],row[f"baseline_{period}_{variable}_nse"],rtol=0,atol=1e-12)
            for column in ("storage_error_trend_million_m3_per_year","storage_error_endpoint_change_million_m3","peak_sample_ratio","high_release_mean_bias_m3s","peak_sample_date_offset_days","high_release_samples"):
                row[f"{period}_{column}"]=diag[column]
            row[f"{period}_worst_storage_bias_month"] = int(sb.month) if sb is not None else None
            row[f"{period}_worst_storage_bias_percent_capacity"] = float(sb.storage_mean_bias/meta.capacity_m3*100) if sb is not None else np.nan
            row[f"{period}_worst_release_bias_month"] = int(qb.month) if qb is not None else None
            row[f"{period}_worst_release_bias_m3s"] = float(qb.release_mean_bias) if qb is not None else np.nan
            for month_bias, label, column in ((sb,"storage","storage_mean_bias"),(qb,"release","release_mean_bias")):
                if month_bias is not None:
                    value=month_bias[column]/meta.capacity_m3*100 if label=="storage" else month_bias[column]
                    unit="% of capacity" if label=="storage" else "m³/s"
                    lines.append(f"- {period}: largest absolute monthly mean {label} bias is {value:+.4g} {unit} in month {int(month_bias.month)} ({int(month_bias.samples)} native samples). Bias is simulated minus observed.")
            lines.append(f"- {period}: storage-error trend {fmt(diag.storage_error_trend_million_m3_per_year)} million m³/year; endpoint error change {fmt(diag.storage_error_endpoint_change_million_m3)} million m³. These describe error, not observed reservoir depletion.")
            lines.append(f"- {period}: simulated/observed largest native release sample ratio {fmt(diag.peak_sample_ratio)}; high-release mean bias {fmt(diag.high_release_mean_bias_m3s)} m³/s over {int(diag.high_release_samples)} samples above the calibration P95 threshold. Largest-peak dates differ by {fmt(diag.peak_sample_date_offset_days)} days; these may be different events, so this is not a lag estimate.")
        zonal=zones[(zones.reservoir_id==identifier)&(zones.parameter_set=="pooled_selected")].copy()
        lines.extend(["", "Zone diagnostics (classification by selected simulated physical storage):", "",
                      "| Period | Zone | Samples | Storage NSE | Release NSE | Release mean bias (m³/s) |",
                      "| --- | --- | --- | --- | --- | --- |"])
        for _, z in zonal.sort_values(["period","physical_storage_zone"]).iterrows():
            lines.append(f"| {z.period} | {int(z.physical_storage_zone)} | {int(z.samples)} | {fmt(z.storage_nse)} | {fmt(z.release_nse)} | {fmt(z.release_mean_bias)} |")
        lines.extend(["", "Sparse or constant-observation zones can have undefined or unstable NSE. Zone membership is model-dependent and may change between profiles; these groups do not establish which equation or zone caused the overall error.", ""])
        for period in ("calibration","validation"):
            native=baseline_comparisons[(baseline_comparisons.reservoir_id==identifier)&(baseline_comparisons.budget==1024)&(baseline_comparisons.period==period)]
            assert len(native)==3 and set(native.seed)=={42,43,44}
            for variable in ("storage","release"):
                values=native["calibrated_"+variable+"_nse"]
                row[f"seed_balanced_{period}_{variable}_nse_min"]=float(values.min())
                row[f"seed_balanced_{period}_{variable}_nse_max"]=float(values.max())
            seed_rows.extend(native.to_dict("records"))
            lines.append(f"Independent seed-balanced {period} profiles span storage NSE {fmt(native.calibrated_storage_nse.min())} to {fmt(native.calibrated_storage_nse.max())}, release NSE {fmt(native.calibrated_release_nse.min())} to {fmt(native.calibrated_release_nse.max())}. These are three archived representatives, not a predictive uncertainty interval.")
        lines.extend(["",f"All-seed descriptive stability screen: {row['descriptive_stability_all_seeds']}. Maximum relative last-128 hypervolume gain {fmt(row['maximum_relative_hv_gain_last128'])}; maximum between-seed front distance {fmt(row['maximum_between_seed_front_distance_nse'])}; bound-normalized balanced-parameter RMS spread {fmt(row['balanced_seed_parameter_rms_spread'])}. These do not certify convergence or parameter identifiability.","",
                      "Observed continuity evidence uses the working model clock and endpoint-trapezoid hypothesis:", ""])
        for _, o in observed[observed.reservoir_id==identifier].iterrows():
            lines.append(f"- {o.period}: observed residual RMSE {fmt(o.observed_residual_rmse_million_m3)} million m³; signed equivalent rate {fmt(o.observed_residual_equivalent_rate_m3s)} m³/s across {int(o.finite_intervals)} finite intervals. This does not prove a missing flux or identify an inflow correction.")
        lines.extend(["", "Recommended checks: inspect source inflow/release timing and storage definitions across periods; compare the high-bias months and high-release events against source operating records. Assess whether fixed monthly rules can represent any documented operating changes. Do not adjust forcing or targets solely to improve held-out scores.", "",
                      f"Further calibration: {row['further_calibration_action'].replace('_',' ')}. No new evaluations are proposed in this assessment.", "",
                      f"[Existing diagnostic figures](../../twenty_reservoir_workflow_20261012/reports/{identifier}/README.md). Physical source certification and production readiness remain unestablished.", ""])
        (reports/f"{identifier}.md").write_text("\n".join(lines),encoding="utf-8")
        summaries.append(row)
    table=pd.DataFrame(summaries)
    table["GRAND_ID"]=table.GRAND_ID.astype("Int64")
    table.to_csv(OUT/"reservoir_performance_and_actions.csv",index=False,float_format="%.17g")
    pd.DataFrame(comparisons).to_csv(OUT/"baseline_selected_comparison.csv",index=False,float_format="%.17g")
    pd.DataFrame(seed_rows).to_csv(OUT/"three_seed_balanced_comparison.csv",index=False,float_format="%.17g")
    selected_seasonal=seasonal[seasonal.parameter_set=="pooled_selected"].copy()
    selected_seasonal.to_csv(OUT/"selected_seasonal_bias.csv",index=False,float_format="%.17g")
    zones[zones.parameter_set=="pooled_selected"].to_csv(OUT/"selected_zone_diagnostics.csv",index=False,float_format="%.17g")
    counts=table.primary_action.value_counts().sort_index().to_dict()
    summary={"reservoirs":20,"actions":{k:int(v) for k,v in counts.items()},"heldout_both_positive":int(((table.selected_validation_storage_nse>0)&(table.selected_validation_release_nse>0)).sum()),
             "heldout_both_improved":int(((table.delta_validation_storage_nse>0)&(table.delta_validation_release_nse>0)).sum()),
             "simulations":0,"optimization_evaluations":0,"selection_changed":False}
    write(OUT/"assessment_summary.json",summary)
    overview=["# Twenty-reservoir predictive-performance assessment","", "This assessment reuses verified archived evidence. It performs no new simulation or optimization and changes no parameter selections.", "",
              f"{summary['heldout_both_positive']} profiles have positive held-out NSE for both objectives; {summary['heldout_both_improved']} improve both relative to baseline. Positive NSE and improvement are separate tests.", "",
              "| Reservoir | GRAND_ID | Held-out storage NSE | Held-out release NSE | Storage change | Release change | Next action |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in table.iterrows():
        overview.append(f"| [{r['name']}](reservoirs/{r.reservoir_id}.md) | {int(r.GRAND_ID) if pd.notna(r.GRAND_ID) else 'unresolved'} | {fmt(r.selected_validation_storage_nse)} | {fmt(r.selected_validation_release_nse)} | {r.delta_validation_storage_nse:+.4g} | {r.delta_validation_release_nse:+.4g} | {r.primary_action.replace('_',' ')} |")
    overview.extend(["", "Actions are a descriptive review order, not production certification or statistical significance tests. The rule first checks nonpositive calibration NSE, then nonpositive held-out NSE, then any deterioration from baseline. Three profiles meet the exploratory-retention screen; others receive review actions. Small score differences are not assumed meaningful. No validation-based seed replacement occurs.", "",
                     "Start detailed source/timing review with Garrison and Oahe because their selected held-out storage scores deteriorate sharply from baseline. Dickson and Amistad need calibration-period review as well. Negative held-out scores that improved over baseline still need diagnosis; they should not be described as successful solely because calibration helped.", "",
                     "Review actions identify hypotheses to examine, not causes established by these diagnostics. In particular, storage-error trends do not measure observed depletion, unmatched maximum-peak dates do not estimate lag, and model-defined zones cannot be compared as fixed observed groups. Monthly peaks refer to native monthly samples, not instantaneous flood peaks.", "",
                     "[Full performance/action table](reservoir_performance_and_actions.csv), [baseline versus selected comparison](baseline_selected_comparison.csv), [three-seed representatives](three_seed_balanced_comparison.csv), [seasonal evidence](selected_seasonal_bias.csv), [zone evidence](selected_zone_diagnostics.csv), [summary](assessment_summary.json).", "",
                     "Further optimization should be considered only after checking source/timing and model assumptions. The existing stability screen is descriptive; it does not prove convergence. No new evaluations are needed to use this assessment. Deferred reservoirs remain parked; Fort Randall's GRAND_ID remains unresolved.", ""])
    (OUT/"README.md").write_text("\n".join(overview),encoding="utf-8")
    write(OUT/"analysis_manifest.json",{"archive_receipt_sha256":sha(ARCHIVE/"final_verification.json"),"consumed_sha256":consumed,"runner_sha256":sha(__file__),
          "artifact_sha256":{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()}})
    print(summary)


if __name__=="__main__":
    main()
