"""Run the twenty historical cases through the supplied-forcing interface once."""
from pathlib import Path

import numpy as np
import pandas as pd

from forcing_batch import FrozenProfiles, run_batch
from simulate_calibrated_reservoirs import ROOT, ARCHIVE, read, write, resolve, sha

OUT = ROOT / "outputs/forcing_workflow_v1_1_20261012"


def main():
    if OUT.exists():
        raise ValueError("Historical verification requires a fresh output directory")
    inputs = OUT / "requests"
    inputs.mkdir(parents=True)
    store = FrozenProfiles()
    jobs = []
    references = {}
    for identifier in sorted(store.registry["profiles_by_reservoir_id"]):
        profile, parameters, config = store.load({"reservoir_id":identifier})
        source = resolve(store.registry["profiles_by_reservoir_id"][identifier]).parent
        reference = store.checked(source / "simulation.csv")
        expected = pd.read_csv(reference,index_col="date",parse_dates=True,float_precision="round_trip")
        references[identifier] = {"path":reference.relative_to(ROOT).as_posix(),"sha256":sha(reference)}
        # Archived inflows already have source preprocessing applied. The CSV
        # intentionally exercises the model_input stage rather than applying it twice.
        forcing = expected[["inflow_m3s","observed_storage_m3","observed_release_m3s"]]
        forcing.to_csv(inputs / f"{identifier}.csv",index_label="date",float_format="%.17g")
        jobs.append({"job_id":identifier, **({"grand_id":profile["grand_id"]} if profile["grand_id"] is not None else {"reservoir_id":identifier}),
                     "forcing":{"csv":f"{identifier}.csv","stage":"model_input","columns":{"date":"date","inflow":"inflow_m3s","storage":"observed_storage_m3","release":"observed_release_m3s"},"units":{"inflow":"m3/s","storage":"m3","release":"m3/s"}},
                     "initial":{"storage":parameters.initial_storage_m3,"storage_unit":"m3","release":parameters.initial_release_m3s,"release_unit":"m3/s"},
                     "sampling":profile["sampling"],"evaluation_periods":{key:profile["periods"][key] for key in ("calibration","validation")}})
    write(inputs / "historical_batch.json",{"schema_version":1,"kind":"dztr_forcing_batch","jobs":jobs})
    if not run_batch(inputs / "historical_batch.json",OUT / "simulations"):
        raise ValueError("Historical forcing batch failed; inspect per-job result.json")
    rows, maximum = 0,0.
    for job in jobs:
        identifier = job["job_id"]
        expected = pd.read_csv(ROOT / references[identifier]["path"],index_col="date",parse_dates=True,float_precision="round_trip")
        actual = pd.read_csv(OUT / "simulations" / identifier / "simulation.csv",index_col="date",parse_dates=True,float_precision="round_trip")
        pd.testing.assert_frame_equal(actual,expected.drop(columns="period"),check_exact=True,check_dtype=False,check_freq=False)
        profile, _, _ = store.load({"reservoir_id":identifier})
        summary = read(OUT / "simulations" / identifier / "result.json")
        for metric in summary["metrics"]:
            for variable in ("storage","release"):
                key=variable+"_nse"
                expected_score=profile["calibration_"+key] if metric["period"]=="calibration" else profile["validation"]["metrics"][key]
                np.testing.assert_allclose(metric[key],expected_score,rtol=0,atol=1e-12)
        rows += len(actual)
        maximum=max(maximum,summary["maximum_balance_residual_m3"])
    write(OUT / "scientific_verification.json",{"status":"passed","reservoirs_verified":20,"trajectory_rows_verified":rows,
          "exact_core_and_observed_trajectory_replay":True,"period_nse_matches":True,"maximum_balance_residual_m3":maximum,
          "historical_simulations":20,"optimization_evaluations":0,"reference_sha256":references})
    print(f"Verified all twenty forcing-interface simulations: {rows} rows, exact replay, maximum mass residual {maximum} m3.")


if __name__=="__main__":
    main()
