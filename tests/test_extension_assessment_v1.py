"""Scientific contracts for source diagnostics and exploratory admission."""
import numpy as np
import pandas as pd
import pytest

from workflows import extension_assessment_v1 as w


def frame(days=5):
    index = pd.date_range("2000-01-01", periods=days, name="date")
    return pd.DataFrame({"storage_m3": np.arange(days)*86400.+1e7,
        "inflow_m3s": np.nan, "release_m3s": 2.}, index=index)[w.COLUMNS]


def test_signed_net_flux_is_exact_and_not_independent_validation():
    source = frame(); source.iloc[2, source.columns.get_loc("storage_m3")] -= 10*86400
    result = w.conditional_inflow(source)
    assert result.inflow_m3s.iloc[1] == 3 and result.inflow_m3s.iloc[2] == -7
    assert result.inflow_m3s.iloc[3] == 13 and np.isnan(result.inflow_m3s.iloc[0])
    assert result.reconstruction_closure_m3.abs().max() < 1e-8
    proof = w.conditional_summary(result)
    assert proof["negative_net_intervals"] == 1
    assert proof["assessment_responses_used_in_forcing"] and not proof["independent_inflow"]
    assert source.inflow_m3s.isna().all()


def test_missing_storage_and_missing_release_are_not_filled_or_bridged():
    source = frame(8); source.loc[source.index[2], "storage_m3"] = np.nan
    source.loc[source.index[5], "release_m3s"] = np.nan
    result = w.conditional_inflow(source)
    assert result.inflow_m3s.isna().tolist() == [True, False, True, True, False, True, False, False]
    assert w.blocks(result.inflow_m3s.notna()) == [(1, 2), (4, 5), (6, 8)]


def test_missing_calendar_labels_are_rejected():
    with pytest.raises(ValueError, match="consecutive daily"):
        w.conditional_inflow(frame().drop(frame().index[2]))


@pytest.mark.parametrize("column", ["storage_m3", "release_m3s"])
def test_unresolved_negative_responses_are_not_clipped(column):
    source = frame(); source.loc[source.index[0], column] = -1
    with pytest.raises(ValueError, match="source decision"): w.conditional_inflow(source)


def test_source_budget_flags_do_not_repair_the_source():
    source = frame(); source.inflow_m3s = 3.
    source.loc[source.index[2], "storage_m3"] += 1e6
    audited = w.source_budget(source, capacity=1e7)
    pd.testing.assert_frame_equal(audited[w.COLUMNS], source)
    assert audited.large_budget_residual_flag.sum() == 2
    assert audited.source_budget_residual_m3.iloc[2] == 1e6
    assert audited.source_budget_residual_m3.iloc[3] == -1e6


def test_horizon_exact_units_preserve_zeros_signed_values_and_gaps(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("date,s_af,i_cfs,r_cfs\n2000-01-01,-1,-2,0\n2000-01-02,,1,2\n")
    result = w.read_horizon(path)
    assert result.storage_m3.iloc[0] == -w.ACRE_FOOT_M3
    assert result.inflow_m3s.iloc[0] == -2*w.CFS_M3S
    assert result.release_m3s.iloc[0] == 0 and np.isnan(result.storage_m3.iloc[1])


def test_incompatible_capacity_blocks_probe_without_parameter_repair():
    source = frame(365*2)
    source["storage_m3"] = 1e8 + np.arange(len(source))
    result, simulation, targets = w.baseline_probe(w.conditional_inflow(source), 1e6, "computed_net_inflow")
    assert result["status"] == "blocked" and not result["calibration_ready"]
    assert result["optimizer_evaluations"] == result["intervals_completed"] == 0
    assert simulation is None and targets.iloc[:, :3].min().min() >= 1e8


def test_interval_probe_independent_mass_balance_and_source_status():
    source = frame(366)
    x = np.arange(len(source))
    source.storage_m3 = 5e7 + 1e6*np.sin(x/20)
    source.release_m3s = 2 + np.sin(x/10)
    result, simulation, targets = w.baseline_probe(w.conditional_inflow(source), 1e8, "computed_net_inflow")
    assert simulation is not None and targets.shape == (12, 6)
    assert result["status"] in {"failed", "completed_descriptive_probe"}
    assert result["max_independent_balance_residual_m3"] <= 1e-4
    assert not result["capacity_adopted_for_production"] and not result["calibration_ready"]
    scored = simulation.interval_completed
    assert not scored.iloc[0] and simulation.loc[scored, "independent_balance_residual_m3"].abs().max() <= 1e-4


def test_verify_detects_changed_artifacts_and_escaping_paths(tmp_path):
    w.write(tmp_path / "summary.json", {"x": 1})
    manifest = {"kind": "reservoir_extension_assessment_v1", "input_sha256": {},
        "artifact_sha256": {"summary.json": w.sha(tmp_path / "summary.json")}}
    w.write(tmp_path / "manifest.json", manifest)
    assert w.verify(tmp_path) == manifest
    w.write(tmp_path / "summary.json", {"x": 2})
    with pytest.raises(ValueError, match="artifact changed"): w.verify(tmp_path)
