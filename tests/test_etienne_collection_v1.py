"""Observation import contracts; descriptive targets never imply model readiness."""
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from workflows import etienne_collection_v1 as w


@pytest.fixture
def workspace():
    root = w.ROOT / "outputs"
    path = Path(tempfile.mkdtemp(prefix="test_etienne_collection_", dir=root)).resolve()
    yield path
    assert path.is_relative_to(root.resolve()) and path.name.startswith("test_etienne_collection_")
    shutil.rmtree(path)


def source_spec(**overrides):
    return {"file": "source.txt", "value_column": 1, "factor": 1., "source_units": "explicit_test_units",
        "missing_values": [], "missing_tokens": [], "duplicate_policy": "reject", "evidence": "fixture evidence", "statistics": [], **overrides}


def test_explicit_units_preserve_zero_and_declared_missing_support(workspace):
    path = workspace / "storage.txt"
    path.write_text("20000101 0\n20000102 -1\n20000103\n20000105 2.5\n\n")
    values, quality, proof = w.read_observations(path, source_spec(factor=1e6, missing_values=[-1]))
    assert values.iloc[0] == 0 and values.iloc[-1] == 2.5e6
    assert values.iloc[1:4].isna().all()
    assert proof["declared_sentinel_rows"] == 1 and proof["missing_calendar_days"] == 1
    assert proof["zero_days"] == 1 and proof["valid_days"] == 2
    assert quality.iloc[3].source_rows == 0


def test_selects_lake_chelan_si_field_without_double_conversion(workspace):
    path = workspace / "flow.txt"; path.write_text("2000-01-01 2210 62.579640\n2000-01-02 0 0\n")
    values, _, proof = w.read_observations(path, source_spec(value_column=2))
    assert values.iloc[0] == 62.579640 and values.iloc[1] == 0
    assert proof["zero_days"] == 1


def test_declared_spreadsheet_errors_are_missing_without_general_token_coercion(workspace):
    path = workspace / "spreadsheet.txt"; path.write_text("20000101 #VALEUR!\n20000102 4\n")
    values, _, proof = w.read_observations(path, source_spec(missing_tokens=["#VALEUR!"]))
    assert np.isnan(values.iloc[0]) and values.iloc[1] == 4
    assert proof["declared_missing_text_rows"] == proof["missing_rows"] == 1
    with pytest.raises(ValueError, match="Unknown observation"): w.read_observations(path, source_spec())


def test_repeated_dates_require_explicit_untimed_storage_proxy(workspace):
    path = workspace / "subdaily.txt"; path.write_text("20000101 2\n20000101 4\n20000102 8\n")
    with pytest.raises(ValueError, match="Duplicate day"): w.read_observations(path, source_spec())
    values, quality, proof = w.read_observations(path, source_spec(duplicate_policy="daily_arithmetic_mean_proxy"))
    assert values.tolist() == [3., 8.]
    assert quality.valid_samples.tolist() == [2, 1]
    assert proof["repeated_date_rows"] == 1 and proof["distinct_source_dates"] == 2
    assert proof["observation_basis"].endswith("proxy")


@pytest.mark.parametrize("content", ["bad-date 1\n", "20000101 inf\n", "20000101 mystery\n", "\n\n"])
def test_unusable_source_is_not_silently_accepted(workspace, content):
    path = workspace / "bad.txt"; path.write_text(content)
    with pytest.raises(ValueError): w.read_observations(path, source_spec())


def test_unrecognized_negative_values_are_preserved_and_gate_estimates(workspace):
    path = workspace / "signed.txt"; path.write_text("20000101 -2\n20000102 3\n")
    values, _, proof = w.read_observations(path, source_spec())
    assert values.iloc[0] == -2 and proof["uninterpreted_negative_rows"] == 1
    frame = pd.DataFrame({"storage_m3": [10., 11.], "release_m3s": values}, index=values.index)
    targets, _ = w.target_components(frame, "hazen")
    assert targets.iloc[:, 3:].isna().all().all()


def test_component_estimation_reuses_hazen_and_preserves_valid_zeros():
    index = pd.date_range("2000-01-01", "2004-12-31", name="date")
    frame = pd.DataFrame({"storage_m3": np.arange(len(index)) + 1., "release_m3s": np.arange(len(index)) % 8}, index=index)
    targets, counts = w.target_components(frame, "hazen")
    for month in range(1, 13):
        sample = frame.loc[frame.index.month == month, "release_m3s"]
        np.testing.assert_array_equal(targets.loc[month].iloc[3:], np.percentile(sample, [10, 45, 85], method="hazen"))
        assert counts.loc[month, "release_samples"] == len(sample)


def fixtures(folder):
    archive = folder / "archive"; archive.mkdir()
    (archive / "compute_monthly_statistics.py").write_text("# fixture unit reader evidence\n")
    index = pd.date_range("2000-01-01", "2007-12-31", name="date")
    x = np.arange(len(index))
    (archive / "storage.txt").write_text("\n".join(f"{d:%Y%m%d} {10+i%100}" for d, i in zip(index, x)))
    (archive / "release.txt").write_text("\n".join(f"{d:%Y%m%d} {i%15}" for d, i in zip(index, x)))
    old_obs = folder / "existing.csv"; pd.DataFrame({"inflow_m3s": 2., "storage_m3": 5., "release_m3s": 1.}, index=index).to_csv(old_obs)
    relative = old_obs.relative_to(w.ROOT).as_posix()
    base = {"kind": "reservoir_collection", "excluded_reservoirs": [{"reservoir_id": "Tuyen", "reason": "user exclusion"}],
        "reservoirs": [{"reservoir_id": "old", "name": "Existing Lake", "first_date": "2000-01-01", "last_date": "2007-12-31",
            "rows": len(index), "observations_file": relative, "observations_sha256": w.native.sha(old_obs), "config_file": "fixture.json", "files_sha256": {relative: w.native.sha(old_obs)}}]}
    w.write(folder / "base.json", base)
    catalog = {"kind": "paper_profile_application_v1", "application_source_sha256": {}, "reservoirs": [{"reservoir_id": "old", "GRAND_ID": 8, "monthly_profile_available": True, "status": "completed"}]}
    w.write(folder / "catalog.json", catalog)
    w.write(folder / "manifest.json", {"artifact_sha256": {"catalog.json": w.native.sha(folder / "catalog.json")}})
    pd.DataFrame([{"GRAND_ID": 99, "RES_NAME": "New Lake", "DAM_NAME": "New Dam", "ALT_NAME": "", "CAP_MCM": 900, "COUNTRY": "Canada"}]).to_csv(folder / "grand.csv", index=False)
    cfg = {"kind": "dztr_etienne_collection_v1", "schema_version": 1, "archive": str(archive), "base_collection": str(folder / "base.json"),
        "base_catalog": str(folder / "catalog.json"), "grand_reference": str(folder / "grand.csv"), "quantile_method": "hazen",
        "existing_aliases": [{"alias": "Old Alias", "reservoir_id": "old"}], "reservoirs": [
            {"reservoir_id": "new", "name": "New Lake", "catalog_terms": ["New Lake"],
                "storage": source_spec(file="storage.txt", factor=1e6), "release": source_spec(file="release.txt")},
            {"reservoir_id": "partial", "name": "Partial Lake", "catalog_terms": ["Partial"], "storage": source_spec(file="partial.txt", factor=1e6)}]}
    shutil.copyfile(archive / "storage.txt", archive / "partial.txt")
    path = folder / "config.json"; w.write(path, cfg)
    return path, old_obs


def test_collection_expands_inventory_and_exports_only_descriptive_parameters(workspace):
    config, old_obs = fixtures(workspace)
    before = w.native.sha(old_obs)
    report = w.build(config, workspace / "package", workspace / "report")
    out, proof = w.verify(workspace / "package")
    assert proof["combined_reservoirs"] == 3 and proof["new_complete_target_tables"] == 1
    assert proof["new_calibration_ready_reservoirs"] == proof["optimizer_evaluations"] == proof["new_verified_GRAND_IDs"] == 0
    rows = w.native.read(out / "parameterization_registry.json")["profiles"]
    assert rows[0]["GRAND_ID"] is None and rows[0]["catalog_candidates"][0]["GRAND_ID"] == 99
    assert rows[0]["capacity_m3"] is None
    for row in rows:
        assert not row["calibration_ready"] and not row["simulation_ready"] and not row["parameters_ready"]
        assert row["fitting_scope"] == "all_available_records_descriptive_no_assessment"
    record, frame = w.load_observations(out, reservoir_id="new")
    assert frame.inflow_m3s.isna().all() and frame.release_m3s.eq(0).any()
    assert w.load_observations(out, grand_id=8)[0]["reservoir_id"] == "old"
    with pytest.raises(ValueError, match="verified"): w.load_observations(out, grand_id=99)
    assert w.native.sha(old_obs) == before
    assert w.build(config, out, report) == report
    table = pd.read_csv(report / "new_reservoir_parameterization_by_GRAND_ID.csv")
    assert table.GRAND_ID.isna().all()
    assert table.loc[table.reservoir_id == "partial", "release_normal_m3s_month_01"].isna().all()
    observation = out / "new/observations_si.csv"; observation.write_text(observation.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"): w.verify(out)


def test_existing_aliases_and_names_cannot_create_duplicate_reservoirs(workspace):
    path, _ = fixtures(workspace)
    cfg = w.native.read(path)
    cfg["reservoirs"][0]["name"] = "Old Alias"; w.write(path, cfg)
    with pytest.raises(ValueError, match="Duplicate"): w.config(path)
    cfg["reservoirs"][0]["name"] = "Existing Lake"; w.write(path, cfg)
    with pytest.raises(ValueError, match="duplicates"): w.build(path, workspace / "package", workspace / "report")


def test_source_paths_cannot_escape_archive(workspace):
    archive = workspace / "archive"; archive.mkdir()
    (workspace / "outside.txt").write_text("data")
    with pytest.raises(ValueError, match="escapes"): w.source_path(archive, "../outside.txt")


def test_release_proxy_aggregation_is_not_accepted(workspace):
    path, _ = fixtures(workspace); cfg = w.native.read(path)
    cfg["reservoirs"][0]["release"]["duplicate_policy"] = "daily_arithmetic_mean_proxy"; w.write(path, cfg)
    with pytest.raises(ValueError, match="discharge"): w.config(path)
