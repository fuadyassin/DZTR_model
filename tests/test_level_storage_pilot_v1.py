"""Real-input pilot reuse/identity and reference selection without assessment."""
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from workflows import level_storage_pilot_v1 as pilot
from workflows import reservoir_v2 as w
from strategies.representative_years import YearSelection
from workflows.geometry_v1 import read_regeom

CONFIG = w.ROOT / "configs/run_manager_gardiner_geometry_v1.json"


@pytest.fixture
def workspace():
    parent = w.ROOT / "outputs"
    folder = Path(tempfile.mkdtemp(prefix="test_level_storage_", dir=parent)).resolve()
    yield folder
    assert folder.is_relative_to(parent.resolve()) and folder.name.startswith("test_level_storage_")
    shutil.rmtree(folder)


def test_plan_is_readonly_preserves_calendar_and_declares_no_optimizer():
    before = {p.name for p in (w.ROOT / "outputs").iterdir()}
    design = pilot.plan(CONFIG)
    assert design["new_optimizer_evaluations"] == 0
    assert not design["assessment_selects_reference"]
    assert design["selection_signature"] == "2680f2925c7cce95717ec7b477f39fe2579e390f888eef825616dec89f5b0b60"
    assert {p.name for p in (w.ROOT / "outputs").iterdir()} == before


def test_all_references_ignore_changed_held_out_values():
    cfg = w.read(CONFIG)
    paths = {k: pilot.app.reference(v) for k, v in cfg["inputs"].items()}
    frame = pilot.read_frame(paths["observations"])
    h = pilot.read_frame(paths["levels"]).level_m
    selection = YearSelection(**w.read(paths["selection"]))
    curve = read_regeom(paths["geometry"], expected_grand_id=253)
    original = pilot.references(cfg, frame, h, curve, selection, 9.4e9)
    mask = selection.mask(frame.index, "assessment")
    changed = frame.copy(); changed.loc[mask, "storage_m3"] *= 20.
    changed_h = h.copy(); changed_h.loc[mask] += 999.
    altered = pilot.references(cfg, changed, changed_h, curve, selection, 9.4e9)
    assert original == altered
    assert original["calibration_storage_alignment"][0].zero_level_m == pytest.approx(493.6684404334709)


def test_completed_pilot_reuses_jobs_and_rejects_tamper(workspace, monkeypatch):
    monkeypatch.setattr(pilot, "plot", lambda *a, **k: None)
    out = workspace / "run"
    before = w.sha(w.ROOT / "outputs/paper_forcing_metadata_v1_20261017/reservoir_parameterization_all_37_by_GRAND_ID.csv")
    result = pilot.run(CONFIG, out)
    # Two top-alignment scenarios have physical target violations; never repair
    # them just to obtain a simulation. The other two run the unchanged solver.
    assert result["new_fixed_profile_simulations"] == 2 and result["new_optimizer_evaluations"] == 0
    for method in ("full_supply_top", "calibration_maximum"):
        assert not w.read(out / method / "profile.json")["parameters_valid"]
    assert pilot.verify(out)["status"] == "verified"
    summary = w.read(out / "summary.json")
    assert summary["level_rows"] == summary["rows"] == 11688
    assert summary["calibration_fraction"] >= .65 and not summary["geometry_calibration_ready"]
    conversion = pd.read_csv(out / "calibration_storage_alignment/level_conversion_si.csv")
    assert conversion.storage_m3.notna().all() and conversion.storage_m3.between(0, 9.4e9).all()
    from tools.verify_level_storage_pilot_v1 import verify as independent_verify
    science = independent_verify(out)
    assert science["independently_verified_target_cells"] == 360
    assert science["original_inputs_rechecked"] == 928 and science["new_simulation_calls"] == 0
    def forbidden(*a, **k): raise AssertionError("A completed resume must not simulate")
    monkeypatch.setattr(pilot.app, "run_frame", forbidden)
    assert pilot.run(CONFIG, out, resume=True)["new_fixed_profile_simulations"] == 0
    assert before == w.sha(w.ROOT / "outputs/paper_forcing_metadata_v1_20261017/reservoir_parameterization_all_37_by_GRAND_ID.csv")
    with (out / "conversion_metrics.csv").open("a") as stream: stream.write("tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        pilot.run(CONFIG, out, resume=True)


def test_config_wrong_datum_and_unknown_flags_fail_before_running(workspace):
    cfg = w.read(CONFIG); cfg["level_datum"] = "unverified-datum"
    path = workspace / "config.json"; w.write(path, cfg)
    with pytest.raises(ValueError, match="datum/identity"):
        pilot.plan(path)
    cfg["invent_bed"] = True; w.write(path, cfg)
    with pytest.raises(ValueError, match="configuration"):
        pilot.plan(path)
