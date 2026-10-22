"""Portable source-selection and static-interpretation checks, without MATLAB."""

import hashlib
import json
from pathlib import Path

import pytest

from dztr.integrations import world_sources
from dztr.integrations.world_sources import discover_world_sources


RULE = """function [qout] = S2Q1JAN(Fu,Ld,Lc,Ln,Lf,qmin,qnorm,qnd,dt,smax,Inflow,qds,cse)
if Fu <= Ld
qout = 0;
elseif Fu > Ld && Fu <= Lc
qout = min(qmin,((Fu-Ld)*smax/(dt)));
elseif Fu > Lc && Fu <= Ln
qout = qmin + (qnorm-qmin)*((Fu-Lc)/(Ln-Lc));
elseif Fu > Ln && Fu <= Lf
if cse==1
qout = qnorm + ((Fu-Ln)/(Lf-Ln))*(qnd-qnorm);
else
qout = qnorm + ((Fu-Ln)/(Lf-Ln))*max((Inflow-qnorm),(qnd-qnorm));
end
else
qout = min(max(((Fu-Lf)*smax/dt),qnd),qds);
end
end
"""
BASE = """% Comments are not active settings: icor = 99;
my_dir='F:\\obsolete\\example';
filename = [my_dir,'\\','coefficents_original.txt']; delimiterIn=' '; headerlinesIn=0;
icor = 0.92; cse=0; Ld=0.1; niter=1000;
coeff=importdata(filename,delimiterIn,headerlinesIn);
coeff2=reshape(coeff(5:end,1),12,19)';
load([my_dir,'\\','res_info_out.mat']);
formatOuttime='yyyy/mm/dd'; strtime='2001/01/01';
date1=datenum(2001,1,1); date2=datenum(2002,12,31);
t1=find(res_info_out.timed(:,1)==date1);
t2=find(res_info_out.timed(:,1)==date2);
stoObs=res_info_out.stoobs(t1:t2,1); flowObs=res_info_out.outflow(t1:t2,1);
flowI=res_info_out.inflow(t1:t2,1);
smax=coeff(1,1); Intflow=coeff(2,1); Intstor1=coeff(3,1); qds=coeff(4,1);
Lc=coeff2(2,:)./smax; Ln=coeff2(3,:)./smax; Lf=coeff2(4,:)./smax;
qmin=coeff2(5,:); qnorm=coeff2(6,:); qnd=coeff2(7,:);
tinterval=1; dt=60*60*(tinterval*24);
datelooping=datenum(strtime,formatOuttime)+tinterval;
flowSIM=zeros(numel(flowObs),1); flowSIM(1,1)=Intflow(1,1);
stoSIM=zeros(numel(flowObs),1); stoSIM(1,1)=stoObs(1,1);
for i = 2:1:numel(stoObs(:,1))
 [yy, mm] = datevec(datelooping);
 stoflo = stoSIM(i-1,1) + (dt/2)*(flowI(i,1)*icor + flowI(i-1,1)*icor - flowSIM(i-1,1));
 Inflow = flowI(i,1);
 Fu = stoflo/smax;
 flowSIM(i,1) = S2Q1JAN(Fu,Ld,Lc(mm),Ln(mm),Lf(mm),qmin(mm),qnorm(mm),qnd(mm),dt,smax,Inflow,qds,cse);
 for it=1:niter
  stoSIM(i,1) = stoflo - (dt/2)*(flowSIM(i,1));
  Fu = stoSIM(i,1)/smax;
  flowSIM(i,1) = S2Q1JAN(Fu,Ld,Lc(mm),Ln(mm),Lf(mm),qmin(mm),qnorm(mm),qnd(mm),dt,smax,Inflow,qds,cse);
 end
 if stoSIM(i,1)>smax
  flowSIM(i,1) = flowSIM(i,1) + ((stoSIM(i,1)-smax)/dt);
  stoSIM(i,1) = smax;
 end
 datelooping = datelooping + tinterval;
end
wrmp=366;
% Plotting section reuses variables, not simulation settings.
icor=17; flowI=[];
"""
WRITER = """filename=[my_dir,'\\','res_info.xlsx'];
resdata=xlsread(filename,sheetA{1,1});
strdate2='2000/01/01'; enddate2='2002/12/31';
strdate='2001/01/01'; enddate='2002/12/31';
res_info_out.timed=timed;
res_info_out.stoobs=resdata(k1:k2,4).*1000000;
res_info_out.inflow=resdata(k1:k2,5);
res_info_out.outflow=resdata(k1:k2,6);
save res_info_out.mat res_info_out
fprintf(fileID,'%12.0f\\n',max_storage); % This format percent is a string.
"""


def write_folder(root, name="ex12_Test", *, base=BASE, rule=RULE, writer=WRITER, constants=None, table=None):
    folder = root / name
    folder.mkdir()
    (folder / "SimRISM14.m").write_text(base, encoding="utf-8")
    (folder / "S2Q1JAN.m").write_text(rule, encoding="utf-8")
    if writer is not None:
        (folder / "exampleRES_fit.m").write_text(writer, encoding="utf-8")
    if constants is None:
        constants = [100, 1, 45, 9]
    if table is None:
        table = [list(range(1, 13))] + [[value] * 12 for value in
                    (20, 50, 80, 1, 2, 3, 0, 1, 2, 2, 3, 4, 10, 40, 70, 30, 60, 90)]
    values = constants + [value for row in table for value in row]
    (folder / "coefficents_original.txt").write_text("\n".join(map(str, values)) + "\n", encoding="utf-8")
    return folder


def test_discovers_static_base_with_actual_named_coefficient_and_source_proofs(tmp_path):
    folder = write_folder(tmp_path)
    (folder / "coefficents_newer.txt").write_text("do not select the latest filename")
    (folder / "TS_SimRISM14_m.m").write_text("cse=999;")
    (folder / "res_info.xlsx").write_bytes(b"observation-workbook")
    (folder / "res_info_out.mat").write_bytes(b"observation-mat")
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    spec, = discover_world_sources(tmp_path)
    assert spec["status"] == "ready"
    assert Path(spec["coefficient_file"]).name == "coefficents_original.txt"
    assert spec["model"] == {
        "capacity_m3": 100, "initial_release_m3s": 1, "initial_storage_m3": 45,
        "channel_capacity_m3s": 9, "channel_capacity_mode": "finite",
        "dead_storage_fraction": 0.1, "dead_storage_expression": "0.1",
        "inflow_correction": 0.92, "flood_mode": "inflow",
    }
    assert spec["initial_storage_source"] == "observed"
    assert spec["initial_release_source"] == "coefficient"
    assert spec["sampling"] == {"frequency": "daily", "timestep_days": 1,
                                "legacy_start": "2001-01-01", "month_clock": "legacy_synthetic"}
    assert spec["warmup_rows"] == 365 and spec["fixed_point_iterations"] == 1000
    assert spec["overflow_mode"] == "matlab" and spec["simulation_supported"]
    assert spec["raw_monthly_targets"]["storage_normal_m3"] == [50] * 12
    assert spec["raw_bounds"]["lower_storage_critical_m3"] == [10] * 12
    assert spec["raw_bounds"]["upper_release_normal_m3s"] == [3] * 12
    assert spec["observation_mapping"]["storage_factor"] == 1e6
    assert spec["observation_mapping"]["workbook_legacy_start"] == "2000-01-01"
    assert spec["observation_mapping"]["date_column_order"] is None
    assert spec["observation_mapping"]["calendar_source"] == "writer_generated_row_order"
    assert len(spec["source_sha256"]) == 6
    for path, expected in spec["source_sha256"].items():
        assert Path(path).is_absolute()
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    assert any(row["field"] == "icor" and "0.92" in row["code"] for row in spec["evidence"])
    json.dumps(spec, allow_nan=False)
    assert before == {path.name: path.read_bytes() for path in folder.iterdir()}


def test_numeric_order_and_ambiguous_folder_isolation(tmp_path):
    write_folder(tmp_path, "ex10_Ten")
    ambiguous = write_folder(tmp_path, "ex2_Two")
    (ambiguous / "SimRIS14.m").write_text(BASE)
    (tmp_path / "unrelated").mkdir()
    rows = discover_world_sources(tmp_path)
    assert [row["reservoir_id"] for row in rows] == ["ex2_Two", "ex10_Ten"]
    assert [row["status"] for row in rows] == ["error", "ready"]
    assert "found 2" in rows[0]["error"]
    assert rows[0]["base_script"] is None


def test_variant_only_folder_is_not_promoted_to_base(tmp_path):
    folder = write_folder(tmp_path)
    (folder / "SimRISM14.m").rename(folder / "TS_SimRISM14_m.m")
    spec, = discover_world_sources(tmp_path)
    assert spec["status"] == "error" and "found 0" in spec["error"]


def test_monthly_row_selection_dead_minimum_and_reported_output_shifts(tmp_path):
    base = BASE.replace("Ld=0.1", "Ld=0.4*min(Lc)").replace("tinterval=1;", "tinterval=30.5;")
    base = base.replace("t1=find(res_info_out.timed(:,1)==date1)", "t1=1")
    base = base.replace("t2=find(res_info_out.timed(:,1)==date2)", "t2=24")
    base = base.replace("stoSIM(1,1)=stoObs(1,1)", "stoSIM(1,1)=Intstor1(1,1)")
    base = base.replace("wrmp=366;", "flowSIM = [flowSIM(1:4,1);flowSIM(1:end-4,1)];\nstoSIM = [stoSIM(4:end,1);stoSIM(end-2:end,1)];\nwrmp=13;")
    writer = WRITER.replace(".*1000000", "").replace("res_info_out.timed=timed;",
        "fufk=mean(resdata((resdata(:,1)==i & resdata(:,2)==j),:),1);\nresdata=resdata2;\nres_info_out.timed=timed;")
    write_folder(tmp_path, base=base, writer=writer)
    spec, = discover_world_sources(tmp_path)
    assert spec["status"] == "ready"
    assert spec["model"]["dead_storage_fraction"] == pytest.approx(0.08)
    assert spec["model"]["dead_storage_expression"] == "0.4*min(Lc)"
    assert spec["initial_storage_source"] == "coefficient"
    assert (spec["selection"]["t1"], spec["selection"]["t2"]) == (1, 24)
    assert spec["warmup_rows"] == 12
    assert spec["legacy_release_shift_rows"] == 4
    assert spec["legacy_storage_shift_rows"] == -3
    assert spec["observation_mapping"]["aggregation"] == "unweighted_monthly_mean"
    assert spec["observation_mapping"]["calendar_source"] == "workbook_month_labels"
    assert spec["observation_mapping"]["date_column_order"] == ["year", "month", "day"]


@pytest.mark.parametrize("expression,expected", [
    ("res_info_out.inflow(t1:t2,1)*1.45", {"factor": 1.45, "offset_m3s": 0}),
    ("res_info_out.inflow(t1:t2,1)+10", {"factor": 1, "offset_m3s": 10}),
    ("(res_info_out.inflow(t1:t2,1).*icorr2)+iadd", {"factor": 1.1, "offset_m3s": 8}),
])
def test_inflow_preprocessing_stays_separate_from_balance_correction(tmp_path, expression, expected):
    base = BASE.replace("flowI=res_info_out.inflow(t1:t2,1);", f"iadd=8; icorr2=1.1;\nflowI={expression};")
    write_folder(tmp_path, base=base)
    spec, = discover_world_sources(tmp_path)
    assert spec["status"] == "ready" and spec["inflow_preprocessing"] == expected
    assert spec["model"]["inflow_correction"] == 0.92


def test_bennett_uses_date_selectors_t3_t4(tmp_path):
    base = BASE.replace("t1", "t3").replace("t2", "t4")
    write_folder(tmp_path, base=base)
    spec, = discover_world_sources(tmp_path)
    assert spec["status"] == "ready"
    assert (spec["selection"]["t1_variable"], spec["selection"]["t2_variable"]) == ("t3", "t4")


def test_no_overflow_variant_and_oahe_equivalent_rule_are_explicit(tmp_path):
    base = BASE.replace(" if stoSIM(i,1)>smax\n  flowSIM(i,1) = flowSIM(i,1) + ((stoSIM(i,1)-smax)/dt);\n  stoSIM(i,1) = smax;\n end\n", "")
    rule = RULE.replace("((Fu-Ld)*smax/(dt))", "(((Fu-Ld)*smax)/dt)")
    write_folder(tmp_path, base=base, rule=rule)
    spec, = discover_world_sources(tmp_path)
    assert spec["status"] == "ready" and spec["overflow_mode"] == "none"


def test_external_forcing_keeps_observation_import_but_blocks_simulation(tmp_path):
    base = BASE.replace("tinterval=1;", "S=readtable('MESH_output_reach14.csv');\nres_info_out.inflow=Arels(:,2);\nflowI=res_info_out.inflow(t1:t2,1)-5;\ntinterval=1;")
    write_folder(tmp_path, base=base)
    spec, = discover_world_sources(tmp_path)
    assert spec["status"] == "ready" and not spec["simulation_supported"]
    assert spec["observation_mapping"]["storage_factor"] == 1e6
    assert spec["inflow_preprocessing"] is None and spec["inflow_source"] == "external"
    assert "external-inflow adapter" in spec["simulation_unsupported_reason"]
    assert any("-5" in code for code in spec["external_inflow_evidence"])


@pytest.mark.parametrize("change,match", [
    (lambda base: base.replace("tinterval=1;", "tinterval=2;"), "sampling interval"),
    (lambda base: base.replace("Ld=0.1", "Ld=system('evil')"), "dead-storage expression"),
    (lambda base: base.replace("stoSIM(1,1)=stoObs(1,1)", "stoSIM(1,1)=mean(stoObs)"), "initial condition"),
    (lambda base: base.replace("(dt/2)*(flowSIM(i,1))", "dt*(flowSIM(i,1))"), "balance"),
    (lambda base: base.replace("qnd=coeff2(7,:)", "qnd=coeff2(6,:)"), "coefficient/time interpretation"),
    (lambda base: base.replace("stoObs=res_info_out.stoobs(t1:t2,1)", "stoObs=res_info_out.stoobs(t1:t2,1)*2"), "observation slice"),
    (lambda base: base.replace("wrmp=366;", "flowSIM=flowSIM*2;\nwrmp=366;"), "post-simulation transformation"),
    (lambda base: base.replace("flowI=res_info_out.inflow(t1:t2,1)", "flowI=log(res_info_out.inflow(t1:t2,1))"), "inflow preprocessing"),
    (lambda base: base.replace("date1=datenum(2001,1,1)", "date1=datenum(2001,2,30)"), "day is out of range"),
    (lambda base: base.replace("strtime='2001/01/01'", "strtime='2000/01/01'"), "Legacy start"),
    (lambda base: base.replace("wrmp=366", "wrmp=0"), "positive integers"),
])
def test_unsupported_base_is_not_silently_interpreted_as_trinity(tmp_path, change, match):
    write_folder(tmp_path, base=change(BASE))
    spec, = discover_world_sources(tmp_path)
    assert spec["status"] == "error" and match in spec["error"]


def test_unknown_release_math_is_rejected(tmp_path):
    write_folder(tmp_path, rule=RULE.replace("qout = 0;", "qout = 1;"))
    spec, = discover_world_sources(tmp_path)
    assert spec["status"] == "error" and "release math" in spec["error"]


def test_dead_equality_and_flood_above_capacity_are_preserved_with_warnings(tmp_path):
    folder = write_folder(tmp_path, base=BASE.replace("Ld=0.1", "Ld=1.0*min(Lc)"))
    path = folder / "coefficents_original.txt"
    values = path.read_text().split()
    values[4 + 3 * 12] = "110"
    path.write_text("\n".join(values))
    spec, = discover_world_sources(tmp_path)
    assert spec["status"] == "ready" and spec["model"]["dead_storage_fraction"] == 0.2
    assert spec["raw_monthly_targets"]["storage_flood_m3"][0] == 110
    assert any("dead storage" in warning for warning in spec["warnings"])
    assert any("exceeds physical capacity" in warning for warning in spec["warnings"])


@pytest.mark.parametrize("change,match", [
    (lambda text: text + "\n0", "232"),
    (lambda text: "nonsense" + text[text.index("\n"):], "numeric"),
    (lambda text: text.replace("\n12\n20\n", "\n13\n20\n", 1), "month header"),
    (lambda text: "0" + text[text.index("\n"):], "positive"),
])
def test_malformed_coefficient_file_is_isolated(tmp_path, change, match):
    folder = write_folder(tmp_path)
    path = folder / "coefficents_original.txt"
    path.write_text(change(path.read_text()))
    spec, = discover_world_sources(tmp_path)
    assert spec["status"] == "error" and match in spec["error"]


def test_infinite_channel_is_explicit_and_json_ready(tmp_path):
    write_folder(tmp_path, constants=[100, 1, 45, float("inf")])
    spec, = discover_world_sources(tmp_path)
    assert spec["status"] == "ready" and spec["model"]["channel_capacity_m3s"] is None
    assert spec["model"]["channel_capacity_mode"] == "unbounded"
    json.dumps(spec, allow_nan=False)


def test_missing_named_file_and_path_traversal_are_not_replaced_by_latest(tmp_path):
    folder = write_folder(tmp_path)
    (folder / "coefficents_original.txt").rename(folder / "coefficents_other.txt")
    spec, = discover_world_sources(tmp_path)
    assert spec["status"] == "error" and "coefficents_original.txt" in spec["error"]
    (folder / "SimRISM14.m").write_text(BASE.replace("coefficents_original.txt", "../coefficents_other.txt"))
    spec, = discover_world_sources(tmp_path)
    assert spec["status"] == "error" and "path traversal" in spec["error"]


def test_coefficient_symlink_cannot_escape_folder(tmp_path):
    folder = write_folder(tmp_path)
    path = folder / "coefficents_original.txt"
    outside = tmp_path / "outside.txt"
    path.rename(outside)
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("Creating file symlinks is unavailable in this environment.")
    spec, = discover_world_sources(tmp_path)
    assert spec["status"] == "error" and "escapes" in spec["error"]


def test_source_changes_during_discovery_are_detected(tmp_path, monkeypatch):
    folder = write_folder(tmp_path)
    original = world_sources._Source.evidence

    def mutate_after_read(self, number, code, field):
        original(self, number, code, field)
        if field == "coefficient_constants":
            (folder / "SimRISM14.m").write_text(BASE + "% concurrent edit\n")

    monkeypatch.setattr(world_sources._Source, "evidence", mutate_after_read)
    spec, = discover_world_sources(tmp_path)
    assert spec["status"] == "error" and "changed during discovery" in spec["error"]


def test_ambiguous_writers_use_only_consensus_mapping_and_never_pick_latest(tmp_path):
    folder = write_folder(tmp_path)
    (folder / "alternateRES_fit.m").write_text(WRITER + "% alternate parameter targets\n")
    spec, = discover_world_sources(tmp_path)
    assert spec["status"] == "ready" and spec["writer_file"] is None
    assert len(spec["writer_candidates"]) == 2 and spec["observation_mapping"]["storage_factor"] == 1e6
    (folder / "alternateRES_fit.m").write_text(WRITER.replace(".*1000000", ""))
    spec, = discover_world_sources(tmp_path)
    assert spec["observation_mapping"] is None
    assert any("mapping is unresolved" in warning for warning in spec["warnings"])


def test_formatter_percent_and_commented_alternatives_remain_distinct():
    assert world_sources._uncomment("fprintf(fid,'%12.0f\\n',x); % comment") == "fprintf(fid,'%12.0f\\n',x);"
    assert world_sources._uncomment("% icor=17;") == ""
