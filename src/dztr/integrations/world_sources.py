"""Discover explicit, auditable configurations in the historical world folders.

This is a limited static reader, not a MATLAB interpreter. It selects the one
unprefixed ``SimRIS[M]<number>.m`` script, follows its named coefficient file,
and rejects unknown simulation or release math. Physical inconsistencies are
reported without rewriting the original targets or constructing the stricter
daily ``ReservoirParameters`` contract.
"""

from __future__ import annotations

from datetime import date
import hashlib
import math
from pathlib import Path
import re
from typing import Any

from dztr.schemas import MONTHLY_TARGET_COLUMNS


_FOLDER = re.compile(r"ex(\d+)_(.+)", re.IGNORECASE)
_BASE = re.compile(r"SimRIS(?:M)?\d+\.m", re.IGNORECASE)
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_RULE = """function [qout] = S2Q1JAN(Fu,Ld,Lc,Ln,Lf,qmin,qnorm,qnd,dt,smax,Inflow,qds,cse)
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
end"""
_BALANCE = """for i = 2:1:numel(stoObs(:,1))
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
{overflow}
datelooping = datelooping + tinterval;
end"""
_OVERFLOW = """if stoSIM(i,1)>smax
flowSIM(i,1) = flowSIM(i,1) + ((stoSIM(i,1)-smax)/dt);
stoSIM(i,1) = smax;
end"""


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _uncomment(line: str) -> str:
    """Keep percent characters in MATLAB strings and ignore true comments."""
    quoted = False
    previous = ""
    index = 0
    while index < len(line):
        char = line[index]
        if char == "'":
            if quoted and index + 1 < len(line) and line[index + 1] == "'":
                index += 2
                continue
            if quoted:
                quoted = False
            elif not previous or previous in "=([,;:+-*/":
                quoted = True
        elif char == "%" and not quoted:
            return line[:index].strip()
        if not char.isspace():
            previous = char
        index += 1
    return line.strip()


class _Source:
    def __init__(self, path: Path, folder: Path, spec: dict[str, Any]):
        resolved = path.resolve()
        if not resolved.is_relative_to(folder.resolve()):
            raise ValueError(f"Source path escapes its reservoir folder: {path.name}.")
        self.path = resolved
        contents = resolved.read_bytes()
        self.contents = contents
        spec["source_sha256"][str(resolved)] = hashlib.sha256(contents).hexdigest()
        self.lines = [(number, _uncomment(line)) for number, line in
                      enumerate(contents.decode("utf-8-sig").splitlines(), 1)]
        self.lines = [(number, code) for number, code in self.lines if code]
        self.text = "\n".join(code for _, code in self.lines)
        self.spec = spec

    def evidence(self, number: int, code: str, field: str) -> None:
        self.spec["evidence"].append({"file": str(self.path), "line": number,
                                      "field": field, "code": code})

    def assignment(self, name: str, *, optional: bool = False) -> str | None:
        pattern = re.compile(r"(?:^|;)\s*" + re.escape(name) + r"\s*=\s*([^;]+)")
        matches = [(number, code, match.group(1).strip())
                   for number, code in self.lines for match in pattern.finditer(code)]
        if not matches and optional:
            return None
        if len(matches) != 1:
            raise ValueError(f"Expected one active assignment for {name}; found {len(matches)}.")
        number, code, expression = matches[0]
        self.evidence(number, code, name)
        return expression

    def number(self, name: str) -> float:
        expression = self.assignment(name)
        if not expression or not re.fullmatch(_NUMBER, expression):
            raise ValueError(f"Unsupported nonliteral numeric assignment for {name}: {expression}.")
        value = float(expression)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        return value


def _date_expression(expression: str | None, name: str) -> str:
    value = re.fullmatch(r"datenum\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", expression or "")
    if value:
        return date(*(int(part) for part in value.groups())).isoformat()
    value = re.fullmatch(r"'(\d{4})[/-](\d{1,2})[/-](\d{1,2})'", expression or "")
    if value:
        return date(*(int(part) for part in value.groups())).isoformat()
    raise ValueError(f"Unsupported calendar assignment for {name}: {expression}.")


def _coefficients(source: _Source) -> tuple[list[float], list[list[float]]]:
    text = source.contents.decode("utf-8-sig")
    tokens = text.split()
    if len(tokens) != 232:
        raise ValueError("MATLAB coefficients require exactly 232 values.")
    try:
        values = [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("MATLAB coefficients must be numeric.") from exc
    # A positive infinite channel cap is a real MATLAB convention. Encode it
    # explicitly below rather than emitting nonstandard JSON Infinity.
    if any(not math.isfinite(value) for index, value in enumerate(values) if index != 3):
        raise ValueError("Only the channel-capacity constant may be positive Infinity.")
    if not math.isfinite(values[3]) and values[3] != math.inf:
        raise ValueError("Channel capacity must be finite or positive Infinity.")
    constants = values[:4]
    table = [values[4 + index * 12:4 + (index + 1) * 12] for index in range(19)]
    if table[0] != list(range(1, 13)):
        raise ValueError("MATLAB coefficient month header must be exactly 1 through 12.")
    if constants[0] <= 0:
        raise ValueError("Reservoir capacity must be positive to interpret storage fractions.")
    seen = 0
    for number, line in enumerate(text.splitlines(), 1):
        if seen < 4 and line.split():
            source.evidence(number, line.strip(), "coefficient_constants")
            seen += len(line.split())
    return constants, table


def _physical_warnings(spec: dict[str, Any], table: list[list[float]]) -> None:
    model = spec["model"]
    capacity = model["capacity_m3"]
    dead = model["dead_storage_fraction"] * capacity
    conditions = {
        "Initial coefficient storage is outside [0, capacity]; preserved provisionally.":
            not 0 <= model["initial_storage_m3"] <= capacity,
        "Dead storage is outside [0, capacity); original expression is preserved.":
            not 0 <= model["dead_storage_fraction"] < 1,
        "Critical storage reaches or falls below dead storage in at least one month.":
            any(value <= dead for value in table[1]),
        "Storage target ordering is degenerate or reversed in at least one month.":
            any(not critical < normal < flood for critical, normal, flood in zip(*table[1:4])),
        "Flood storage exceeds physical capacity in at least one month.":
            any(value > capacity for value in table[3]),
        "Negative monthly targets are present; original values are preserved.":
            any(value < 0 for row in table[1:7] for value in row),
        "Release target ordering is reversed in at least one month.":
            any(not critical <= normal <= flood for critical, normal, flood in zip(*table[4:7])),
        "Calibration bounds contain negative values or lower exceeds upper.":
            any(low < 0 or high < low for lower, upper in
                ((table[7:10], table[10:13]), (table[13:16], table[16:19]))
                for low_row, high_row in zip(lower, upper) for low, high in zip(low_row, high_row)),
    }
    spec["warnings"].extend(message for message, condition in conditions.items() if condition)
    if model["initial_release_m3s"] < 0 or (model["channel_capacity_m3s"] is not None and
                                            model["channel_capacity_m3s"] < 0):
        spec["warnings"].append("Negative physical release constant is preserved for review.")


def _initial_source(script: _Source, variable: str, references: dict[str, str]) -> str:
    expression = _compact(script.assignment(variable) or "")
    if expression not in references:
        raise ValueError(f"Unsupported initial condition for {variable}: {expression}.")
    return references[expression]


def _read_inflow(script: _Source) -> dict[str, float]:
    expression = _compact(script.assignment("flowI") or "")
    plain = r"res_info_out\.inflow\(t[13]:t[24],1\)"
    if re.fullmatch(plain, expression):
        return {"factor": 1.0, "offset_m3s": 0.0}
    match = re.fullmatch(plain + r"\*([+-]?[\d.eE]+)", expression)
    if match and re.fullmatch(_NUMBER, match[1]):
        return {"factor": float(match[1]), "offset_m3s": 0.0}
    match = re.fullmatch(plain + r"([+-][\d.eE]+)", expression)
    if match and re.fullmatch(_NUMBER, match[1]):
        return {"factor": 1.0, "offset_m3s": float(match[1])}
    match = re.fullmatch(r"\(" + plain + r"\.\*(\w+)\)\+(\w+)", expression)
    if match:
        return {"factor": script.number(match[1]), "offset_m3s": script.number(match[2])}
    raise ValueError(f"Unsupported inflow preprocessing: {expression}.")


def _reporting_shifts(script: _Source) -> tuple[int, int]:
    release, storage = 0, 0
    for number, code in script.lines:
        normalized = _compact(code)
        for variable, expected, offset in (
            ("flowSIM", "flowSIM=[flowSIM(1:4,1);flowSIM(1:end-4,1)];", 4),
            ("stoSIM", "stoSIM=[stoSIM(4:end,1);stoSIM(end-2:end,1)];", -3),
        ):
            if re.match(variable + r"=\[", normalized):
                if normalized != expected:
                    raise ValueError(f"Unsupported reported-output transformation: {code}.")
                script.evidence(number, code, "legacy_" + variable + "_shift")
                if variable == "flowSIM":
                    release = offset
                else:
                    storage = offset
    return release, storage


def _writer_mapping(source: _Source) -> dict[str, Any]:
    storage = _compact(source.assignment("res_info_out.stoobs") or "")
    match = re.fullmatch(r"resdata\(k1:k2,4\)(?:\.\*([\d.eE+-]+))?", storage)
    if not match or (match[1] and not re.fullmatch(_NUMBER, match[1])):
        raise ValueError("Unsupported workbook storage conversion in writer.")
    factor = float(match[1]) if match[1] else 1.0
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Writer storage factor must be finite and positive.")
    for field, column in (("inflow", 5), ("outflow", 6)):
        if _compact(source.assignment("res_info_out." + field) or "") != f"resdata(k1:k2,{column})":
            raise ValueError("Unsupported workbook flow conversion in writer.")
    aggregation = "none"
    normalized = _compact(source.text)
    if "resdata=resdata2;" in normalized:
        if "mean(resdata((resdata(:,1)==i&resdata(:,2)==j),:),1)" not in normalized:
            raise ValueError("Unsupported workbook monthly aggregation.")
        aggregation = "unweighted_monthly_mean"
    # Most writers synthesize dates and never interpret columns 1..3. The
    # observation adapter must independently resolve those calendar columns.
    return {
        "storage_column": 4, "inflow_column": 5, "release_column": 6,
        "storage_factor": factor,
        "measurement_columns": [4, 5, 6],
        "date_column_order": ["year", "month", "day"] if aggregation != "none" else None,
        "year_column": 1 if aggregation != "none" else None,
        "month_column": 2 if aggregation != "none" else None,
        "day_column": 3 if aggregation != "none" else None,
        "year_offset": None, "aggregation": aggregation,
        "date_mode": "synthetic_row_order",
        "calendar_source": "workbook_month_labels" if source.spec["sampling"]["frequency"] == "monthly" else "writer_generated_row_order",
        "workbook_legacy_start": _date_expression(source.assignment("strdate2"), "strdate2"),
        "writer_start": _date_expression(source.assignment("strdate"), "strdate"),
        "writer_end": _date_expression(source.assignment("enddate"), "enddate"),
        "source_start": _date_expression(source.assignment("strdate2"), "strdate2"),
        "source_end": _date_expression(source.assignment("enddate2"), "enddate2"),
    }


def _read_writers(folder: Path, spec: dict[str, Any]) -> None:
    candidates: list[_Source] = []
    for path in sorted(folder.glob("*.m"), key=lambda value: value.name.casefold()):
        if "res_fit" not in path.name.casefold():
            continue
        source = _Source(path, folder, spec)
        if "res_info_out.stoobs" in source.text and re.search(r"\bsave\s+res_info_out\.mat", source.text):
            candidates.append(source)
    spec["writer_candidates"] = [str(source.path) for source in candidates]
    spec["writer_file"] = str(candidates[0].path) if len(candidates) == 1 else None
    mappings = []
    for source in candidates:
        try:
            mappings.append(_writer_mapping(source))
        except ValueError as exc:
            spec["warnings"].append(f"Writer {source.path.name} requires explicit review: {exc}")
            mappings.append(None)
    spec["observation_mapping"] = mappings[0] if mappings and mappings[0] is not None and all(
        mapping == mappings[0] for mapping in mappings) else None
    if len(candidates) > 1:
        spec["warnings"].append("Multiple observation/parameter writers are present; no authoritative writer is selected.")
    if not candidates:
        spec["warnings"].append("No local observation writer establishes workbook units or calendar conversion.")
    if candidates and spec["observation_mapping"] is None:
        spec["warnings"].append("Workbook mapping is unresolved; saved artifacts need independent unit/calendar review.")


def _read_folder(folder: Path, spec: dict[str, Any]) -> None:
    candidates = sorted((path for path in folder.iterdir() if path.is_file() and _BASE.fullmatch(path.name)),
                        key=lambda value: value.name.casefold())
    spec["base_script_candidates"] = [str(path.resolve()) for path in candidates]
    if len(candidates) != 1:
        raise ValueError(f"Expected one unprefixed base simulation script; found {len(candidates)}.")
    script = _Source(candidates[0], folder, spec)
    spec["base_script"] = str(script.path)
    # Stop before scoring/plotting sections, where similarly named arrays and
    # date variables are reused. This deliberately reads the actual base run.
    warmup = next((index for index, (_, code) in enumerate(script.lines)
                   if re.match(r"wrmp\s*=", code)), None)
    if warmup is None:
        raise ValueError("Base script has no explicit warm-up selector wrmp.")
    script.lines = script.lines[:warmup + 1]
    script.text = "\n".join(code for _, code in script.lines)
    external_forcing = bool(re.search(r"\b(?:readtable|readmatrix|csvread)\s*\(|res_info_out\.inflow\s*=", script.text))
    if re.search(r"\b(?:eval|run)\s*\(", script.text):
        raise ValueError("Base script dynamically evaluates code; static interpretation is unsupported.")
    spec["simulation_supported"] = not external_forcing
    if external_forcing:
        spec["simulation_unsupported_reason"] = "Base script replaces observations with external forcing; an explicit external-inflow adapter is required."
        spec["warnings"].append(spec["simulation_unsupported_reason"])

    filename = script.assignment("filename") or ""
    literals = re.findall(r"'((?:[^']|'')*)'", filename)
    if len(literals) != 2 or literals[0] not in ("\\", "/"):
        raise ValueError("Unsupported coefficient filename expression; require [my_dir,'\\','filename.txt'].")
    leaf = literals[1]
    if "/" in leaf or "\\" in leaf or ":" in leaf or leaf in (".", "..") or not leaf.endswith(".txt"):
        raise ValueError("Coefficient reference must name a local .txt file without path traversal.")
    coefficients = _Source(folder / leaf, folder, spec)
    spec["coefficient_file"] = str(coefficients.path)
    constants, table = _coefficients(coefficients)
    capacity, release, storage, channel = constants
    dead_expression = script.assignment("Ld") or ""
    dead_normalized = _compact(dead_expression)
    if re.fullmatch(_NUMBER, dead_normalized):
        dead = float(dead_normalized)
    else:
        match = re.fullmatch(r"(?:(" + _NUMBER + r")\*)?min\(Lc\)", dead_normalized)
        if not match:
            raise ValueError(f"Unsupported dead-storage expression: {dead_expression}.")
        dead = (float(match[1]) if match[1] else 1.0) * min(value / capacity for value in table[1])
    icor, cse = script.number("icor"), script.number("cse")
    if icor < 0 or cse not in (0, 1):
        raise ValueError("Require nonnegative icor and explicit cse 0 or 1.")
    spec["model"] = {"capacity_m3": capacity, "initial_release_m3s": release,
                     "initial_storage_m3": storage,
                     "channel_capacity_m3s": channel if math.isfinite(channel) else None,
                     "channel_capacity_mode": "finite" if math.isfinite(channel) else "unbounded",
                     "dead_storage_fraction": dead, "dead_storage_expression": dead_expression,
                     "inflow_correction": icor, "flood_mode": "storage" if cse == 1 else "inflow"}
    spec["raw_monthly_targets"] = dict(zip(MONTHLY_TARGET_COLUMNS, table[1:7]))
    spec["raw_coefficient_table"] = table
    spec["raw_bounds"] = {
        **dict(zip(("lower_" + key for key in MONTHLY_TARGET_COLUMNS), table[13:16] + table[7:10])),
        **dict(zip(("upper_" + key for key in MONTHLY_TARGET_COLUMNS), table[16:19] + table[10:13])),
    }
    _physical_warnings(spec, table)
    spec["initial_storage_source"] = _initial_source(script, "stoSIM(1,1)",
                                                    {"stoObs(1,1)": "observed", "Intstor1(1,1)": "coefficient"})
    spec["initial_release_source"] = _initial_source(script, "flowSIM(1,1)", {"Intflow(1,1)": "coefficient"})
    timestep = script.number("tinterval")
    if timestep not in (1, 30.5):
        raise ValueError(f"Unsupported sampling interval {timestep}; require 1 or 30.5 days.")
    start = _date_expression(script.assignment("date1"), "date1")
    end = _date_expression(script.assignment("date2"), "date2")
    legacy_start = _date_expression(script.assignment("strtime"), "strtime")
    if end < start or legacy_start != start:
        raise ValueError("Legacy start must match selection start and dates must be ordered.")
    spec["sampling"] = {"frequency": "daily" if timestep == 1 else "monthly",
                        "timestep_days": timestep, "legacy_start": legacy_start,
                        "month_clock": "legacy_synthetic"}
    selection: dict[str, Any] = {"start": start, "end": end, "t1": None, "t2": None}
    for output, alternatives, target in (("t1", ("t1", "t3"), "date1"), ("t2", ("t2", "t4"), "date2")):
        present = [(name, script.assignment(name, optional=True)) for name in alternatives]
        present = [(name, expression) for name, expression in present if expression is not None]
        if len(present) != 1:
            raise ValueError(f"Expected one observation selector for {output}.")
        name, expression = present[0]
        if timestep == 30.5:
            if not re.fullmatch(r"\d+", expression):
                raise ValueError("Monthly selection requires explicit positive row selectors.")
            selection[output] = int(expression)
            if selection[output] <= 0:
                raise ValueError("MATLAB row selectors are one-based and positive.")
        elif _compact(expression) != f"find(res_info_out.timed(:,1)=={target})":
            raise ValueError("Daily selection requires explicit date-matching selectors.")
        selection[output + "_variable"] = name
    if timestep == 30.5 and selection["t2"] < selection["t1"]:
        raise ValueError("Monthly row selectors must be ordered.")
    spec["selection"] = selection
    observation_slice = selection["t1_variable"] + ":" + selection["t2_variable"]
    for variable, field in (("stoObs", "stoobs"), ("flowObs", "outflow")):
        expected = f"res_info_out.{field}({observation_slice},1)"
        if _compact(script.assignment(variable) or "") != expected:
            raise ValueError(f"Unsupported observation slice or conversion for {variable}.")
    wrmp, iterations = script.number("wrmp"), script.number("niter")
    if wrmp < 1 or not wrmp.is_integer() or iterations < 1 or not iterations.is_integer():
        raise ValueError("Warm-up and iteration counts must be positive integers.")
    spec["warmup_rows"] = int(wrmp) - 1
    spec["fixed_point_iterations"] = int(iterations)
    if external_forcing:
        spec["inflow_preprocessing"] = None
        spec["inflow_source"] = "external"
        # Record Brazeau's actual last assignment without guessing how to
        # align external files. Keep its observation import available.
        assignments = [(number, code) for number, code in script.lines
                       if re.match(r"(?:flowI|res_info_out\.inflow)\s*=", code)]
        for number, code in assignments:
            script.evidence(number, code, "unsupported_external_inflow")
        spec["external_inflow_evidence"] = [code for _, code in assignments]
    else:
        spec["inflow_preprocessing"] = _read_inflow(script)
        spec["inflow_source"] = "observed"
    spec["legacy_release_shift_rows"], spec["legacy_storage_shift_rows"] = _reporting_shifts(script)

    normalized = _compact(script.text)
    with_overflow = _compact(_BALANCE.format(overflow=_OVERFLOW))
    without_overflow = _compact(_BALANCE.format(overflow=""))
    if with_overflow in normalized:
        overflow_mode = "matlab"
        balance = with_overflow
    elif without_overflow in normalized:
        overflow_mode = "none"
        balance = without_overflow
    else:
        raise ValueError("Unsupported base-script balance, fixed-point order, or overflow math.")
    spec["overflow_mode"] = overflow_mode
    suffix = normalized.split(balance, 1)[1]
    for reporting in ("flowSIM=[flowSIM(1:4,1);flowSIM(1:end-4,1)];",
                      "stoSIM=[stoSIM(4:end,1);stoSIM(end-2:end,1)];"):
        suffix = suffix.replace(reporting, "")
    if suffix != "wrmp=" + _compact(script.assignment("wrmp") or "") + ";":
        raise ValueError("Unsupported post-simulation transformation before scoring.")
    for variable, expected in (("coeff", "importdata(filename,delimiterIn,headerlinesIn)"),
                               ("coeff2", "reshape(coeff(5:end,1),12,19)'"),
                               ("smax", "coeff(1,1)"), ("Intflow", "coeff(2,1)"),
                               ("Intstor1", "coeff(3,1)"), ("qds", "coeff(4,1)"),
                               ("Lc", "coeff2(2,:)./smax"), ("Ln", "coeff2(3,:)./smax"),
                               ("Lf", "coeff2(4,:)./smax"), ("qmin", "coeff2(5,:)"),
                               ("qnorm", "coeff2(6,:)"), ("qnd", "coeff2(7,:)"),
                               ("dt", "60*60*(tinterval*24)"),
                               ("datelooping", "datenum(strtime,formatOuttime)+tinterval")):
        # datelooping's only subsequent assignment belongs to the exact loop
        # already checked. Inspect initialization separately for this field.
        if variable == "datelooping":
            initialization = [(number, code) for number, code in script.lines
                              if re.match(r"datelooping\s*=", code) and "datenum" in code]
            if len(initialization) != 1:
                raise ValueError("Unsupported month-clock initialization.")
        elif _compact(script.assignment(variable) or "") != expected:
            raise ValueError(f"Unsupported coefficient/time interpretation for {variable}.")
    for required in ("coeff2=reshape(coeff(5:end,1),12,19)';", "smax=coeff(1,1);",
                     "Intflow=coeff(2,1);", "Intstor1=coeff(3,1);", "qds=coeff(4,1);",
                     "Lc=coeff2(2,:)./smax;", "Ln=coeff2(3,:)./smax;", "Lf=coeff2(4,:)./smax;",
                     "qmin=coeff2(5,:);", "qnorm=coeff2(6,:);", "qnd=coeff2(7,:);",
                     "dt=60*60*(tinterval*24);", "datelooping=datenum(strtime,formatOuttime)+tinterval;"):
        if required not in normalized:
            raise ValueError(f"Unsupported coefficient/time interpretation; missing {required}.")
    rule = _Source(folder / "S2Q1JAN.m", folder, spec)
    spec["operating_rule_file"] = str(rule.path)
    allowed = {_compact(_RULE), _compact(_RULE.replace("((Fu-Ld)*smax/(dt))", "(((Fu-Ld)*smax)/dt)"))}
    if _compact(rule.text) not in allowed:
        raise ValueError("Unsupported S2Q1JAN release math; do not assume Trinity equivalence.")
    _read_writers(folder, spec)
    if spec["inflow_preprocessing"] is not None and spec["inflow_preprocessing"] != {"factor": 1.0, "offset_m3s": 0.0}:
        spec["warnings"].append("Base script changes raw inflow before the balance and release rule; preserve this separately from icor.")
    if timestep == 30.5:
        spec["warnings"].append("MATLAB uses fixed 30.5-day steps and synthetic month dates, which drift from calendar months.")
    if overflow_mode == "none":
        spec["warnings"].append("Base script omits overflow/spill adjustment; legacy storage may exceed capacity.")
    if spec["legacy_release_shift_rows"] or spec["legacy_storage_shift_rows"]:
        spec["warnings"].append("MATLAB shifts reported outputs after simulation; physical balance outputs must remain separate.")


def discover_world_sources(source_root: str | Path) -> list[dict[str, Any]]:
    """Return JSON-ready per-folder specs in numeric ``ex`` order.

    Each folder is isolated: ambiguity, a missing named file, path traversal,
    or unsupported math produces an ``error`` record while other folders are
    still inspected. Nothing is executed, inferred from the latest timestamp,
    or written to ``source_root``. Source hashes are checked again at the end
    of each folder's read to detect concurrent edits.
    """
    root = Path(source_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("World source root must be a directory.")
    folders = sorted((path for path in root.iterdir() if path.is_dir() and _FOLDER.fullmatch(path.name)),
                     key=lambda path: (int(_FOLDER.fullmatch(path.name)[1]), path.name.casefold()))
    results = []
    for folder in folders:
        spec: dict[str, Any] = {
            "reservoir_id": folder.name, "folder": str(folder.resolve()),
            "name": _FOLDER.fullmatch(folder.name)[2].replace("_", " "),
            "base_script": None, "coefficient_file": None, "operating_rule_file": None,
            "writer_file": None, "source_sha256": {}, "model": None, "sampling": None,
            "selection": None, "initial_storage_source": None, "initial_release_source": None,
            "warmup_rows": None, "legacy_release_shift_rows": 0, "legacy_storage_shift_rows": 0,
            "status": "ready", "warnings": ["Static base-script selection does not establish which script last produced saved artifacts."],
            "evidence": [],
        }
        try:
            if not folder.resolve().is_relative_to(root):
                raise ValueError("Reservoir folder resolves outside the requested source root.")
            _read_folder(folder, spec)
        except (ValueError, OSError, UnicodeError, OverflowError) as exc:
            spec["status"] = "error"
            spec["error"] = str(exc)
        for name in ("res_info.xlsx", "res_info_out.mat"):
            path = folder / name
            if path.is_file():
                try:
                    resolved = path.resolve()
                    if not resolved.is_relative_to(folder.resolve()):
                        raise ValueError(f"Observation path escapes its reservoir folder: {name}.")
                    spec["source_sha256"][str(resolved)] = hashlib.sha256(resolved.read_bytes()).hexdigest()
                except (OSError, ValueError) as exc:
                    spec["status"] = "error"
                    spec["error"] = str(exc)
        for filename, expected in spec["source_sha256"].items():
            try:
                actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"Source changed during discovery: {Path(filename).name}.")
            except (OSError, ValueError) as exc:
                spec["status"] = "error"
                spec["error"] = str(exc)
                break
        results.append(spec)
    return results
