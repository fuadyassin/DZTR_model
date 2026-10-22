"""Reproduce the committed Gardiner/Trinity parameters and audit local datasets."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dztr.data import find_reservoir_file, read_reservoir_observations

from .constants import MONTH_NAMES
from .quantiles import estimate_percentiles
from .targets import estimate_generalized_parameters


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_observation_folder(folder: str | Path) -> pd.DataFrame:
    """Report availability without computing any parameters for placeholders."""
    rows = []
    for path in sorted(Path(folder).glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        row = {"file": path.name}
        try:
            observations = read_reservoir_observations(path)
            quality = observations.quality
            row.update(status=quality.status, row_count=quality.row_count, missing_days=quality.missing_days)
            row.update(quality.missing_counts)
            row["issues"] = " | ".join(quality.issues)
        except (ValueError, TypeError) as exc:
            row.update(status="invalid", issues=str(exc))
        rows.append(row)
    return pd.DataFrame(rows)


def reproduce_benchmarks(repo_root: str | Path, output_dir: str | Path) -> dict:
    """Write reproducible SI estimates, an input audit, and a comparison report.

    The 780 saved quantiles per reservoir are compared in the legacy workbook's
    units using rtol=1e-10, atol=1e-8. P99 is newly computed and not present in the
    reference workbook. Original workbooks are never modified.
    """
    root = Path(repo_root).resolve()
    output = Path(output_dir)
    config = json.loads((root / "configs" / "benchmarks.json").read_text(encoding="utf-8"))
    reference_path = root / config["reference_parameters"]
    catalog = pd.read_excel(root / config["catalog"])
    reference = pd.read_excel(reference_path)
    folder = root / config["observations_folder"]
    report = {
        "quantile_method": "linear",
        "units": {"storage": "m3", "flow": "m3/s"},
        "comparison_units": {"storage": "million_m3", "flow": "m3/s"},
        "comparison_tolerance": {"rtol": 1e-10, "atol": 1e-8},
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "reference_sha256": _sha256(reference_path),
        "benchmarks": [],
    }
    targets, quantiles = [], []
    for reservoir in config["reservoirs"]:
        identifier = reservoir["grand_id"]
        path = find_reservoir_file(folder, identifier)
        observations = read_reservoir_observations(path)
        calculated = estimate_percentiles(observations).frame
        saved_rows = reference.loc[reference["GRAND_ID"] == identifier]
        metadata_rows = catalog.loc[catalog["GRAND_ID"] == identifier]
        if len(saved_rows) != 1 or len(metadata_rows) != 1:
            raise ValueError(f"Expected exactly one reference and catalog row for GRAND_ID {identifier}.")
        saved, metadata = saved_rows.iloc[0], metadata_rows.iloc[0]
        comparison = {"compared_values": 0, "mismatches": [], "max_abs_errors": {}}
        for variable, metric, scale, unit in (
            ("storage_m3", "S", 1e6, "storage_million_m3"),
            ("release_m3s", "Q", 1.0, "release_m3s"),
            ("inflow_m3s", "I", 1.0, "inflow_m3s"),
        ):
            errors = []
            for item in calculated.loc[calculated["variable"] == variable].itertuples():
                column = f"{metric}{item.percentile:g}"
                if item.month:
                    column += f"_{MONTH_NAMES[item.month - 1]}"
                if column not in saved.index:
                    if item.percentile == 99:
                        continue
                    raise ValueError(f"Missing reference parameter {column} for GRAND_ID {identifier}.")
                actual, expected = item.value / scale, float(saved[column])
                comparison["compared_values"] += 1
                errors.append(abs(actual - expected))
                if not np.isclose(actual, expected, rtol=1e-10, atol=1e-8):
                    comparison["mismatches"].append(column)
            comparison["max_abs_errors"][unit] = max(errors, default=0.0)
        if comparison["compared_values"] != 780:
            raise ValueError(f"Expected 780 historical quantiles for GRAND_ID {identifier}.")
        known_channel = None if pd.isna(metadata["Q_DS_MAX"]) else float(metadata["Q_DS_MAX"])
        parameters = estimate_generalized_parameters(observations, channel_capacity_m3s=known_channel)
        monthly = parameters.monthly_targets.reset_index()
        monthly.insert(0, "grand_id", identifier)
        monthly["channel_capacity_m3s"] = parameters.channel_capacity_m3s
        targets.append(monthly)
        calculated = calculated.copy()
        calculated.insert(0, "grand_id", identifier)
        quantiles.append(calculated)
        p99 = float(calculated.loc[(calculated["variable"] == "release_m3s") & (calculated["month"] == 0) & (calculated["percentile"] == 99), "value"].iloc[0])
        issues = list(parameters.issues)
        if not np.isclose(float(metadata["SMAX"]), float(metadata["CAP_MCM"]) * 1e6):
            issues.append("SMAX and CAP_MCM disagree; retained as separate catalog values pending scientific review.")
        report["benchmarks"].append({
            "grand_id": identifier,
            "name": reservoir["name"],
            "source": path.relative_to(root).as_posix(),
            "source_sha256": _sha256(path),
            "daily_records": len(observations.frame),
            "start_date": observations.frame.index.min().strftime("%Y-%m-%d"),
            "end_date": observations.frame.index.max().strftime("%Y-%m-%d"),
            **comparison,
            "matches_reference": not comparison["mismatches"],
            "release_p99_m3s": p99,
            "channel_capacity_m3s": parameters.channel_capacity_m3s,
            "catalog_smax_m3": float(metadata["SMAX"]),
            "catalog_capacity_m3": float(metadata["CAP_MCM"]) * 1e6,
            "issues": issues,
        })
    audit = audit_observation_folder(folder)
    report["dataset_counts"] = dict(Counter(audit["status"]))
    report["passed"] = all(item["matches_reference"] for item in report["benchmarks"]) and not report["dataset_counts"].get("invalid", 0)
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(targets, ignore_index=True).to_csv(output / "monthly_targets_si.csv", index=False)
    pd.concat(quantiles, ignore_index=True).to_csv(output / "quantiles_si.csv", index=False)
    audit.to_csv(output / "dataset_audit.csv", index=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository containing the bundled workbooks.")
    parser.add_argument("--output", type=Path, default=Path("outputs/parameterization"))
    args = parser.parse_args()
    report = reproduce_benchmarks(args.repo, args.output)
    for item in report["benchmarks"]:
        print(f"{item['name']} ({item['grand_id']}): {item['compared_values']} reference quantiles, {len(item['mismatches'])} mismatches")
    print(f"Dataset status: {report['dataset_counts']}")
    print(f"Artifacts: {args.output.resolve()}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
