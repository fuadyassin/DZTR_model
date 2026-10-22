"""Publish verified population-study results, optionally waiting for completion.

This helper reads saved artifacts and produces a figure and Markdown note. It
does not run a search, change inputs, or score any reservoir simulations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from dztr.calibration.checkpoint import _replace_checkpoint
from dztr.calibration.population_study import verify_population_study
from plot_population_study import plot_population_comparison, read_verified_tables


NOTE_NAME = "population_results.md"
POLL_SECONDS = 30.0


def _manifest_digest(folder):
    return hashlib.sha256((folder / "study_manifest.json").read_bytes()).hexdigest()


def _final_source_reports(verified):
    reports = []
    for job in verified["jobs"]:
        proof = job["source_proof"]
        source = Path(proof["directory"]) / "summary.json"
        report = json.loads(source.read_text(encoding="utf-8"))
        if (report.get("checkpoint_identity") != proof["identity"]
                or report.get("search_evaluations") != job["search_evaluations"]):
            raise ValueError("A final source report changed after its population-study verification.")
        reports.append((job, report))
    return reports


def _number(value):
    return f"{float(value):.6f}"


def _metric(row, name):
    return (f"{_number(row[f'{name}_median'])} "
            f"({_number(row[f'{name}_min'])}–{_number(row[f'{name}_max'])})")


def render_results_note(grouped, agreement, verified, source_reports, manifest_digest):
    """Render descriptive, calibration-only findings from verified saved data."""
    settings = verified["settings"]
    final_count = int(settings["evaluation_checkpoints"][-1])
    populations = tuple(int(value) for value in settings["population_sizes"])
    seeds = tuple(int(value) for value in settings["seeds"])
    tolerance = float(settings["agreement_tolerance"])
    if len(source_reports) != verified["job_count"]:
        raise ValueError("Every completed population job requires a verified source report.")
    compared = int(verified["compared_search_evaluations"])
    expected = len(populations) * len(seeds) * final_count
    newly_scored = int(verified["new_search_evaluations_this_invocation"])
    restored = int(verified["reused_search_evaluations_this_invocation"])
    if compared != expected or newly_scored + restored != compared:
        raise ValueError("Study search accounting does not match the complete equal-budget job matrix.")
    lines = [
        "# Verified population study results",
        "",
        f"The completed study compares populations {', '.join(map(str, populations))} "
        f"across {len(seeds)} seeds at {final_count:,} cumulative search trials per job. "
        "All comparisons below use calibration objectives. Validation scores did not "
        "determine population comparisons, representative selection, or evaluation budgets.",
        "",
        "[Comparison figure](population_comparison.png) · [Study summary](summary.json) · "
        "[Population statistics](population_summary.csv)",
        "",
        "## Final-budget calibration performance",
        "",
        "Each cell gives the seed median followed by the seed minimum and maximum. "
        "These are descriptive seed ranges, not confidence intervals. Higher values "
        "are better. Hypervolume uses the fixed [1, 1] reference in negative-NSE coordinates.",
        "",
        "| Population | Hypervolume: median (min–max) | Best release NSE: median (min–max) | Best storage NSE: median (min–max) |",
        "| --- | --- | --- | --- |",
    ]
    final = grouped.loc[grouped.evaluations.eq(final_count)]
    for population in populations:
        rows = final.loc[final.population_size.eq(population)]
        if len(rows) != 1:
            raise ValueError("Final population statistics must contain exactly one row per planned population.")
        row = rows.iloc[0]
        lines.append(f"| {population} | {_metric(row, 'hypervolume')} | "
                     f"{_metric(row, 'max_release_nse')} | {_metric(row, 'max_storage_nse')} |")
    lines.extend([
        "",
        "Best release and best storage scores can belong to different archive members; "
        "these columns do not describe a single joint parameter set.",
        "",
        "## Seed-front agreement and target variability",
        "",
        "Front distances compare every seed pair using the full calibration Pareto fronts "
        "and symmetric Hausdorff distance with the infinity norm, in raw NSE units. "
        f"The illustrative agreement tolerance is {_number(tolerance)}. Target ranges "
        "describe the balanced representatives across seeds, divided by the fixed "
        "calibration bound width; the median and maximum summarize the 72 coordinates. "
        "These ranges describe variability, not parameter uncertainty or identifiability.",
        "",
        "| Population | Maximum seed-front distance | All pairs within tolerance | Balanced-target range: median / maximum |",
        "| --- | --- | --- | --- |",
    ])
    for population in populations:
        distances = agreement.loc[agreement.population_size.eq(population)
                                  & agreement.evaluations.eq(final_count), "front_distance"]
        if len(distances) != len(seeds) * (len(seeds) - 1) // 2:
            raise ValueError("Final front agreement must include every planned seed pair.")
        maximum = float(distances.max())
        row = final.loc[final.population_size.eq(population)].iloc[0]
        lines.append(f"| {population} | {_number(maximum)} | "
                     f"{'Yes' if maximum <= tolerance else 'No'} | "
                     f"{row.parameter_spread_median:.2%} / {row.parameter_spread_maximum:.2%} |")
    lines.extend([
        "",
        "## Search and reporting accounting",
        "",
        f"The study compares **{compared:,} search trials** "
        f"({len(populations) * len(seeds)} jobs × {final_count:,}). The last study-runner "
        f"invocation records **{newly_scored:,} new trials** and **{restored:,} reused trials**. "
        "The budget includes the starting baseline and every rejected search candidate.",
        "",
    ])
    calibration_replays = sum(int(report["reporting_calibration_evaluations"])
                              for _, report in source_reports)
    validation_runs = sum(int(report["reporting_validation_evaluations"])
                          for _, report in source_reports)
    lines.extend([
        f"The final saved source exports separately record **{calibration_replays:,} "
        f"selected-candidate calibration replays** and **{validation_runs:,} validation "
        "reporting runs**. These counts describe the last export per source. Earlier "
        "adaptive-stage exports and independent timing/profiling experiments are "
        "outside this accounting; this note does not claim total project-wide model calls. "
        "This post-processing helper performed no search trials or model evaluations.",
        "",
        "## Retrospective stopping diagnostics",
        "",
        "These statuses come from each source report's final calibration-archive "
        "assessment. They are provisional resource diagnostics and do not establish "
        "global convergence. The matched study budgets were fixed; retrospective "
        "statuses did not give any population a smaller comparison budget.",
        "",
        "| Population | Seed | Final status | Earliest stopping candidate | Later invalidation recorded |",
        "| --- | --- | --- | --- | --- |",
    ])
    for job, report in sorted(source_reports, key=lambda item: (item[0]["population_size"], item[0]["seed"])):
        candidate = report.get("earliest_stopping_candidate_evaluations")
        lines.append(f"| {job['population_size']} | {job['seed']} | {report['stopping_status']} | "
                     f"{'None' if candidate is None else f'{int(candidate):,}'} | "
                     f"{'Yes' if report.get('candidate_invalidated', False) else 'No'} |")
    lines.extend([
        "",
        "This is a descriptive pilot for one reservoir and a small seed set. "
        "Population size changes both initialization and the number of generations "
        "available at a fixed budget. Matching seed labels provides reproducibility "
        "controls; it does not create common random draws between populations. "
        "The results do not establish a fleet-wide population default, global "
        "optimizer convergence, or unique reservoir parameters.",
        "",
        "The study and source manifests were verified before publication. "
        f"Study manifest SHA256: `{manifest_digest}`.",
        "",
    ])
    return "\n".join(lines)


def finish_population_study(folder):
    """Verify complete sources, export the figure, and atomically publish a note."""
    folder = Path(folder).resolve()
    manifest_digest = _manifest_digest(folder)
    verified = verify_population_study(folder)
    grouped, agreement, table_summary = read_verified_tables(folder)
    if table_summary != verified or _manifest_digest(folder) != manifest_digest:
        raise ValueError("Population study changed while its final tables were being read.")
    source_reports = _final_source_reports(verified)
    text = render_results_note(grouped, agreement, verified, source_reports, manifest_digest)
    plot_population_comparison(grouped, agreement, folder)
    temporary = None
    output = folder / NOTE_NAME
    try:
        with tempfile.NamedTemporaryFile(prefix=".population_results_", suffix=".tmp",
                                         dir=folder, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        if (verify_population_study(folder) != verified
                or _manifest_digest(folder) != manifest_digest):
            raise ValueError("Population study changed before its final note could be published.")
        _replace_checkpoint(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def wait_and_finish(folder, *, wait=False, timeout_hours=24.0):
    """Wait for a successful completion marker, with bounded local polling."""
    folder = Path(folder).resolve()
    if not isinstance(wait, bool):
        raise ValueError("wait must be a boolean.")
    if isinstance(timeout_hours, bool) or not math.isfinite(timeout_hours) or timeout_hours <= 0:
        raise ValueError("timeout_hours must be finite and positive.")
    deadline = time.monotonic() + float(timeout_hours) * 3600
    if wait:
        print(f"Waiting for a complete study in {folder} (timeout {timeout_hours:g} hours).", flush=True)
    while True:
        summary_path = folder / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            failures = summary.get("failed_job_count", 0)
            if failures:
                raise RuntimeError(f"Population study reports {failures} failed job(s); "
                                   "resolve or resume them before publishing results.")
        if (folder / "study_manifest.json").is_file():
            return finish_population_study(folder)
        if not wait:
            raise FileNotFoundError("The study has no complete study_manifest.json; "
                                    "use --wait to wait for the running jobs.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out after {timeout_hours:g} hours waiting for study completion: {folder}")
        time.sleep(min(POLL_SECONDS, remaining))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/trinity_population_study"))
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    args = parser.parse_args(argv)
    root = args.repo.resolve()
    folder = (root / args.output_dir).resolve()
    if not folder.is_relative_to(root):
        parser.error("Study output must remain within --repo.")
    try:
        note = wait_and_finish(folder, wait=args.wait, timeout_hours=args.timeout_hours)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    print("Verified complete study and source manifests; published figure and results note without model calls.", flush=True)
    print(note, flush=True)


if __name__ == "__main__":
    main()
