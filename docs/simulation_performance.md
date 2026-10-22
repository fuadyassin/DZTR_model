# Preparing monthly targets for daily simulation

Daily simulation reads an immutable monthly value tuple from a cache local to
that simulation. Each calendar month is prepared when first needed, then reused
by the solver, release-rule diagnostic and final zone calculation. A new call
starts a fresh cache, so changes to the mutable monthly target table between
calls are observed. Months that the simulation never visits are not read.
Targets are expected to remain unchanged while an individual simulation runs.

The release branches, implicit root calculations, MATLAB fixed iteration count,
overflow corrections and arithmetic ordering are preserved. Public release and
solver calls continue reading current parameter values with their existing
validation. This changes lookup work, not calibration selection or stopping.

## Comparing the original and prepared implementations

The standalone benchmark compares four verified historical archives: Trinity
seeds 42, 43 and 44 at 2,048 evaluations, and Gardiner seed 42 at 1,024. For each
archive it replays the generalized baseline and distinct best-release,
best-storage and balanced candidates. These archives currently supply sixteen
distinct archive/candidate pairs, or thirty-two calibration objective replays.

```powershell
python -B examples/benchmark_prepared_simulation.py `
  --reference-repo D:\2_CodeProjects\GitHub\Repos\DZTR_model `
  --output outputs/prepared_simulation_benchmark
```

Run this from the repository containing the prepared implementation while the
reference repository retains the original source files. Use a new output folder
for each execution. `--reference-repo` supplies historical archives, configuration
and observations. `--reference-code-root` can separately supply a preserved
original source tree containing all 29 consumed modules under `src/dztr`.
When omitted, the code root defaults to the reference repository. The benchmark
requires that only `model/balance.py` and `simulation/runner.py` differ between
consumed source trees and that the reference sources, observations and software
match the historical identities.

After integrating the prepared implementation into the main repository, run
from that repository using its preserved original code snapshot:

```powershell
python -B examples/benchmark_prepared_simulation.py `
  --reference-repo . `
  --reference-code-root outputs/simulation_performance/reference_code `
  --output outputs/prepared_simulation_benchmark_repeat
```

The archive/data repository may equal the current repository when an explicit,
distinct code root is supplied. The three smaller frozen simulation files under
`tests/reference_cases/uncached_simulation` support regression tests; this
benchmark uses the complete 29-module snapshot to verify the historical consumed
source identity. Preserve that snapshot before updating the main source tree.
The staged benchmark used `outputs/prepared_simulation_reference`; the guarded
installer preserves the same code in the main repository at
`outputs/simulation_performance/reference_code`.

Original release, balance, runner and objective modules are loaded under separate
aliases. Reference bindings are changed only inside those aliases; production
module globals are not patched. Every comparison requires exact full daily
frames, objective costs, metrics, feasibility, paired counts and diagnostic
issues, including agreement with the archived calibration scores. No validation
replays, optimizer trials or checkpoint continuation are performed.

Timing covers each complete calibration objective evaluation, including
candidate validation, simulation and metric calculation. Problem construction,
source verification and report writing are outside the timed interval. Mode
order alternates by candidate pair. One replay per mode and candidate remains a
small selected sample; concurrent machine load and ordering limit the meaning of
the ratios. These measurements do not establish optimizer throughput or a fleet
speedup.

The output contains `candidate_comparison.csv`, `replay_attempts.csv`,
`summary.json` and `benchmark_manifest.json`. The summary records input hashes,
both sets of consumed code hashes, historical archive proofs, replay accounting
and aggregate CPU and wall ratios. Source and reference hashes must remain
unchanged throughout the experiment, and historical artifacts are reverified at
completion.

## Checkpoints and historical results

Balance and runner source files contribute to adaptive checkpoint identity.
Changing them creates a new identity even when daily output is exactly equal.
Start new searches in new output folders with the prepared implementation.
Historical checkpoint files and their identities remain unchanged; their stored
proofs support historical verification without asserting that they match the
current source tree. Never rewrite an old identity to bypass a resume check.

A future comparison using the prepared implementation has a separate config,
`configs/trinity_population_study_prepared.json`. It retains populations 16/32/64,
seeds 42/43/44 and evaluation checkpoints 512/1,024/2,048, with an empty
`reuse_runs` list so every search starts under the new code identity. Run it after
integration with a new output directory:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -B -m dztr.calibration.population_study `
  --repo . `
  --config configs/trinity_population_study_prepared.json `
  --output outputs/trinity_population_study_prepared `
  --workers 3
```

This fresh prepared-code comparison has not been launched. The original
population comparison remains a historical result. Its completed study and
recorded source proofs remain available for read-only verification. The original
population config still refers to the old external archives and correctly
rejects their reuse after consumed code changes.

The local `examples/integrate_prepared_simulation.py` helper is available with
`--plan PATH` and `--plan-sha256 SHA256`. It requires a pinned reviewed plan,
verified complete population-study publication, unchanged staged file hashes
and unchanged destination files. It installs balance before runner and records
an integration receipt with recovery information if interrupted. Its optional
`--wait` mode waits for publication before making source changes.

Installed reference sources and replay proofs live under
`outputs/simulation_performance`: the original replay report is in
`benchmark_original`, and the separate frozen-code repeat is in
`benchmark_frozen`. The measurement locations below describe the staging run.

## Verified two-reservoir replay pilot

The 2026-10-06 run completed all sixteen candidate pairs and thirty-two
calibration replays. Every full daily frame, objective cost, metric, feasibility
result, paired count and diagnostic issue matched exactly. Calibration scores
also matched the historical archived scores. Both source trees, input hashes,
four checkpoint/archive proofs and all three report artifact hashes passed
verification. There were zero validation replays, optimizer trials, optimizer
resume attempts or rewritten checkpoint identities.

| Replayed candidates | Pairs | Original CPU seconds | Prepared CPU seconds | CPU ratio | Wall ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trinity, three archived seeds | 12 | 6.719 | 0.938 | 7.17× | 7.03× |
| Gardiner, one archived seed | 4 | 2.563 | 0.328 | 7.81× | 7.32× |
| Combined | 16 | 9.281 | 1.266 | 7.33× | 7.11× |

The combined wall times were 9.341 seconds for the original implementation and
1.314 seconds for the prepared implementation. Ratios divide the total original
time by the total prepared time; they are not averages of per-candidate ratios.
Windows CPU timing is coarse at these short durations, and the original
population experiment was running concurrently. The result supports reducing
monthly lookup work for these feasible calibration evaluations. It does not
establish a universal speedup, a population default, convergence or the runtime
of a complete optimizer. Gardiner remains an exploratory second-reservoir case
with the physical metadata limitations documented in its configuration.

At measurement time, the change and benchmark were isolated in the unmerged
worktree `D:\2_CodeProjects\GitHub\Repos\DZTR_model_worktrees\monthly_cache`.
The main repository's original implementation and running population study were
unchanged. The preserved report copy is
[`benchmark_original/summary.json`](../outputs/simulation_performance/benchmark_original/summary.json).

A second execution used the separate complete frozen code root, with the same
historical archives, and independently verified another sixteen candidate pairs
and thirty-two calibration replays. This run supports the post-integration
reproduction path; it is a repeat of the selected candidates, not sixteen new
candidates or a larger reservoir cohort.

| Frozen-code repeat | Original seconds | Prepared seconds | Ratio |
| --- | ---: | ---: | ---: |
| CPU | 6.859 | 0.938 | 7.32× |
| Wall | 6.944 | 0.919 | 7.55× |

All frames and scores matched exactly again, all historical source proofs and
report artifact hashes passed, and there were zero validation replays, optimizer
trials or identity changes. These repeat timings carry the same qualifications
as the first run. The separate report is
[`benchmark_frozen/summary.json`](../outputs/simulation_performance/benchmark_frozen/summary.json).
Each measured script version is additionally preserved as `benchmark_source.py`
inside its report folder; its SHA256 matches that report's `benchmark_sha256`.
These additive source snapshots are outside the original three-artifact manifest.

Staged regression verification passed all 621 existing tests and 51 new parity
tests. A locally built wheel also passed import and exact source-byte checks for
all 39 packaged modules. The new checks cover both solver modes, flood modes and
overflow modes;
calendar boundaries and leap day; spill diagnostics; negative storage and exact
fixed-iteration ordering; lazy month preparation and edits between calls;
discontinuity errors; and exact seeded search and same-version checkpoint resume.
Independent review confirmed that step-input and solver arithmetic statements
match the original abstract syntax trees, with public signatures and input
validation ordering preserved.
