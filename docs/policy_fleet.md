# Policy-aware batch calibration

The executable preparation is `outputs/policy_fleet_20261012`. It retains the complete 41-reservoir admission registry and action queue. Twenty reservoirs have 60 declared calibration jobs: nine use strict root rejection and eleven use nearest previous simulated storage. Bull Lake's existing interval results remain separate cached evidence. No new reservoir calibrations were launched during implementation.

After execution is authorized, one command from the repository runs the reviewed jobs and common reports:

```powershell
python -B tools/policy_fleet.py --run-authorized
```

Three separate processes run concurrently. Each writes a process log and atomic native generation checkpoints. A job failure is recorded without cancelling other jobs. Completed jobs are reported even when the fleet contains failures. Existing output directories are preserved.

Each fresh job attempts at most two declared initializations. It first tries calibration-only generalized parameters, then the saved critical-gap witness when available. Physical constraints and the unchanged empirical bounds are checked before a model call. A candidate must also complete warmup/calibration with the declared root policy and pass the existing conservation/objective checks. If none succeeds, that job stops before optimization and retains its initialization ledger. This is a bounded preflight, not an uncounted initialization search.

The search cap remains 512 evaluations, population 32, seeds 42/43/44. Initialization checks, up to three selected calibration replays and up to four validation attempts are separately reported. Invalid physical candidates can be rejected without a model call; a validation attempt can likewise fail its forcing/support gate before simulation. The cap does not establish convergence.

Checkpoint identity binds the policy, minimum-storage tie rule, initialization protocol, source hashes, numerical code, runner, reviewed job plan, sampling, periods, settings and Python/library versions. Incomplete jobs with saved generation checkpoints can continue at the same declared budget:

```powershell
python -B tools/policy_fleet.py --resume-job ex1_Glen_canyon 42
```

Identity checks precede replay. Continuation makes one separately recorded baseline calibration check, then uses saved optimizer state and random-generator state. Completed jobs cannot be resumed. A failure before the first checkpoint cannot be resumed through this command; preserve its diagnostic directory and create a separately reviewed fresh experiment. Changing budgets, policies or inputs requires a new plan; this controller does not migrate older checkpoint identities.

Representative IDs are saved before held-out evaluation. A held-out forcing or score-support failure remains a validation report failure, with blank metrics and the reason retained. The selected calibration trajectory is saved when a held-out trajectory is unavailable. It is not labelled as a successful held-out simulation, and calibration success is not discarded. Validation is never used to select initializations or representatives.

Preparation is available without evaluations through `--prepare --output outputs/<fresh_directory>`. It verifies the frozen admission evidence and inputs, creates the job plan and pins the new runner dependencies. The numerical model, canonical source data, prior archives and existing checkpoint formats remain unchanged. Source definitions and inherited clocks remain provisional; numerical conservation is not physical source certification.
