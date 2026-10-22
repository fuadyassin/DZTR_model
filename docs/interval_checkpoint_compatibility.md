# Interval checkpoint compatibility

Fresh interval-search checkpoints and previously extended checkpoints now follow the same continuation contract. A fresh checkpoint containing the current resume adapter is accepted directly. It does not need a fabricated historical bridge.

Continuation preserves saved candidate vectors, objective costs, survivors, random-generator state, bounds, and CSV prefixes. It evaluates only the additional budget. The original archive remains unchanged; write each extension into a new directory. Verification files in the parent directories of the population-32 and population-64 archives identify the files authorized for continuation.

For example, the following extends one completed population-32 job to a total of 1,024 evaluations. This is an example only; the population comparison did not execute this extension.

```python
from pathlib import Path
from dztr.calibration.interval_resume import resume_interval_seed

root = Path(r"D:\2_CodeProjects\GitHub\Repos\DZTR_model")
parent = root / "outputs/bull_lake_interval_population_20261009/runs/population_32"
report = resume_interval_seed(
    parent / "seed_42",
    root / "configs/reviews/bull_lake_interval_policy.json",
    root / "outputs/bull_lake_population32_seed42_1024/seed_42",
    total_evaluation_budget=1024,
    checkpoints=(512, 640, 768, 896, 1024),
)
```

Run from the repository's configured Python environment with `src` on the import path. Population-aligned total budgets and checkpoints are required. A new parent verification manifest must be produced and reviewed before chaining another extension from its output. The API does not silently certify its own output for subsequent reuse.

Changes to numerical modules, forcing inputs, bounds, objective settings, or software identity remain grounds for refusal. If the resume adapter alone changes, an explicit `approved_adapter_migration` dictionary must identify exactly:

- `old_sha256`: the adapter hash stored in the checkpoint;
- `new_sha256`: the adapter hash currently on disk;
- `prior_verification_path`: the absolute path to the actual verified parent manifest;
- `prior_verification_sha256`: that manifest's hash.

The API verifies these fields against the source checkpoint and current files. It records the adapter migration in a chained identity receipt. This exception does not authorize changes to hydrological equations or optimization code. Adapter migrations require review of the change before supplying that dictionary; matching hashes alone do not demonstrate scientific equivalence.

The population study exercised this explicit migration on the historical fresh seed-44 checkpoint: 512 inherited evaluations plus 16 new evaluations, total 528. Its saved vectors, costs, and CSV prefixes were preserved exactly. The resulting chained identity was accepted for subsequent continuation. Focused tests also compare fresh and chained continuations with an uninterrupted search, check malformed receipts and identity chains, and reject unrelated numerical-code changes.

See [the population study](interval_population_study.md) for the experiment results and [its compatibility receipt](../outputs/bull_lake_interval_population_20261009/compatibility_check.json) for the actual migration hashes.
