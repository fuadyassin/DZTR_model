## Codex efficiency and scientific validation

Work autonomously: inspect files, edit code, execute commands, and run appropriate tests.

- Prefer one agent unless independent parallel reviews are clearly necessary.
- Avoid repeated repository-wide analysis when relevant files are already identified.
- Run focused tests during implementation; run the complete regression suite once at the end of a substantial change.
- Preserve numerical reproducibility, hydrological equations, mass balance, and scientific correctness.
- Treat optimization experiments as independent computational jobs with persistent checkpoints and logs.
- Avoid frequent progress polling and repeated analysis of incomplete experiment results.
- Reuse verified experiment outputs instead of rerunning calculations unnecessarily.
- Use concise final reports showing changes, test results, scientific checks, and remaining limitations.
- Never sacrifice correctness or skip necessary verification to conserve usage.

## Cache and temporary storage

- The user has limited free space on C:. Keep project-controlled caches and temporary files under this repository's `.cache` directory on D:.
- Before Python, tests, plotting, pip, or background experiment commands in PowerShell, run `& .\tools\Set-ProjectCache.ps1` in the same shell invocation. Child processes inherit the redirected environment.
- Continue to use `python -B` and fresh, explicitly checked pytest temporary directories under this repository when appropriate. Never let a fixed pytest base directory remove cache, data, checkpoint, or result directories.
- Keep persistent scientific inputs, outputs, checkpoints and logs in their existing versioned repository locations. Cache routing must not alter numerical code, experimental identities, or verified results.
- Application-managed Codex state, credentials, history and plugin files are separate from project caches; preserve them unless a specific migration is authorized and verified.
