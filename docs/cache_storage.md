# Project cache storage on D:

The user has limited C: drive space. Use the repository's `.cache` directory for
project-controlled temporary files and caches. Scientific inputs, versioned
outputs and optimizer checkpoints retain their existing D: locations.

Activate the cache environment in every fresh PowerShell session:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools\check_project_cache.py
```

Or launch a command through the helper:

```powershell
& .\tools\Set-ProjectCache.ps1 -ProjectCommand @('C:\Users\YassinF\AppData\Local\anaconda3\python.exe', '-B', 'tools\check_project_cache.py')
```

The helper sets the process environment, inherited by child workers:

| Storage | Repository directory |
| --- | --- |
| Python/pytest default temporary files | `.cache/temp` |
| pip downloads and wheels | `.cache/pip` |
| Matplotlib font/config cache | `.cache/matplotlib` |
| Python bytecode for imports, including packages installed on C: | `.cache/python` |
| XDG-aware libraries | `.cache/xdg` |
| Optional Numba compiled cache | `.cache/numba` |
| Optional joblib temporary/memory-mapping files | `.cache/joblib` |

`.cache` is ignored by Git. The local `.env` provides the same values for Python
tools that load it, including VS Code debugging with its Python environment-file
support. A terminal does not automatically load `.env`; activate the helper there.
The supplied `.env` uses this workspace's absolute path and must be updated if
the repository is relocated. The PowerShell helper computes its path dynamically.

Continue to use `python -B` where appropriate. For regression tests, pass a fresh
repository-contained `--basetemp` and `-p no:cacheprovider`; pytest can remove its
base directory, so never designate `.cache`, `data` or `outputs` themselves as that
base. The earlier explicit `outputs/pytest_*` locations are already on D:.

The existing user pip cache is migrated from C: by `tools/Move-PipCache.ps1`.
Every file is SHA256-verified at the destination and checked again at the source
before its C: copy is removed. The source directory remains. User-level
`PIP_CACHE_DIR` points to D: for new sessions. The two existing Matplotlib font
caches are also moved, and user-level `MPLCONFIGDIR` points to D:.
Existing IDE/terminal processes can retain their old environment: activate the
helper immediately, or restart the IDE to pick up persistent user settings.
Migration receipts record the previous settings.

Project routing covers these environment-aware tools. Windows, MiKTeX and other
applications can retain their own logs/configuration. Codex stores local state,
including configuration, credentials and history, separately under its user
directory; this project helper does not relocate that active application state.
See [official OpenAI documentation](https://learn.chatgpt.com/docs/config-file/config-advanced)
for those state locations and configuration controls.

[Project cache verification](../outputs/cache_storage_20261014/project_cache_verification.json),
[pip migration receipt](../outputs/cache_storage_20261014/pip_cache_migration.json).
No model equations, parameter values, study identities or saved experiments are
changed by cache routing.
