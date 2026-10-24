# Project-local cache storage

Use the repository's `.cache` directory for project-controlled temporary files
and caches. The cache helper resolves this directory from the checkout location,
so the repository can be placed on a drive with sufficient free space.
Scientific inputs, versioned outputs and optimizer checkpoints retain their
configured locations.

Activate the cache environment in every fresh PowerShell session:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools\check_project_cache.py
```

Or launch a command through the helper:

```powershell
& .\tools\Set-ProjectCache.ps1 -ProjectCommand @('python', '-B', 'tools\check_project_cache.py')
```

The helper sets the process environment, inherited by child workers:

| Storage | Repository directory |
| --- | --- |
| Python/pytest default temporary files | `.cache/temp` |
| pip downloads and wheels | `.cache/pip` |
| Matplotlib font/config cache | `.cache/matplotlib` |
| Python bytecode for imports | `.cache/python` |
| XDG-aware libraries | `.cache/xdg` |
| Optional Numba compiled cache | `.cache/numba` |
| Optional joblib temporary/memory-mapping files | `.cache/joblib` |

`.cache` and local `.env` files are ignored by Git. An optional private `.env`
can configure tools that load environment files, but no `.env` is distributed.
A terminal does not automatically load it; activate the helper in each fresh
PowerShell session. The helper computes its paths dynamically. For Linux setup,
see the [portable application guide](portable_application_v1.md).

Continue to use `python -B` where appropriate. For regression tests, pass a fresh
repository-contained `--basetemp` and `-p no:cacheprovider`; pytest can remove its
base directory, so never designate `.cache`, `data` or `outputs` themselves as that
base. For example, `outputs/pytest_example_01` is suitable only if it does not
already exist.

Cache routing applies to the listed environment-aware tools. It does not move
unrelated application settings, credentials or history. It also does not change
model equations, parameter values, study identities or saved experiments.
