# Permanent reservoir inputs

**Public checkout:** only Gardiner, Trinity, Oahe and Garrison are shared.
Use the [four-example manifest](../public_examples/manifest.json) with
`--manifest data/public_examples/manifest.json`. The full manifest and the other
reservoir observations described below remain local and ignored by Git.

This collection contains the **41 active reservoirs** imported from the reviewed
MATLAB examples: 27 daily and 14 monthly. Tuyen Quang is intentionally excluded.
Brazeau's observations are included; its external forcing/storage interpretation
still requires review before simulation.

The [manifest](manifest.json) lists each reservoir's observation file, model
configuration, sampling frequency, selected dates, hashes and historical
readiness statuses. Paths resolve relative to the repository root declared by
the manifest. [Configurations](../../configs/reservoirs) contain fixed physical
settings and explicit warmup/calibration/validation periods.

Each reservoir directory contains:

- `observations_si.csv`: date, storage in m3, inflow and release in m3/s.
- `observation_provenance.json`: source units, calendar reconciliation and data review evidence.
- `matlab_source.json`: selected script, constants, raw targets, clocks and source hashes.
- `legacy_targets_si.csv` and `legacy_bounds_si.csv`: unmodified historical tables.

Observations and evidence are byte-for-byte copies of the verified import.
Records cover the selected source simulation period, not necessarily every row
in the original workbook. Missing or signed measurements remain unchanged.
Monthly samples retain monthly resolution and explicit duration settings.

The E-drive paths inside provenance identify original sources; they are not
runtime dependencies. Normal runs require this data folder, its manifest and
the matching configs, with the Python package/dependencies available. The
complete original MATLAB/optimization archive remains outside this collection.

```powershell
python examples/run_reservoir_collection.py --reservoir Trinity --output outputs/trinity_new
python examples/run_reservoir_collection.py --all --output outputs/collection_new
```

Modern runs fit targets/bounds using calibration observations only, carry the
continuous state into validation and write results under the chosen output
folder. An explicit `--mode legacy` uses packaged historical settings and clocks.
Use a new output folder for each run. Scientific blockers are reported per
reservoir; a batch with blocked members completes its reports and returns exit
code 1. `--optimize N` explicitly adds N search evaluations per selected member;
no search is launched by the default baseline command.

The [package receipt](package_receipt.json) pins the original normalized inputs
and every packaged file. Existing numerical archives were preserved. Repeating
`python examples/package_world_reservoirs.py` accepts identical destinations and
refuses to overwrite differing files. Repackaging uses the original normalized
import outputs; normal simulation does not need those outputs.

The manifest's readiness fields describe the verified historical import. Fresh
readiness and simulation results are written to each new collection run; changing
an experiment requires explicit review and matching updated hashes.
