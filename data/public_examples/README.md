# Public demonstration data

The shared repository includes observation examples for **four reservoirs from the paper**. The rest of the reservoir observation collection, raw station downloads, naturalized-flow donor series and prepared source records remain local and ignored by Git.

| Reservoir | GRanD ID | Normalized observation file | Demonstration purpose |
|---|---:|---|---|
| Gardiner | 253 | `data/reservoirs/ex14_Gardiner/observations_si.csv` | Parameter estimation, managed workflow and level/geometry conversion |
| Trinity | 131 | `data/reservoirs/ex12_Trinity/observations_si.csv` | Original MATLAB comparison and legacy examples; existing source/readiness qualifications still apply |
| Oahe | 870 | `data/reservoirs/ex6_Oahe/observations_si.csv` | Mixed-year calibration and storage/release trade-offs |
| Garrison | 753 | `data/reservoirs/ex5_Garrison/observations_si.csv` | Daily/monthly handling and mixed-year calibration comparisons |

Each normalized example retains its original provenance, MATLAB source metadata and legacy target/bound tables. These inputs and their hashes are unchanged. Sharing an example does not certify its source definitions or guarantee that every solver/calibration configuration succeeds.

Gardiner and Trinity also retain their legacy Excel observation workbooks in `ResDataInflowStorageRelease/` for the documented original estimator and MATLAB-comparison examples. These are alternative representations of two selected reservoirs, not extra reservoir datasets. The Trinity regression archive under `tests/reference_cases/` remains part of the selected Trinity example.

Gardiner's additional normalized level records, station-component records and geometry curve are the only extra observation/geometry demonstration files allowed by `.gitignore`. Raw download pages and other acquisition files remain excluded. Basic GRanD identity, capacity and coordinate metadata remain shareable; the full local observation inventories are excluded.

Use the four-reservoir [manifest](manifest.json) with the existing collection CLI:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B examples/run_reservoir_collection.py --manifest data/public_examples/manifest.json --reservoir Gardiner --output outputs/public_gardiner_new
```

Repeat `--reservoir` to request another example, or use `--all` with this explicit public manifest. The collection reports scientific blockers for each requested case. It does not silently change parameters, timing or root policy to obtain a successful run. No new calibration or simulation was run when preparing this sharing manifest.

The full `data/reservoirs/manifest.json` and larger catalogs describe the private local collection. Their observation files are not distributed with a fresh checkout. Collection-wide analyses and archive-dependent tests require restoring the corresponding private inputs. Some archived run-manager configs also refer to ignored prepared results under `outputs/`; users must generate those stages or supply their own records.

The exact sharing rules are in the root [`.gitignore`](../../.gitignore). New files under `data/` are excluded unless explicitly allowed. All existing local datasets remain on disk. Future commits remove the unselected legacy workbooks from the tracked tree; previously committed copies remain in Git history unless a separate history cleanup is performed.
