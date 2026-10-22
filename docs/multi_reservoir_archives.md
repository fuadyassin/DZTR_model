# Multi-reservoir archived calibration results

The combined archive catalog contains Bull Lake's three population-32 runs and the six original Gardiner, Oahe and High Aswan pilot jobs. It exports existing results only: no new optimization, simulation, validation, or parameter selection takes place.

The provisional population-32/512-evaluation decision remains a separate working configuration. Historical cohort pilots retain **population 16, budget 128, seeds 42/43**, their original sampling and solver, and their original calibration/validation split. Old runs are never relabelled with the newer decision.

Use the [41-reservoir comparison](../outputs/multi_reservoir_archive_20261010/reservoir_comparison.csv) to see available catalog results, original settings, source-review snapshots and limitations. It includes every active input, including those without results in this export. Tuyen remains excluded. The [nine-job comparison](../outputs/multi_reservoir_archive_20261010/job_comparison.csv) provides archived balanced-role calibration scores and separately labelled historical validation scores.

This is an inventory and evidence comparison, not a fleet performance ranking. NSE denominators, operating periods, observations and solver protocols differ by reservoir. No seed or candidate is chosen from validation performance.

## Contracts

Result records use the existing `reservoir_calibration_result` contract: identity, optimizer settings, physical constants, bounds, periods, checkpoint references, feasibility and four monthly parameter roles. Sampled pilots additionally record native sampling, solver/overflow settings, historical source reviews, and validation provenance. Their baseline and three Pareto-selected roles retain the archived evaluation IDs. Monthly storage targets are in m³ and release targets in m³/s; exporting does not alter parameters.

Bull Lake's original standardized packages are copied without changes. Sampled pilot fronts, full vectors, trials, generation histories, selection receipts and historical validation records are preserved byte for byte. A Pareto-vector subset and monthly parameter tables are derived from their archived IDs without recalculation.

Native CSV headers are preserved: cohort trial/front costs are `release_cost`/`storage_cost`, and vector columns are `0` through `71`; interval archives use `negative_release_nse`/`negative_storage_nse` and `x00` through `x71`. Both minimize negative NSE. Readers should use the archive format and objective contract, rather than assuming identical native headers. The monthly SI table headers and the comparison-table headers are common across reservoirs.

Historical cohort validation is stored in `validation_metrics` and each role's `historical_validation`, labelled as imported reporting. `validation_scored_by_export` remains false. Bull Lake's unscored validation remains null. Retaining historical held-out evidence does not produce a new validation experiment.

Checkpoint files remain in their original archives with exact identities and hashes. The sampled pilots' historical numerical identity predates the interval implementation; this export does not certify compatibility with a current continuation runner or migrate those checkpoints. No source-identity checks are weakened.

## Scientific limits

- Gardiner retains daily integration and inherited source-clock/storage assumptions.
- Oahe and High Aswan retain **30.5-day monthly integration**, with calendar labels selecting operating months.
- Oahe's archived balanced storage validation NSE is negative for both seeds and lower than its generalized baseline. The parameters remain experimental; additional optimization is not an established remedy.
- High Aswan's inherited 50,000 m³/s channel limit and physical timing/storage definitions remain unverified.
- Bull Lake retains its signed computed-net forcing policy and segmented conditional-hindcast interpretation; these population-32 runs have no held-out scores.

All jobs remain `budget_reached_convergence_unconfirmed`. Exploratory eligibility is a historical review under explicit assumptions, not independent physical certification or authorization for new jobs. Other input records keep their original review reasons; deferred reservoirs are not promoted by this export.

## Reuse

The new [pinned worklist](../configs/reviews/multi_reservoir_archive.json) records the selected historical jobs and their actual populations and budgets. The original exporter and prior worklists are unchanged.

```powershell
python -B tools/standardize_cohort_archives.py --worklist configs/reviews/multi_reservoir_archive.json --output outputs/my_multi_reservoir_export
```

Choose a fresh output directory. The adapter verifies prior exports, cohort manifests, original input hashes, archived selection and checkpoint vectors/costs before writing results. Its reporting code stays outside `src/dztr` to preserve numerical checkpoint identities.

See [the completed report](../outputs/multi_reservoir_archive_20261010/README.md), [result index](../outputs/multi_reservoir_archive_20261010/result_index.json), and [the original archive contract](archive_workflow.md).
