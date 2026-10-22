# Standard reservoir inputs and archived results

The working decision is **population 32, a 512-evaluation cap, and seeds 42, 43 and 44**. This is a provisional computational configuration supported by the Bull Lake comparison. It is not a sufficient budget established across reservoirs. Current outputs are labelled `budget_reached_convergence_unconfirmed`.

No additional optimization, parameter scoring, simulation replay or held-out validation is scheduled by this workflow. It exports existing evidence only. The pinned worklist is [archive_workflow.json](../configs/reviews/archive_workflow.json).

## Input records

The existing [permanent input manifest](../data/reservoirs/manifest.json) remains the source of membership and file identities. The standardized catalog records all 41 active reservoirs and retains Tuyen's exclusion. Each input record carries the observation/configuration hashes, explicit SI units, native daily or monthly sampling, periods, physical constants, percentile method, and the original readiness snapshot.

Historical `readiness` fields are labelled as snapshots, not current calibration authorization. The exporter does not reassess forcing, solve roots, repair parameters, or promote blocked reservoirs. Every record has `new_job_authorized: false`. Reservoir-specific source reviews remain necessary before future jobs; Bull Lake's signed computed-net policy is not transferred to other reservoirs.

## Result records

The first export contains the three verified population-32 Bull Lake jobs. Each result has a stable job ID, reservoir ID, seed, optimization settings, periods, bounds, source policy, original checkpoint identity and hash, objective definitions, feasibility counts, stopping settings and separate validation status. Paths in `result_index.json` are relative to the export directory; archived-source and input references are relative to the repository root. Original absolute paths inside historical checkpoint identities are preserved faithfully.

The full Pareto front and its native 72-value vectors are copied byte for byte. Four candidate roles are exported per seed: generalized baseline, best release, best storage and balanced. Roles retain the archived candidate IDs; a baseline need not be a member of the final Pareto front. Several roles may legitimately refer to the same candidate. There is no new cross-seed selection or single claimed optimal parameter set.

Each role also gets a 12-row monthly parameter CSV with:

- `month` from 1 through 12;
- critical, normal and flood storage targets in **m³**;
- critical, normal and flood release targets in **m³/s**.

The native vector stores six 12-month blocks: Sc/C, Sn/C, Sf/C, Qc, Qn and Qf. Exporting multiplies only the first three blocks by that reservoir's capacity and transposes to monthly rows. It does not sort, clip, fit, or alter the parameters. Invalid ordering is refused. Release targets are not generally capped by channel capacity: the DZTR emergency operating rule applies the channel cap in its own zone.

Calibration scores are copied from the archived candidate records. `validation_metrics` is null and `validation_scored_by_export` is false. These exports retain Bull Lake's segmented conditional-hindcast interpretation, source clock uncertainty, quarantine masks and observed block anchors. They are not independently validated predictive parameter sets.

## Reuse

From the repository root, create a fresh export directory:

```powershell
python -B tools/standardize_archives.py --worklist configs/reviews/archive_workflow.json --output outputs/my_archive_export
```

The worklist pins the collection, decision evidence, archive verification manifests and reservoir policies. Every archived file is checked against its verified manifest. The exporter requires the source checkpoint identity to bind the same configuration, observations and policy, refuses input drift and existing output directories, and produces a checksum manifest. It imports no DZTR simulation or optimization modules.

To register further *existing* calibration results, add explicitly identified verified archives to a new worklist. The current adapter accepts reviewed interval archives with the same objective contract; other archive formats need an explicit adapter. Declaring additional catalog reservoirs does not trigger calculations or authorize jobs. Result indexes can be combined later without pooling observations, changing time steps or mixing held-out scores into calibration selection.

The reporting tool stays outside `src/dztr` because existing continuation identities hash that numerical source tree. This preserves continuation compatibility without weakening its hash checks.

See the [completed export](../outputs/archive_workflow_20261010/README.md) for the actual counts, parameter tables and verification results.
