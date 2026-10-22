# Local reservoir observations and parameter inputs

**Sharing scope:** the public repository retains only Gardiner, Trinity, Oahe and
Garrison observation examples. Their [public manifest and guide](public_examples/README.md)
describe the selected files, Gardiner level/geometry extras and runnable commands.
All other observation collections and source downloads below remain local and
ignored by Git. The inventory counts in this document describe the full local
workspace, not the data distributed in a fresh checkout.

The current [combined reservoir index](catalogs/reservoir_collection_v2_20261021.json) contains **62 local reservoir records**. Its [CSV inventory](catalogs/reservoir_inventory_v2_20261021.csv) is the simplest place to browse names, dates, observation paths and target/readiness fields.

- [Original collection](reservoirs/README.md): 41 imported MATLAB reservoir examples, preserved with their original inputs and provenance.
- [Etienne extension](reservoirs_etienne_v1_20261021/parameterization_registry.json): 21 additional reservoir records, 28 archived observation files and descriptive monthly target estimates.
- [Expansion overview and coverage plot](../outputs/reservoir_expansion_v1_20261021/collection_overview.md).

The 37-reservoir paper inventory remains its own reference cohort. The 21 additions are not labeled as members of that paper cohort. Tuyen Quang remains excluded.

The new records include seven with storage and release, ten with storage only and four with release only. Their target estimates use all available records for descriptive purposes. They require inflow, adopted physical metadata, storage/timing reconciliation and calibration-only refitting before simulation or calibration. Missing values, missing targets and unverified GRAND IDs remain explicit.

`catalogs/grand_v1_1_reference.csv` is a separate global metadata reference with 6,862 entries; it does not imply that observations or calibrated DZTR profiles exist here for every entry. Candidate matches for the new reservoirs are retained as search leads, not verified ID assignments.

Use the existing run manager after a record meets its source and modeling requirements. The [extension guide](../docs/reservoir_collection_extension_v1.md) documents normalization, file access and admission steps.
