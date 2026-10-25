# Researcher release process

The public researcher application release combines four
parts: a stable command, frozen-profile examples, a curated scientific
assessment and a tested source/wheel bundle. Existing scientific versions,
private observations, checkpoints and source gates remain preserved.

`tools/build_researcher_bundle_v2.py` builds an allowlisted ZIP with code,
documentation, public configuration templates and only the already-approved
Gardiner, Trinity, Oahe and Garrison observation examples. Frozen application
profiles are supplied for Gardiner, Oahe and Garrison. Other observation records,
raw downloads, donor series, `.git`, `.cache`, presentations and experiment
outputs are excluded. The bundle manifest records every included file and hash,
the source commit, version and whether the working tree was modified.

Portability validation builds a wheel from the copied source, installs it into
a separate environment, runs the public command from the extracted
bundle and runs inflow-only examples from outside the source tree. This checks
operation without private archive paths; it is not an independent scientific
validation dataset. GitHub Actions runs these checks on Ubuntu 24.04 and
Windows 2022 with Python 3.12.7 and reference dependencies. Environment versions
and test receipts are retained. Parameter generation and fixed baselines are
tested with the approved public records and at least 65% calibration support.

The source bundle is currently required for workflow orchestration; the wheel
alone supports frozen-profile simulation. The release is not an offline
dependency installer. A fresh machine needs its Python dependencies installed.

The publication sequence is:

1. Keep the reviewed public-data allowlist and MIT license together with
   the release version and its software citation metadata.
2. Freeze the release version, dependencies, tests, source identity and bundle
   checksums; distinguish reproducibility from source/forecast certification.
3. Build this documentation with strict local-link validation and publish it to
   GitHub Pages after both platform checks pass. Local archive links are
   labelled as references; private result directories are not published.
4. Archive the approved portable release on Zenodo, obtain an **actual software
   DOI**, then add that registered DOI to citation metadata, the documentation
   and repository links. The model paper's DOI remains a separate reference.

Published versions of the tagged public source are archived at
[Zenodo](https://doi.org/10.5281/zenodo.23606592). The wheel, application bundle,
rendered documentation and checksums are GitHub release assets. DOI links are
added to main-branch metadata after archival; the original tag and sealed files
remain fixed. Remaining source investigations stay deferred. See
[release and citation](release_v0_3_1.md) for the current metadata.
