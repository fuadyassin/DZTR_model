# DZTR 0.3.1: release and citation

Version 0.3.1 is a repository and packaging cleanup release. It preserves the
reservoir equations, observation handling, optimization strategies, calibrated
parameters and three frozen public application profiles from version 0.3.0.

## What changed

- Removed 11 Jupyter checkpoint copies and six Python bytecode files from Git.
- Excluded generated caches and notebook checkpoints from application bundles.
  Public CI rejects tracked generated files, including Windows-style paths.
- Kept AGENTS.md for repository development and excluded it from the application
  ZIP and installed wheel.
- Replaced personal workflow wording and machine-specific commands in public
  documentation with neutral researcher instructions.
- Made bundle metadata follow the distribution version, including the release
  date and the corresponding release configuration.

These changes introduce no new calibration experiment or scientific performance
claim. Existing source gates and benchmark qualifications remain in place.

## Downloads and verification

The [v0.3.1 GitHub release](https://github.com/fuadyassin/DZTR_model/releases/tag/v0.3.1)
provides the application source ZIP, Python wheel, rendered documentation and
SHA256 checksums. The source bundle is required for
workflow orchestration; the wheel supports frozen-profile simulation on its own.
Install Python dependencies separately. Use the
[portable application guide](portable_application_v1.md) for examples.

The [public release checks](https://github.com/fuadyassin/DZTR_model/actions/workflows/public-release.yml)
verify Ubuntu and Windows installation, public parameter generation, exact
historical and restart reproduction, water balance and sealed resume. Each
result identifies the source commit and reference environment. MIT licensing
and third-party input terms remain unchanged.

## Citation and archive scope

**Software:** Fuad Yassin (2026). DZTR: reservoir parameterization, simulation
and calibration workflow. Version 0.3.1. Zenodo.
[Software DOI: 10.5281/zenodo.23631633](https://doi.org/10.5281/zenodo.23631633).
Machine-readable metadata is in `CITATION.cff`. The
[all-versions DOI](https://doi.org/10.5281/zenodo.23606592) identifies the software
across releases; it is distinct from an individual release DOI.

**Model paper:** Yassin, F., Razavi, S., Elshamy, M., Davison, B., Sapriza-Azuri,
G., and Wheater, H. (2019). Representation and improved parameterization of
reservoir operation in hydrological and land-surface models. Hydrology and Earth
System Sciences, 23, 3735–3764.
[Model-paper DOI](https://doi.org/10.5194/hess-23-3735-2019).

Zenodo's GitHub integration archives the full tagged public source. AGENTS.md
remains part of that repository snapshot. The application source ZIP, wheel,
rendered documentation and checksums are separate GitHub release assets.
Unselected observations, private experiment outputs, caches and presentations
remain excluded from Git and release bundles.

The [v0.3.0 release and DOI](release_v0_3_0.md) retain their original files and
checksums. Adding a new release does not replace the earlier archive. Cite the
version used for a study; the model paper remains a separate reference. Current
main-branch citation metadata adds the registered DOI without replacing the
frozen release files.
