# DZTR 0.3.0: release and citation

Version 0.3.0 packages the public researcher workflow, standalone frozen profiles,
cross-platform checks and this documentation. Software and documentation use
the MIT license. Third-party input provenance and original data terms remain
applicable; see the repository's `THIRD_PARTY_NOTICES.md`.

## Installable application

Use the [v0.3.0 GitHub release](https://github.com/fuadyassin/DZTR_model/releases/tag/v0.3.0)
for the sealed source/application ZIP, Python wheel, rendered documentation and
SHA256 checksums. The [public workflow checks](https://github.com/fuadyassin/DZTR_model/actions/workflows/public-release.yml)
record installation, public tests, parameter generation, fixed-profile
applications and verified resume on Ubuntu and Windows. Each result identifies
the tested source commit and environment.

The source bundle is needed for workflow orchestration. The wheel supports
frozen-profile application independently of the source checkout. Private
calibration archives, donor series and unselected observations are excluded.
Use the [quick start](portable_application_v1.md) for the three public examples.

The package release version is available through `dztr --version` or installed
distribution metadata. Archived numerical modules retain their original bytes,
including the historical core's 0.2.0 marker, to preserve old experiment hashes.
The release metadata is separate from those frozen scientific identities.

## Citation

**Software:** Fuad Yassin. DZTR: reservoir parameterization, simulation and
calibration workflow. Version 0.3.0. Zenodo.
[Software DOI: 10.5281/zenodo.23606593](https://doi.org/10.5281/zenodo.23606593).
Machine-readable citation metadata is in `CITATION.cff`.

**Model paper:** Yassin, F., Razavi, S., Elshamy, M., Davison, B., Sapriza-Azuri,
G., and Wheater, H. (2019). Representation and improved parameterization of
reservoir operation in hydrological and land-surface models. Hydrology and Earth
System Sciences, 23, 3735–3764. [Model-paper DOI](https://doi.org/10.5194/hess-23-3735-2019).

The version DOI identifies the published v0.3.0 source snapshot. The
[all-versions DOI](https://doi.org/10.5281/zenodo.23606592) identifies the software
across releases. Use the version DOI when reporting results from this release.

Zenodo's GitHub integration archived the full tagged public source, including
code, public examples and Markdown documentation. The installable wheel,
allowlisted application bundle, rendered documentation and checksums are GitHub
release assets. The frozen release files retain their original checksums;
these current documentation pages and main-branch citation metadata add the
registered DOI without moving the release tag or replacing its files.

## Scientific scope

Passing software checks establishes the tested installation and reproducibility
behavior. The [held-out assessment](scientific_assessment_v1.md) separately
describes performance. Conditional forcing, approximate geometry, regional
donor assumptions, unresolved source cases and unproven optimizer convergence
retain their documented qualifications. Remaining source investigations are
deferred; publication does not promote them to admitted calibration cases.
