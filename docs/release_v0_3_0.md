# DZTR 0.3.0: release and citation

Version 0.3.0 packages the public researcher workflow, standalone frozen profiles,
cross-platform checks and this documentation. Software and documentation use
the MIT license. Third-party input provenance and original data terms remain
applicable; see the repository's `THIRD_PARTY_NOTICES.md`.

## Installable application

Use the [GitHub releases](https://github.com/fuadyassin/DZTR_model/releases) for
the sealed source/application ZIP, Python wheel and SHA256 checksums once the
release is published. The [public workflow checks](https://github.com/fuadyassin/DZTR_model/actions/workflows/public-release.yml)
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
calibration workflow. Version 0.3.0. GitHub repository and `CITATION.cff`.

**Model paper:** Yassin, F., Razavi, S., Elshamy, M., Davison, B., Sapriza-Azuri,
G., and Wheater, H. (2019). Representation and improved parameterization of
reservoir operation in hydrological and land-surface models. Hydrology and Earth
System Sciences, 23, 3735–3764. [Model-paper DOI](https://doi.org/10.5194/hess-23-3735-2019).

The software DOI remains pending actual Zenodo archival. It will be added to
the citation file and this page after registration. The paper DOI is not used
as a software DOI. A GitHub/Zenodo connection requires a one-time account setup;
manual deposition of the sealed files is an alternative.

## Scientific scope

Passing software checks establishes the tested installation and reproducibility
behavior. The [held-out assessment](scientific_assessment_v1.md) separately
describes performance. Conditional forcing, approximate geometry, regional
donor assumptions, unresolved source cases and unproven optimizer convergence
retain their documented qualifications. Remaining source investigations are
deferred; publication does not promote them to admitted calibration cases.
