# Paper-first exploratory workflow

The published [Table 1](https://hess.copernicus.org/articles/23/3735/2019/hess-23-3735-2019-t01.xlsx)
defines **37 unique reservoirs**. Its wrapped Falcon name is one reservoir, and
the paper's Seminoe maps explicitly to the locally spelled `ex40_Semione` folder.
All 37 have local observations. Travers, Mcgregor, Glenmore and Brazeau are the
four additional active cases. Tuyen is outside this cohort's scope.

The downloaded table and its SHA256 are packaged in `data/paper_cohort_v1`.
Membership is used to order the programme, while dates, coordinates and capacities
remain separately interpreted source fields. The table contains rounded metadata
and apparent discrepancies, so it is not an automatic GRAND crosswalk. Existing
verified GRAND IDs remain unchanged.

## Declared exploratory treatments

`workflows/paper_exploration.py` stages a source-specific assumption record before
any optimization. The first physically and calibration-feasible treatment in the
following fixed order is chosen; scores are recorded but do not rank treatments:

1. Retain inherited capacity and the inherited no-release floor.
2. Try the prepared feasible initializer, then a sufficient critical-gap witness
   if one exists within the unchanged calibration percentile box.
3. If preparation or fixed simulation still fails, try lower assumed floor
   fractions from the declared sequence 0.1, 0.05, 0, retaining only values lower
   than the original. These are model-policy hypotheses, not measured dead volume.
4. For Fort Peck and Falcon only, repeat that sequence under the nominal capacity
   documented in Table 1, with an explicit gross-storage-coordinate hypothesis.
   That interpretation is exploratory rather than certified by the capacity label.

Every simulation uses the existing `nearest_previous_storage` policy. It selects
among admissible roots using previous simulated storage with the existing
deterministic tie break. A missing root remains a failure. The selected physical
parameters, native timestep, root policy and initializer are pinned before search.

Observed storage outside nominal model limits is retained and scored. Targets and
simulated storage stay within declared model limits. This allows model-observation
disagreement without asserting that surcharge, observation timing or storage
coordinates have been independently resolved. Initial-state limits and independent
mass balance remain mandatory. Flows and observations are never clipped, sorted,
imputed or converted to absolute values to obtain admission.

Big Horn's existing `1.1 * inflow + 8` source transform is applied once; its
physical inflow correction remains separate. All other source transforms are
also recorded. Targets and bounds retain calibration-only Hazen estimation.
Continuous state crosses unscored years, while held-out response values never
enter preparation, admission or parameter choice.

## Executable stages

```powershell
python -B tools/paper_exploration.py prepare --output outputs/paper_exploration_new
python -B tools/paper_exploration.py run --output outputs/paper_exploration_new
python -B tools/report_paper_exploration.py --output outputs/paper_exploration_new
```

Preparation requires a fresh output directory. The study runner processes
reservoirs in paper-first order, with three independent seed processes per
reservoir. Each search uses population 32, seeds 42/43/44 and 512 evaluations.
It writes persistent checkpoints, immutable budget snapshots, logs and study
execution status. Completed reservoir releases are verified and reused on resume.
Use the original versioned controller and V2 runner with unchanged identities.

The reporter runs the fixed-initializer assessment only after freezing each
calibration-selected profile, caches that diagnostic, and verifies that selection
hashes remain unchanged. A feasible bounds initializer is labeled as such rather
than assumed to be generalized parameters. Failed assessments remain visible and
never trigger seed replacement. The report joins previous immutable releases and
the new releases into one parameter table without changing the old registries.

The current treatment batch is Big Horn, Toktogul, Chardara, Kayrakkum, Palisades,
Fort Peck, Falcon and American Falls. Their chosen treatment details, preflight
attempts and source limitations are in the [study](../outputs/paper_exploration_v1_20261013/README.md).

## Cases that need separate routes

Canyon Ferry's confirmed calendar/rate errors and missing months require input
repair. Trinity/Ghost require their explicit source-coordinate/timing profiles.
Flaming Gorge, Navajo and Seminoe require a signed-net-flux/loss interpretation and
adapter; no zero clipping is applied. Bennett retains its gross/active-coordinate
review. These seven paper records keep their priority and concrete source tasks.
Bull Lake's existing signed-net interval calibration stays separate.

The four nonpaper cases remain tracked with secondary priority. The current
protocol uses mixed years and at least 65% calibration support with the modern
checkpointed search; paper membership does not change that protocol into an exact
reproduction of the published experiments. All assumption-based profiles remain
exploratory, with convergence, operating intent and source certification reported
separately from successful completion.
