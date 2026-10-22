# Representative-year calibration strategy, version 1

Use complete years spanning dry, normal and wet inflow regimes for calibration rather than requiring an early continuous calibration block. Model simulation remains continuous; parameter fitting and objective scoring use explicit, disjoint row masks. The first complete year is warmup. Default calendar years can be replaced by a declared water-year start month.

This is a new versioned experiment protocol. It does not overwrite the previously frozen twenty-reservoir profiles, their GRAND_ID table or chronological results. It requires new targets, bounds, objective identities and checkpoints. No reservoir optimization jobs have been launched as part of preparation.

## Selection algorithm

1. Require consecutive native daily/monthly dates and finite nonnegative adjusted inflows. Identify complete years; partial edge years remain unscored context. Reserve the first complete year for warmup.
2. Classify eligible years using lower/upper annual-inflow terciles: dry below the lower threshold, wet above the upper threshold, normal between them. Keep tied values together; constant forcing produces only a normal group and explicit missing-regime warnings. These are reservoir-inflow categories, not independently observed meteorological classifications.
3. Compute annual time-weighted mean inflow, native-sample P10 inflow and previous-two-year mean inflow. Monthly mean weighting uses calendar month lengths for a classification proxy, without changing the solver's fixed monthly clock. A recorded measurement is not automatically established to be a monthly mean or an endpoint value.
4. Select whole years within each regime, including eligible forcing extremes and a normal central year. Greedy coverage spreads selections in normalized forcing, antecedent-flow, low-flow and calendar-time space. The seed breaks ties reproducibly; it is not chosen from simulation scores.
5. Allocate at least 65% of eligible years and **at least 65% of all recorded native rows**, including warmup/context in the denominator. Also require at least that fraction of all finite storage and release observations separately. Round up by whole years; some records need more than 65%. Preserve at least one assessment year, and one per regime when feasible. Reject impossible coverage rather than silently weakening the minimum.

Only forcing features and observation-availability counts affect selection. Observed storage/release values and model performance do not. Whole years preserve all seasons, but annual stratification cannot guarantee every operating change or drought sequence is represented. Antecedent forcing and continuous state partially address memory; source operating definitions still matter. Calendar spread is not a substitute for explicit known operating-era metadata.

## Fitting and scoring

Monthly empirical targets and P5–35/P35–75/P75–95 bounds are recomputed from calibration observations only, using Hazen percentiles. Unselected outcomes do not enter fitting. Masks and their SHA256 signatures are authoritative; a bounds date envelope is only metadata, not an instruction to fit every date inside it.

If raw generalized targets violate physical ordering but their empirical box contains feasible candidates, preserve the raw targets and separately construct a physically feasible bounds midpoint as the optimizer initialization. This is not a clipped or corrected generalized baseline. Physical constants, capacity, dead-storage fraction and raw bounds remain unchanged. If the empirical box is itself infeasible, preparation reports that problem instead of launching a search.

Each candidate runs once through continuous native forcing from the original initial state to the last calibration year. Intermediate assessment/context years remain in the state evolution, but their observed storage/release are not used or assimilated. Future forcing after the final calibration year cannot reject a calibration candidate. Assessment is an explicit separate call that runs the full continuous record and scores only assessment dates. No annual state resets or concatenation of disconnected years occurs.

The default objective gives each represented calibration regime equal weight:

```text
release cost = -mean(NSE_release_dry, NSE_release_normal, NSE_release_wet)
storage cost = -mean(NSE_storage_dry, NSE_storage_normal, NSE_storage_wet)
```

This prevents larger wet-period variance from automatically dominating a pooled NSE. Pooled NSE and regime NSE are also reported. If a regime is genuinely absent, only represented regimes can be scored and the coverage warning remains; undefined NSE in an active regime is an error, not silently dropped. The alternative `pooled_nse` cost mode is available through the API and changes the checkpoint identity. Equal-regime weighting is an explicit application choice, not a claim of universally optimal calibration.

## Commands and API

Prepare all twenty currently calibrated samples:

```powershell
python -B tools/prepare_representative_calibration.py --output outputs/my_representative_plan --minimum-fraction 0.65
```

Select a subset or a water-year convention explicitly:

```powershell
python -B tools/prepare_representative_calibration.py --reservoir-id ex5_Garrison ex6_Oahe --water-year-start-month 10 --output outputs/my_water_year_plan
```

An optimization run is a separate computational job with its own persistent checkpoint and progress log. This example starts a new search; it was **not executed** during preparation:

```powershell
python -B tools/calibrate_representative_years.py outputs/my_representative_plan --reservoir-id ex5_Garrison --seed 42 --population 32 --evaluation-budget 512 --output outputs/my_garrison_search
python -B tools/calibrate_representative_years.py outputs/my_representative_plan --reservoir-id ex5_Garrison --seed 42 --population 32 --evaluation-budget 1024 --output outputs/my_garrison_search --resume
```

Identity includes the exact row selection, cost mode, prepared inputs, model constants, numerical source, strategy/runner hashes and runtime versions. The existing checkpoint loader rejects a changed identity. Old chronological checkpoints must not be continued against this objective, even if populations or bounds look similar. Searches log trials, vectors, Pareto costs and generation history. They do not select parameters using assessment scores or promote profiles automatically.

The reusable API is in [representative_years.py](../strategies/representative_years.py): `select_representative_years`, `fit_selected_years`, `RepresentativeYearObjective`. It sits outside the frozen numerical package so earlier checkpoint identities remain unchanged.

## Interpretation and literature

The prepared [twenty-reservoir plan](../outputs/representative_years_v1_20261012/README.md) is a representative historical assessment design. Interleaving years and continuous state introduce temporal dependence; it is not a blind prospective forecasting test. The record has already been examined, so the new assessment must not be marketed as untouched external validation. For forecasting or changing operating regimes, add a separately declared temporal or differential-condition test and freeze it before selecting parameters. Do not tune split seeds from assessment performance.

Noncontinuous hydrological-year tests have precedent in the [general differential split-sample study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7451689/). [Shen, Tolson and Mai (2022)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021WR031523) examine how data splitting affects hydrological model robustness and challenge a universal preference for older-data calibration/newer-data validation. [Razavi and Tolson (2013)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2012WR013442) discuss representative information in calibration periods. Those studies support examining representativeness; they do not establish this exact selector, the 65% requirement or a guarantee of improved DZTR predictions. The 65% minimum is the user's requirement.
