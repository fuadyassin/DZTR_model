# Parameterization conventions

## Input contract

`ReservoirObservations.frame` has a timezone-naive daily `DatetimeIndex` named
`date`, sorted chronologically. Values use these canonical columns:

| Field | Internal unit | Recognized source examples |
| --- | --- | --- |
| `storage_m3` | m³ | `Storage(mil_m3)`, `Storage(mcm)`, `Storage(m3)` |
| `inflow_m3s` | m³/s | `Inflow(cms)`, `inflow_m3s` |
| `release_m3s` | m³/s | `Outflow(cms)`, `release_m3s` |

Storage in million m³ is multiplied by 1,000,000 at ingestion. Bare source names
require explicit units for populated observations. A conflicting explicit unit
is rejected. Dates come from `Date`/`date`, `year/month/day`, or a daily index;
conflicting date representations raise. Subdaily, timezone-aware, duplicate,
missing or invalid dates are rejected.

Missing variables and cells stay NaN. No interpolation, zero filling or invented
daily sequence occurs. Nonnumeric, infinite and negative measurements raise.
Empty or entirely missing data are `unavailable`. A file with at least two rows,
all three variables present, and every measurement zero is a suspected
`placeholder`. This classification can include a truly empty reservoir record;
inspect its provenance before treating it as usable data.

## Quantile estimates

The shared default set is P5 through P100 at five-point increments, plus P99.
All three variables use the same definitions. The estimator returns tidy rows:
`variable`, `month`, `percentile`, `value`, `sample_count`. Month zero denotes
the full supplied period; months 1–12 group observations by calendar month
across that period. Each nonmissing observation has equal weight. Empty samples
produce NaN and a zero sample count, including an entirely missing variable.

The default is explicitly NumPy's `linear` interpolation, preserving the previous
calculation convention. An explicit `quantile_method="hazen"` reproduces the
historical MATLAB `prctile` algorithm; NumPy's `midpoint` is a different method.
The returned estimates record the chosen method. These are non-exceedance
percentiles, not exceedance probabilities. [NumPy documentation](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)

## Generalized targets

The six monthly targets use P10/P45/P85 for storage and release. A complete
result requires both variables in all twelve months. No global fallback is
substituted for missing monthly data. Inflow is optional at this stage. Equal
storage targets are retained and reported as a degenerate-zone issue for the
future release solver, rather than perturbing the observations.

`channel_capacity_m3s` uses a supplied physical value, or overall release P99
when omitted. The benchmark preserves the catalog values of 2500 m³/s for
Gardiner and 300 m³/s for Trinity, and reports release P99 separately. The
targets follow [Yassin et al. (2019), Sections 3.1 and 3.3](https://hess.copernicus.org/articles/23/3735/2019/#section3).

## Legacy boundary

`dztr_param.calculate_percentiles` and `update_extended_data` delegate to the
normalized calculation but return storage in **million m³**. Existing workbook
values and external exporters retain their historic convention. Generated
schemas and calculations agree on P65, P99 and P100. Newly computed NaNs replace
stale estimates for missing samples. The updater returns a new DataFrame,
preserves metadata, supports new IDs on current pandas, and warns before
skipping missing, ambiguous, placeholder or unavailable data. Invalid data raise.

## Reproduction and provenance

The CLI compares 780 existing quantiles per benchmark in original workbook units
using rtol=1e-10 and atol=1e-8, allowing floating-point roundoff in unit conversion
and Excel serialization. P99 has no saved reference and is checked independently
in tests. Reports include source SHA-256 digests, periods, dependency versions,
maximum errors by variable and availability counts.

The saved parameters describe the entire historical record. Reproducing them is
a parameter regression check, not evidence of simulation accuracy or
out-of-sample performance. The calibration/validation baseline derives distributions
from its calibration period and reserves validation data; see
[the baseline notes](calibration_baseline.md).

## Simulation configuration and remaining questions

- Trinity's `SMAX` is 2,533,500,000 m³, while `CAP_MCM` implies 2,633,500,000 m³.
  The supplied MATLAB coefficient files explicitly select 2,533,500,000 m³;
  the Trinity simulation adopts that value without overwriting the GRanD field.
- Capacity, initial state, dead storage, inflow factor and flood mode are explicit
  `ReservoirParameters` fields separate from statistical target estimation.
  Evaporation and precipitation are not separate forcings in this MATLAB port;
  it preserves the original multiplicative inflow adjustment.
- Legacy level-to-storage acquisition assumes the observed maximum level maps
  to full dam height. Verify the physical datum before using derived storage.
- CSV/TXT exporters and HydroLAKES metadata need separate format and unit
  checks before integration.
