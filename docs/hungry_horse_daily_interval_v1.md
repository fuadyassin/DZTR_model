# Hungry Horse: daily-mean DDS calibration and GRAND 299 profile

This versioned scenario uses all complete calendar years in the combined flow
archive: **1988–2016**. It uses the whole separate USBRPN storage series and the
original USBR/HORIZON inflow and release series. No records are silently spliced
between these products.

The source review verified the USBRPN storage values against current USBR
daily AF records on 20–26 August 2006. The archived 152,000 cfs inflow on
23 August conflicts with the provider's estimated unregulated-flow product.
That product is a different quantity, so it is not substituted. The anomalous
inflow interval is explicitly quarantined, and all original source values
remain preserved.

USACE's 2022 Hungry Horse Water Control Manual, PDF page 19, distinguishes:

| Storage definition | Acre-feet | Use in this scenario |
|---|---:|---|
| Normal full pool, elevation 3,560 ft | 3,467,178 | Recorded separately from maximum storage |
| Maximum gross pool, elevation 3,565 ft | 3,586,783 | DZTR hard storage ceiling, including surcharge |
| Physical dead storage, elevation 3,196 ft | 37,131 | DZTR dead-storage threshold |
| Active joint-use storage | 2,982,025 | Distinct from total/gross storage |

The hard storage ceiling is not a normal operating target. Monthly targets are
estimated from the calibration observations. Channel capacity remains a
calibration-only Hazen Q99 benchmark proxy. The report retains these modeling
assumptions rather than treating them as a certified operating rule.

[USACE manual](https://water.usace.army.mil/cda/documents/wc/3390/HungryHorse_WCM_10102022_redacted.pdf#page=19),
[USBR daily storage retrieval](https://www.usbr.gov/pn-bin/daily?station=hgh&format=html&year=2006&month=8&day=20&year=2006&month=8&day=26&pcode=af&pcode=fb),
[USBR daily flow retrieval](https://www.usbr.gov/pn-bin/daily?station=hghm&format=html&year=2006&month=8&day=20&year=2006&month=8&day=26&pcode=qu&pcode=qd).
Copies of the provider responses, the manual and the source-review receipt are
pinned in the prepared evidence directory. The PDF reader used for source review
was installed only in `.cache/vendor/pypdf` on D:, not in the Python environment;
it is not a runtime dependency of calibration.

Run the complete workflow:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/calibrate_daily_intervals_v1.py run --config configs/hungry_horse_daily_interval_v1_20261021.json --output outputs/hungry_horse_daily_interval_v1_20261021
```

The command prepares inputs, runs three independent DDS jobs using seeds
42/43/44 and 512 evaluations each, selects by equal-weight pooled storage/release
NSE loss, assesses every seed and exports the shared GRAND-indexed profile.
It pins scientific code and prepared inputs. Each job saves its RNG state,
full trial history and best cost every 16 evaluations. Completed jobs and
publications are verified and reused. A checkpoint can continue within its
frozen horizon of 512; a different horizon is a different DDS experiment.

For a stopped worker, after confirming its process has exited, the saved
checkpoint can be continued with:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/calibrate_daily_intervals_v1.py worker --output outputs/hungry_horse_daily_interval_v1_20261021 --seed 42
```

An existing `running.lock` prevents concurrent workers for the same seed. Do not
remove it while that worker is running. A terminated worker can leave a stale
lock; inspect the recorded process ID before removing only that lock file.

Calibration uses whole mixed dry/normal/wet years. Annual regimes use means of
available reported inflows, with at least 95% daily coverage. No missing daily
inflows are filled. Fixed flow exclusions and a 30-day warmup after each block
restart are applied before selecting years. The actual fitting mask covers
**67.12% of the entire 10,593-day calendar**: 21 calibration years, seven
held-out years and one initial warmup year. Targets, raw percentile bounds,
channel Q99 and gap initial states use calibration data only.

Each finite-forcing block starts from a **fixed calibration-derived monthly
normal storage**. It never starts from a held-out storage observation, and
parameters do not change the frozen initial-storage reference. Year-role
boundaries do not reset model state. Restarts occur only at forcing gaps;
initial rows and the next warmup days are excluded from fitting and scoring.

The mean-flow adapter integrates each available daily inflow as its own
86,400-second volume and compares modeled mean total outflow, including spill,
with observed mean release. The array path calls the unchanged canonical root
solver; all exported representatives are replayed through the full canonical
interval simulator. No equations or roots are altered for speed.

Selected parameters load through the existing `reservoir_v2.load_profile`
shared registry using GRAND 299. **Replay them through the mean-flow adapter**:

```powershell
& .\tools\Set-ProjectCache.ps1
python -B tools/calibrate_daily_intervals_v1.py replay --output outputs/hungry_horse_daily_interval_v1_20261021
```

This profile's route is `daily_mean_interval_scalar_v1`; the existing native
endpoint simulation adapter is a different clock and must not be substituted.
The profile records its initialization recipe and complete prepared scenario.

Outputs include the per-seed baseline comparison, monthly bias, DZTR-zone
diagnostics, seed agreement, checkpoint improvement, figures, selected parameter
registry and a new 62-reservoir parameterization export. Earlier inventories,
data and calibrated profiles are preserved.

Reported inflow independence and the exact historical provider clock remain
unverified. The explicit interval-end convention and source combination make
this an exploratory assessment. Held-out responses do not select seeds or
parameters, but this does not establish fully independent predictive validation.
