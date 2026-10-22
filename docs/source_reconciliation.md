# Source reconciliation before DZTR calibration

The [Ghost and Trinity review](../outputs/source_reconciliation_20261008/README.md) traces source records, conversions, editing history, timing and storage definitions. It includes a separate dated provider reference for Trinity and a ledger for every previously flagged or extreme date. Existing canonical observations and MATLAB reproduction settings are preserved.

For each reservoir, record the following facts before creating a new calibration data profile:

| Field | What it must describe |
| --- | --- |
| Source identity | Provider, station and sensor IDs, source file/version, retrieval time and hash. |
| Units and transformations | Raw units, conversion factors, formulas, edits and their provenance. Keep raw values separate from normalized values. |
| Storage coordinate | Total, live or active volume; datum, source rating curve and curve version. |
| Storage clock | Actual snapshot time, timezone and relation to the named date. |
| Flow interval | Instantaneous sample or interval mean; interval start/end, timezone, date label and total release components. |
| Inflow derivation | Independent gauges, modeled forcing or storage-balance computation; drainage coverage and information shared with calibration targets. |
| Quality and missingness | Original flags, missing records, signed values, edited records, objective masks and any forcing replacement policy. |
| Limits | Physical dead volume, normal full pool, flood surcharge and DZTR operating floor as separately named quantities in the same storage coordinate. |
| Periods | Warmup, calibration and validation boundaries; transformations and parameter estimates learned only from calibration. |

A source correction must have evidence independent of improved model fit. Provider revisions remain separate dated versions. A raw numeric storage conversion can be corrected directly, while an already filled column may require returning to raw records to avoid scaling flow increments. Preserve signed computed inflow in the source layer even if a particular simulator requires a later documented forcing decision.

For total end-of-day storage and named-day mean flows, the interval balance is `S[D] - S[D-1] = (I[D] - Q[D] - losses[D]) * duration`. That convention needs the actual observation definitions; a generic date label is insufficient. Changing diagnostic accounting does not automatically change the simulator's integration rule. Compare incomplete source versions on common support and report coverage; gaps prevent a continuous whole-period closure claim.

The [calibration requirements](../outputs/source_reconciliation_20261008/calibration_gates.json) document verified facts and open decisions for Ghost and Trinity. They are machine-readable review records, rather than checks enforced by the existing CLI. A future runner should inspect its selected source profile and requirements before starting an optimizer, and write the selected source hash, clocks, masks and storage definitions into every result.

After the source profile is defined, re-estimate percentile targets from calibration records in that profile's coordinate and units. Review model bounds, initial state and floor with the same definitions; retain validation for assessment. For hundreds of reservoirs, report incomplete source definitions per reservoir so that an unresolved sample does not interrupt all completed source reviews.
