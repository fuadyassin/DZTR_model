# Gardiner historical water-level input snapshot

Station `05HF003`, LAKE DIEFENBAKER AT GARDINER DAM, Saskatchewan. Historical daily means for 1980-01-01 through 2011-12-31 were retrieved from ECCC's GeoMet API on 2026-10-18 UTC (2026-10-17 local project date).

`acquisition/levels_si.csv` has all 11,688 dates, in metres, preserving provider numerical values. Provider symbols comprise 11,110 unflagged, 183 estimated and 395 partial-day values. Symbols are not grounds for silently removing observations; the pilot reports their diagnostics separately. The acquisition stage does not derive storage, infer a datum conversion or download new forcing. Its observation CSV has empty inflow/release/storage columns because only the level station was requested.

The twelve raw daily-mean response pages and their hashes, station query, daily calendar and retrieval time are retained under `acquisition/raw/` and `acquisition/provenance.json`. Missing values would remain missing; none was gap-filled. The raw API responses are serialized as JSON without stripping provider fields.

`station_metadata.json` identifies PFRA datum. `station_datum.html` records the official modern conversion to CGVD2013:EPOCH2010, adding −0.313 m, published November 16, 2022 after a 2021 GNSS occupation. This does not by itself certify historical datum continuity or any reservoir-bottom elevation. `operator_full_supply.html` states the full supply level as 556.87 m without an explicit vertical-datum/geometry-depth tie. `metadata_receipt.json` pins the retrieved source snapshots.

Sources: [ECCC hydrometric daily means](https://api.weather.gc.ca/collections/hydrometric-daily-mean/items?STATION_NUMBER=05HF003&datetime=1980-01-01/2011-12-31), [station metadata](https://api.weather.gc.ca/collections/hydrometric-stations/items?STATION_NUMBER=05HF003&f=json), [vertical datum](https://wateroffice.ec.gc.ca/report/datum_e.html?stn=05HF003), [WSA full supply level](https://wsask.ca/infrastructure/infrastructure-l1-1/lake-diefenbaker/).

Credit: Environment and Climate Change Canada, historical hydrometric data, retrieved 2026-10-18 UTC. Approximate geometry remains the separately archived ReGeom dataset; no geometry was inferred from these gauge levels alone.
