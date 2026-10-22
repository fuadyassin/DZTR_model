# Calibrated simulation workflow, version 1.0

The twenty selected profiles can now be simulated directly from the exported SI parameter table. The command checks the registry, profile/table identities and exact monthly targets, frozen numerical source, forcing inputs, historical clocks and periods. It runs one continuous historical simulation per reservoir and compares every column with the archived selected trajectory. It independently reconstructs trapezoidal mass balance. It performs no optimization and does not select parameters using held-out results.

From the repository root, with the environment containing the project's dependencies:

```powershell
python -B tools/simulate_calibrated_reservoirs.py --all --output outputs/my_twenty_simulation
python -B tools/simulate_calibrated_reservoirs.py --grand-id 253 597 --reservoir-id ex4_Fort_randall --output outputs/my_subset_simulation
```

Choose a new output directory for each batch. GRAND_ID aliases and internal IDs can be mixed; duplicate aliases produce one job. Fort Randall remains available by internal ID because its GRAND_ID is unresolved. Unknown IDs fail rather than substituting another reservoir.

Each reservoir gets `simulation.csv`, the exact monthly SI targets, profile metadata and `result.json`. The batch gets `batch_summary.json`, `readiness.csv` and a hash manifest. Failures are recorded per reservoir and the command exits unsuccessfully if any job fails. These are historical replay commands; future forcing requires a separate explicit input interface.

The verified release is [twenty_workflow_release_v1_20261012](../outputs/twenty_workflow_release_v1_20261012/README.md). All twenty reproduce exactly. A reproducible simulation does not certify physical source definitions, optimizer convergence or production suitability. Readiness records retain held-out NSE separately; the existing diagnostic reports contain seasonal, drift, peak and zone evidence. Deferred reservoirs remain recorded in the prior workflow and are outside this release.

The release refers to the existing immutable calibration archive and packaged observations; it is a repository workflow snapshot, not a standalone data distribution. New calibration or forcing conventions require a new version and verification.
