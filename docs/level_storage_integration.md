# Level and geometry preparation before parameterization

The original `dztr_param/res_data_est_from_hydrometric.py` already converts observed gauge levels into estimated storage using GRanD-indexed ReGeom tables. The [2026-10-17 review](../outputs/level_storage_review_v1_20261017/README.md) documents that work, checks 77 local curves and explains how to integrate it before data-availability route selection.

Implement the next stage as a versioned geometry profile plus a batch level-to-storage adapter. Make its reference level, curve coordinate, units, timing, storage definition and uncertainty explicit. Prefer usable level-derived storage to the constant storage-fraction fallback, while preserving its provenance as an estimate.

The new adapter and parameterization routes remain proposed. Existing calibrated profiles, source gates and numerical equations were not changed by the review.
