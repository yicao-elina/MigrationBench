# MLIP candidate energy-surface validation (2026-10-01)

## Scope and evidence boundary

This is an independent candidate single-point campaign. Existing endpoint relaxations and NEB campaigns were read-only inputs; no candidate was relaxed and no existing campaign directory was overwritten. Absolute MACE energies are retained per model for traceability but are not compared across model families because their zero points and energy conventions differ.

## Coverage

- **neb_intermediate**: 26 candidates.
- **stable_endpoint**: 24 candidates.
- **near_saddle**: 15 candidates.
- **NEB source coverage**: 26/36 paths had a local provenance-backed final MACE image; 10 were not executed and were not replaced by IDPP.
- **MD high-energy/large-displacement**: `md_missing`; no auditable real MD snapshots were available.

## Single-point result boundary

- **FT-600K**: 65/65 successful single points; failures remain in `failure_ledger.csv`.
- **FT-600K-grouped-s123**: 0/65 successful single points; failures remain in `failure_ledger.csv`.
- **FT-600K-grouped-s234**: 0/65 successful single points; failures remain in `failure_ledger.csv`.
- **FT-600K-grouped-s345**: 0/65 successful single points; failures remain in `failure_ledger.csv`.
- **FT-MultiT**: 0/65 successful single points; failures remain in `failure_ledger.csv`.
- **Foundation**: 65/65 successful single points; failures remain in `failure_ledger.csv`.
- **MPA0**: 65/65 successful single points; failures remain in `failure_ledger.csv`.
- **MPA0-grouped-s123**: 0/65 successful single points; failures remain in `failure_ledger.csv`.
- **Scratch**: 0/65 successful single points; failures remain in `failure_ledger.csv`.
- **Scratch-grouped-s123**: 0/65 successful single points; failures remain in `failure_ledger.csv`.
- **Scratch-grouped-s234**: 0/65 successful single points; failures remain in `failure_ledger.csv`.
- **Scratch-grouped-s345**: 0/65 successful single points; failures remain in `failure_ledger.csv`.
- **Scratch5-grouped-s123**: 0/65 successful single points; failures remain in `failure_ledger.csv`.
- **Scratch5-grouped-s234**: 0/65 successful single points; failures remain in `failure_ledger.csv`.
- **Scratch5-grouped-s345**: 0/65 successful single points; failures remain in `failure_ledger.csv`.

Successful model comparisons use max force, Cr position spread, coordination range, and finite-geometry checks. `model_anomaly` is a screening flag, not a claim of model accuracy or OOD proof.

## Scientific interpretation

Stable endpoints are the most complete category: the FT-600K, Foundation, and MPA0 CPU-compatible weights evaluated all 24 stable candidates. The near-saddle and NEB-intermediate sets are intentionally single-point-only; their forces are diagnostic of local model response and must not be interpreted as relaxed barriers.

The current CPU-compatible results show a strong regime distinction: stable endpoints have low FT-600K residual forces, while intermediates and near-saddle images have larger forces, as expected for non-stationary structures. Foundation and MPA0 also produce finite values but generally larger forces on these non-endpoint candidates. This is evidence for force-surface sensitivity, not evidence that either model is more accurate.

The 13 historical/seed weights that failed during model deserialization are classified as runtime failures, with original tracebacks retained. They are not counted as scientific failures, OOD cases, or successful evaluations. GPU/QoS access is still required before claiming complete seed-ensemble coverage.

## DFT single-point priority

1. Near-saddle candidates from `cov_020`, `cov_035`, `cov_002`, `cov_027`, and `cov_031`, especially those marked `model_anomaly` or with high successful-model max force.
2. NEB intermediates on the highest-barrier paths and all source-complete paths with force/coordination disagreement.
3. Stable endpoints `unique_044`, `unique_065`, `unique_067`, `unique_080`, `unique_096`, `unique_097`, `unique_109`, and `unique_151`, preserving the prior consensus/disagreement rationale.

## Reproduction

The local manifest/build script is `build_candidates.py`; remote execution used `run_rockfish.slurm` with `run_single_points.py` under `/scratch16/pclancy3/yi/mace_candidate_eval_20261001`. The successful remote job was Rockfish job 31475643 (CPU-compatible subset); job 31475439 records the earlier full CPU attempt with the original legacy/seed weights and its runtime failures.
