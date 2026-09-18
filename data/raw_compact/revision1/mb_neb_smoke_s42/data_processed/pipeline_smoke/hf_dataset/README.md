# MigrationBench

This dataset stores migration-barrier benchmark artifacts with lossless provenance.

## Tables

- `configurations`: structures and source hashes.
- `calculations`: energies, forces when real, convergence status, raw-log provenance.
- `neb_paths`: ordered NEB paths and canonical barrier metadata.
- `derived_barriers`: protocol-specific barriers and manuscript inclusion gates.

## Current Build Note

Generated from `/home/ycao73/revision1_pipeline_20260909/data_processed/pipeline_smoke/mlff_emt/mlff_neb_images.extxyz` with source sha256 `7760a3cf98d21702dc5ae8a10f2aa7e3ae1253f5a1bce590d50695c32912807d`.
Protocol: `mlff_pre_neb`. Convergence status: `converged`.

Rows with `force_label_status=fabricated_zero_quarantined`, placeholder energies, digitized PNG values,
or unconverged/diverged status are retained for audit but excluded from manuscript-allowed derived barriers.
