# Collaborator DFT/NEB dataset

This directory is the processed, provenance-first layer for DFT and NEB
calculations collected from Rockfish collaborators. Raw calculation trees stay
on Rockfish and are never copied into GitHub. Every processed record must point
to an immutable source path plus file hashes, parser version, and quality-gate
status.

## Current sources

- `vwu16` / Victor Wu: `/scratch16/pclancy3/victor`
- `jpilla1` / Joshitha: `/data/jedison2/jpilla1` (currently not readable by
  `ycao73`; ingestion is blocked until owner/group read permission is granted)

The user's Rockfish shortcuts are under:

`/scratch16/pclancy3/yi/MigrationBench_collaborators/`

## Files

- `schema.sql`: SQLite schema for systems, calculations, artifacts, structures,
  trajectories, results, provenance, and quality gates.
- `source_inventory.csv`: human-auditable source roots and access state.
- `calculation_catalog.csv`: normalized calculation-level catalog; `status`
  distinguishes observed files from scientifically accepted results.
- `quality_gates.csv`: required checks before a record can support a paper claim.
- `publication_blueprint.md`: end-to-end processing plan and manuscript/reviewer
  mapping.

The authoritative database will be generated as `migrationbench_dft_neb.sqlite`
by the ingestion scripts after raw access and parser validation are complete.
