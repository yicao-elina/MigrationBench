# Paper provenance audit — 2026-10-13

This is an additive, read-only-first audit package for the MigrationBench paper
and Revision 1. It separates model identity from manuscript labels and links
raw inputs, model/checkpoint hashes, jobs, derived tables, figures, and text
claims.

## Immediate publication decision

The file

`/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/finetuned_MACE_compiled.model`

with SHA256

`dc41a98908dd25b150fb5c8dd0a28d95f5feeac37022682a7c389c475444d08e`

is treated as `legacy_compiled_omat`, not as official zero-shot MACE-OMAT-0.
Its raw trajectories and instability diagnostics are retained for forensic
history, but it is excluded from the publication model ranking and must not be
used to support claims about the public `mace-omat-0-medium.model`.

The public/OMAT-derived file

`/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/X-FORCE/xforce/models/Foundation-MACE-matpes-pbe-omat-ft.model`

with SHA256

`e618ad582b84239905b9c3b77ce6e9ce111b0ecd1533223a1a6aac7a696b8aa0`

is a separate OMAT-FT model. Existing rows that use this file but call it
`Foundation` should be relabelled `OMAT-FT`, not deleted.

The Revision 1 `0.41 eV` self-NEB claim is currently not joined to a unique
model hash. It must not be attributed to either the legacy compiled checkpoint
or the public OMAT-0 medium checkpoint until the original run manifest is
recovered. This is a traceability gap, not evidence that the numerical value is
correct or incorrect.

## Evidence roots

- Original four-model MD manifests and checkpoint audits:
  `analysis/mace_1ns_publication_figure_20261005/` and
  `analysis/phase1_static_force_comparison_20261002/`.
- Revision 1 article source and source ledger:
  `/Users/alina/Project/migrationbench-paper-repro/paper/revision1/` and
  `/Users/alina/Project/migrationbench-paper-repro/data/processed/manuscript_source_ledger.csv`.
- DFT/NEB inventory and acceptance gates:
  `data/collaborator_dft_neb/` and
  `/Users/alina/Project/migrationbench-paper-repro/analysis/`.
- Figure evidence inventory:
  `analysis/figure_evidence_audit_20261008/`.

## Status vocabulary

`VERIFIED` means the artifact and its content hash are identified and the
relevant scientific gate passes. `LINEAGE_SUPPORTED` means the training
relationship is supported by scripts/manifests but still needs the exact source
checkpoint hash. `PROVENANCE_PARTIAL` means the result exists but at least one
model, input, protocol, or sampling link is unresolved. `EXCLUDE` means the
artifact remains archived but is not allowed into the current publication
tables/figures. `RECONSTRUCTION` means the result is reproduced from surviving
evidence rather than exact original raw output.

## Safety rule

Do not delete raw models, trajectories, DFT outputs, or old figures as part of
this audit. Exclude them through the publication registry and build new
paper-facing tables/figures from the canonical registry. Deletion requires a
separate explicit archival decision after checksums and retention copies exist.
