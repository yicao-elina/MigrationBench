# Twelve-member MACE source-geometry ensemble audit — 2026-10-23

## Scope

This is an additive, read-only static evaluation of the 47 immutable MACE base
input geometries. It is not an endpoint relaxation campaign and it does not
overwrite an existing MACE or DFT result. The purpose is to quantify
model-to-model disagreement at the source geometries before a matched-geometry
DFT error audit is possible.

## Submitted computation

- Remote root: /scratch16/pclancy3/yi/dft_mace_47_source_geometry_ensemble_20261023/
- Slurm array: 32373911, indices 0–11, concurrency cap 2, A100 GPU
- Protocol: one model per array member; 47 static MACE energy/force evaluations;
  no optimizer and no cell or atom-coordinate updates.
- Registry: 12 hash-verified checkpoints:
  - three Foundation-family comparator members;
  - FT-600K seeds 123/234/345;
  - FT-MultiT seeds 123/234/345;
  - Scratch seeds 123/234/345.

All 12 paths were verified against their remote SHA-256 values before
submission. The Scratch path/hash reconciliation is retained in
tables/MODEL_REGISTRY_HASH_RECONCILIATION.tsv; it corrects two
path-versus-compiled-checkpoint registrations and one malformed 65-character
source hash without changing any remote checkpoint.

## Terminal evidence

All array members 32373911_0 through 32373911_11 are COMPLETED with
ExitCode=0:0. Each model wrote:

- one 47-row source_geometry_predictions.tsv;
- one compressed per-atom force archive;
- one provenance.json recording observed model hash, input-manifest hash,
  software environment, and n_success=47, n_failure=0.

Thus the returned evidence is 12 models × 47 structures = 564 successful,
hash-verified static predictions. Raw scheduler logs, predictions, force
archives, and provenance files are retained under
raw/source_geometry_ensemble/.

## Runtime warnings retained

Every array stderr contains PyTorch/MACE TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
warnings. Six members also report a TorchScript-archive dispatch warning; all
stdout records show n_failure=0, and no scheduler log contains a traceback.
Cuequivariance acceleration was unavailable. These are runtime/performance
warnings, not evidence of a model or numerical failure, but they remain part
of the audit trail.

## Interpretation boundary

The output is a model-disagreement diagnostic on source geometries. It is not
a DFT error benchmark because the locally available package lacks the final DFT
XYZ structures needed to evaluate MACE energy and forces on the exact same
geometry, cell, PBC, and atom order.
