# MigrationBench paper reproduction

This repository is the canonical source for the MigrationBench revision, its
paper artifacts, and the compact evidence needed to audit every reported
result. Overleaf is a publication mirror; Rockfish is a compute backend, not a
source of truth.

## Reproduction levels

1. **Paper reproduction (offline):** regenerate registries, manuscript macros,
   tables, figures, and release checks from the committed compact evidence.
2. **Full computational reproduction:** rerun QE/MACE jobs on Rockfish using
   the recorded inputs, manifests, job IDs, environment, and source hashes.
   Wavefunctions, charge densities, complete QE `outdir` trees, and model
   checkpoints are intentionally not stored in Git.

## Quick start

```bash
conda env create -f environment.yml
conda activate migrationbench
make reproduce
make test
make audit
```

For the ARCH Skipjack cluster, use the validated module-based recipe in
[`docs/skipjack_environment.md`](docs/skipjack_environment.md). It keeps
PyTorch supplied by `pytorch/2.10.0-b200-py3.11` and installs the remaining
MACE/ASE packages into a project-local prefix.

The supported source-checkout test entry point is `make test`; it sets
`PYTHONPATH=src` explicitly.

`make paper` additionally compiles the main text, SI, and response letter when
the full LaTeX toolchain is installed. To rebuild one target, use Snakemake,
for example `snakemake -s workflow/Snakefile figures/publication/fig6_shap_revised.pdf`.

The Sec. 3.5/Fig. 6 SHAP release uses only the X-FORCE three-model compact
inputs in `data/raw_compact/revision1/shap/`. Re-run the aligned workflow with:

```bash
PYTHONPATH=scripts python scripts/shap_analysis.py \
  --xyz data/raw_compact/revision1/shap/sampled_trajectory_Cr2_temp.xyz \
  --predictions data/raw_compact/revision1/shap/sampled_trajectory_Cr2_temp_predictions_with_binding1.csv \
  --output /tmp/migrationbench-shap
```

The scope and source hashes are recorded in `docs/audit/shap_source_lineage.md`.
The 600-row four-model input, 2-concentration analysis, and Dual-X ID/OOD/NEB
SHAP results are not inputs to the MigrationBench Fig. 6 claim.

## Repository map

- `configs/`: stable project/model/path definitions and experiment configs.
- `workflow/`: executable Snakemake DAG and Rockfish profile.
- `src/migrationbench/`: submission, monitoring, ingestion, analysis,
  plotting, and provenance code.
- `data/raw_compact/`: immutable compact evidence copied from historical runs.
- `data/processed/`: canonical analysis inputs; `data/published/`: review exports.
- `registry/`: run/claim/reviewer ledgers and content hashes.
- `figures/`, `tables/`: generated and publication-ready artifacts.
- `paper/`: main text, SI, response letter, bibliography, and journal assets.
- `docs/history/`: migration decisions and source snapshots.

## Operating rules

- Analysis and plotting code reads `data/processed`, never arbitrary cluster
  paths.
- A new run is accepted only after its manifest and compact outputs are synced,
  hashed, and entered in `registry/runs.csv` (and `runs.parquet` when PyArrow is
  available).
- A manuscript number enters through `paper/results_generated.tex`; the same
  value must not be independently copied into main text, SI, and response.
- `python scripts/sync_overleaf.py --overleaf PATH` stops on remote divergence
  by default. It never silently overwrites independent Overleaf edits.
- Before release, `make audit` and `make test` must pass with a clean worktree.

## Historical sources

The migration inventory, source hashes, inclusion policy, and known gaps are
documented in `docs/history/MIGRATION_REPORT.md`. The remote branch
`archive/main-before-paper-repro-2026-09-17` preserves the previous GitHub
`main` exactly.
