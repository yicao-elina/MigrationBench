# Canonical repository migration report

Migration date: 2026-09-17

## Sources examined

1. OneDrive submission working tree: original manuscript assets and the full
   untracked `Revision1/` audit package.
2. Overleaf Git clone: publication history through commit `fb3e83b` and a
   distinct response-letter lineage.
3. `/Users/alina/Project/26MIgrationBench`: the newer executable Rockfish
   workflow, current configs/state, tests, 185 compact run directories, and
   later material-transfer/protocol-sensitivity work.
4. GitHub `yicao-elina/MigrationBench`: previous `main` at `266e2d1` and `dev`
   at `d48e652` before the canonical-repository migration.

Absolute source paths are intentionally documented only here as migration
history and are not consumed by the workflow.

## Canonical-source decisions

- Main and SI use the OneDrive `Revision1` copies, which contain the audit and
  reviewer-marker mechanism.
- The response letter uses the complete Revision1 TeX pack. Supporting review
  material is retained under `paper/response_to_reviewers/`.
- Current executable scripts/configs/state come from the newer local
  `26MIgrationBench` working tree, including its uncommitted September 17
  monitoring and QE protocol-sensitivity updates.
- Original paper figures remain under `paper/Fig/`; revised publication figures
  are under `figures/publication/`.
- Historical prose audits are evidence, not executable state, and live under
  `docs/audit/` and `docs/workflow/`.

## Data inclusion policy

Committed evidence includes text inputs/outputs, scheduler logs, runtime and
restart manifests, compact extxyz snapshots, small CSV/JSON/Parquet tables,
and publication artifacts. Files larger than the compact threshold are omitted
when a smaller manifest/summary exists.

Explicitly excluded: QE wavefunctions, charge density, complete `outdir` and
`.save` trees, large trajectory histories, virtual environments, caches, and
model checkpoints. The original Revision1 contamination consisted of an
approximately 34 MB `wfc1.dat` and 8.2 MB `charge-density.dat`; neither is in
this repository.

## Traceability contract

The intended chain is:

`reviewer item -> claim_id -> experiment/run_id -> compact raw artifact + SHA256 -> parser -> processed row -> figure/table -> generated LaTeX macro -> manuscript/response`.

`registry/claims.yaml`, `registry/reviewer_map.yaml`, `registry/runs.csv`, and
`registry/artifacts.sha256` provide the machine-readable spine. The imported
`manuscript_source_ledger.csv`, `claim_map.md`, and `pending_values.csv` retain
the earlier audit vocabulary while it is progressively normalized.

## Known limitations

- Some imported historical scripts still expose command-line assumptions from
  the Rockfish-era layout. They are preserved for evidentiary completeness;
  new workflow rules must use repository-relative paths.
- The current 0.336 eV 1-4 DFT barrier remains explicitly labelled as an
  unconverged candidate in the historical ledger.
- Full computational reproduction requires Rockfish access, QE modules,
  pseudopotentials, and external model checkpoints. Paper reproduction does not.
- The imported manuscript still contains historical reviewer placeholders.
  The release audit requires that placeholders remain registered until resolved.

