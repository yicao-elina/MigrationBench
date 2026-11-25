# Endpoint-47 DFT–MACE site and disagreement package

This additive analysis package audits the 47 Discovery DFT endpoints without
modifying an existing DFT, NEB, MD, or MACE relaxation campaign.

## Main deliverables

- Figures: figures/FIG1 through figures/FIG4 in PDF, SVG, and PNG.
- Paper-facing interpretation: NATURE_READY_RESULTS_AND_DISCUSSION_20261023.md.
- Static-ensemble scheduler and provenance audit:
  SOURCE_GEOMETRY_ENSEMBLE_AUDIT_20261023.md.
- Data-quality and claim limits: tables/DFT47_DATA_QUALITY_REPORT.tsv and
  tables/CLAIM_EVIDENCE_MATRIX.tsv.
- Endpoint-level data: processed/DFT_MACE47_INTEGRATED_ENDPOINT_TABLE.tsv.
- Overleaf-ready handoff: paper_handoff/OVERLEAF_HANDOFF.md,
  paper_handoff/MIGRATIONBENCH_INTEGRATION_MAP.md,
  paper_handoff/OVERLEAF_HANDOFF_MANIFEST.tsv, and
  paper_handoff/PAPER_CLAIM_EVIDENCE_MATRIX.tsv.

## Reproduction order

1. Run scripts/build_attribution_package.py to rebuild source-site descriptors,
   DFT evidence categories, and existing four-model relaxation summaries.
2. Run scripts/integrate_source_geometry_ensemble.py after the raw
   source_geometry_ensemble results are present.
3. Run scripts/build_publication_tables.py.
4. Run scripts/render_publication_figures.py with the local Plot Atlas source
   directory on PYTHONPATH.

## Non-negotiable evidence boundary

The package reports MACE ensemble disagreement on immutable source geometries.
It does not report a calibrated DFT error because the final DFT XYZ structures
are not locally available for same-geometry MACE evaluation. Do not subtract
the currently stored QE and MACE total energies or relabel geometric zones as
stable, metastable, or unstable.

## Paper handoff

The `paper_handoff/` directory is the release boundary for manuscript work.
`MIGRATIONBENCH_INTEGRATION_MAP.md` explains where each result belongs in the
main text or Supplementary Information, what upstream result it consumes, and
what downstream claim it enables. `OVERLEAF_HANDOFF.md` gives the exact copy,
compile, and provenance checks for a later Overleaf handoff. The handoff is
intentionally additive: it does not replace the existing grouped-split,
endpoint/NEB, MD, or transport evidence packages.
