# Endpoint-47 DFT–MACE evidence package

## Summary

This PR adds an additive, provenance-tracked evidence package for the 47
Discovery DFT source geometries and the corresponding MACE diagnostics. It
includes four publication figures, three table families, raw/static ensemble
records, model hash reconciliation, failure/exclusion ledgers, and an
Overleaf-ready manuscript integration map.

## Evidence included

- 47-site lattice/geometry distribution across within-gap, gap/QL interfacial,
  and within-QL regions.
- Restricted 21-endpoint retry-campaign DFT relative-energy panel; the three
  time-limited endpoints remain excluded from ranking claims.
- Four-model, three-start legacy MACE relaxation sensitivity.
- Twelve-model static source-geometry ensemble: 3 foundation-family
  comparators plus 3 seeds each for FT-600K, FT-MultiT, and Scratch; 564/564
  predictions completed with hash-verified provenance.
- Exploratory model-disagreement associations with continuous vertical/local
  coordination descriptors.

## Scientific boundaries

This PR does not establish stable/metastable/unstable labels, migration
barriers, transport coefficients, or calibrated MACE–DFT absolute errors.
The current DFT–MACE error ledger remains ineligible until matched final DFT
XYZ structures are archived and evaluated on the same geometry.

## Overleaf handoff

Start with `paper_handoff/OVERLEAF_HANDOFF.md` and
`paper_handoff/MIGRATIONBENCH_INTEGRATION_MAP.md`. The richer claim-to-artifact
matrix is `paper_handoff/PAPER_CLAIM_EVIDENCE_MATRIX.tsv`; all release hashes
are written to `paper_handoff/PAPER_RELEASE_MANIFEST.json`.

## Validation

- Existing package build, integration, table, and figure scripts are recorded
  under `scripts/`.
- Model and source-input hashes are retained in `tables/` and raw provenance
  records.
- The package preserves empty scheduler log files as evidence gaps rather than
  treating scheduler completion as a scientific gate.

