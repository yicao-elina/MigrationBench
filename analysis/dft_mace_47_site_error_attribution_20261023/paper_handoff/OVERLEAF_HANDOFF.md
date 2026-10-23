# Overleaf handoff checklist

This directory is the manuscript-facing release boundary for the endpoint-47
DFT–MACE evidence package. The package is additive and should be pulled into
the paper revision branch only after the associated Git commit and Overleaf
version are recorded together.

## Import order

1. Read `MIGRATIONBENCH_INTEGRATION_MAP.md` and the parent
   `NATURE_READY_RESULTS_AND_DISCUSSION_20261023.md`.
2. Copy the four PDF figures from `../figures/` into the manuscript figure
   asset directory. Prefer PDF for the paper; retain SVG/PNG for inspection.
3. Convert or copy the three TSV tables into the paper's table-source
   directory without editing values by hand.
4. Use `PAPER_CLAIM_EVIDENCE_MATRIX.tsv` to map every sentence to a figure,
   table, ledger, and allowed claim status.
5. Add the package commit SHA and the Overleaf Git commit/version to the
   release manifest before merging the manuscript text.

## Compile and visual QA

From a clean checkout, rebuild with:

```text
python analysis/dft_mace_47_site_error_attribution_20261023/scripts/build_attribution_package.py
python analysis/dft_mace_47_site_error_attribution_20261023/scripts/integrate_source_geometry_ensemble.py
python analysis/dft_mace_47_site_error_attribution_20261023/scripts/build_publication_tables.py
python analysis/dft_mace_47_site_error_attribution_20261023/scripts/render_publication_figures.py
```

Then compile the Overleaf project, inspect every panel at journal scale, and
compare the rendered values against the TSV tables. Record the resulting
Overleaf commit/version in `PAPER_RELEASE_MANIFEST.json`; do not silently
replace the PDF figures or hand-edit derived table values.

## Scientific gates carried into the manuscript

- The 21-endpoint energy panel is a restricted retry-campaign reference, not a
  global stability or metastability classification.
- `unique_077`, `unique_078`, and `unique_093` remain failed/nonconverged and
  are excluded from ranking claims.
- Empty scheduler `.out/.err` files remain a logging evidence gap even where
  the job exit code is zero.
- The 12-model ensemble is complete for static source-geometry predictions
  (564/564), but it is not a DFT-calibrated error estimate because matched DFT
  final XYZ structures are not locally archived.
- The legacy `Foundation` label means an OMAT-derived fine-tuned checkpoint;
  preserve that wording in captions and methods.

## Do not merge these claims from this package

Do not infer thermodynamic stability, migration barriers, transport
coefficients, or absolute MACE–DFT errors from these artifacts alone. Those
claims require their own completed evidence packages and gates.

