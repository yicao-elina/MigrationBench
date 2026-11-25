# Cr three-path Scratch-600K R05

This is the user-certified 2026-11-01 release bundle for the complete
Scratch-600K-Lmax1 Cr three-path figure. The figure keeps the page-13 legend
order and the R04 typography/layout contract while adding the certified
Scratch-600K data to all three path columns.

## Deliverables

- `figures/`: publication PDF, SVG, and PNG.
- `scripts/make_cr_three_path_2x3_R05.py`: standalone renderer.
- `data/`: plotted profile table and coverage audit.
- `source/`: materialized campaign/model/path manifests, available endpoint
  records, source-table snapshots, and explicit pointer boundaries for inputs
  evicted by the File Provider.
- `provenance/`: source/output hashes, render record, and the original R05
  analysis manifest. `POINTER_ONLY_INPUTS.tsv` records the original source
  paths and hashes for provider-backed inputs that are not materialized in the
  current checkout.

Rebuild from the repository root with:

```bash
python results/plots/cr_three_path_fixed_vs_self_scratch600k_20261101_R05/scripts/make_cr_three_path_2x3_R05.py
```

The renderer writes outputs into this bundle's `figures/` and audit tables
into `data/` by default. The manifest records the exact source hashes and
the three-output hashes. When macOS has evicted provider-backed inputs, use
the original analysis-root R05 commit and the pointer table to re-materialize
them before rerunning the historical-source portion of the renderer.

## Scientific boundary

The Scratch-600K campaign is the canonical Lmax=1 set with seeds 123/234/345,
81 atoms, 10 images, and the three paths `in-gap`, `gap→QL`, and `within-QL`.
All 9/9 fixed-path profiles are finite. The self-relaxed endpoint-only NEB
row includes 7/9 converged primary-evidence cases. The two diagnostic-only
cases are retained in the source campaign records but excluded from the main
performance values: seed123 × within-QL and seed234 × in-gap.

The self-relaxed curves are endpoint-only MLIP NEB proxies, not DFT MEP
barriers. The DFT reference vector is retained explicitly in the renderer;
the original reference PDF is an external pointer with its SHA256 recorded in
the manifest. Large trajectory/iteration extxyz files remain remote and are
listed with paths and hashes in
`source/scratch_600k_migration_20261101/raw/rockfish/scratch_600k_migration_20261101/OMITTED_OR_POINTER_ONLY.tsv`.

The original analysis-root R05 bundle was committed as
`ea3a30247ed60414f0b5ac056afeeb722cf63925`. This canonical `results/plots`
bundle is the organized release copy; its local reproducibility boundary is
explicit rather than hiding provider-backed inputs as empty local files.
