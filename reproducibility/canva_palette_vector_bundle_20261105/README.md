# Canva P01–P10 reproduction bundle

This additive package keeps the latest locally reproducible Canva quantitative figures together with the plotting code, the plotted/processed inputs, and source-lineage records. It is designed so that a future figure can be found from its page ID, then traced from final PNG/PDF/SVG to the local renderer, the local plotted data, and the upstream source pointer.

## Scope and page coverage

The quantitative vector products are CANVA-P03–P10. P01 and P02 are retained in `manifests/CANVA_PAGE_COVERAGE.tsv` as historical workflow/structure pages, but are not represented by fabricated quantitative data or a new plot renderer.

Each P03–P10 product has all three final formats:

- PNG for review;
- editable SVG;
- vector PDF.

P03 and P04 were regenerated with non-rasterized scatter layers. The vector QA report records PDF image counts, drawing counts, text blocks, and SVG `<image>` tag counts.

## Visual standard

- Foundation: `#8FC7AB`
- Fine-tuning: `#29A7B6`
- Scratch: `#327FB1`
- DFT/reference: `#303840`
- 600 K: open marker; Multi-T: filled marker
- seed 123 / 234 / 345: circle / triangle / square
- Foundation checkpoint: diamond
- fixed path / self-relaxed: solid / dashed lines when overlaid
- Foundation checkpoint-specific plots use the green family: MACE-MP `#2F6B4F`, MACE-OMAT-0 `#5F9E73`, MACE-MH `#8FC7AB`.

## Reproduce

Run from this package directory with a Python environment containing NumPy, Matplotlib, and PyMuPDF:

```text
python3 reproduce_all.py
python3 scripts/verify_bundle.py
python3 scripts/build_source_hash_manifest.py
```

`reproduce_all.py` runs the four local renderers. They now resolve their inputs from `data/source_archive/` inside this package; they do not depend on the original worktree paths. The renderers write to `figures/` and write renderer-level manifests at the package root. The verification script refreshes `CANVA_PALETTE_VECTOR_MANIFEST.json` and `manifests/OUTPUT_VECTOR_QA.tsv`.

## Provenance and status boundaries

- `SOURCE_PROVENANCE.tsv` is the figure-level map from page to output, renderer, plotted data, source archive, and scientific/publication status.
- `SOURCE_POINTERS.tsv` records external raw trajectories, checkpoints, and provider-backed paths that are not materialized in Git.
- `FAILURE_EXCLUSION_REGISTRY.tsv` records unavailable/provider-dataless files and nonconverged or diagnostic-only boundaries; they are not silently promoted to primary evidence.
- `manifests/SOURCE_FILE_HASHES.tsv` is generated over every materialized file under `data/source_archive/`; text-file hashes use LF normalization so they remain stable across Git checkouts.
- Existing source campaign manifests, ledgers, release registries, and audit tables are retained under `data/source_archive/` wherever they are materialized.

The bundle is a reproducible visual package, not a claim that every historical Canva input can be regenerated exactly. Large trajectories, model checkpoints, and provider-backed files remain explicit pointers with their original paths or source-manifest references.

## Figure index

| Page | Final stem | Renderer | Primary local data |
|---|---|---|---|
| P03 | `CANVA-P03_parity_2x4_palette_updated` | `scripts/render_rmse_palette_20261102.py` | `analysis/rmse_train_test_20261026/raw/rockfish_parity` |
| P04 | `CANVA-P04_frame_rmse_palette_updated` | `scripts/render_rmse_palette_20261102.py` | `analysis/rmse_train_test_20261026/raw/rockfish_parity` |
| P05 | `CANVA-P05/md_six_panel_palette_updated` | `scripts/render_md_six_panel_palette_20261102.py` | `analysis/md_1ns_updated_publication_figure_20261025/md15_analysis_r03` plus RDF tables |
| P06 | `CANVA-P06_neb_cr_r05_palette_updated` | `scripts/render_neb_palette_20261102.py` | Cr R05 tables plus Scratch-600K endpoint archive |
| P07 | `CANVA-P07_foundation_class_3x3_palette_updated` | `scripts/render_foundation_class_palette_20261102.py` | Foundation class 3×3 R03 tables |
| P08 | `CANVA-P08_foundation_class_4x3_palette_updated` | `scripts/render_foundation_class_palette_20261102.py` | Foundation class 4×3 R03 tables |
| P09 | `CANVA-P09_cr_interlayer_sliding_palette_updated` | `scripts/render_neb_palette_20261102.py` | Cr grouped R04 tables and fixed/self source records |
| P10 | `CANVA-P10_foundation_interlayer_sliding_palette_updated` | `scripts/render_neb_palette_20261102.py` | Foundation 10-model grouped Plot Atlas tables |

The machine-readable output paths and hashes are in `CANVA_PALETTE_VECTOR_MANIFEST.json`.
