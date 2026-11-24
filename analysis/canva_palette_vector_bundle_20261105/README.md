# Canva P03–P10 palette vector bundle

This dated bundle contains the latest reproducible Canva parallel plots using the MigrationBench visual standard. It is additive: historical plot roots and the earlier palette exports are not overwritten.

## Visual encoding

The complete shared palette and encoding reference is in [`PALETTE_REFERENCE.md`](PALETTE_REFERENCE.md), with machine-readable values in [`MIGRATIONBENCH_FIGURE_PALETTE.json`](MIGRATIONBENCH_FIGURE_PALETTE.json).

- Foundation: `#8FC7AB`
- Fine-tuning: `#29A7B6`
- Scratch: `#327FB1`
- DFT/reference: `#303840`
- Campaign task types: Base `#FABAB9`, Perturb A `#F26EA6`, Perturb B `#74297E`.
- 600 K: open marker; Multi-T: filled marker
- seed 123 / 234 / 345: circle / triangle / square
- foundation checkpoints: diamond
- fixed path / self-relaxed: solid / dashed lines where both are shown

P06 is the R05 Cr three-path version, including the independent Scratch-600K data group. P07/P08 expand Foundation checkpoints in the green family; P10 uses the same family for Foundation sliding.

## Outputs

Each figure has a PNG review render and a PDF in this Git bundle. SVG exports are native vector outputs; P03's 54 MB editable SVG is preserved in the Rockfish archive because of its size, while its PDF and PNG remain here. P03 and P04 were regenerated with scatter layers explicitly non-rasterized; earlier PDF versions contained embedded raster images.

The exact paths, SHA-256 hashes, source pointers, archive location, and vector QA counts are in CANVA_PALETTE_VECTOR_MANIFEST.json.

## Reproduction

Use the repository's Anaconda environment and run from this directory:

```text
/Users/alina/anaconda3/bin/python3 source/render_rmse_palette_20261102.py
/Users/alina/anaconda3/bin/python3 source/render_md_six_panel_palette_20261102.py
/Users/alina/anaconda3/bin/python3 source/render_neb_palette_20261102.py
/Users/alina/anaconda3/bin/python3 source/render_foundation_class_palette_20261102.py
```

The renderer scripts write their freshly generated files under `source/figures/`; the committed release copies are under `figures/`. The source data and scientific-status boundaries remain those recorded by the existing local campaign manifests and source scripts.

P01/P02 are workflow/structure pages rather than quantitative parallel plots, so they are not silently represented by this P03–P10 plot bundle.
