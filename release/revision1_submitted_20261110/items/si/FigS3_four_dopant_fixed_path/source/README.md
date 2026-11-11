# Four-dopant fixed-path NEB figure, palette and font update

This package re-renders the Cr/Cu/Li/Na fixed-path figure for the SI with larger type and the shared MigrationBench palette. The three path headings use legible, dark pink text accents from the shared palette in `analysis/canva_palette_vector_bundle_20261105/MIGRATIONBENCH_FIGURE_PALETTE.json`; model curves retain the MACE foundation-class green family and screened QE curves remain charcoal dashed lines.

## Source and reproducibility

- Original SI PDF: `output/revision1_camera_ready/paper/revision1/figures/main/SI/fig_NEB-fixed-path_Cu-Li-Na-Cr.pdf` (SHA-256 recorded in `RENDER_MANIFEST.json`).
- Original R03 plotting implementation: `analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/archive/legacy_figure_products_20261024/build_foundation_class_4x3_r03.py`.
- Latest palette re-render precedent: `analysis/canva_palette_vector_bundle_20261105/source/render_foundation_class_palette_20261102.py` (CANVA-P08).
- Plot inputs, copied locally for reproducibility: the R03 grouped MACE class profile TSV and screened QE path reference TSV in `source_data/`.

The data values and normalized image coordinates are unchanged. Run from the repository root:

```bash
python analysis/four_dopant_fixed_path_palette_20261109/render_four_dopant_fixed_path.py
```

The script exports PDF, editable SVG, and PNG under `figures/`. The figure is a diagnostic fixed-image energy comparison; the source QE records retain convergence limitations as stated in the SI caption.
