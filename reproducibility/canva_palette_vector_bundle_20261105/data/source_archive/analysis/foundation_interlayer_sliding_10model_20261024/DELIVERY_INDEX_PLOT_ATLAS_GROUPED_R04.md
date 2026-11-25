# Grouped interlayer-sliding figure — Plot Atlas R04

R04 updates the method label from `self-relaxed endpoint-only` to `self-relaxed path`. Data and visual styling remain the accepted R03 version: bold typography, +2 pt typography update, strengthened spines, DFT-NEB dashed trace, and pointwise sample-SD uncertainty bands for the three model classes.

## Main figure

- [PNG](results/figures/plot_atlas_grouped_r04/interlayer_sliding_10model_grouped_2x2_plot_atlas_r04.png)
- [SVG](results/figures/plot_atlas_grouped_r04/interlayer_sliding_10model_grouped_2x2_plot_atlas_r04.svg)
- [PDF](results/figures/plot_atlas_grouped_r04/interlayer_sliding_10model_grouped_2x2_plot_atlas_r04.pdf)

## Standalone panels

- [Fixed bulk](results/figures/plot_atlas_grouped_r04/fixed_bulk_grouped_plot_atlas_r04.png)
- [Fixed Cr-in-gap](results/figures/plot_atlas_grouped_r04/fixed_cr_in_gap_grouped_plot_atlas_r04.png)
- [Self bulk](results/figures/plot_atlas_grouped_r04/self_bulk_grouped_plot_atlas_r04.png)
- [Self Cr-in-gap](results/figures/plot_atlas_grouped_r04/self_cr_in_gap_grouped_plot_atlas_r04.png)

Each standalone panel is also available as SVG and PDF in the same output directory.

## Provenance

- [R04 provenance manifest](manifests/FIGURE_PROVENANCE_PLOT_ATLAS_GROUPED_R04.json)
- [Plot script](scripts/plot_interlayer_grouped_profile_atlas_r01.py)

The R04 render was reconstructed from the intact `SLIDING_MATRIX.tsv`, `IMAGE_PROFILES.tsv`, and `QE_REFERENCE.tsv` sources because the redundant `RESULT_MATRIX.tsv` container was empty while retaining its old file size.
