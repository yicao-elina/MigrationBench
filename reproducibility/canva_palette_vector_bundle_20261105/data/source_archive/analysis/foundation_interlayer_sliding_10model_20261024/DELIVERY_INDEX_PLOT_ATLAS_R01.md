# Fixed-path 10-model Plot Atlas reconstruction R01

This is an additive visual reconstruction of the frozen campaign figure set.
The original `results/figures/` products are unchanged.

## Figures

- `results/figures/plot_atlas_r01/fixed_path_bulk_10model_plot_atlas_r01.{png,svg,pdf}`
- `results/figures/plot_atlas_r01/fixed_path_cr_in_gap_10model_plot_atlas_r01.{png,svg,pdf}`
- `results/figures/plot_atlas_r01/fixed_path_interlayer_sliding_10model_2system_plot_atlas_r01.{png,svg,pdf}`

## Reproduction

```bash
python analysis/foundation_interlayer_sliding_10model_20261024/scripts/plot_interlayer_10model_plot_atlas_r01.py \
  --root /Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/foundation_interlayer_sliding_10model_20261024
```

## Inputs and boundaries

- `RESULT_MATRIX.tsv`: 20 fixed-path rows selected from the 40-row campaign matrix.
- `IMAGE_PROFILES.tsv`: 200 fixed-path image-profile rows, 10 images per model/system.
- `QE_REFERENCE.tsv`: 20 reference rows retained with the `SCF_WARNING` boundary.
- Model identity is preserved from the result matrix; `MatPES-PBE OMAT-FT` is labelled `MatPES-OMAT-FT`.
- Fixed-path curves are `FINITE_DIAGNOSTIC`, not self-relaxed DFT MEP barriers.
- Self-relaxed endpoint-only MLIP NEB quantities are not mixed into these panels.

## Visual review

PASS: PNG raster inspection completed for the two standalone panels and the
1×2 composite. The Plot Atlas grammar uses Arial-first typography, 2.0 pt
solid open-circle model traces, 1.45 pt dashed open-diamond QE reference,
boxed 1.05 pt axes, fixed normalized-coordinate ticks, no decorative grid,
and a shared model-ordered legend. Outputs are exported as PNG, SVG, and PDF.

The exact source/output hashes are recorded in
`manifests/FIGURE_PROVENANCE_PLOT_ATLAS_R01.json`.
