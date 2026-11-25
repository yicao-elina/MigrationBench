# Fixed-path errors and overlap audit figure

Editable and reproducible PDF/SVG/PNG figure rendered in the current MigrationBench palette.

## Data and methods

Panel a uses the canonical Cr in-gap fixed-path absolute barrier errors from `output/revision1_camera_ready/paper/revision1/data_processed/current_release/fig5_plotted_values_as_supplied.tsv` (`column_id=migration_in_gap_fixed`, `absolute_error_eV`). It shows four trained model groups with three individual training seeds each, plus one Foundation group containing MP-0B3, OMAT-0, and MPA0. Seed markers follow the project convention (circle, triangle, square for seeds 123, 234, and 345); open markers are 600 K and filled markers are Multi-T. Diamonds and horizontal whiskers show mean ± sample SD for each group. For Foundation, this summarizes variation across the three distinct checkpoint models, not training-seed variation; individual checkpoint values remain visible as green diamonds.

The canonical DFT reference retains a convergence warning; these fixed-path results are diagnostics and should not be read as validation of a converged activation barrier.

Panel b retains the legacy 25-image train–path overlap audit. Each bar is the minimum training-set distance divided by its predeclared near-duplicate threshold. Values above one lie outside the threshold.

The compact plotting input and source provenance are recorded in `source_data/grouped_retraining_overlap_data.json`.

## Rebuild

From the project root:

```bash
python analysis/grouped_retraining_overlap_palette_20261109/plot_grouped_retraining_overlap.py
```

The script writes `figures/grouped_retraining_overlap_palette_updated.pdf`, `.svg`, and `.png`. Typography follows the project standard: Arial/Helvetica/DejaVu Sans fallback, 11 pt ticks, 12 pt axis labels, 14 pt panel headings, PDF font type 42, and editable SVG text. Palette: scratch `#327FB1`, fine-tuning `#29A7B6`, foundation `#8FC7AB` with related shades, reference ink `#303840`.
