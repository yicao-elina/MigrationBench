# Fixed-path Fig. 4 visual-style update (yellow scratch split, R02)

## Plan and implementation

1. Preserve the existing Fig. 4b/4c data table, model hashes, marker mapping,
   training-seed line styles, uncertainty calculation, axis limits, and
   fixed-path scientific boundary.
2. Split the single `from_scratch` display color into two named subfamily
   colors, while retaining the project palette rather than importing the
   Foundation/MACE colors from the supplied R03 reference figure.
3. Match the supplied R03 plotting grammar at the standalone-panel level:
   Arial-first typography, bold axis labels, four visible spines, outward
   major/minor ticks, no grid, white background, and an unframed figure-level
   legend.
4. Export additive PDF, SVG, and PNG products and record their hashes in a
   versioned manifest. The pristine Stage-1 products and their original
   manifest remain unchanged.

## Model color mapping

| Model family | Color | Role |
|---|---|---|
| Foundation | `#CF4520` | Existing project red |
| FT-600K | `#002E71` | Existing project dark blue |
| FT-MultiT | `#67ACE5` | Existing project light blue |
| Scratch-600K | `#FF9F1B` | Existing project orange retained as the darker warm tone |
| Scratch-MultiT | `#F4C95D` | New lighter golden-yellow companion tone |

The two scratch subfamilies retain the filled-diamond marker family; their
solid/dash-dot summary-line distinction and per-seed line styles are unchanged.
The colors are intentionally related by hue but remain separable in the
legend, traces, and uncertainty bands.

## Scientific and data boundary

This is a visual-only revision of the completed Stage-1 fixed-path products.
The observable remains `FIXED_PATH_SINGLE_POINT`; the DFT reference remains
`COMPLETED_BUT_SCF_WARNING`; no result is relabelled as self-relaxed or as a
validated migration barrier.

## Reproduction

```text
python scripts/plot_fig4_fixed.py \
  --config config/campaign.yaml \
  --style-file config/FIG168_MODEL_STYLE_YELLOW_SCRATCH_R02.json \
  --output-tag yellow_scratch_R02 \
  --manifest-name FIG4_FIXED_PATH_PLOT_MANIFEST_yellow_scratch_R02.json
```

## Outputs

- `results/figures/fig4b_interlayer_sliding_pristine_fixed_path_20261024_yellow_scratch_R02.pdf`
- `results/figures/fig4b_interlayer_sliding_pristine_fixed_path_20261024_yellow_scratch_R02.svg`
- `results/figures/fig4b_interlayer_sliding_pristine_fixed_path_20261024_yellow_scratch_R02.png`
- `results/figures/fig4c_interlayer_sliding_cr_in_gap_fixed_path_20261024_yellow_scratch_R02.pdf`
- `results/figures/fig4c_interlayer_sliding_cr_in_gap_fixed_path_20261024_yellow_scratch_R02.svg`
- `results/figures/fig4c_interlayer_sliding_cr_in_gap_fixed_path_20261024_yellow_scratch_R02.png`
- `results/figures/FIG4_FIXED_PATH_PLOT_MANIFEST_yellow_scratch_R02.json`
