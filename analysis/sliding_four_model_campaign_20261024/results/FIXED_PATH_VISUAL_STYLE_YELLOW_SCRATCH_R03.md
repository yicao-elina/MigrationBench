# Fixed-path Fig. 4 visual-style update (R03)

This R03 revision keeps the R02 warm-yellow scratch palette and simplifies the
legend to model names only:

`DFT` · `Foundation` · `FT-600K` · `FT-MultiT` · `Scratch-600K` · `Scratch-MultiT`

Foundation is now rendered as a pointwise mean ± 1 SD across its three
registered checkpoint variants, using the same summary-band grammar as the
other model groups. This is an ensemble summary across checkpoint variants,
not a training-seed uncertainty estimate.

The fixed-path data table, model hashes, axis limits, marker family, scratch
subfamily colors, and scientific boundary are unchanged from R02. The DFT
reference remains `COMPLETED_BUT_SCF_WARNING`, and the panels remain
fixed-path single-point diagnostics rather than self-relaxed or validated
migration barriers.

## Reproduction

```text
python scripts/plot_fig4_fixed.py \
  --config config/campaign.yaml \
  --style-file config/FIG168_MODEL_STYLE_YELLOW_SCRATCH_R03.json \
  --output-tag yellow_scratch_R03 \
  --manifest-name FIG4_FIXED_PATH_PLOT_MANIFEST_yellow_scratch_R03.json
```

## Outputs

- `results/figures/fig4b_interlayer_sliding_pristine_fixed_path_20261024_yellow_scratch_R03.pdf`
- `results/figures/fig4b_interlayer_sliding_pristine_fixed_path_20261024_yellow_scratch_R03.svg`
- `results/figures/fig4b_interlayer_sliding_pristine_fixed_path_20261024_yellow_scratch_R03.png`
- `results/figures/fig4c_interlayer_sliding_cr_in_gap_fixed_path_20261024_yellow_scratch_R03.pdf`
- `results/figures/fig4c_interlayer_sliding_cr_in_gap_fixed_path_20261024_yellow_scratch_R03.svg`
- `results/figures/fig4c_interlayer_sliding_cr_in_gap_fixed_path_20261024_yellow_scratch_R03.png`
- `results/figures/FIG4_FIXED_PATH_PLOT_MANIFEST_yellow_scratch_R03.json`
