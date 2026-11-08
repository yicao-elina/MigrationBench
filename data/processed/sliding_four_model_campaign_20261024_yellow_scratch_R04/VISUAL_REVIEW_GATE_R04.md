# Fig. 4 visual review gate (R04, frozen)

This is the acceptance contract for the R04 fixed-path Fig. 4b/4c render.
The reference grammar is the supplied Foundation-class R03 figure and its
renderer. MigrationBench model colors remain the project colors; geometry and
rendering style are the locked reference elements.

## Required plotted marks

- Display summary curves only; do not display individual seed/checkpoint
  traces.
- Every colored model summary is pointwise mean ± 1 SD.
- Colored summary curves use solid lines, linewidth `2.0`, open-circle markers,
  marker size `4.3`, white marker face, same-color marker edge, and edge width
  `0.95`.
- Every SD band uses alpha `0.20` and no edge.
- DFT uses a black dashed line with dash pattern `(0, (4, 2))`, linewidth
  `1.45`, open-diamond markers, marker size `3.6`, white face, `#222222`
  edge, and edge width `0.8`.

## Required visual grammar

- Legend contains names only: `DFT`, `Foundation`, `FT-600K`, `FT-MultiT`,
  `Scratch-600K`, `Scratch-MultiT`.
- No `mean`, `SD`, `n=3`, `checkpoint`, or methodological text appears in the
  legend.
- Arial is embedded in PDF; four spines are visible in `#111111` at linewidth
  `1.05`.
- Major ticks point outward, length `4.5`, width `1.0`; minor ticks point
  outward, length `2.5`, width `0.8`; no grid lines.
- White background, no clipped labels, no overlapping legend or panel label,
  and one-page PDF/SVG/PNG outputs.

## Scientific invariants

- Input table remains the Stage-1 `fixed_profiles.csv` with 300 unique
  model/system/image rows.
- Foundation aggregation is across the three registered checkpoint variants,
  not training seeds.
- Observable remains `FIXED_PATH_SINGLE_POINT`; DFT status remains
  `COMPLETED_BUT_SCF_WARNING`.

R04 is accepted only after an independent visual-review subagent returns
`PASS` against this contract for both Fig. 4b and Fig. 4c.
