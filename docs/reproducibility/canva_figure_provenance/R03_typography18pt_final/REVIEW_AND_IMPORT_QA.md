# R03 typography and Canva import QA

This is the publication-final review package generated from PR #14 commit `935e492`. Figure 1 is the approved R06 clean raw-PCA package; Figures 2, 3, 4 and 6 remain in this directory. Historical exploratory outputs are removed from the active tree and retained only in Git history/archive snapshots.

## Pre-import checks

- Source renderer uses Arial, bold panel titles, 18 pt authored minimum text, 20 pt axis/panel labels, 18 pt ticks and legends, and a 22 pt figure title.
- Shared legends are placed below the overall figure for Figures 2, 3, 4 and 6. Figure 1C keeps its source-split legend below panel C because it is panel-specific.
- Figure 1C retains complete source-split clustered coverage.
- Figure 2C/F retain four model families, train/test clusters, n=3 replicate points and sample-SD whiskers.
- Figure 3B retains the complete Foundation-reference paired clusters.
- Figure 4 retains the approved Foundation-reference paired structure.
- Figure 6B/C retain the complete five-family dumbbell and 15-model coordination clusters from the audited status table.
- Energy profile axes use `ΔE (eV)` and image/path axes use `Reaction Coordinate`.
- PDF and SVG are emitted as vector outputs; PNG is only the visual QA raster preview.
- Reproduction inputs used by the final renderers are vendored under `../data/`; large raw trajectories, checkpoints and remote scratch collections remain pointer-only.

## Data-boundary notes

The complete summary panels are not reduced because auxiliary provider-backed files are unavailable locally. One Figure 2 force archive is omitted only from the auxiliary frame-level scatter layer. Figure 6 line summaries omit only auxiliary files that do not materialize in this checkout; Figure 6B/C remain status-table complete. No synthetic rows, imputation, or extrapolation are used.

## Visual QA result

The five PNG previews were inspected after rendering. The rejected intermediate Figure 6 made by mechanically enlarging the legacy SVG is not part of this directory. The final Figure 6 was regenerated from the source plotting logic and checked for clipping, overlap, legend placement and vector text.

## Canva import inputs

Use the matching `.svg` files for editable vector import. Keep the matching `.pdf` files as archival vector exports and the `.png` files as review previews. Import each figure as a separate Canva design/page so text, paths, bars, markers, whiskers and shaded bands remain independently selectable where Canva's SVG importer preserves them.
