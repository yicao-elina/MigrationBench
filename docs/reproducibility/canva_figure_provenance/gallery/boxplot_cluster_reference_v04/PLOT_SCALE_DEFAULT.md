# Default plot scale: seed-resolved clustered boxplots

This is the default project grammar for publication figures that compare
model families across categories while exposing seed-level variation.

## Data and marks

- Keep all finite frame-level observations as points inside their exact
  family/category/seed box; use deterministic horizontal jitter.
- Use log10 values for the RMSE distribution when the source metric spans
  orders of magnitude.
- Use the boxplot median, IQR, and 1.5-IQR whiskers; suppress fliers only
  because all finite frame points are already displayed.
- Encode model family with the R04 fill palette:
  `Scratch-600K #FF9F1B`, `FT-600K #002E71`,
  `Scratch-MultiT #F4C95D`, `FT-MultiT #67ACE5`.
- Encode seed independently with a neutral white legend and hatch pattern:
  seed 123 = no hatch, seed 234 = `///`, seed 345 = `\\\\\\`.

## Scale and stroke defaults

- Use Arial first, with a sans-serif fallback; white figure background.
- Tick labels: 20 pt bold.
- Axis labels: 22 pt bold.
- Panel titles: 22 pt bold; composite section headings: 23 pt bold; figure
  titles/legend title: 24 pt bold where used.
- Major ticks: 8 pt length, 1.8 pt width; minor ticks: 4 pt length,
  1.2 pt width.
- Main axes and box edges are black and visibly heavy: 1.8 pt axes,
  1.45 pt box/whisker/cap edges, 2.0 pt median.
- Hatch stroke: approximately 1.35 pt.
- The v16 reference uses `box_width=0.135` and seed offsets `[-0.16, 0,
  0.16]` for the two-family temperature clusters. For four-family clusters,
  reduce the width proportionally (the source scripts remain authoritative for
  each concrete layout).
- Do not add decorative gridlines. Preserve generous whitespace and keep all
  text, brackets, and legend handles clear of the plotting marks.

## Composition defaults

- For new composites, place a shared legend below the full chart when the
  panels share one legend; its text should visually match the axis labels
  rather than becoming a small afterthought. The exact v16 gallery render is
  intentionally preserved with its historical top legend so its pixels and
  source remain a faithful reference case.
- For the preferred square composite, use a 16 x 16 inch canvas and a 2 x 5
  grid: temperature-resolved panels occupy the left three columns and the
  overall Scratch-vs-FT overview occupies the right two columns.
- Use explicit gutters between the two chart families. Keep the first x-axis
  segment joined to its y-axis and reserve a dedicated right gutter for the
  overview y-axis label.
- Group headers use a thick upper rule and a thinner lower rule. Keep model
  family colors, hatch semantics, and panel ordering unchanged across the
  standalone and composite versions.

## Canonical reference

The exact implementation is the Figure 2 v16 chain ending at commit
`941baf8`, especially `_figure02_latest_seed.py`,
`_figure02_latest_layout.py`, `_figure02_latest_compact.py`, and
`build_figure02_latest.py`. This document describes the default grammar;
the generated gallery manifest records the exact script/input hashes for a
particular reproduction.

