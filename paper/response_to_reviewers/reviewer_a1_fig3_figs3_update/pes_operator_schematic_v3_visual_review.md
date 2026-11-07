# Visual QA record — PES operator schematic v3

Date: 2026-10-19

## Passed checks

- The shared DFT/MLIP PES legend is in a dedicated figure-level band and does
  not touch either 3D z-axis or surface.
- The five atom snapshots are compact QL cross-sections with visibly stacked
  layers; each shows the red connected Cr trajectory and gold current image.
- The fixed and self-relaxed top panels retain separate black/blue and
  black/red path grammars, including their floor projections.
- The lower-left fixed section uses the same synthetic surface evaluation as
  the upper paths. Black and blue section connectors make the mapping explicit.
- The lower geometry and re-score panels use short, non-overlapping legends;
  callouts are in white space and do not cover plotted curves.
- Axis labels and tick labels are visibly larger than v2, including the 3D
  q-axis/energy labels and all lower-panel quantitative axes.
- PDF, SVG, and PNG output exists; the SVG contains vector plot elements plus
  embedded raster atom thumbnails only.

## Scientific/provenance boundaries retained

- v3 is a reconstruction/operator schematic, not a quantitative PES.
- Atom positions are coordinate-derived structural anchors; they do not prove
  DFT self-consistent convergence.
- Both lower energy panels are synthetic; no real fixed-path profile is shown
  and no relaxed-path DFT re-score or barrier ranking is inferred.
- The v2 outputs remain untouched for comparison and rollback.
