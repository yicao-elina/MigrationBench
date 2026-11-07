# v4 visual review

Date: 2026-10-19

## Checks performed

- Rendered the main figure to PNG, PDF, and SVG.
- Rendered the DFT and MLIP atom paths with POV-Ray as five transparent
  coordinate-derived frames each, then composed each branch as an independent
  five-panel PDF/SVG/PNG figure.
- Rasterized all three PDFs with Poppler at 180 dpi and inspected the pages.
- Confirmed the main SVG contains no embedded raster `<image>` elements.
- Confirmed the main SVG contains no residual `arrow`, `geometry shift`,
  `model mismatch`, `delta`, `step`, or old red-colour strings.
- Confirmed the main figure has no atom PNG dependency and the two atom-panel
  manifests set `labels`, `arrows`, and `arrowheads` to `false`.
- Ran Python syntax checks and `git diff --check`.

## Acceptance result

PASS for the requested clean split: the main PES operator schematic is a
label-light vector figure with open space for later atom placement, while DFT
and MLIP atom trajectories are delivered as separate five-panel figures.  v3
remains unchanged as the rollback/reference release.
