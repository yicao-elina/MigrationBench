# v5 visual review

Date: 2026-10-19

## Checks performed

- Rendered v5 to PDF, SVG, and PNG after adding five image-anchor markers.
- Visually inspected the raster output at full resolution.
- Confirmed the fixed profile has five vertically aligned DFT/MLIP pairs and
  the labels `A`, `1`, `TS`, `3`, `B`.
- Confirmed the self-relaxed q1-q2 panel has five DFT markers, five MLIP
  markers, matching labels on the MLIP path, and non-directional paired
  segments.
- Confirmed the lower-right re-score panel has five markers on each curve.
- Confirmed no arrows, arrowheads, atom thumbnails, high-saturation red, or
  old callout text were reintroduced.
- Ran syntax checks, PDF/SVG generation, and `git diff --check`.

## Acceptance result

PASS for the requested point-to-atom correspondence.  v4 remains available as
the clean no-anchor rollback; v5 is the recommended reviewer-response figure.
