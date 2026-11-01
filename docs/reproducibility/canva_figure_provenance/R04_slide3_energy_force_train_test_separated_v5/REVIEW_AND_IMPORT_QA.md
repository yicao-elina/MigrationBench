# Slide 3 separated energy/force train-test parity — R04 v5 QA

## Scope

This additive revision preserves v4 data, layout, typography, force limits,
alpha values, and shared metric-block axis labels.

## Display change

- The Energy and Force colorbar titles are moved farther away from their
  right-side logarithmic tick labels using increased label padding.
- Both colorbar title regions were visually checked for overlap at the final
  `22 pt` typography scale.

## Artifact checks

- Python source compilation passed.
- PNG preview was visually inspected: colorbar titles and tick labels are
  separated, with no overlap; the rest of the figure remains unchanged.
- PDF is a single-page vector export.
- SVG parsed successfully as XML and contains vector colorbar rectangles and
  hexbin polygons.
- Manifest records input hashes and output hashes.

## Canva handoff

The SVG is the editable vector source. The PDF is the vector review/export
copy and the PNG is the raster preview. Nothing was imported into Canva in
this revision.
