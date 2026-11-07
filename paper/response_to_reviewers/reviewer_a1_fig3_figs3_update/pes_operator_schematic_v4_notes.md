# PES operator schematic v4 notes

## Scope

v4 is the clean main-figure release requested after the v2/v3 visual review.
The main figure contains the two synthetic reduced-coordinate PES operators,
their smooth paths and q1-q2 floor projections, enlarged axes, and a concise
DFT/MLIP surface legend.  Atom thumbnails are deliberately not embedded.

The corresponding atom visuals are released independently as:

- `fig3d_figS3_atom_path_dft_fivepanel_v4.*`
- `fig3d_figS3_atom_path_mlip_fivepanel_v4.*`

This preserves a blank upper band in the main figure for final author-side
placement if desired.

## Visual policy implemented

- Removed atom thumbnails, directional arrows, arrowheads, vertical energy
  mismatch arrows, geometry-shift connectors, `delta E_fixed`, step 1/2 text,
  model-mismatch callouts, the bottom decomposition equation, and the
  `schematic` labels.
- Retained only panel titles, axis labels/ticks, plotted curves, and concise
  legends needed to distinguish DFT and MLIP.
- Replaced the previous high-saturation red emphasis with muted Plot Atlas
  colours.
- Kept the lower panels synthetic and consistent with the upper synthetic PES
  surfaces; no unavailable self-consistent DFT re-score is represented as real
  data.

## Palette lock

The analytical figure uses the registered Plot Atlas `nature_steered_dynamics`
palette: DFT `#9B9B9B`, MLIP `#337FB0`, with white-blended surface tints
`#EDEDED` and `#DAE8F1`.  The standalone atom figures use the registered
`tissue_pastel` palette with a fixed element mapping: Te `#9A95C7`, Sb
`#9CD5D4`, and Cr `#FFDE95`.  These mappings are recorded in the v4 manifests
and are not branch-specific.

## Provenance boundary

The QL atom frames are coordinate-derived structural visual anchors from the
61-atom DFT fixture and current Foundation MLIP path.  They are not evidence
of a converged self-consistent DFT NEB, and no historical `0.410 eV` record is
used.  The main and atom manifests retain the input hashes and output hashes.
