# Visual QA record — PES operator schematic v2

Date: 2026-10-18

## Passed checks

- The DFT and MLIP surfaces have distinct z-dependent grey and blue shading, with separate translucent wireframes and legend entries.
- The fixed operator shows one black dashed floor projection for the common DFT geometry path, while the black/blue curves differ only in surface height.
- The self-relaxed operator shows the black dashed DFT path, the red MLIP path, the black/red floor projections, lateral image-to-image shift markers, and the vertical central-image energy mismatch.
- Five coordinate-derived atom snapshots are present for each branch and labeled `A`, `1`, `TS`, `3`, `B`.
- The lower geometry and re-score curves are smooth rather than right-angle polylines; callouts no longer overlap the lower-panel legends.
- PDF, SVG, and PNG outputs exist; the PNG is 3092 × 1959 RGBA, and the atom render PNGs are RGBA with embedded base64 images in the SVG.
- `plot_pes_operator_schematic_v2.py` and `render_atom_frames_v2.py` pass Python bytecode compilation, and `git diff --check` reports no whitespace errors for the new source/notes files.

## Deliberate limitations retained

- The top surfaces are reduced-coordinate visual schematics, not quantitative 3N-dimensional DFT/MLIP PES reconstructions.
- The current Foundation MLIP self-relaxed path is a provisional geometry anchor; it is not presented as a completed DFT NEB.
- The lower-right re-score curves are an explicitly documented operator definition, not matched same-system DFT data.
- No historical 0.410 eV self-NEB record is used.
