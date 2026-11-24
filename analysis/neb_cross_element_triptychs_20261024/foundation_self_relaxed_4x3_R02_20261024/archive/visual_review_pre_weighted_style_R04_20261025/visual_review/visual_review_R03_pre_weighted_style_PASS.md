PASS

Independent fresh visual review, 2026-10-25, of the updated Foundation self-relaxed 4×3 R03 products listed in VISUAL_QA_R03.json (created 2026-10-25T19:47:56.624709+00:00). No figures, code, data, or QA files were modified. The user's current compact-legend requirement supersedes the earlier requirement to place the full diagnostic qualification inside the figure legend.

## Direct inspection and visual gates

Decoded all thirteen current PNGs, opened and re-rendered all thirteen PDFs at 144 dpi, and rendered all thirteen SVGs with CairoSVG at scale 2. Visually inspected the composite PDF/SVG renders and all twelve standalone panels in PNG/PDF/SVG comparison sheets. This verdict is based on rendered content as well as metadata. Temporary renders are outside the campaign.

- Every legend contains exactly four entries: **MACE-MP, MACE-OMAT-0, MACE-MH, DFT-NEB**. Family sample-count suffixes remain. Neither “converged / included” nor “failure / excluded” appears. Visual inspection and extracted PDF text agree across all thirteen products.
- All twelve dark dashed/open-diamond reference traces are visible in the composite and each corresponding standalone. Their styling is clearly distinct from the colored solid/open-circle family means and translucent SD bands. QA records ten reference points per panel, 120 total. Reference extrema and model uncertainty bands fit within the displayed vertical limits.
- Composite letters a–l and standalone local a are now outside the axes and clear of traces, spines, ticks, titles, and legends. In particular, the prior Cr gap→QL letter/reference collision is resolved. No label/legend collision, clipped text, or plot masking was observed.
- The 4×3 matrix remains Cr/Cu/Li/Na by in-gap diffusion/gap→QL/within-QL. Explicit gutters, complete independent y axes/ticks, shared composite x/y labels, consistent family colors, and the archived reference's profile-ensemble grammar remain acceptable. Standalones retain identifying titles, individual axis names/units, and corresponding quantitative curves/bands/reference shapes.
- No decorative grid or Blender/molecular layer is present.

## Typography at intended placement

Calculated placed font sizes from extracted PDF spans and actual tight-cropped PDF widths:

`placed_font_pt = source_font_pt × intended_width_mm × 72 / (25.4 × PDF_page_width_pt)`.

The composite's smallest text is **11.759867 pt at 180 mm width**. Standalone minima range **13.648416–14.729372 pt at 100 mm width**. All exceed the required 11 pt floor. Every PDF embeds ArialMT/Arial-BoldMT TrueType subsets. Arial-first typography with Helvetica/DejaVu Sans fallback remains intact.

## Scientific qualification retained

Read the complete current CAPTION_FOUNDATION_SELF_RELAXED_4x3_R03.md and inspected GROUND_TRUTH_REFERENCE_MANIFEST_R03.json. The caption explicitly expands DFT-NEB to **QE/DFT reference · SCF warning; diagnostic**, retains SCF-warning/failed-SCF-convergence status, and identifies canonical fixed-path single-point NEB diagnostics aligned by image index i/9. It explicitly excludes same-geometry DFT rescoring of relaxed MLFF images, validated DFT minimum-energy paths, and unconditional ground truth. Nonuniform Cr source coordinates are retained separately.

The manifest retains `QE_DFT_FIXED_PATH_SINGLE_POINT_REFERENCE`, `PROVISIONAL_DIAGNOSTIC_REFERENCE`, `same_geometry_dft_rescoring=false`, `unqualified_ground_truth_claim=false`, and raw `COMPLETED_BUT_SCF_WARNING` / `failed_scf_convergence` statuses. QA distinguishes the compact legend label from the full scientific-boundary label. Thus the compact visual label does not remove the qualification from the accompanying release materials.

## Export and QA integrity

Exactly **39 current R03 products** are present: composite plus twelve distinct element/path panels, each PNG/SVG/PDF (thirteen per format). Every recorded output SHA-256 and byte count matches. Reference manifest/profile hashes and the archived authoritative reference hash also match their QA entries. All PNGs decode, PDFs open, and SVGs render. All PDFs are single-page vector documents with zero embedded raster images. Cross-format labels, traces, bands, and geometry agree. QA counts, font settings, placement widths, reference pointer, and scientific-boundary metadata are coherent with inspected artifacts.

## Non-blocking notes and scope

Endpoint marker centers at x=0/1 remain on the axes boundaries, with conventional partial marker-edge clipping already accepted in prior reviews; no reference point is omitted or vertically truncated. Standalone tight crops vary slightly with tick-label widths; actual file widths were used for font calculations. Keep the recorded 180/100 mm placements or recalculate typography if reduced.

This PASS certifies the requested visual/export and accompanying qualification gates. Raw model identity, geometry, NEB convergence, reference alignment/provenance, and QE scientific validity remain within the separate independent data-review gate. Only this report was written inside the campaign.
