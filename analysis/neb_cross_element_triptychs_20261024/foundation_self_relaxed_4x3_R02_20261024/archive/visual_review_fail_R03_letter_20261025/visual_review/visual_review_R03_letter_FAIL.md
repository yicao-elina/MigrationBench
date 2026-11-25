FAIL

Independent visual review of current R03 products, 2026-10-25. No figures, code, data, or QA files were modified.

## Numbered actionable feedback

1. Move the panel letter away from the QE/DFT trace in **Cr gap→QL**: composite panel **b** and `panel_Cr_gap_to_ql_self_relaxed_R03` (local letter **a**). In PNG, PDF, and SVG renders, the nearly horizontal dark dashed reference near zero runs through the lower part of the letter at the upper left. The open diamond immediately to its right compounds the crowding. Place the letter outside the axes above the upper-left corner, as in the archived reference, or choose a verified empty location within this panel. Check all panel letters after relocation and re-export/update hashes. Do not mask or omit reference points to hide the overlap. This is the sole blocking visual repair found.

## Evidence and passing checks

Freshly decoded all 13 R03 PNGs, opened all 13 R03 PDFs, and rendered all 13 R03 SVGs with CairoSVG. Re-rendered every PDF at 144 dpi and SVG at scale 2, and visually inspected the composite and every standalone in all formats using detail/comparison renders. Compared against the previously inspected archived authoritative reference and Plot Atlas profile-ensemble contract, including the frozen R03 addendum.

- Exactly 39 current R03 outputs: composite plus 12 element/path panels, each PNG/SVG/PDF. All 39 recorded byte counts and SHA-256 values in VISUAL_QA_R03.json independently match the files.
- Dark dashed open-diamond QE/DFT reference visibly exists in all twelve composite panels and all twelve standalones. All twelve QA panel states report ten reference points, 120 total. Reference extrema and model uncertainty bands fit within the displayed y ranges; no vertical truncation was observed. Conventional partial endpoint-marker clipping at x=0/1 persists and is not the blocking finding.
- The full label **“QE/DFT reference · SCF warning; diagnostic”** is present and readable in the composite and all standalone legends. No legend clipping, inter-entry collision, or title collision was found. Reference diamonds/dashes clearly differ from colored model open-circle/solid traces. “Converged / included” has a different circle symbol; the manifest explicitly retains warning/failure statuses and `same_geometry_dft_rescoring=false`.
- Composite actual PDF width is 780.982666 pt. At recorded 180 mm width, its smallest text is **11.759867 pt**. Every standalone has actual PDF width 372.438751 pt; at recorded 100 mm width its smallest text is **11.797110 pt**. Both exceed the required 11 pt floor, using actual tight-cropped widths. The longer standalone diagnostic label has therefore not reduced placed text below the threshold.
- Four rows remain Cr/Cu/Li/Na; columns remain in-gap diffusion/gap→QL/within-QL. Explicit gutters, twelve complete independent boxed axes/ticks, shared composite x/y labels, Arial-first typography, and consistent model colors/markers/bands remain acceptable. No grid or Blender layer is visible.
- All PDFs are readable single-page vector documents with no embedded raster images. Composite PNG is 4338 × 4630 pixels; standalone PNGs are 2069 × 1999. SVG/PDF/PNG geometry and trace appearances agree. Standalone titles, limits, curves, bands, and reference shapes match their composite counterparts.
- QA JSON coherently records the new reference label/style, 120 points, panel states, intended widths, font sizes, reference pointer, and output inventory. Its completeness flags do not detect the panel-letter overlap and do not constitute an independent visual PASS.

## Scientific boundary

This is visual/export QA, not independent validation of raw QE or NEB calculations. The reference is explicitly an SCF-warning fixed-path diagnostic, not same-geometry DFT rescoring of the relaxed MLIP path or a validated DFT MEP. Raw `COMPLETED_BUT_SCF_WARNING` and `failed_scf_convergence` statuses remain recorded. The separate data-review gate must establish reference provenance/alignment; this report does not promote diagnostic data to converged ground truth.
