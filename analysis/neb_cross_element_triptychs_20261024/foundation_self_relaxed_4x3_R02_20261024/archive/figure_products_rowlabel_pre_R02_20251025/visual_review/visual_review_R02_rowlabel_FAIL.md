FAIL

Independent fresh visual re-review, 2026-10-25, of the current regenerated Foundation self-relaxed 4×3 R02 products. The previous FAIL verdict was not used as evidence for this decision. Typography and authoritative-reference repairs now pass. One current composite spacing defect remains; no figure or data files were changed.

## Required actionable repair

1. **Separate composite row labels from first-column y tick labels.** In the current composite PNG, PDF, and SVG, Cu, Li, and Na are too close to adjacent negative tick labels. In particular, Cu's upper/lower edges nearly touch the minus signs of −0.3/−0.6 in panel d; Li nearly touches −0.3 in g; Na nearly touches −0.4 in j. A 288 dpi PDF detail crop confirms the near-contact visually. Extracted PDF text bounds corroborate the collision of layout boxes: row labels occupy x=38.064–60.424 pt, while these tick-label boxes start at x=57.580 pt, a 2.844 pt horizontal overlap. Cu intersects both tick boxes vertically; Li and Na each intersect the indicated tick box. This describes overlapping text layout bounds and inadequate visible clearance, not a claim that all glyph ink overlaps. At 180 mm placement the crowding persists and makes element labels look attached to numeric signs.

   Move the row annotations farther left, and/or reserve a dedicated row-label column between the shared y label and tick labels. A practical starting change is to shift the annotation from axes x=−0.29 to approximately −0.34 (about 8.6 source pt left with the present axes width), then verify both its clearance from ticks and from the shared y label. Aim for at least 3 pt clear space at intended placement, while retaining the ≥11 pt text floor. Re-export the composite in all three formats, update output hashes in VISUAL_QA_R02.json, and visually re-check the Cu/Li/Na rows. This is a layout repair; retain the current scientific curves and independent y axes.

No further blocking defect was identified. The Cu gap→QL standalone has a slightly wider tight bounding box than the other standalones; this is documented below as non-blocking, not a required repair.

## Fresh inspection method

Read the current renderer and current VISUAL_QA_R02.json against the frozen ACCEPTANCE_CRITERIA.md and the already-read Plot Atlas independent-panel/profile contracts. Parsed every current SVG, opened every current PDF, and decoded every current PNG. Rendered all 13 PDFs with PyMuPDF at 144 dpi, all 13 SVGs with CairoSVG at scale 2, and inspected the composite and all twelve standalone panels in each format using full-detail renders/comparison sheets. Inspected the native 400 dpi composite PNG and a 288 dpi local PDF crop of the row-label defect. Re-rendered and inspected the explicitly authoritative archived reference. Temporary render files are outside the campaign; this report is the only campaign file written by this reviewer.

## Acceptance results

### Typography at recorded physical placement: PASS

Computed physical font sizes from extracted PDF spans and each PDF's actual cropped page width, not from the nominal figure canvas:

`placed_font_pt = PDF_span_font_pt × (intended_width_mm × 72 / 25.4) / PDF_page_width_pt`.

| Product | Measured page width | Intended width | Smallest extracted source font | Smallest placed font |
|---|---:|---:|---:|---:|
| Composite | 780.982666 pt | 180 mm | 18 pt | 11.7599 pt |
| Eleven standalones | 312.467072 pt | 100 mm | 17 pt | 15.4221 pt |
| Cu gap→QL standalone | 321.920197 pt | 100 mm | 17 pt | 14.9692 pt |

All text spans pass the explicitly requested ≥11 pt minimum. Composite sizes are 18/19/20 pt; standalone sizes are 17/18/20.5 pt. The composite PDF height is 833.428772 pt; standalone height is 312.008881 pt. ArialMT and Arial-BoldMT are used in every PDF with embedded TrueType subsets. Renderer/SVG font priority remains Arial, Helvetica, DejaVu Sans. Legend and tick source sizes agree with the new QA JSON. The 180/100 mm placements are now explicit and testable. The former typography objection is closed.

### Authoritative reference: PASS

The user has explicitly designated the archived reference for this re-review, and the current QA JSON records the resolution rather than hiding the missing historical path. Resolved file:

`/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/archive/legacy_figure_products_20261024/figures/foundation_class_3x3_R03/foundation_model_class_3x3_R03.pdf`

Independently recomputed SHA-256 matches its recorded value:

`59d6202451f9f94ff21f3fba9924e025001c0af6c406c2bdf2ca1fc947023f84`.

The reference's family palette, boxed independent axes, open markers, translucent bands, outward ticks, no-grid appearance, and frameless legend grammar are retained. Its three-row fixed-path/QE content is a style reference only; the requested four-row self-relaxed content and status legend correctly differ. The former reference-pointer objection is closed.

### Matrix, axes, marks, and standalone correspondence: PASS except required row-label spacing

- Exactly four rows ordered Cr, Cu, Li, Na; three columns ordered in-gap diffusion, gap→QL, within-QL. Composite panel letters run a–l in row-major order.
- Explicit inter-column/inter-row gutters, twelve independent complete boxed y axes with numeric ticks, and shared x/y names outside the composite grid. Every panel displays its own normalized-coordinate x ticks from 0 to 1.
- No decorative grid; no Blender or molecular rendering layer. Family colors remain MACE-MP gold #d5a238, MACE-OMAT-0 red #d85a45, MACE-MH green #5d8b56.
- Solid mean curves, open-circle markers, and translucent bands remain visible and distinguishable, including broad within-QL bands. First-image energies visibly start at zero. No empty panel, interpolation used to conceal missing status, or extraneous explanatory strip is visible.
- The composite's two-row legend is readable, frameless, and collision-free. It reports 5/5, 2/2, 2/2 and “converged / included.” Each standalone has a readable single-column legend, a unique element/path title, axes-associated x/y labels outside the plotting box, and complete tick labels. No standalone legend/title/axis-label collision was found.
- All twelve standalone profiles preserve their matching composite curve/band shapes, family mappings, x ticks, and y limits. Local standalone letter “a” is an acceptable independent-panel convention. Both exports call the same draw_panel recipe. Layout and mark sizes differ intentionally for the respective placements.

### Export inventory, integrity, and QA coherence: PASS

- Exactly 39 current products: 12 element/path combinations × 3 formats plus 1 composite × 3 formats; 13 PNG, 13 SVG, 13 PDF. The 39 JSON output paths are unique, and every recorded byte count and SHA-256 matches the file on disk.
- Composite PNG: 4338 × 4630 pixels at approximately 400 dpi. Eleven standalone PNGs: 1736 × 1734; Cu gap→QL: 1788 × 1734, also approximately 400 dpi. Resolution is sufficient at the recorded placement widths.
- Every PDF is a readable single-page vector document with no embedded raster images. Every SVG is well-formed and renders successfully; vector text/marks/transparency agree with PNG/PDF. SVG text remains font-dependent when opened elsewhere; PDFs embed their fonts.
- Twelve axes, twelve standalone panels, formats, font sizes, intended widths, reference path/hash, and no-grid/no-Blender/shared-label style declarations agree with inspected artifacts and source. QA JSON is a coherent export/style manifest; it does not detect the row-label clearance defect and is not itself an independent visual PASS.

## Non-blocking notes and scope

The claim that every standalone page is 312.467 pt wide has one exception: Cu gap→QL is 321.920 pt because its wider negative decimal tick labels expand the tight crop. Its actual minimum placed font remains 14.969 pt, comfortably above threshold. Use each file's actual page width for placement calculations. A common crop could improve alignment in a future montage but is unnecessary for this gate.

The single-column standalone legend occupies substantial vertical space, but preserves readable separation and does not obscure data. Endpoint circles are partially clipped at x=0/1 in the conventional axis-edge treatment, consistently across formats; this remains non-blocking.

This is visual/export QA. The displayed “converged / included” status is not an independent verification of raw model identity, NEB forces, geometry, or scientific publication readiness; those remain within the independent data-review gate. The current FAIL is solely the newly measured composite row-label/tick clearance defect, not either resolved prior objection.
