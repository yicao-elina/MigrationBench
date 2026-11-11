# Formal visual review — R03 fixed-path trajectory figure

**Review date:** 2026-10-23  
**Review round:** R03  
**Reviewer mode:** manual visual inspection, with R03 JSON/manifest evidence used only as supporting provenance; automated QA is not treated as acceptance.  
**Candidate status:** **FAIL — not visually/publication accepted**  
**Scientific release status:** **BLOCKED**; the DFT SCF-warning gate, fixed-path seed gate, and curve-level provenance gate remain unresolved.

## 1. Scope and evidence inspected

The review was limited to the frozen criteria and the supplied R03 outputs:

- Frozen criteria: [`FROZEN_VISUAL_ACCEPTANCE_CRITERIA.md`](<../visual_review/FROZEN_VISUAL_ACCEPTANCE_CRITERIA.md>)
- Integrated raster: [`fixed_path_trajectory_composite_R03.png`](../integration/generated_R03/fixed_path_trajectory_composite_R03.png), 4200 × 3850 px, RGB, recorded at 300 dpi
- Integrated PDF: [`fixed_path_trajectory_composite_R03.pdf`](../integration/generated_R03/fixed_path_trajectory_composite_R03.pdf), one page, 1008 × 924 pt
- Integrated SVG: [`fixed_path_trajectory_composite_R03.svg`](../integration/generated_R03/fixed_path_trajectory_composite_R03.svg)
- Blender top: [`triptych_top_R03.png`](../blender/triptych_top_R03.png), 4200 × 1750 px
- R03 integration evidence: [`INTEGRATION_REPORT_R03.json`](../integration/INTEGRATION_REPORT_R03.json)
- R03 Blender evidence: [`scene_manifest_R03.json`](../blender/scene_manifest_R03.json) and [`RENDER_QA_R03.json`](../blender/RENDER_QA_R03.json)
- R03 data/QA evidence: [`figure_data_manifest_R03.json`](../figure_data_manifest_R03.json) and [`QA_REPORT_R03.json`](../QA_REPORT_R03.json)
- R01 comparison baseline: [`REVIEW_R01.md`](REVIEW_R01.md)

The integrated PDF/SVG were checked as deliverables, not merely by their existence. The SVG is an image-only wrapper containing one embedded PNG image; the PDF has no extractable embedded fonts. Thus the integrated export does not preserve the lower plot as an editable/vector figure.

## 2. VIS-01–VIS-16 formal results

| ID | Result | Manual evidence | R03 change and required action |
|---|---|---|---|
| **VIS-01** | **PASS** | The composite has three equal visual columns. The upper titles and lower plot boxes share the same left/center/right column positions. R03 integration evidence reports boundaries 0/1400/2800/4200 px with zero top–lower boundary error, and the raster inspection agrees. | **Fixed relative to R01.** Keep the zero-pixel shared-column contract in the release artifact. |
| **VIS-02** | **PASS** | Column order is consistently **within gap → gap2QL → withinQL**. Each upper subtitle and lower title refers to the same path class. | **Preserved from R01.** No reordering is needed. |
| **VIS-03** | **PASS** | In all three upper columns, the Cr path is rendered with numbered real-path ghost positions 0–9 and leader rails. At native size and a 50%-width readability stress view, all ten badges remain individually recoverable; gap2QL is tight but not ambiguous. The R03 manifest records ten images and ten markers per path. | **Major improvement over R01**, where image 1–8 visibility was only pending/partial in the integrated raster. Keep the labels at this contrast and spacing. |
| **VIS-04** | **PASS** | Every column includes `image 0 → 9`; the numbered endpoint/ghost sequence is visually ordered and does not depend on color alone. | **Improved over R01** by adding explicit numbered badges and leader rails. |
| **VIS-05** | **PASS** | The figure-level strip explicitly states: `ghosted spheres = real retained path-image overlay` and `α = image order only, not uncertainty/error`. The same alpha convention is used across the three columns. | **Fixed a R01 failure.** This is now clear enough that the ghost should not be read as a confidence band. Keep this wording in the final caption/package. |
| **VIS-06** | **PASS** | The host lattice, layer visibility, Cr highlighting, orthographic view, and apparent atom scale are comparable across columns; no endpoint or key host layer is visibly clipped. The R03 manifest records common orthographic/projection/unwrap policy. | **Improved over R01** through the common-scale render and independent path-focused cameras. Do not crop the rails or host layers in a later export. |
| **VIS-07** | **PASS** | The upper half contains atoms and path overlays only: no synthetic PES surface, artificial energy height, hand-drawn barrier, or schematic trajectory is present. | **Preserved.** The visual remains geometry-only and does not use the reference PES schematic as data. |
| **VIS-08** | **PASS** | The lower curves are polygonal/direct point-to-point profiles; no smoothing spline or extra visual extrema are apparent. R03 metadata states no interpolation. | **Preserved from R01.** Keep direct ten-image connections. |
| **VIS-09** | **PASS** | The four family colors are visibly distinct and match the frozen palette: Foundation red-orange, From-scratch orange, FT-600K dark blue, and FT-MultiT light blue. DFT is graphite rather than a family color. | **Preserved.** No color change is required. |
| **VIS-10** | **PASS** | DFT/QE is graphite/black and the visible status strip says `SCF warning; diagnostic`. The lower semantic key says `DFT warning`; it does not call the line ground truth. | **Improved over R01** by moving the warning into a clean figure-level status area. The label must remain in the final export. |
| **VIS-11** | **FAIL** | The curves use circle/square/triangle seed-like markers, but the visible semantic key only says `seeds: 123/234/345` beside one generic circular marker. It does not explicitly map **123 → circle, 234 → square, 345 → triangle**. `checkpoint variant` is separate, but the key does not fully disambiguate every marker used in the curves. | **R03 regresses in audit specificity relative to R01.** Add three explicit legend entries for training seeds 123, 234, and 345, and retain `checkpoint variant (not training seed)` as a separate entry. Do not use the visual key to imply that the withheld band is valid. |
| **VIS-12** | **PASS** | State is not encoded by color alone: the lower key separately shows DFT warning, band withheld, diagnostic status, failure `×`, and checkpoint-variant semantics using line/marker differences. | **Fixed the R01 ambiguity** around the withheld-band/status styles. Keep the status row separate from the four family-color row. |
| **VIS-13** | **FAIL** | The final composite shows a generic `× failure` key and the status strip says `× failure`, but it no longer identifies the affected arm/path or the actual `MODEL_LOAD` failure. No connected numeric line is drawn for the failed arm, which is correct, but a reader cannot determine from the figure alone that FT-MultiT seed 123 failed on all three paths. | **R03 removes the R01 in-axes overlap, but removes too much failure specificity.** Add one compact, non-overlapping status entry such as `× FT-MultiT seed 123 — MODEL_LOAD (all three paths)`. Do not add a fabricated curve or zero-valued replacement. |
| **VIS-14** | **PASS** | The figure-level status strip, four-family legend row, and semantic-status row sit above the lower axes and do not cover the curves. The upper atom images and labels are also unobstructed. | **Major improvement over R01**, where in-axes diagnostic prose overlapped data. Keep the current outside-axes placement, while adding the path-specific failure text compactly. |
| **VIS-15** | **PENDING** | `s (0–1)` plus `s = normalized reaction coordinate` makes the x meaning recoverable. The y label is only `ΔE (eV)`; the visible figure does not state `relative energy` or `ΔE_i = E_i − E_0`, although that definition is present in the R03 data contract. | **Partially fixed relative to R01.** Add a visible short definition, preferably `relative energy, ΔE_i = E_i − E_0 (eV)`, without putting a paragraph inside the axes. |
| **VIS-16** | **PASS** | The supplied PNG has no visible clipping of titles, labels, endpoint atoms, ghost rails, legend rows, or axes. A 50%-width raster stress view retained readable plot labels and individually recoverable 0–9 badges. | **Improved over R01** in spacing and typography. This pass is for the supplied raster’s visual legibility; it does not waive the separate final journal-width gate in PUB-05/PUB-08. |

### Visual gate result

**VISUAL GATE: FAIL.** VIS-11 and VIS-13 fail, and VIS-15 remains pending. Therefore R03 does not receive a visual pass even though the alignment, ghost semantics, and legend placement are substantially improved.

## 3. PHY/data and physics gate

These statuses are not a substitute for re-running or upgrading the science. `PASS` below means that the R03 evidence preserves the required data meaning for the plotted/withheld objects; it does not mean the entire study is scientifically released.

| ID | Result | Review conclusion |
|---|---|---|
| PHY-01 | **PASS** | R03 records the three path IDs, retained extxyz inputs, source hashes, PBC/cell information, endpoint hashes, and atom-order information. |
| PHY-02 | **PASS** | Each path is represented as ten retained images indexed 0–9; no visual resampling or silent deletion is evident. |
| PHY-03 | **PASS** | The R03 scene evidence records real retained positions and frame-to-frame MIC unwrap; no synthetic coordinates are used in the upper render. |
| PHY-04 | **PASS** | The data contract defines relative energy as `ΔE_i = E_i − E_0` in eV. The visible y-axis wording still needs the VIS-15 editorial completion. |
| PHY-05 | **PASS** | The observable is recorded as `FIXED_PATH_SINGLE_POINT`; no self-relaxed geometry is visually substituted into the upper path. |
| PHY-06 | **PASS** | Fixed-path, self-relaxed, and MIC-linear observables are kept separate in the R03 contract; the candidate does not draw a self-relaxed barrier as fixed-path data. |
| PHY-07 | **PASS** | The plotted numeric rows are finite; the three load-failure blocks are not drawn as numeric curves. |
| PHY-08 | **PASS** | No force error, SCF residual, or path-threshold residual is drawn as an energy uncertainty band. |
| PHY-09 | **PASS** | The R03 manifest records the four model families and the finite/failure block counts; the figure does not visually claim a clean complete benchmark. |
| PHY-10 | **PASS** | Foundation is represented as checkpoint/model variants, not as a training-seed SD ensemble. |
| PHY-11 | **PASS** | From-scratch branches remain discrete; no branch spread is presented as a valid training-seed uncertainty band. |
| **PHY-12** | **BLOCKED** | A legal mean ± 1 sample SD band requires three verified same-path, same-observable training seeds. The required fixed-path seed matrix is not verified. |
| PHY-13 | **PASS** | The R03 contract defines the intended SD as sample SD across verified training seeds, while the visible candidate correctly withholds the band. |
| PHY-14 | **PASS** | Missing/load-failure rows are not filled, interpolated, or silently converted into zeros. |
| PHY-15 | **PASS** | R03 evidence excludes the historical/61-atom and unrelated path packages from the rendered geometry. |
| **PHY-16** | **BLOCKED** | Remote fixed-profile reads remain unavailable at runtime; R03 uses the documented Tier-1 fallback and does not have raw per-profile hashes for the remote artifacts. This is not sufficient for curve-level final provenance. |

**PHY/DATA GATE: BLOCKED by PHY-12 and PHY-16.** The figure must not be described as a complete four-family fixed-path uncertainty release.

## 4. DFT gate

| ID | Result | Review conclusion |
|---|---|---|
| DFT-01 | **PASS** | The three reference paths are visibly labeled `DFT/QE: SCF warning; diagnostic`, and the R03 data evidence records `COMPLETED_BUT_SCF_WARNING`. |
| **DFT-02** | **BLOCKED** | The warning-bearing references cannot be called unconditional `DFT ground truth`. No image, scheduler completion, or successful render upgrades this gate. |
| DFT-03 | **PASS** | SCF warning is shown as a diagnostic status, not as an energy-error band. |
| **DFT-04** | **BLOCKED** | Clean image-wise convergence evidence, finite-value/force evidence, and the required updated registry release are not present. |

**DFT GATE: BLOCKED.** R03 does not and must not upgrade the black/graphite diagnostic curves to ground truth.

## 5. Seed gate

| ID | Result | Review conclusion |
|---|---|---|
| SEED-01 | **PASS** | R03 keeps training seed, checkpoint variant, NEB seed, and failure/status fields distinct in the data evidence. |
| SEED-02 | **PASS** | Foundation variants are not described as random training seeds; the figure has a separate checkpoint-variant semantic. |
| SEED-03 | **PASS** | From-scratch branches remain discrete and are not aggregated into a seed SD band. |
| **SEED-04** | **BLOCKED** | FT-600K seed 123 is base-unproven and FT-MultiT seed 123 is a `MODEL_LOAD` failure; there are no three verified training seeds for a legal band. The candidate correctly withholds the band. |
| **SEED-05** | **PENDING** | The R03 manifest records the withheld-band reasons, but the visible figure only says `bands withheld` and `seeds: 123/234/345`; it does not show the actual verified count and exclusions next to the semantic key. |

**SEED GATE: BLOCKED.** A withheld band is the correct status; it is not a passed uncertainty analysis.

## 6. Publication gate

| ID | Result | Review conclusion |
|---|---|---|
| PUB-01 | **PASS** | The R03 status/contract boundary keeps retained geometry, diagnostic DFT/QE, fixed-path model profiles, and withheld uncertainty conceptually separate. |
| **PUB-02** | **BLOCKED** | The fixed-path family/seed numeric matrix is not complete and fully verified; the R03 candidate must not claim a complete composite benchmark. |
| PUB-03 | **PASS** | The R03 JSON evidence records 33 finite diagnostic blocks, 3 `MODEL_LOAD` failure blocks, and 30 DFT warning rows. The figure-level failure key still needs the path-specific text required by VIS-13. |
| PUB-04 | **PASS** | No fixed/self barrier ranking or unsupported DFT ranking is asserted in the candidate. |
| **PUB-05** | **PENDING** | The supplied composite is 1008 pt wide (14 in). The candidate evidence does not establish the approved final Wiley body width. At a 7-in placement the current 11–12-pt-at-14-in layout would be reduced to roughly half-size, so final body typography cannot be accepted without a target-width render. |
| PUB-06 | **PASS** | The three upper/lower columns are equal and aligned; the legends sit outside the axes and do not occlude curves. |
| **PUB-07** | **FAIL** | PNG/PDF/SVG files exist and are hash-recorded, but the integrated SVG is a single embedded 4200×3850 PNG and the integrated PDF has no embedded fonts. The composite export is therefore raster-wrapped rather than a font-preserving/vector publication export. |
| **PUB-08** | **PENDING** | The supplied color raster has been visually checked for clipping and a 50%-width readability stress case, but a final approved body-width grayscale/color-vision check is not evidenced. |
| **PUB-09** | **BLOCKED** | The remote fixed-profile read limitation and unavailable raw per-profile hashes remain a reproducibility/provenance blocker. |
| **PUB-10** | **BLOCKED** | VIS-11/VIS-13 fail, VIS-15/PUB-05/PUB-08 are not complete, PUB-07 fails, and PHY/DFT/SEED gates remain blocked. |

**PUBLICATION GATE: BLOCKED.** The R03 image is a diagnostic candidate, not a final正文图.

## 7. R03 relative to R01

### Confirmed improvements

1. R03 adds individually numbered 0–9 badges and leader rails for every upper path; R01 left image-level recovery pending/ambiguous.
2. R03 explicitly says that ghost alpha is image-order/path-overlay semantics, not uncertainty/error; this fixes the R01 ghost ambiguity.
3. R03 moves SCF/failure/status prose out of the axes and separates the family legend from the semantic-status legend; this fixes the major R01 overlap problem.
4. R03 adds an explicit normalized reaction-coordinate note and produces an integrated PDF/SVG/P​NG set.
5. R03 enforces the top/lower 1400-px column boundaries with zero reported alignment error, and manual inspection finds the same alignment.

### Remaining or newly exposed failures

1. The shortened R03 seed legend no longer maps circle/square/triangle to 123/234/345 explicitly; VIS-11 is therefore not accepted.
2. R01’s path-specific `FT-MultiT 123: MODEL_LOAD` failure wording was moved out of the axes but not replaced by an equally specific non-overlapping status entry; the generic `× failure` is insufficient for VIS-13.
3. The x-axis meaning is improved, but the visible y-axis still does not state `relative energy` or `ΔE_i = E_i − E_0`; VIS-15 remains pending.
4. The integrated PDF/SVG addition is not a true vector/font-preserving release: the composite SVG is one embedded raster and the PDF has no embedded fonts.
5. R03 is visually cleaner, but the approved final publication width and final grayscale/color-vision proof are still absent; this keeps PUB-05/PUB-08 pending.

## 8. Required R04 changes before re-review

- Add explicit seed-marker entries: `training seed 123` (circle), `training seed 234` (square), and `training seed 345` (triangle); retain `checkpoint variant (not training seed)` separately.
- Add a compact path-specific failure entry outside the axes: `× FT-MultiT seed 123 — MODEL_LOAD (all three paths)`. Keep the failed arm unconnected and do not fabricate numeric values.
- Make the visible energy definition explicit: `relative energy, ΔE_i = E_i − E_0 (eV)`; preserve `s = normalized reaction coordinate`.
- Provide a true publication export: vector lower curves/text and font-preserving PDF/SVG, with the Blender top retained as an explicitly documented raster layer if necessary. Do not merely wrap the whole composite PNG in PDF/SVG.
- State the target journal body width and regenerate typography for that width; then recheck 100% raster, grayscale, and color-vision views with no clipping or status loss.
- Keep the following unchanged until new evidence exists: `DFT/QE — SCF warning; diagnostic`, withheld seed bands, the `MODEL_LOAD` failure ledger, and the remote-profile provenance limitation.

## Final decision

**R03 is not accepted.** The candidate is a meaningful visual improvement over R01, but it does not receive a visual pass or publication sign-off. In particular, DFT SCF warnings, missing/failed seed evidence, and unavailable raw profile provenance remain hard blockers and are not upgraded by this review.
