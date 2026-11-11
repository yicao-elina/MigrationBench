# MigrationBench schematics: blueprint

Six main figures, one story: **method → static accuracy → MD → local migration →
interlayer sliding → distribution & attribution**. 600 K / Multi-T runs through
every test; fixed-path vs self-relaxed is separated *inside* the path figures;
SOAP-PCA and SHAP explain the differences at the end.

This blueprint covers the **schematic layer** of each figure (structures, motion
cues, flow, operators, icons) and reserves **data slots** where the quantitative
plots built from source data are dropped in. Schematics never carry numbers.

## 1. Design principles (from the Nature references)

| Principle | Reference cue | Rule here |
|---|---|---|
| One metaphor per figure | active-learning loop; GAME-Net pipeline | Each figure opens with a panel **a** that tells the figure's question visually; data panels follow in reading order |
| Soft matte 3D objects on flat UI | matte isometric cubes; soft-shaded adsorbate spheres | Atoms are matte spheres (soft key light upper-left, no specular hotspot, darker rim, soft contact shadow) on flat light-grey containers |
| Thin flow arrows | 0.5–0.75 pt curved black arrows | Process flow = 0.6 pt `#3a3f47` arrows; dashed = optional / inferred link |
| Colour only for meaning | slate cubes + one accent colour | Neutral greys for scaffolding; colour encodes species, initialization, thermal coverage, task |
| Quiet typography | sans-serif, sentence case, no boxes around labels | Arial; 8 pt bold panel letters, 7 pt titles, 6 pt labels, 5 pt notes |
| Whitespace and grid | generous gutters | 183 mm double-column width, 3 mm gutters, 2 mm container radius |

## 2. Visual language

**Atoms** (matching `Figure1_Ideal_Visual.pdf`): Te `#c27aa6` (mauve-pink), Sb `#5f9ea9`
(teal-grey), Cr `#eba92c` (gold, the hero; always drawn last and slightly larger).
> Note: the later six-figure drafts showed Te teal / Sb pink; the earlier Figure 1 convention
> (Te pink / Sb teal) was chosen. Swapping is one line in `figkit/palette.py`.

**Initialization** (fill colour of model glyphs and markers)
Foundation / zero-shot `#5fa86b` · Fine-tuning `#2a9d9a` · Scratch / specialist `#2f5f98`

**Thermal coverage** (marker fill): open = 600 K (single T), filled = Multi-T.
**Seeds** (marker shape): circle s123, triangle s234, square s345. Zero-shot: diamond.

**Tasks** (pink → purple ramp, pointwise → collective)
Static `#f5d3de` · MD `#efbdd0` · In-gap `#e9a3c0` · Gap-to-QL `#df86ad` · Within-QL `#c97bb4` · Sliding `#9b72b8`
**Operators**: fixed path `#f2b3c8` (light bar), self-relaxed `#e46aa2` (strong bar).
**Run status**: complete `#6aa88a` · failed `#d0606a` · pending/audit `#b8bcc4`.

**Motion cues** (how a still image shows dynamics)
- Stroboscope: successive Cr positions, light → dark gold, last one solid.
- Transition state: dashed ring + `‡`.
- Streaks: thin trails for collective (layer) motion; block arrows for shear.
- Displacement arrows for small host relaxations, with the magnification printed.
- Thermal "breathing": ghosted offsets around mean positions (labelled schematic).

## 3. Component library (`figkit/`)

| Component | Source | Used in |
|---|---|---|
| Cr81 bilayer slab (perspective) | real NEB image 0 | 1a, 2a, 3a, 6a |
| In-gap / Gap-to-QL / Within-QL vignettes | real NEB images 0–9 | 1c, 4a |
| Pristine / Cr-in-gap sliding slabs | real DFT sliding path | 1c, 5a |
| Cr local environment (ball-and-stick) | real coordinates ≤ 3.4 Å | 1d, 3a, 6a |
| Thermal snapshot (MD) | schematic offsets on real lattice | 1c, 3a |
| Model glyphs (Foundation / FT / Scratch) | vector icons | 1b, 2a, 3a |
| Thermal glyphs (600 K / Multi-T) | vector icons | 1b, all keys |
| Operator glyphs (fixed / self-relaxed) | vector icons | 1d, 4a, 5a |
| Diagnostics glyphs (SOAP, PCA, distance, SHAP) | vector icons | 1e, 6a |
| Data slot (axes frame + what goes there) | vector | 2–6 |

## 4. Figure layouts (183 mm wide)

**Fig. 1 | Method: what is compared, and how** (a–e, schematic only)
a hero Cr-doped Sb₂Te₃ bilayer with Cr path arc → b 3 × 2 design grid (initialization × thermal coverage)
with model and thermometer glyphs → c task atlas: five 3D vignettes on the pink→purple ramp
("pointwise → collective") → d two measurement operators E_θ(q_s^ref) vs E_θ(q_s^θ) →
e diagnostics entry (SOAP-PCA + SHAP) pointing to Fig. 6.

**Fig. 2 | Static accuracy: how accurate are E and F on the same configurations?**
a schematic: one shared configuration set → E (scalar) and F (per-atom arrows) → three initializations
× two thermal coverages. Slots: b RMSE statistics, c energy parity, d force parity (full train/test in SI).

**Fig. 3 | MD: how long does each model run stably?**
a schematic: timeline strip of a stable run (ordered snapshots) vs a failing run (disordering, ✕ at
failure step); local-coordination glyph (Cr + neighbours, r_ij). Slots: b run length, c T, d P, e MSD,
f local coordination (RDF / VACF in SI).

**Fig. 4 | Local migration: how do three environments change the path response?**
a three columns of real-path vignettes (In-gap, Gap-to-QL, Within-QL) with stroboscopic Cr;
under each, operator strip fixed vs self-relaxed. Slots: 3 × 2 reaction-coordinate panels + replicate statistics.

**Fig. 5 | Interlayer sliding: how do collective displacement and doping change performance?**
a pristine vs Cr-in-gap slabs with shear arrows and streaks (real sliding path) × operator strip.
Slots: b–c peak energies (pristine, Cr-in-gap), d doped − pristine response.

**Fig. 6 | Distribution & attribution: which coverage differences and features accompany error?**
a pipeline: configurations → SOAP local environments → shared PCA space (training cloud vs task points)
→ distance to training set → SHAP attribution. Slots: b task dot-matrix, c SOAP-PCA, d training distance, e SHAP.

## 5. Production

- Pure SVG built in Python (`figkit`), exported to PDF (cairo, TrueType Arial embedded) and 600 dpi PNG.
- Every structure is projected from real coordinates except the MD thermal snapshot and the
  failing-run snapshot, which are labelled *schematic*.
- Components are also exported one-per-file for Illustrator reuse.
- Integration into the manuscript goes through `figurectl` (paper repo rule), one FIG at a time.

## 6. Build

```bash
cd figures/nature_schematics
../../.venv-fig/bin/python build_all.py      # → out/fig1_method … fig6_attribution (.svg/.pdf/.png)
```

`figkit/` holds the reusable pieces (palette, SVG/text layout, matte atom renderer,
structures from real coordinates, icons, layout). `figs/figN_*.py` assembles each figure.
