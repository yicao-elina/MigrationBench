# PES operator schematic: interpretation and caption

## What the figure must communicate

The two panels are not two versions of the same scalar error. They are two
different operators applied to an MLIP:

1. **Fixed-path scoring.** The DFT image sequence is held fixed. The MLIP is
   evaluated at the same geometries, so the geometry coordinates are shared and
   the vertical separation between the surfaces is the pointwise energy error,
   \(\delta E_i=E_{\mathrm{MLIP}}(R_i^{\mathrm{DFT}})-E_{\mathrm{DFT}}(R_i^{\mathrm{DFT}})\).
2. **Self-relaxed NEB.** Only the endpoints are fixed. The MLIP generates its
   own image sequence, \(\Gamma_{\mathrm{MLIP}}=\mathrm{NEB}_{\mathrm{MLIP}}(R_A,R_B)\),
   which can differ from the DFT path in reduced configuration space. The
   evaluation therefore has two stages: path/geometry relaxation, followed by
   DFT re-scoring of the new MLIP geometries.

For a matched DFT re-score, the path-dependent discrepancy can be written as

\[
\Delta E_i =
\underbrace{E_{\mathrm{DFT}}(R_i^{\mathrm{MLIP}})-E_{\mathrm{DFT}}(R_i^{\mathrm{DFT}})}_{\text{geometry/path contribution}}
+
\underbrace{E_{\mathrm{MLIP}}(R_i^{\mathrm{MLIP}})-E_{\mathrm{DFT}}(R_i^{\mathrm{MLIP}})}_{\text{energy-model contribution}}.
\]

The two terms should not be presented as a literal vector addition in the
physical 3D surface. The first is a displacement in configuration space that
changes the DFT energy; the second is the vertical energy mismatch evaluated at
the new geometry. The schematic uses a lateral arrow and a vertical arrow to
make this distinction explicit.

## Evidence boundary for the current release

- The lower-left profile is drawn from the current processed 1–4 DFT candidate
  profile and the FT-600K predictions on the same image sequence.
- The atomic thumbnails are rendered from the 61-atom 1–4 path files and are
  visual anchors for the migration event, not an additional barrier estimate.
- The PES surfaces and the self-relaxed energy comparison are reduced-coordinate
  schematics. The right-hand atomic strip uses the finite FT-600K pre-NEB path
  for visual context; it is not labeled as a converged self-consistent DFT
  result.
- The figure must not be used to reinstate or rank the historical 0.410 eV
  record. That record remains excluded because its system and calculator
  lineage do not match the current fixed-path operator.

## Suggested caption

**Figure A1 | Fixed-path and self-relaxed NEB are distinct projections of an
MLIP–DFT energy landscape.** Schematic reduced-coordinate energy landscapes
showing a DFT surface (dark grey) and an MLIP surface (blue) together with
atomistic snapshots of Cr migration in layered Sb\(_2\)Te\(_3\). **a,** In
fixed-path scoring, the DFT-defined image sequence is held fixed and the MLIP
is evaluated at the same geometries; the vertical separation therefore measures
\(E_{\mathrm{MLIP}}(R_i^{\mathrm{DFT}})-E_{\mathrm{DFT}}(R_i^{\mathrm{DFT}})\).
The lower profile shows the current 1–4 path data on this common image
sequence. **b,** In self-relaxed NEB, only the endpoints are supplied and the
MLIP generates a different path. The operator consequently separates into
(1) a geometry/path shift in reduced configuration space and (2) energy
re-scoring of the relaxed MLIP geometries. The surfaces and the self-relaxed
comparison are schematic definitions of the two-stage operator; they do not
assert a completed same-system DFT re-score or a universal model ranking.

## Production recommendation

Keep the current vector surface/path/text layer. For the final manuscript
version, replace only the atom thumbnails with transparent 3D renders from
Blender, POV-Ray, or OVITO using a single orthographic camera and the same
element colors. Do not render the labels or equations in Blender: keep those
as vector text in the Python/SVG layer. This preserves editability and avoids
the small grey subtitle problem in the earlier audit figure.
