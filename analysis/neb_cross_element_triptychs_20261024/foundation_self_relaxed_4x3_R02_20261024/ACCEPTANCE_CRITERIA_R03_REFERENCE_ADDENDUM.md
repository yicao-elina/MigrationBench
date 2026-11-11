# R03 reference-layer acceptance addendum

This addendum applies to the user-requested revision that adds a NEB QE/DFT
reference trace to the Foundation self-relaxed 4×3 figure. The original R02
criteria remain in force; this file adds gates for the new reference layer.

## Data/provenance gates

1. The reference layer must contain exactly 4 elements × 3 canonical paths ×
   10 ordered image points = 120 finite points.
2. Every reference point must resolve to a preserved source table, raw-source
   path, source hash, source record, canonical path ID, and raw convergence
   status. The `2L-Cu__...` strings remain legacy path stems, not element
   identity.
3. The reference x coordinate must be aligned by image index (`i/9`) with the
   self-relaxed profile coordinate. Any source coordinate used to create that
   overlay must be retained in the manifest; no silent interpolation or
   smoothing is allowed.
4. The manifest must explicitly state that the available reference is a
   canonical fixed-path QE/DFT diagnostic and is **not** same-geometry DFT
   rescoring of the self-relaxed MLFF path. Raw `COMPLETED_BUT_SCF_WARNING` or
   `failed_scf_convergence` statuses must remain visible in the manifest.
5. The plotted legend label may use the compact label `DFT-NEB`; the full
   qualified label `QE/DFT reference · SCF warning; diagnostic` must remain in
   the caption, manifest, and review record. The figure must not make an
   unconditional ground-truth, converged DFT MEP, or physical barrier claim.

## Visual gates

1. The reference trace is present and visible in all 12 panels, using the
   Plot Atlas reference grammar: dark dashed line with open-diamond markers.
2. The legend distinguishes model-family mean/SD traces from the reference
   trace without collision, clipping, or ambiguous color semantics.
3. The prior R02 visual contract remains unchanged: Arial-first typography,
   explicit three-column gutters, independent complete y axes, shared x/y
   labels, the exact NEB x-axis label `Reaction Coordinate`, no decorative
   grid, no Blender, and readable composite plus all 12 standalone panels at
   intended placement.
4. The y limits include both the MLIP uncertainty bands and the reference
   points; no reference point may be clipped or silently omitted.

## Independent release gate

The data reviewer must independently PASS the reference manifest and its
alignment/boundary. The visual reviewer must independently PASS the composite
and all 12 standalone panels after the new trace is rendered. Any FAIL creates
a repair/re-review round. A new R03 release manifest is valid only when both
reports begin with `PASS` and all reference checks are true.
