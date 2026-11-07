# PES-operator schematic v3

## Review-comment resolution

### Comment 1 — legend and collision control

The DFT/MLIP surface legend is now a single figure-level legend in a dedicated
white band between the two panel titles. It is no longer anchored next to a
3D z-axis or on top of a surface. The lower-panel legends use short labels and
separate upper-left/upper-right anchors; callouts are placed in the remaining
white regions. The final PNG was inspected at full resolution after rendering.

### Comment 2 — packed QL view and trajectory

The v3 atom renderer uses a direct QL cross-section: camera along `+y` with
`z` vertical. The local slab is cropped tightly enough to show stacked Te/Sb/Te
layers rather than a loose 3D cloud. The five Cr positions are added as
semi-transparent red trajectory spheres connected by POV-Ray cylinders, while
the current image is a gold solid sphere. Both DFT and MLIP branches use this
same visual grammar. The underlying coordinates remain the audited five-image
61-atom paths; the render is a structural visual anchor only.

### Comment 3 — unified synthetic energy panels

The lower-left fixed-path energy panel no longer uses the real DFT/FT-600K
profile. It evaluates the same synthetic DFT/MLIP surfaces used in the upper
3D panels along the fixed geometry path. Two restrained black/blue connectors
show that the lower curves are sections of the upper synthetic paths. The
lower-right relaxed-path re-score remains synthetic as well, so both energy
panels now have the same schematic status.

### Comment 4 — axis typography

Axis labels, tick labels, titles, legends, and path markers were enlarged and
the canvas was expanded to `9.35 × 6.35` inches. The v3 output was checked as
PDF, SVG, and PNG, including the intended final-size raster preview.

## Evidence boundary

- `surface_status`: synthetic reduced-coordinate PES; z-dependent color is a
  visual distinction and is not a quantitative PES reconstruction.
- `geometry_status`: coordinate-derived DFT fixture and current Foundation
  MLIP self-relaxed geometry provide the QL and path visual anchors.
- `fixed_energy_profile_status`: synthetic section only; the real fixed-path
  profile is intentionally excluded from v3 for visual consistency.
- `self_rescore_status`: synthetic operator definition only; no matched
  same-system DFT re-score or barrier claim is implied.
- No historical 0.410 eV record is used.

## Rebuild

```bash
python render_atom_frames_v3.py --output-dir atom_renders_v3
/Users/alina/Documents/Codex/plot_library/.venv/bin/python \
  plot_pes_operator_schematic_v3.py \
  --output fig3d_figS3_pes_operator_schematic_v3
```

The atom renderer uses the installed ASE POV-Ray writer and local POV-Ray
executable. OVITO and Blender remain optional alternatives, not dependencies
of this reproducible version.
