# Cr-centered local-environment Blender style contract

- Source: `analysis/neb_blender_visual_refinement_20261025/mlff_neb_images.extxyz`
- Selection: frame 6; Cr is selected by the explicit `Cr` species rule.
- Local environment: the Cr atom plus every atom within 5.5 Å in the source
  coordinate frame; no atom is independently moved.
- Bonds: no topology is present in the EXTXYZ source. Any rods are therefore a
  geometry-based visual heuristic using the documented covalent-radius cutoff;
  the render is not a topology proof.
- Projection: orthographic, zero roll, no depth of field, white field, safe
  border, no ground plane or horizon.
- Material: nonmetallic Principled BSDF, roughness 0.48 for atoms and 0.52 for
  rods, specular IOR level 0.28, coat 0.0; broad warm-neutral key and low
  cool-neutral fill for a high-end matte/waxy appearance.
- Palette: Cr warm copper-gold focal atom; Sb restrained teal; Te muted plum;
  bonds are endpoint-colour-mixed and lower saturation.
- Deliverables: 1600 px preview, 3600 px square final, 16-bit RGB white PNG and
  16-bit RGBA transparent PNG, plus `.blend` and selection manifest.

