Figure 1 production kit v2 — coordinate-based preview assembly

This audit package contains a reviewable Figure 1 preview assembled from the
supplied kit, real EXTXYZ coordinates, and the real 600 K grouped validation
configuration pool. It is PREVIEW_ONLY: do not release high-resolution Blender
outputs until the user explicitly approves the preview.

Main outputs:
  Figure1_Ideal_Visual.pdf / .svg / .png
  latex/Figure1_LaTeX_Assembly.pdf
  Figure1_Production_Guide.pdf
  panels/ and vector/ standalone layers

Coordinate-based Blender assets:
  assets/production/A01_material_hero.png
  assets/production/A02_local_environment.png
  assets/production/A03_in_gap.png
  assets/production/A04_gap_to_ql.png
  assets/production/A05_within_ql.png
  assets/production/A06_collective_sliding.png
  assets/production/A07_thermal_sampling.png

All seven assets use the mother-scene style from the molecular Blender skill:
scene-linear Te/Sb/Cr values, 64×32 smooth non-metallic spheres, the approved
low-elevation oblique orthographic camera direction, four broad softboxes,
Standard view and exposure 0.52. Output corners remain transparent; the black
background visible in the preview viewer is not baked into the PNGs.

Scientific boundaries:
  A01/A03/A04/A05 are coordinate renders from supplied one-Cr EXTXYZ paths.
  A02/A07 reuse real 600 K grouped validation frames as descriptive snapshots;
  they are not a new MD run or independent replica campaign.
  A06 is a RECONSTRUCTION: actual host coordinates plus a documented rigid
  +1.20 Å upper-QL display shift; it is not a sampled sliding trajectory.
  Panel b is a workflow schematic and does not audit checkpoints or predictions.
  Panel e is the supplied 30-point MIC geometry calculation; no energy, force,
  OOD or training-coverage claim is joined to it.

Rebuild the preview with:
  Blender 4.0.2 -b --python source/render_actual_assets.py -- --package-root <package> --aimd-source <valid.xyz>
  python source/compose_actual_assets.py
  python source/build_figure1.py
  python source/build_production_guide.py
  cd latex && xelatex -interaction=nonstopmode -halt-on-error Figure1_LaTeX_Assembly.tex

The source ledger, hashes, claim matrix, exclusions and gap report are under
provenance/. The original concept assets and attached production guide remain
as reference specification/history; current resolved asset roles are in
data/asset_resolution.json.
