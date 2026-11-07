# PES-operator schematic v2: evidence, rendering, and rebuild notes

## Frozen visual acceptance criteria

1. The two upper panels must read as two operators, not as two scalar error plots.
2. DFT and MLIP must be visually separable as translucent, z-dependent shaded surfaces.
3. The fixed-path panel must show one common DFT geometry path, its floor projection, and the vertical same-geometry energy mismatch.
4. The self-relaxed panel must show the black DFT path, the red MLIP-relaxed path, both floor projections, the lateral geometry shift, and the second-stage energy mismatch on the relaxed path.
5. Five coordinate-derived atom snapshots must anchor the endpoints, three interior images, and the TS-like central image. The snapshots are structural anchors, not claims of DFT self-consistent convergence.
6. The lower geometry panel must use the actual reduced coordinates from the five-image DFT fixture and current Foundation MLIP self-relaxed path, with smooth curves rather than right-angle polylines.
7. The lower re-score panel must be explicitly treated as an operator definition because a matched same-system DFT re-score on the self-relaxed path is not available. It must not be used to claim a barrier, ranking, or completed self-consistent DFT result.

## Evidence boundary

- `fixed-path energy profile`: derived from the current processed 1–4 DFT candidate profile and FT-600K fixed-path predictions.
- `fixed-path geometry`: five-image 61-atom DFT fixture.
- `self-relaxed geometry`: five-image 61-atom Foundation MLIP path from the current self-NEB campaign. It is a provisional MLIP optimization/preconditioner path for this schematic, not a publication-grade DFT NEB result.
- `self-relaxed re-score`: synthetic smooth curves used only to make the second operator leg legible. No historical 0.410 eV record is used.
- The two-dimensional `(q1, q2)` coordinates are a transparent reduced-coordinate projection defined from the Cr endpoint displacement and the orthogonal central-image DFT residual. They are a visualization coordinate system, not a claim that the 3N-dimensional PES has only two physical degrees of freedom.

## Rendering chain

- The editable surfaces, paths, equations, labels, and lower panels are generated with Python/Matplotlib under the Plot Atlas publication style and exported to PDF, SVG, and PNG.
- The five atom snapshots are rendered from the actual XYZ/extXYZ coordinates with ASE 3.28 and POV-Ray. The output PNGs have transparent alpha and are embedded into the SVG/PNG figure as visual anchors.
- The local installation contains `povray`, `ffmpeg`, ImageMagick, ASE, Matplotlib, and Plotly. Blender, OVITO/OVITOS, VMD, and Tachyon are not currently installed; they are therefore not required by this reproducible version.
- ASE documents writing POV-Ray scene files and using POV-Ray for atomistic visualization: <https://wiki.fysik.dtu.dk/ase/ase/io/io.html?highlight=read> and <https://wiki.fysik.dtu.dk/ase/faq.html>. OVITO remains a reasonable upgrade path for richer trajectory rendering and alpha-enabled viewport output: <https://www.ovito.org/manual/usage/rendering.html> and <https://ovito.org/manual/python/introduction/rendering.html>. Blender's Atomic Blender add-on is another option for larger, more cinematic scenes: <https://docs.blender.org/manual/id/4.1/addons/import_export/mesh_atomic.html>.

## Rebuild commands

```bash
python render_atom_frames_v2.py --output-dir atom_renders_v2
MIGRATIONBENCH_PLOT_ATLAS=/path/to/plot_library \
PYTHONPATH=/path/to/plot_library/src python \
  plot_pes_operator_schematic_v2.py \
  --output fig3d_figS3_pes_operator_schematic_v2
```

Expected outputs:

- `fig3d_figS3_pes_operator_schematic_v2.pdf`
- `fig3d_figS3_pes_operator_schematic_v2.svg`
- `fig3d_figS3_pes_operator_schematic_v2.png`
- `fig3d_figS3_pes_operator_schematic_v2.manifest.json`
- `pes_operator_v2_path_coordinates.csv`
- `atom_renders_v2/atom_render_manifest.json`

The POV-Ray run currently emits a non-fatal user-level configuration warning about `~/.povray/3.7/povray.conf`; all ten atom frames still render successfully with transparent backgrounds.
