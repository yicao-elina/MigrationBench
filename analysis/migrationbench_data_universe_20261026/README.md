# MigrationBench configuration-space universe — 2026-10-26

This additive package renders three views of the same provenance-aware
configuration table: retained SOAP/PCA coordinates, UMAP, and t-SNE. The
current local release uses the previously audited grouped Cr/Sb/Te SOAP/PCA
coordinates as the common representation because the full raw MLIP MD SOAP
vectors are not retained in this checkout. UMAP and t-SNE are therefore
explicitly labelled as nonlinear views of the retained two-dimensional SOAP/PCA
space, not as independent claims about a universal latent space.

The package also contains a deterministic Blender selector and renderer for a
Cr-centered local environment extracted from a real NEB EXTXYZ frame. The
source has coordinates and energies but no explicit bond topology, so bonds in
the render are geometry-based visual heuristics and are labelled as such in
the manifest.

## Outputs

- `figures/migrationbench_data_universe_pca.png/.pdf/.svg`
- `figures/migrationbench_data_universe_umap.png/.pdf/.svg`
- `figures/migrationbench_data_universe_tsne.png/.pdf/.svg`
- `derived/data_universe_points.csv`
- `derived/data_universe_manifest.json`
- `blender/cr_local_environment_manifest.json`
- `blender/render_cr_local_environment.py`
- `blender/style_contract.md`

## Rebuild

```bash
python analysis/migrationbench_data_universe_20261026/run_manifold_plots.py
/Applications/Blender.app/Contents/MacOS/Blender -b --python \
  analysis/migrationbench_data_universe_20261026/blender/render_cr_local_environment.py -- \
  --input analysis/neb_blender_visual_refinement_20261025/mlff_neb_images.extxyz \
  --output-dir analysis/migrationbench_data_universe_20261026/blender/rendered \
  --frame 6 --width 3600 --height 3600 --samples 256 --save-blend
```

