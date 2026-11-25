# MigrationBench configuration universe — 2026-10-26

This is an additive, provenance-bound analysis package for the Cr-centred local
configuration universe. It keeps fixed-path DFT images, MLIP self-relaxed NEB
images, MLIP interlayer self-relaxed images, grouped reference configurations,
and a separately sampled MD layer as explicit source layers.

## Common embedding contract

All rows use the same non-periodic, Cr-centred SOAP descriptor:

```text
species = Cr, Sb, Te; r_cut = 5.0 Å; n_max = 4; l_max = 4; sigma = 0.5 Å
average = off; centers = every Cr; pbc = false
```

PCA is fitted on grouped training reference local environments only. UMAP and
t-SNE are fitted on the first 30 whitened PCA coordinates with fixed seed 42;
they are visualizations of the same input space, not mutually comparable
coordinates or a performance score. Training-manifold distance is a descriptor
space diagnostic, not a universal task-difficulty metric.

## Scientific boundary

- `Foundation` 61-atom self-NEB paths are current and converged in the local
  registry, but remain provisional model-PES paths rather than DFT MEPs.
- `FT-600K` 61-atom self-NEB paths are available and are kept separate from
  the 81-atom legacy/provisional geometries.
- `Scratch-MultiT` and `FT-MultiT` self-NEB geometries are exploratory/pending
  for the missing or archived subsets; they are shown with open markers where
  appropriate.
- `Scratch-600K` has no local self-NEB EXTXYZ in the audited matrix and is
  represented in the coverage table as missing rather than imputed.
- The DFT interlayer paths are retained with their `COMPLETED_BUT_SCF_WARNING`
  status. The DFT Cr 1–4 path is the current audited candidate profile and is
  not promoted to a formally converged production barrier.
- The interlayer self-relaxed package currently has readable Foundation paths;
  archived compressed/dataless placeholders are retained as missing evidence.

## Rebuild

From this directory:

```
python run_structure_universe.py
python repair_path_profiles.py
python blender/render_all_clusters.py
python compose_publication_universe.py
python coordination_distortion.py
python coordination_reference_comparison.py
python compose_comprehensive_task_map.py
```

The MD layer is pinned in `derived/md_map_sample.extxyz` after extraction from
the remote trajectories. To refresh it, rerun `extract_md_map_sample.py` with
the audited Rockfish input manifest, copy the result and its manifest back into
`derived/`, and then rerun the final command.

The main deliverables are in `figures/`; the point table, cluster table,
coverage matrix, render inputs, Blender manifests, and style metadata are in
`derived/` and `blender/`. The separate PCA/UMAP/t-SNE panels remain available
for independent review alongside the composed publication figure. The
task-conditioned comprehensive display is exported as
`figures/migrationbench_configuration_universe_comprehensive.png/.pdf/.svg`,
with its auditable point table in
`derived/comprehensive_task_map_points.csv`.

## Comprehensive task-conditioned map

`compose_comprehensive_task_map.py` makes one deliberately readable universe
display from a single common local-structure pipeline. The four two-word-or-
fewer level labels are defined by the physical regime; method and path
protocol are encoded as sublayers:

| level | display label | actual layer |
|---|---|---|
| L0 | `L0 DFT Reference` | static DFT/reference basin |
| L1 | `L1 Thermal Equilibrium` | DFT-AIMD or MLIP-MD thermal basin |
| L2 | `L2 NEB Migration` | directed migration paths |
| L3 | `L3 Layer Sliding` | interlayer sliding paths |

The MD layer contributes 72 uniformly spaced frames from 15 source branches.
Its local SOAP vectors are recomputed with exactly the same contract as the
static/path layers; because each MD frame contains many Cr centres, the map
uses the mean local descriptor and mean shell-distortion summary over all Cr
centres in that frame. Circle area encodes the Cr-shell RMS bond-length
distortion relative to the Cr-containing reference shell. Translucent hulls
show the task-level configuration regions. Sublayer saturation/lightness
distinguishes DFT from MLIP or AIMD from MD; solid path lines denote fixed or
sliding paths and dashed lines denote self-relaxed paths.

The L0 label is placed at the displayed median of the reference manifold, so
it remains inside the broad L0 cluster rather than outside the canvas region.

To make task clusters visible on one page, the grouped reference manifold keeps
its original UMAP footprint, while each foreground task keeps its internal
relative UMAP geometry and is placed in a separate display region. Therefore
the display placement is a presentation transform: distances between different
task islands in this particular panel must not be interpreted as new metric
distances. The original common-space PCA/UMAP/t-SNE coordinates and the full
provenance columns remain available for quantitative follow-up. The remote
source paths and SHA256 hashes for the MD sample are retained in
`derived/md_map_sample_manifest.json`; this layer is descriptive coverage and
does not change the underlying MD acceptance or transport status.

## Coordination-sphere rendering and distortion

The Blender outputs are coordination ball-and-stick models: Cr-neighbour
cylinders are drawn only when the minimum-image distance is within 3.75 Å, and
the surrounding local atoms remain visible as matte spheres. These cylinders
are distance-qualified coordination guides, not inferred covalent bond orders.
The `PERFECT_REFERENCE` render is the Cr-containing host endpoint used as the
common coordination reference. Some grouped reference frames contain more than
one Cr; the first audited Cr centre is used as the focal centre and its index is
recorded in the render manifest.

`figures/coordination_distortion.*` reports shell coordination number, bond
length spread, and paired bond-length deviation for every plotted frame.
`figures/coordination_reference_comparison.*` places the reference and three
representative renders beside the same-shell distortion metrics.
One retained grouped-training frame has no neighbour within the common shell;
its deviation is kept undefined and is excluded from finite distortion
summaries rather than being treated as zero.
