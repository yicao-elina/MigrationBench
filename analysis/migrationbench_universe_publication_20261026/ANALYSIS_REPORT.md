# Configuration-universe analysis report

## What is in the current common-space map

The current map contains 2,724 local configurations:

| source layer | count | interpretation |
|---|---:|---|
| grouped reference configurations | 2,362 | train/valid/test reference structures |
| MLIP self-relaxed Cr NEB | 245 | available 61- and 81-atom path images |
| MLIP interlayer self-relaxed | 30 | available Foundation 2×2 path images |
| DFT interlayer NEB | 10 | Bulk and Cr-in-gap AXSF path images |
| DFT Cr fixed path | 5 | audited 1–4 candidate path images |
| MD thermal sample | 72 | uniformly sampled frames from 15 Rockfish source branches |

The descriptor is a common non-periodic Cr-centred SOAP with 390 features.
PCA is fitted only on grouped training local environments. The first two PCA
components explain 0.3487 and 0.1646 of the fitted variance, respectively.
UMAP and t-SNE use the same whitened 30-dimensional PCA input and fixed seed
42. They are complementary visualization views; their axes are not physically
interpretable and should not be compared by absolute coordinate values.

The original static/path coordination table retains its 2,652 rows, and the
comprehensive table retains all 2,724 map rows after adding the MD layer. One
grouped-training frame (train frame 1783) has zero neighbours within the 3.75 Å
shell, so its paired bond-length deviation is undefined rather than imputed as
zero. MD shell statistics are aggregated as the mean over all Cr centres in
each frame. The distortion summaries therefore retain undefined values where
the common shell is absent rather than converting them to zero.
The exported metrics distinguish signed total coordination-number change
(`coordination_number_delta`) from the species-wise Sb/Te count mismatch
(`species_count_delta`).

## What the map says

1. The grouped reference set forms the broad background manifold. Model-generated
   path images occupy both dense reference-like regions and sparse extrapolative
   regions; the latter are not automatically “wrong”, but they are farther from
   the training descriptor distribution.
2. Foundation and FT-600K have the best local path coverage in the current
   audited checkout. Scratch-MultiT path geometry is present as exploratory
   material but includes non-converged paths. FT-MultiT has missing local EXTXYZ
   for part of the 61/81 path matrix. Scratch-600K has no local self-NEB or
   interlayer self-relaxed EXTXYZ in the audited matrix.
3. A cluster representative is selected by medoid-like distance in the common
   local-structure space. The rendered Cr environment is therefore a real
   source frame, not a synthetic cartoon. Cluster colour is shared by the
   map label, representative star, accent ring, and Blender output.
4. The local SOAP distance is a coverage/extrapolation diagnostic. It is not a
   model-quality score, a DFT error estimate, or a universal ordering of task
   difficulty. Task layers are kept explicit by marker shape and path lines.

The compact level labels are physical regimes rather than generic model names:
`L0 DFT Reference`, `L1 Thermal Equilibrium`, `L2 NEB Migration`, and `L3
Layer Sliding`. Static DFT, DFT-AIMD, MLIP-MD, DFT fixed, MLIP self-relaxed,
DFT sliding, and MLIP sliding remain explicit sublayers in the point table and
the figure legend. Thus self-relaxed is treated as a path-generation protocol,
not as a synonym for penetration.

The Blender representation is a coordination ball-and-stick view. A cylinder
is drawn from the focal Cr to a neighbour only when the same minimum-image
distance is within 3.75 Å; the cylinder is a visual coordination guide and
does not assert a covalent bond order. The `PERFECT_REFERENCE` scene is the
Cr-containing host endpoint used to define the common shell (CN=8, 2 Sb + 6 Te,
mean distance 2.839 Å, shell standard deviation 0.117 Å). An undoped pristine
Sb2Te3 cell has no Cr centre, so it is not substituted for this Cr-centred
reference. Some grouped reference frames contain multiple Cr atoms; the first
audited Cr is used as the focal centre and the selected index is retained in
the manifest and distortion table.

## NEB and interlayer interpretation

The Cr self-relaxed paths are model-PES paths. They are not interchangeable
with the DFT fixed path and are not used to claim a common DFT minimum-energy
path. The DFT 1–4 profile is included as the current audited candidate and
retains its non-formal-convergence caveat. The DFT interlayer profiles retain
the `COMPLETED_BUT_SCF_WARNING` status. MLIP interlayer self-relaxed images are
endpoint/path proxies, not DFT-NEB barriers.

## MD boundary

The comprehensive display now includes a uniform sample of 72 frames from 15
remote Rockfish MD source branches (up to five frames per branch when
available). The sampled EXTXYZ is retained locally as
`derived/md_map_sample.extxyz`, and the remote path, frame count, selected
frames, and source SHA256 are recorded in
`derived/md_map_sample_manifest.json`. The same non-periodic, Cr-centred SOAP
contract is recomputed for these frames; aggregation over their 2,050-atom
cells is explicitly a mean over Cr centres. This is a descriptive local
configuration-coverage layer, not a full-trajectory density estimate and not
a new 1-ns transport or scientific-acceptance claim. The earlier whole-cell MD
atlas remains a separate analysis product.

## Files for review

- `figures/migrationbench_universe_publication.png/.pdf/.svg`: composed figure
- `figures/migrationbench_configuration_universe_comprehensive.png/.pdf/.svg`:
  single task-conditioned display; task colour, Cr-shell distortion area, and
  separated task-level configuration regions
- `figures/structure_universe_pca.*`, `structure_universe_umap.*`,
  `structure_universe_tsne.*`: independent manifold panels
- `figures/path_geometry_energy_profiles.*`: separate path-energy diagnostic
- `figures/coordination_distortion.*`: same-shell distortion metrics and
  process/model distributions
- `figures/coordination_reference_comparison.*`: reference-versus-representative
  coordination ball-and-stick comparison
- `derived/structure_universe_points.csv`: every plotted row and provenance
- `derived/coordination_distortion_points.csv`: coordination metrics for every
  plotted frame
- `derived/comprehensive_task_map_points.csv`: transformed display coordinates,
  task layer, model/protocol metadata, and distortion marker size used by the
  comprehensive figure
- `derived/comprehensive_task_map_vectors.npz`: common PCA/UMAP/t-SNE vectors
  used by the comprehensive map
- `derived/comprehensive_task_map_fit.json`: fit contract, counts, labels, and
  provenance summary for the comprehensive map
- `derived/md_map_sample.extxyz`: pinned MD frame sample used in the map
- `derived/md_map_sample_manifest.json`: remote MD source paths and SHA256
  manifest for that sample
- `derived/coverage_matrix.csv`: available/missing geometry matrix
- `derived/cluster_representatives.csv`: rendered representative registry
- `derived/coordination_shells.json`: reference shell and per-frame neighbour
  distances
- `blender/cluster_renders/`: bright white-background, transparent, and `.blend`
  coordination ball-and-stick outputs

## Comprehensive task-conditioned display

The comprehensive panel is intended to answer a visual question: where do
configurations from different task layers sit relative to the broad reference
manifold, and how strongly is the Cr coordination shell distorted? It uses the
same local SOAP/PCA/UMAP/t-SNE contract for 2,724 rows, including the 72-frame
MD sample. The reference layer is shown as a pale background manifold. For the
foreground task layers, the within-layer UMAP geometry is preserved but the
layer medians are separated on the canvas to prevent the small DFT/path sets
from being hidden by the much larger reference set. Parent hues encode the
four physical regimes, sublayer saturation/lightness encodes DFT versus MLIP
or AIMD versus MD, and path-line style distinguishes fixed/sliding from
self-relaxed protocols. The task labels are placed at the displayed group
medians, including L0 inside the reference cluster.

This separation is a display layout, not a re-fitting of the descriptor or a
claim that the empty space between task islands is a physical distance. Circle
area is the finite Cr-shell RMS bond-length deviation; the one retained
zero-neighbour reference row remains undefined and is not assigned a zero
distortion.
