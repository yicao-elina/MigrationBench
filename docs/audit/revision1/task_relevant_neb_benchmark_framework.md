# Task-Relevant NEB Benchmark Framework

Date: 2026-09-09

## Core Object

A migration benchmark task is a tuple

`tau = (S, a_m, x_0, x_1, G, C, R)`

where `S` is the host structure, `a_m` is the migrating species, `x_0` and `x_1` are endpoint sites, `G` is an ordered path or path-generating protocol, `C` is the electronic-structure configuration, and `R` is the relaxation/convergence rule.

The DFT reference is not just a number. It is the barrier, the path geometry, and the convergence trace:

`Delta E_DFT(tau) = max_i E_DFT(X_i*) - E_DFT(X_0*)`

where `X_i*` are images optimized under the declared NEB protocol.

## Why Equilibrium RMSE Is Not Enough

Energy/force RMSE over ordinary relaxed or thermal frames answers a distribution-average question. Migration barriers ask a decision question concentrated near rare, non-equilibrium saddle regions. A model can have low global RMSE but still fail the migration task by:

- ranking endpoints incorrectly,
- moving the path into the wrong channel,
- smoothing or exaggerating the saddle,
- giving good energies on fixed DFT images but bad forces perpendicular to the path,
- converging cheaply to a model-favored path that DFT later rejects.

So the benchmark should report task-level metrics, not only frame-level RMSE.

## Recommended Metrics

For a model `M` and a task `tau`, store at least four errors:

1. Fixed-path barrier error:
   `epsilon_fixed = Delta E_M(G_DFT) - Delta E_DFT(G_DFT)`

2. Self-relaxed path error:
   `epsilon_self = Delta E_DFT(G_M*) - Delta E_DFT(G_DFT*)`

3. Model self-reporting error:
   `epsilon_report = Delta E_M(G_M*) - Delta E_DFT(G_M*)`

4. Path geometry error:
   `d_path = min_alignment_distance(G_M*, G_DFT*)`, plus saddle-image displacement and coordination/void-score shifts.

This separates "model evaluates a known path badly" from "model relaxes into a different path" and from "model predicts its own relaxed path energy badly."

## Continuous Path Descriptors

Instead of hard labels only, every path should carry continuous descriptors:

- `gap_score`: how much the migrating atom occupies a low-density channel or interstitial void.
- `penetration_score`: how much it overlaps high-density host layers or highly coordinated environments.
- `coordination_profile`: coordination count by image.
- `min_host_distance_profile`: closest host contact by image.
- `path_length_A` and endpoint displacement.
- `saddle_environment`: descriptors of the maximum-energy image.

Labels such as `in_gap`, `deep_penetration`, or `mixed_transition` are then documented bins on top of these numbers. This makes the theory extensible beyond layered 2D materials: replace the slab z-gap component with a general free-volume/local-density descriptor from Voronoi volume, neighbor density, or learned local environment embeddings.

## Measurement Logic

Each path measures a different failure mode:

- In-gap/open-channel paths test whether the model preserves low-barrier diffusion in sparse environments.
- Deep-penetration paths test whether the model penalizes chemically crowded, high-coordination transition states.
- Mixed paths test whether the model can handle topological changes in the migration mechanism.
- Cross-material or collaboration-folder paths test whether these task errors generalize beyond the Sb2Te3 geometry and beyond the original training distribution.

The final dataset should support queries like:

- barrier error as a function of `penetration_score`,
- path geometry error as a function of `gap_score`,
- whether fine-tuning improves saddle regions without harming endpoint stability,
- whether the model systematically compresses high-barrier paths or over-stabilizes crowded Cr environments.

## Collaboration-Folder Generalization Split

The Rockfish folder `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/hpc_collaboration_20260306` should be treated as a cross-domain evidence source. It contains:

- `collaborative_dataset/COLLECTION_MANIFEST.json`: 720 collected files, about 1.83 GB, with collaborator records.
- `mlff_datasets/neb_collab_phase2__20260306_203401__tr80_va10_te10__s42/metadata/dataset_summary.json`: NEB collaboration split with seed 42.
- `mlff_datasets/farhan_perovskites_maidmf__20260310_162437__tr80_va10_te10__s42/metadata/dataset_summary.json`: Farhan/perovskite split with seed 42.
- `analysis/sampling_comparison_farhan/tables/*.csv`: sampling comparisons including energy, force, RMSD, and SOAP-distribution metrics.

These data should not be mixed into the Sb2Te3 DFT reference table. They should become a separate `source_domain=collaboration` or `source_domain=perovskite` split used to test whether model improvements learned from MigrationBench transfer across chemistries and structural motifs.

The same formalism still applies: each record must define the structure, moving species or deformation coordinate, non-equilibrium path/task, calculator, convergence status, split, and provenance SHA. When a path is not a literal NEB trajectory, its task descriptor should be generalized to a deformation/reaction coordinate with the same fixed-path and self-relaxed error decomposition.

## Manuscript Implication

The revised text should avoid presenting the foundation-model self-NEB value as directly comparable to the fixed-path DFT benchmark unless it is DFT-re-evaluated on the same task definition. The defensible claim is:

"Global force/energy accuracy is insufficient to validate migration barriers. MigrationBench therefore evaluates models on explicitly defined path tasks, reporting both fixed-path energy errors and relaxed-path errors with path-geometry metadata. This distinguishes energy-surface errors from path-selection errors and exposes failures in high-coordination transition environments."
