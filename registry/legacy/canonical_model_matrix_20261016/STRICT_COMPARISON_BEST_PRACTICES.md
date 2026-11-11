# Strict model-comparison practice

This campaign separates a model's manuscript label from its canonical identity.
Every comparison row must resolve to a dataset release, source-group split,
architecture, foundation checkpoint (or null), optimizer protocol, seed, model
hash, and acceptance status.

## Non-negotiable pre-submission gates

1. Freeze a versioned dataset release and hash every split file and split summary.
2. Treat whole source trajectories as atomic split units. A frame-level random
   split is exploratory only and cannot support a strict generalization claim.
3. Define the matrix before training. Within one family, only the training seed
   may vary. Foundation checkpoints are separate model families, not seeds.
4. Require the launch script to record data hashes, foundation hash, seed,
   architecture, optimizer, environment, scheduler job ID, and output paths.
5. Refuse overwrite if a target model directory already contains a compiled
   model. Historical outputs are immutable evidence.
6. Accept a model only after finite/loadable compiled-model checks, element and
   PBC checks, source-group leakage checks, and test metrics are recorded.
7. Register every accepted model under a canonical ID containing the dataset and
   protocol family. Never reuse a random-split ID for a grouped-split model.
8. Keep historical heterogeneous models available as `EXCLUDE_FROM_STRICT`
   comparators; do not pool them into seed variance or report them as replicas.
9. For the shared 639 candidates, compare paired forces and force tails. Do not
   rank unrelated model families by absolute energies without a common energy
   reference and a separate validation argument.

## Required artifacts per family

- immutable source inventory and source SHA256 list;
- split manifest with source trajectory IDs and frame counts;
- train/valid/test SHA256 values;
- one launch script and one preflight record;
- one model row per seed/checkpoint in `MODEL_REGISTRY.tsv`;
- model hash and compiled-model load audit;
- failure/exclusion status and reason;
- downstream 639-candidate evaluation manifest.

## Current decisions

The canonical from-scratch release uses the existing deduplicated Multi-T
grouped source universe. The 600 K source universe is contained within it, so
concatenating the two prepared datasets would duplicate trajectories and create
cross-split collisions. The 3D bulk/doped source remains excluded from this
2-layer release until a separate domain-shift audit is accepted.

The pending historical job `32055099_[0-1]` is not accepted because it points
to the unresolved `data/` release. It is kept as a separate provisional record;
its output must never enter the canonical registry or strict variance.
