# Foundation interlayer-sliding: canonical 10-model × 2-system × 2-operator campaign

This is an additive campaign package for the complete canonical Foundation
roster. It deliberately leaves the accepted R13 three-model campaign and the
MatPES-only extensions immutable.

## Frozen scope

- 10 canonical model identities in `MODEL_REGISTRY.tsv`.
- 2 systems: `bulk` (80 atoms) and `cr_in_gap` (81 atoms).
- 2 separate observables: `FIXED_PATH_SINGLE_POINT` and
  `SELF_RELAXED_ENDPOINT_ONLY_NEB`.
- 10 retained images for every row, therefore 40 primary rows and 400 model
  profile rows.
- Fixed path: score the retained QE b-path2 geometries without relaxation.
- Self relaxed: endpoint-only ASE IDPP interpolation followed by MACE FIRE
  NEB, fixed endpoints, 500 steps, `fmax=0.05 eV/A`, `k=0.1`, seed 42,
  `climb=false`.

## Provenance boundary

The QE reference is retained as `COMPLETED_BUT_SCF_WARNING` context. It is not
called a converged DFT ground truth. Self-relaxed profiles are MLIP PES paths,
not DFT minimum-energy paths and are not pooled with fixed-path profiles.

## Remote execution

The formal remote root is
`/scratch16/pclancy3/yi/foundation_interlayer_sliding_10model_20261024`.
The source input structures are copied from the accepted R13 input package and
are hashed again in each row manifest. Raw model checkpoints remain at their
immutable Rockfish paths; their SHA256 values are frozen in the registry.

## Acceptance

The acceptance contract is in `ACCEPTANCE_FREEZE_R01.json` and `.md`. A row is
scientifically usable only when its model/input/runtime/output gates pass:
fixed rows are finite diagnostics and self rows are converged endpoint-only
MLIP paths. Collection, plot, data review, and visual review must all pass
before the campaign is reported as complete.
