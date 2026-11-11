# From-scratch MD failure attribution

This campaign reproduces the three already registered branches without
touching the original ablation campaign:

| training seed | model family | scientific failure in original | model/input/MD provenance |
|---:|---|---:|---|
| 123 | `ablation_lmax_2` | 795.6 ps | original model SHA and `md_seed=20261005` |
| 234 | `ablation_lmax_2` | 52.4 ps | original model SHA and `md_seed=20261005` |
| 345 | `baseline` | 83.0 ps | original model SHA and `md_seed=20261005` |

The reproduction uses the original 2050-atom periodic input, 600 K Langevin
NVT, `dt=1 fs`, `friction=0.01 fs^-1`, 200-step frame saving, and the original
scientific gate: `T <= 5000 K`, `max |F| <= 100 eV/A`, and `d_min >= 0.5 A`.
The model-family difference is retained as provenance; these are not three
replicas of one identical neural-network parameterization.

## Attribution questions

1. Does the same model/input/MD-seed reproduce the onset step within the
   saved-frame resolution?
2. Which observable triggers first: temperature runaway, force runaway, or
   short-range collision?
3. Do local Cr-centered coordination, bond lengths, angle disorder, and
   force tails change before the threshold?
4. Does the configuration move outside the training-set descriptor support
   before the force runaway, and is it closer to test/NEB configurations?
5. Does cross-model force disagreement increase before the threshold?

## Descriptor and comparison contract

The distribution comparison uses the same DScribe SOAP convention as the
existing NEB analysis (`species=[Cr,Sb,Te]`, `r_cut=5 A`, `n_max=4`,
`l_max=4`, `average='inner'`, `sigma=0.5 A`). Training/validation/test
structures are transformed with this descriptor; existing NEB SOAP vectors
are used directly only when their 390-feature shape matches. Distances are
nearest-neighbour distances in a training-fitted standardized PCA space.
They are novelty diagnostics, not calibrated uncertainty and not causal proof.

## Planned publication figures

- `fig_reproduction_threshold_onset`: original versus exact reproduction for
  temperature, maximum force, and minimum pair distance in the final window.
- `fig_failure_distribution_attribution`: training/validation/test/NEB SOAP
  embedding, nearest-reference distances, stable-versus-precursor novelty,
  and Cr-centered local geometry.
- A supplementary force-disagreement panel will be added when the
  counterfactual three-model force evaluation completes.

The plot implementation follows the Plot Atlas `timecourse`,
`feature_embedding`, `joint_scatter`, and `steered_dynamics_nature` grammar:
Arial typography, bold tick labels, heavier spines, no decorative grids,
explicit thresholds, and PNG/SVG/PDF export. Unavailable measurements remain
explicitly unavailable.

## All-replicate execution contract

After the second reproduction root is terminally validated, run
`run_attribution_all.slurm`. It reads the protected original root plus
`mace_from_scratch_reproduce_20261011` and
`mace_from_scratch_reproduce_20261011_r02`, and writes an independent
`results_all_replicates/` tree. The diagnostic script emits `metrics_all.csv`,
`replica_ledger.csv`, `run_summary.csv`, `frame_features.csv`,
`window_feature_summary.csv` (from the report writer),
`force_disagreement.csv`, `snapshot_manifest.csv`, `reference_embedding.csv`,
`initial_state_audit.csv`, and `diagnostic_manifest.json`.

Selected force/geometry frames are evaluated with the exact model recorded in
each run manifest when the trajectory does not contain per-atom forces. The
recorded runner gate remains the authority for onset: temperature > 5000 K,
recorded maximum force > 100 eV/A, minimum distance < 0.5 A, or non-finite
values. SOAP/PCA nearest-neighbour distances are distribution-novelty
diagnostics, not calibrated uncertainty or causal proof; threshold-stopped
branches remain `unstable — no Dxy reported`.

The third run is fail-closed through `prepare_third_replicate.py`. Without
`--submit` it performs the terminal status, scientific stop-reason, model/input
hash, atom/PBC, and source-file checks before creating the isolated `_r02`
root. `--submit` is only used after that preflight succeeds; it records the
array mapping and submitted job ID in
`third_replicate_submission_manifest.json`.
