# Phase 3 reduced periodic DFT validation

This isolated campaign tests whether fixed-geometry QE single points can be evaluated at the same local migration site represented in the 2050-atom MACE trajectory.

## Decision

The 2050-atom single point is not the practical validation unit. The preferred reduced model is the exact 82-atom ordered periodic block copied from the 2050 trajectory: 2 Cr + 32 Sb + 48 Te, with cell `(a/5, b/5, c)` and the original atom identities retained. The 81-atom model removes only the second Cr from that same block and retains the selected Cr site. It is a controlled one-Cr composition approximation, not an exact replacement for the 2050 system.

The historical 61-atom `Cr1Sb24Te36` endpoints do not share this 2L cell/composition and are therefore not used as same-site evidence.

## Current evidence

- Full-trajectory source: `/scratch16/pclancy3/yi/mace_four_model_MD_continuation_20260930/1-from-scratch/segment_from_000200000.extxyz`
- Exact extracted frames: steps 460000, 461800, 462000, 462200; source SHA256 values are in `ordered_block_audit.csv`.
- Local block audit: `ordered_block_audit.csv` and `ordered_block_manifest.json`.
- QE method: QE 7.3.1 CPU, PBE+D3, spin-polarized, cold smearing, 70/280 Ry, 2x2x2 k-points, fixed SCF; no relaxation, NEB, retraining, or SOC claim.
- QE environment smoke: job 31512007 completed with finite energy/forces, SCF convergence, and `JOB DONE`.
- Production 81-atom jobs: direct one-process `pw.x` jobs 31512028/31512029 (shared) and 31512033/31512034 (express) are queued because the account QOS CPU limit is currently full. No DFT energy/force comparison is claimed until output is parsed; the complete scheduler/retry history is in `job_ledger.csv` and `status_20261002.md`.

## Important limitation

The MACE diagnostics show a rapidly growing force outlier: the 81-atom block maximum is 4.88, 45.93, and 229.97 eV/A at steps 461800, 462000, and 462200, respectively. These are OOD/explosion diagnostics, not DFT conclusions. A failed SCF on such a frame must be reported as a validation/coverage limitation, not converted into a physical barrier or model-accuracy claim.
