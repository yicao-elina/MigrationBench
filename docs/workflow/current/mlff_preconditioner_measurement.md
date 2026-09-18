# Measuring MLFF Preconditioning For DFT NEB

## Scientific Question

The benchmark asks whether an MLFF initializer reduces the DFT work needed to
reach the same DFT minimum-energy path. It does not ask only whether the MLFF
has a low random-frame energy or force RMSE.

For MLFF optimizer iteration `t`, define the path residual

`R_t = max_i ||F_i^NEB(t)||`,

the residual integral

`A_T = sum_t (R_t + R_(t+1)) (t_(t+1)-t_t) / 2`,

and the log reduction rate

`k_T = log(R_T/R_0) / T`.

Lower `A_T`, more negative `k_T`, and fewer steps to fixed residual thresholds
indicate a more efficient MLFF preconditioner. These metrics remain diagnostic
until the MLFF path is handed to DFT.

## DFT Handoff Test

For every candidate initializer, preserve its frozen final MLFF images and run
the same QE calculator identity, CPU resources, force target, and restart
policy. Compare:

1. initial DFT NEB residual;
2. DFT ionic iterations and total SCF iterations to each force threshold;
3. residual-versus-cost integral;
4. final path distance and mechanism identity;
5. final forward and reverse DFT barriers.

A speedup is valid only when both initializers converge to the same DFT
mechanism under the accepted calculator identity. A different basin or path is
a mechanism-selection result, not an acceleration result.

`scripts/migrationbench/compare_qe_neb_initialization_ab.py` implements this
comparison without using live-prefix ratios. It accumulates path iterations and
SCF iterations across an explicitly ordered restart lineage, compares the final
bands with periodic minimum-image distances, checks the final DFT barrier, and
requires an N24-accepted calculator identity. Its default same-mechanism gates
are `0.15 A` all-atom path RMS, `0.25 A` migrant-curve RMS, and `0.05 eV`
forward-barrier agreement. These thresholds are versioned command inputs rather
than implicit judgments.

Real Rockfish Slurm job `30853151` passed the full suite (`84 passed`), including
tests that a running prefix cannot produce a speedup, a mechanism switch is
reported separately, and periodic-equivalent migrant paths have zero distance.
The follow-up hardening binds each lineage's first `neb.in` to the corresponding
schema-1.1 pair-manifest hash and records initial residual, residual AUC, and
first path iteration reaching fixed residual thresholds. Job `30853162` exposed
a missing synthetic cell-distance field and is archived as a failed test; the
fail-closed correction passed all 84 tests in Rockfish job `30853165`.

The executable production conditions and matched-input manifest are defined in
`docs/qe_neb_handoff_gate.md`. Diagnostic paths with unverified endpoints are
blocked from QE production even when their file and calculator checks pass.

## Two-Step Rockfish Smoke

The same 61-atom, seven-image historical-curvature-warp candidate was run in
real CPU Slurm jobs with two optimizer steps:

| Model | Job | Initial residual (eV/A) | Final residual (eV/A) | Final/initial | Residual AUC (eV/A step) |
|---|---:|---:|---:|---:|---:|
| MACE-OMAT fine-tuned | 30850337 | 14.0893 | 3.5688 | 0.2533 | 15.9807 |
| MACE-MP-0 medium foundation | 30850360 | 10.7868 | 3.6589 | 0.3392 | 13.3371 |

Neither run reached 1.0, 0.5, 0.1, or the requested 0.05 eV/A threshold. The
OMAT run has the steeper relative two-step decrease, while the foundation run
has the lower finite-window residual integral. Two steps are insufficient to
rank either model for production or DFT acceleration.

An exporter audit found that ASE zero-fills fixed endpoints in
`NEB.real_forces`. Earlier iteration exports therefore had invalid zero endpoint
force labels even though internal-image residuals were correct. Corrected v3
histories read calculator forces before NEB projection; endpoint NEB force and
residual remain absent. The correction ledger is
`data_processed/mlff_history_corrections/endpoint_true_force_correction.json`.

The MLFF barrier proxies changed from 5.9107 to 2.4487 eV for OMAT and from
5.4634 to 2.6135 eV for the foundation model. They are optimizer diagnostics,
not migration barriers and are ineligible for the manuscript.

## Leakage And Proposal Dependence

Using an MLFF to propose a path does not contaminate the final DFT energies or
forces. It can, however, bias which mechanisms are discovered. Every record
must therefore retain `proposal_model_path_sha256`, `evaluation_model`, source
trajectory hashes, and split-group identifiers.

An independent model benchmark must evaluate every model on the same frozen
candidate paths. A self-proposed path may be used to measure end-to-end search
performance, but it must be labeled proposal-dependent and cannot replace the
fixed-path benchmark. No target DFT frame, neighboring image from the same
trajectory, or symmetry duplicate may cross a grouped train/test boundary.

## Artifacts

- Comparison configuration: `configs/mlff_preconditioner_smoke_comparison.json`
- Comparison table and plot:
  `data_processed/mlff_preconditioner_comparison/nonlinear_historical_warp_smoke/`
- Lossless exporter: `scripts/migrationbench/export_mlff_neb_iteration_history.py`
- Comparator: `scripts/migrationbench/compare_mlff_preconditioner_runs.py`
- DFT-NEB initialization comparator:
  `scripts/migrationbench/compare_qe_neb_initialization_ab.py`
- Rockfish validation:
  `data_processed/cluster/mb_nebablineage_s42_30853165/`
