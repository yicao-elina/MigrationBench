# Trust-Region MACE NEB Preconditioner

## Motivation

An unconstrained foundation-model NEB can satisfy its projected-force tolerance
after internal images have moved into a different basin. A lower MACE energy is
not, by itself, evidence that the resulting band is a useful initializer for the
intended DFT mechanism.

The trust-region branch treats MACE as a local path preconditioner. For every
internal image it minimizes the NEB built from

`E_opt(R) = E_MACE(R) + 0.5 sum_a k_a ||MIC(r_a-r_a,ref)||^2`,

where `MIC` is the minimum-image displacement from the immutable input band.
The migrant and host atoms have separately configured force constants. Endpoints
remain fixed by NEB and receive no tether.

## Lossless Recording

Every optimizer iteration stores both quantities separately:

- base MACE energy and atomic forces;
- restrained optimization energy and atomic forces;
- tether energy and maximum displacement from the reference image;
- optimizer iteration, image index, coordinates, cell, and PBC.

The physical barrier proxy is always recomputed from `E_MACE`; the restraint
energy is never added to a reported barrier. A restrained run sets
`optimizer_converged=false` and records convergence only in
`optimizer_converged_under_optimization_potential`, preventing the ordinary
MACE-to-QE gate from accepting it accidentally.

## Acceptance

A restrained branch may advance only to `ready_for_dft_preconditioner_review`.
It must have complete input/model/runtime/history hashes, preserved endpoints,
minimum pair distance at least `1.8 A`, maximum Cr image step at most `2.5 A`,
and no internal base-MACE energy change larger than `5 eV`. It is then compared
with the unrestrained and direct DFT controls. It is not an endpoint-stability
label, an MLFF MEP, or a manuscript barrier.

## First Rockfish Experiment

Job `30853021`, seed `42`, applies `k_Cr=4.0 eV/A^2` and
`k_host=0.5 eV/A^2` to the historical `1-4` path for at most 150 FIRE steps.
It uses the explicit MACE-MP-0 checkpoint, 8 CPU cores, 48 GB, and a 2-hour
Slurm allocation. Its immutable submission configuration is
`configs/mlff_trust_region_1-4_s42.json`. Job `30852994`, seed `52`, is retained
as an independent implementation replicate, not the primary causal pair.

The causal comparison is:

1. identical `1-4` input band and model;
2. unrestrained job `30852128` versus restrained job `30853021`, both seed 42;
3. compare path displacement, energy collapse, continuity, and DFT work after
   handoff, never just optimizer convergence.

This primary comparison is diagnostic because its source candidate explicitly
records `endpoint_status=unverified_historical_neb_endpoints` and
`production_eligible_before_mace=false`. It can establish whether the tether
preserves useful historical curvature, but its final images cannot be attached
to endpoint evidence produced later. The topology audit also identifies image 4
as an internal basin candidate, giving the provisional segment sequence
`image 1 -> image 4 -> image 5`.

After independent repeat relaxations accept those basin structures, the
production sequence is therefore:

1. build separately hash-bound `1->4` and `4->5` nonlinear candidates using
   historical/trust curvature only as a donor;
2. map the accepted endpoints exactly onto each reconstructed band;
3. rerun unrestrained or trust-region MACE on each new candidate;
4. compare and hand off only that new, endpoint-bound MACE result.

No historical trust output is grandfathered into production merely because its
coordinates look similar to newly accepted endpoints.

`scripts/migrationbench/compare_mlff_trust_region.py` is the fail-closed terminal
comparator. It requires exact physical identity between the seed-42 pair and a
complete dual-potential history before a restrained branch can be labeled
`ready_for_dft_ab_design`. Even that label keeps `qe_handoff_allowed=false`
until endpoint and paired DFT gates are available.

For an unaccepted historical candidate, the comparator instead emits
`trust_region_diagnostic_ready_as_curvature_donor_requires_endpoint_remap`.
Rockfish job `30853208` validates this distinction (`85 passed, 1 skipped`).
The skip is the newer dual-manifest helper intentionally absent from the remote
runner while active jobs remain frozen to their submitted code. It must be
eliminated by a full rerun after those jobs terminate.

Real Rockfish Slurm job `30853032` validates the comparator and fail-closed
classification in the full pipeline suite; all `78` tests pass. A subsequent
data-semantics job, `30853107`, passes all `80` tests, including a synthetic
two-iteration band whose base-MACE and restrained labels deliberately differ.
That test verifies that exported physical energies, forces, and barriers come
from base MACE while optimization-only columns retain the tethered values.

Because the local runner continues to evolve while Slurm jobs are active, each
submitted input directory contains a frozen `submitted_code/` snapshot and
`SHA256SUMS`. The primary and replicate both ran against runner hash
`95279374bb7930644ba75bce872208aa7886a1a477275a7546287e82fe6ae129`.
Later code changes cannot silently rewrite the meaning of these runs.

The HF staging builder is dual-potential aware. For restrained runs it uses
only base-MACE energy and force as physical labels, retains optimization and
regularization quantities in separate columns, and fails if hashes, frame
counts, coordinates, or iteration/image keys are not aligned. The dataset
validator independently checks the energy decomposition and potential labels.
The implementation and validator were exercised by Rockfish job `30853107`;
its archived output is
`data_processed/cluster/mb_dualdata_s42_30853107/slurm-30853107.out`.
