# QE NEB Handoff Gate

## Purpose

The handoff layer creates a matched pair of explicit-image QE NEB inputs:

1. the frozen nonlinear candidate before MLFF relaxation;
2. the final MLFF-preconditioned images.

Both inputs use the same engine template, path controls, image count, CPU
protocol, and complete QE calculator identity. Their only intended difference
is the initial path geometry.

## Production Conditions

`qe_neb_from_images.py --run-role production` refuses to write an eligible
input unless all applicable checks pass:

- the candidate endpoints are labeled `accepted_local_minima`;
- the candidate is geometry-valid, nonduplicate, and production eligible;
- the candidate manifest records the SHA-256 of the earlier independent
  endpoint-acceptance artifact, and its two endpoint hashes match that artifact;
- the MLFF manifest is hash-bound to the candidate and its input images;
- the MLFF output image hash matches the supplied handoff images;
- MLFF optimization converged and its final NEB residual is at most the stated
  handoff threshold;
- the generated QE identity is complete and equals the N24 accepted identity.

Trust-region MACE uses a separate initialization role,
`trust_region_mlff_preconditioned`. It cannot satisfy the ordinary unrestrained
MLFF check by construction. Instead, it additionally requires:

- convergence under the explicitly recorded base-plus-tether optimization
  potential, while `optimizer_converged` remains false;
- hash-complete dual-potential extxyz and CSV histories;
- a hash-bound comparator artifact whose decision is exactly
  `trust_region_candidate_ready_for_dft_ab_design`;
- an exact binding from that comparator to the selected MLFF manifest and a
  matching base-potential physical identity.

The comparator alone never authorizes QE. Production generation combines it
with the same independent endpoint, candidate-geometry, and accepted-QE-
calculator gates listed above. This keeps tethered optimization quantities out
of the physical barrier while still allowing a validated trust-region band to
be tested as an initializer.

The matched pair is production eligible only when both individual gates pass.
Converged DFT paths must additionally reach the same final mechanism before
ionic-step or SCF-step acceleration is reported.

Every production, warm-started, and restart QE-NEB wrapper records both
`neb.x` and `pw.x`, not only the driver executable. The downstream A/B
comparator validates `neb.in`, Slurm job id, and both binary hashes for every
segment contributing to cumulative path/SCF work. One provenanced terminal
segment cannot repair an unprovenanced parent segment.

The provenance direction is deliberately acyclic:
`endpoint acceptance -> nonlinear candidate -> MLFF result -> QE handoff`.
An earlier implementation expected the endpoint artifact to know the hash of a
future candidate manifest; that circular requirement was removed and covered
by a forward-binding regression test.

## Diagnostic Smoke

The current bundle uses unverified `1-6/path108` image-2 and image-5 endpoint
candidates. It is therefore diagnostic and not submit-eligible. Direct and
MACE-preconditioned inputs each contain seven images and share complete QE
calculator identity `7111008577b76397`.

Rockfish Slurm job `30850443` is retained as a failed infrastructure attempt:
strict undefined-variable handling was enabled before `/data/apps/go.sh`, so
the job exited on an unset `RF_CONDA_SITE`. No compute-node validation from
that attempt is accepted.

After moving strict handling after environment activation and excluding copied
local validation output, Slurm job `30850445` completed in two seconds with
exit code `0:0`. On the compute node, both inputs passed file-hash, image-card,
complete-identity, and shared-identity checks.

Handoff bundle schema `1.1` additionally binds each `neb.in` to its individual
handoff manifest, candidate and MLFF sources, initialization role, and, for a
trust-region branch, its comparator acceptance artifact. The validator remains
read-compatible with legacy diagnostic schema `1.0` bundles. Rockfish jobs
`30853132` and `30853133` are retained as failed fixture-portability diagnostics:
they exposed missing historical MACE and engine-template files in the remote
code root. The self-contained replacement in job `30853141` passed the full
suite (`82 passed`) and generated and validated a schema-1.1 pair on the compute
node.

Runtime smoke `30857573` is retained as a failed environment-order diagnostic:
activating Conda after loading QE removed the QE executables from `PATH`, so the
strict binary gate rejected the record. The corrected compute-node job
`30857639` activated Conda first, then loaded QE 7.3.1; it captured and validated
real `neb.x` and `pw.x` hashes plus the input and Slurm job id, and passed both
QE-NEB A/B decision tests (`0:0`). Production QE wrappers do not activate the
MACE Conda environment.

## Artifacts

- Pair generator: `scripts/migrationbench/prepare_qe_neb_handoff_pair.py`
- Gated input generator: `scripts/migrationbench/qe_neb_from_images.py`
- Validator: `scripts/migrationbench/validate_qe_neb_handoff_pair.py`
- Slurm smoke wrapper: `scripts/migrationbench/submit_slurm_qe_handoff_smoke.sh`
- Diagnostic bundle: `data_processed/qe_handoff_smoke/1-6_histwarp_s44/pair/`
- Compute-node result: `cluster/mb_qehandoff2_sm_s44_30850445/`
- Schema-1.1 compute-node test:
  `data_processed/cluster/mb_handoffgate_s42_30853141/`
- QE runtime comparator smoke:
  `data_processed/cluster/mb_qenebprov_s42_30857639/`
- Failed environment-order smoke:
  `data_processed/cluster/mb_qenebprov_s42_30857573/`
