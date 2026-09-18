# Nonlinear NEB Candidate Generation

## Role

`generate_nonlinear_neb_candidates.py` is the deterministic geometry layer
between accepted DFT endpoint minima and MACE pre-NEB. It generates several
initial curves instead of silently reducing the workflow to one Cartesian
interpolation. It never assigns energy or force labels.

## Periodic Endpoint Mapping

For atom `a`, the target endpoint displacement is the minimum-image vector
`d_a = MIC(R_B,a - R_A,a)`. The continuous baseline is

`R_a(s) = R_A,a + s d_a`, for `s in [0,1]`.

The last written coordinate may differ from the wrapped input coordinate by a
lattice vector. Its PBC deviation from the requested endpoint must be below
`1e-8 A`. This unwrapped representation prevents a fake long spring at a
periodic boundary.

## Candidate Families

1. `linear_mic`: periodic linear diagnostic baseline.
2. `clearance_arc`: the migrant receives
   `A sin(pi s) n`, where `n` is perpendicular to its endpoint chord. Distinct
   directions are separated by at least 15 degrees; current amplitudes are
   0.25 and 0.50 A.
3. `historical_curvature_warp`: a historical path is unwrapped and resampled by
   configuration-space arc length. Its deviation from its own endpoint chord
   is added to the new endpoint chord. Endpoints remain exact under PBC.

This is curve transfer, not Cartesian averaging of different mechanisms.

## Geometry And Duplicate Gates

Every candidate records minimum all-pair distance, minimum migrant-host
distance, migrant image step, configuration path length, host motion, and
tortuosity. Current smoke thresholds are:

- minimum all-pair distance: `1.7 A`;
- minimum migrant-host distance: `1.4 A`;
- maximum migrant inter-image step: `3.0 A`.

Duplicate detection is deliberately task relevant. Two paths are duplicates
only if both their migrant-path RMS is at most `0.05 A` and their all-atom path
RMS is at most `0.02 A`. An all-atom RMS alone diluted distinct Cr arcs in an
early smoke and was rejected.

## Real Rockfish Smoke

The unverified `1-6/path108` image-2 to image-5 pair generated eight unique
61-atom, seven-image candidates: one MIC baseline, six clearance arcs, and one
historical-curvature warp. All eight passed the local geometry gate.

Rockfish CPU Slurm job `30850304` validated the frozen v1 input with the `mace`
environment and ASE:

- Slurm state `COMPLETED`, exit code `0:0`, elapsed 10 seconds;
- 8/8 candidates accepted;
- every path has 7 images and every image has 61 atoms;
- complete PBC on every image;
- no energy/force arrays and no attached calculator;
- submitted and compute-node manifest SHA-256 both
  `19c159ed2b2423f275ccbf6fd6476f7dfc7937a3c93e5c1ffbd80eaaaa0d51f9`.

The v1 manifest abbreviated `clearance_arc` as `clearance` in one metadata
field. The v2 manifest fixes only that label; all eight extxyz hashes are
identical to the Rockfish-validated v1 files.

## Production Gate

This smoke uses unverified endpoint candidates, so all rows have
`production_eligible_before_mace=false`. No production MACE calculation may be
launched from it. After N17 accepts an endpoint pair and N24 accepts its QE
calculator identity, regenerate the candidate bundle with
`--endpoint-status accepted_local_minima`, `--initial-structure`,
`--final-structure`, and `--endpoint-acceptance`. Production mode refuses QE
input coordinates: both structures must be the final repeat-relaxed extxyz
files, and their exact SHA-256 values must match the accepted pair artifact.
The hash-binding gate has a regression test that rejects either changed
endpoint. Then run MACE independently for each
unique geometry-gate-passing branch and retain the linear curve as a baseline.

## MACE Handoff Smoke

The historical-curvature-warp candidate was passed through the complete MACE
runner on Rockfish. Jobs `30850337` and `30850360` preserve all three snapshots
(initial plus two optimizer steps), all 21 image records, true atomic forces on
every image, and projected NEB forces only on the 15 internal-image records.
Endpoints are never assigned fake zero NEB forces.

Job `30850337` used the MACE-OMAT fine-tuned checkpoint with SHA-256
`e618ad582b84239905b9c3b77ce6e9ce111b0ecd1533223a1a6aac7a696b8aa0`.
Job `30850360` used the explicit MACE-MP-0 medium checkpoint with SHA-256
`01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845`
and float64. The explicit path is required because the MACE API default is not
a stable model identity.

Both jobs reduced the NEB residual to about 3.6 eV/A but remained strongly
unconverged. They validate the geometry-to-MLFF interface and lossless history
capture, not a production initializer or a final barrier. Task-relevant metrics
and the downstream DFT comparison gate are specified in
`docs/mlff_preconditioner_measurement.md`.

## Artifacts

- Generator: `scripts/migrationbench/generate_nonlinear_neb_candidates.py`
- ASE validator: `scripts/migrationbench/validate_nonlinear_candidate_manifest.py`
- CPU Slurm template:
  `scripts/migrationbench/submit_slurm_nonlinear_candidates_smoke.sh`
- Corrected portable bundle:
  `data_processed/nonlinear_candidate_smoke/1-6_iter108_img2_to_img5_v2_s42/`
- Frozen Rockfish result: `cluster/mb_nlpath_smoke_s42_30850304/`
- MLFF measurement: `docs/mlff_preconditioner_measurement.md`
- Two-model comparison:
  `data_processed/mlff_preconditioner_comparison/nonlinear_historical_warp_smoke/`

## Multi-Frame Curvature Donors

Historical trajectories stored as multi-frame ASE-readable files can be used
with `--historical-images` plus `--historical-template-qe`. Inclusive
`--historical-start-image-qe` and `--historical-end-image-qe` bounds select a
specific basin-to-basin segment. The donor contributes only the deviation from
its endpoint chord; newly accepted endpoint structures remain exact in the
generated candidate.

For historical `1-4`, the topology-defined plan is image `1->4` and image
`4->5`. The machine-readable placeholders and gates are in
`configs/1-4_endpoint_remap_plan_s42.json`. Rockfish job `30853440` passed the
full available suite (`86 passed, 1 skipped`) and verified that images 1-4 are
read from the five-frame `sb2te3.xyz` with the QE cell and atom ordering. The
trust jobs are now terminal and fully archived, so the active-runner freeze has
been removed. The latest runner, launcher, monitor, and wrapper were synced to
the canonical Rockfish code root. The first deployment smoke `30861664`
correctly failed because one updated A/B comparator had not been synchronized;
after syncing it, replacement job `30861741` passed all `97` tests with zero
skips.

`scripts/migrationbench/materialize_endpoint_remap_candidates.py` consumes the
plan only after receiving exactly one accepted endpoint-pair artifact for each
planned segment. It verifies endpoint image numbers, repeat-relax acceptance,
accepted-structure hashes, the shared endpoint calculator identity, and donor
hashes before invoking the nonlinear generator. It then requires at least one
production-eligible candidate per segment and emits a hash-bound materialization
manifest for the new MACE queue. Rockfish job `30853464` validates the fail-
closed acceptance checks (`87 passed, 1 skipped` at the former frozen-runner
boundary).

The real historical `1-4` trust-region runs `30853021` and `30852994` both
converged under the tethered optimization potential in 31 steps and reproduced
the same base-potential outputs across seeds. The unrestrained control collapsed
by `5.178 eV`; the trust run limited the largest internal base-energy change to
`4.740 eV` and reduced migrant RMS drift from `0.750 A` to `0.168 A`. It does
not make the old band production eligible: the base-MACE residual remains
`0.190 eV/A`, its barrier proxy is zero, and its endpoints are not accepted.
The formal decision is
`trust_region_diagnostic_ready_as_curvature_donor_requires_endpoint_remap`.

The full materialization path is also exercised independently rather than only
through the acceptance helper. Rockfish CPU Slurm job `30853484` constructed
test-only accepted structures for historical images 1, 4, and 5, hash-bound two
synthetic acceptance artifacts, and invoked the complete materializer. The
`1->4` segment produced 10 geometry-passing unique candidates; `4->5` produced
10 geometry-passing candidates with one correctly deduplicated, leaving 9.
The targeted end-to-end test passed in 4.45 seconds. These structures are test
fixtures only and do not satisfy the real endpoint-acceptance placeholders.
