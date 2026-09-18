# Geometry-Transform DFT Relaxation Speed Experiment

## Question

For a fixed QE NEB iteration and image, does a deterministic local geometry transform reduce the DFT relaxation work without changing the final basin?

## Transform

Only the Cr coordinate is changed. Host coordinates, cell, pseudopotentials, cutoffs, k-points, electronic thresholds, ionic algorithm, random seed, MPI ranks, memory, walltime, and QE `max_seconds` are held fixed.

The transformed coordinate minimizes

`Q(r) = lambda |r-r0|^2 + sum_j softplus((d_safe-d_j(r))/sigma)^2`

under periodic boundary conditions and `|r-r0| <= 0.25 A`. The first term limits distortion; the second reduces short Cr-host contacts smoothly.

This is the deterministic, small-displacement version of the site score proposed
for endpoint search. In the more general notation

`Q_site(r|H) = w_E E_hat(r|H) + w_F ||F_hat(r|H)|| + w_U U(r|H) + w_C P_contact(r|H) + w_R ||r-r0||^2`,

the present experiment isolates the last two terms. It tests whether removing an
unphysical local contact, without using the DFT answer, reduces subsequent DFT
work. Energy, force, embedding, and uncertainty terms will only be activated
after independently relaxed endpoint labels exist; otherwise the transform
would be tuned on the evaluation target.

## Canonical Pairs

| Pair | Exact source | Direct | Transformed |
|---|---|---|---|
| `1-6/image2` | `sb2te3.path108`, image 2 | `30844024 -> 30851800` | `30844035 -> 30851803` |
| `1-6/image5` | `sb2te3.path108`, image 5 | `30844025 -> 30851807` | `30844038 -> 30851806` |
| `1-7/image2` | `sb2te3.path112`, image 2 | `30844039 -> 30851802` | `30844065 -> 30851801` |
| `1-7/image5` | `sb2te3.path112`, image 5 | `30844053 -> 30851808` | `30844069 -> 30851809` |

Every source `.pathN`, generated input, and manifest has a SHA-256 hash. The `.pathN` energy is checked against the same complete `neb.out` iteration before input generation.

## Coverage-Gap Selected Points

A second, independent A/B experiment targets the three historical images that
violate the `1.8 A` Cr-host clearance gate while retaining acceptable endpoints.
For each offending neighbor `j`, the transformed Cr coordinate is updated under
minimum-image periodic boundary conditions as

`r_(k+1) = r_k + min(delta, d_min - d_j) (r_k-r_j)/|r_k-r_j|`.

The update stops when every Cr-host distance is at least `d_min=1.8 A`. Only Cr
moves; the host, cell, atom ordering, and endpoints are unchanged. This is a
deterministic feasibility projection, not an energy minimization and not a DFT
label.

| Pair | Direct job | Transformed job | Cr shift (A) | Initial minimum distance (A) |
|---|---|---|---:|---:|
| `1-3/image2` | `30850842` | `30850844` | 0.423327 | 1.376673 -> 1.800000 |
| `1-3/image3` | `30850843` | `30850845` | 0.390710 | 1.409290 -> 1.800000 |
| `1-5/image2` | `30850846` | `30850847` | 0.391505 | 1.408495 -> 1.800000 |

All six calculations use two CPU MPI ranks, 160 GB, a 24-hour Slurm walltime,
and QE `max_seconds=84600`, allowing QE to stop cleanly before Slurm. A clean
timeout is continued from the latest printed geometry, with its hash stored
separately from the last force-evaluated geometry and every parent
hash and cumulative ionic/SCF count preserved.

An independent input audit found `6/6` complete inputs, zero parse errors, and
one shared calculator identity, `7111008577b76397`. Thus the A/B coordinate is
the intended experimental variable. Manuscript use still waits for the separate
N24 acceptance of this calculator identity.

## Metrics

Primary work metrics are the number of ionic steps and accumulated SCF iterations. Walltime and CPU-hours are secondary because node performance and queue placement vary.

For a comparable pair, define

`S_ionic = N_ionic,direct / N_ionic,transform`

and

`S_scf = N_scf,direct / N_scf,transform`.

A value greater than one indicates less DFT work after transformation.

## Acceptance Gate

A speedup is reported only when both jobs:

1. End with QE `JOB DONE` and BFGS convergence.
2. Have final maximum atomic force no larger than `0.05 eV/A`.
3. Reach the same basin: Cr displacement between final structures <= `0.25 A`, host RMSD <= `0.15 A`, and local Cr-neighbor signature RMS <= `0.15 A`.
4. Every direct and transformed continuation segment passes exact `relax.in`,
   Slurm job-id, and `pw.x` runtime-provenance validation.

If the final basins differ, the result is a basin switch, not an acceleration. If only one member converges, the pair remains censored rather than assigning an artificial speedup.

## Current State

The completed 50-step parent segments give the following censored comparison.
The energy penalty is the transformed first-step DFT energy minus the direct
first-step DFT energy under the identical calculator.

| Pair | Shift (A) | Initial penalty (eV) | Ionic steps D/X | SCF iterations D/X | Current fmax D/X (eV/A) | Current geometry |
|---|---:|---:|---:|---:|---:|---|
| `1-6/image2` | 0.250 | +0.184 | 50 / 50 | 911 / 847 | 0.075 / 0.071 | same by diagnostic gate |
| `1-6/image5` | 0.143 | +0.177 | 50 / 50 | 899 / 933 | 0.782 / 0.066 | different by diagnostic gate |
| `1-7/image2` | 0.250 | +0.179 | 50 / 50 | 858 / 833 | 0.073 / 0.071 | same by diagnostic gate |
| `1-7/image5` | 0.132 | +0.166 | 50 / 50 | 853 / 908 | 0.081 / 1.048 | same by diagnostic gate |

The transform therefore does not lower the starting DFT energy. The two image-2
pairs are approximately tied, while `1-6/image5` has a much smaller transformed
force but currently occupies a different geometry basin. None of these prefix
observations is a speedup result.

The original eight exact-iteration jobs ended cleanly at QE's inherited default
`nstep=50`, but none reached BFGS convergence. They are therefore terminal
prefixes, not completed A/B measurements. The r2 jobs listed above continue
from each branch's latest printed geometry with `nstep=200`, two CPU MPI ranks,
160 GB, a 24-hour walltime, and `max_seconds=84600`. The continuation manifest
stores that geometry hash separately from the last force-evaluated geometry and
retains cumulative ionic-step and SCF work across the lineage.

Numerical speed comparisons remain pending same-basin convergence and N24
calculator acceptance. Rockfish smoke job `30852040` passed all 69 current
tests, including the corrected continuation, fixed remote code-root export,
and periodic path-repair gates. The earlier infrastructure-only attempt
`30851788` is archived because an accidentally duplicated test module caused
collection failure before scientific execution. The six coverage-gap jobs are
still tracked independently. Their direct coordinates are unchanged even
though archived manifests inherited the paired transform displacement field;
the correction is recorded in `configs/qe_geometry_transform_relax_ab_s47.json`,
and the generator now emits an identity transform for future direct branches.

The active r2 calculations were submitted before per-job runtime capture was
enabled. They are valid paired diagnostics, but their accumulated work cannot
support a manuscript speedup because every segment contributing to the ratio
must have `runtime_provenance.json`. A later continuation alone does not repair
an unprovenanced parent cost segment; a production claim requires a fully
provenanced rerun of both arms.

The comparator now enforces this over the full `relax_lineage.json`, not only
the terminal directory. Rockfish CPU smoke `30857352` passed the final-decision
and multi-segment provenance tests (`2 passed`, exit `0:0`).

At the authorized `2026-09-13T15:26Z` Rockfish checkpoint, all six coverage-gap
jobs were running normally. Their latest complete QE blocks contained `25/23`
BFGS steps for direct/transformed `1-3/image2`, `28/27` for `1-3/image3`, and
`29/29` for `1-5/image2`. These are live, right-censored prefixes. Total-force
snapshots are not the maximum per-atom force used by the acceptance gate, so
they are deliberately not converted into speedups.

The companion MACE A/B calculations and new historical-path MACE runs are
diagnostics only. The strict terminal audit found zero accepted QE handoffs:
the corrected periodic `1-3` and `1-5` bands collapsed to zero MACE barrier
proxies and changed mechanism or geometry, while new `1-4`, `1-7`, and `1-8`
runs failed energy-collapse or path-continuity gates. MACE force convergence
alone is therefore insufficient to define the DFT initializer.

`update_qe_exact_geometry_transform_ab.py` is the single update entry point. It
refuses a live Rockfish poll when the newest status is less than three hours old,
then monitors all eight active branches and rebuilds both pair tables, all ionic
curves, force plots, and the combined `exact_iter/summary.json`.

Real Rockfish Slurm job `30852910` exercised the latest provenance, history,
geometry, and basin-collapse gates; all `74` tests passed. The next scientific
comparison runs only after the three-hour cadence gate or a terminal job event.

At the cadence-authorized `2026-09-13T18:27Z` checkpoint, the three selected
direct jobs remained healthy and running at `36/48/45` ionic steps with latest
complete maximum forces `0.06134/0.08137/0.07988 eV/A`. Their transformed
partners reached a clean `nstep=50` stop without BFGS convergence at forces
`0.05517/0.09074/0.06554 eV/A`. Hash-bound 24-hour continuations with
`nstep=200`, `max_seconds=84600`, 2 MPI ranks, and 160 GB were submitted as
jobs `30860584`, `30860628`, and `30860689`; all entered RUNNING in the initial
health snapshot. No pair is a speedup, and all three current geometries differ
under the diagnostic basin thresholds.

The exact-iteration checkpoint at `2026-09-13T18:33Z` found all eight r2 jobs
healthy and running. Cumulative direct/transformed ionic steps were `100/101`,
`96/97`, `103/91`, and `90/91` for `1-6/image2`, `1-6/image5`,
`1-7/image2`, and `1-7/image5`. Only `1-7/image2` remained in the same
diagnostic basin. All four formal decisions remain `pending_not_converged`.

The selected-point comparator also no longer dereferences a Rockfish-only
source path from the manifest. It reads atom order and cell from the locally
synced direct `relax.in`, which is the actual A/B input and is independently
hash-bound. This fixed the failed local post-processing attempt without
re-querying the six jobs.
