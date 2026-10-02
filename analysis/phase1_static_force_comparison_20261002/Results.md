# Phase 1 results

## Completion and provenance

All 16 requested single-point evaluations completed with finite energies,
forces, and stress tensors: 4 models × 4 common snapshots, 16/16 complete and
0/16 failed. The formal evaluation was Rockfish Slurm job 31511914 on host
`gpu14`. Model hashes and source-trajectory hashes are in
`results/models_observed.json` and `logs/trajectory_sha256.txt`.

| model | completed | observed model SHA256 (prefix) | within-model ΔE, 460000→462200 (eV) |
|---|---:|---|---:|
| from-scratch | 4/4 | `da17ab17a982…` | -107.7 |
| naive-FT | 4/4 | `6829d16e68bd…` | +308.4 |
| multi-T FT | 4/4 | `0c132dd8b4b9…` | +595.1 |
| Foundation-OMAT | 4/4 | `dc41a98908dd…` | +194.7 |

## Force and geometry behavior

The common input geometry moves from a minimum periodic pair distance of 2.393 Å
at step 460000 to 2.360 Å, 1.979 Å, and 0.629 Å at the subsequent selected
steps. The fixed cell volume is 150,190.66 Å^3. The maximum atom-level pairwise
force spread increases from 1.29 eV Å^-1 at step 460000 to 5.07, 55.22, and
4,565.24 eV Å^-1. This is a clear geometry/force-integrity failure regime, not
ordinary small model noise.

At the first two snapshots, pairwise mean vector-force differences remain about
0.14–0.23 eV Å^-1. At step 462000 they increase to 0.17–0.46 eV Å^-1, and at
step 462200 the largest pairwise mean difference is 6.69 eV Å^-1 (multi-T FT vs
naive-FT). The largest-disagreement atoms are Te at step 460000 and Sb at the
later three steps; the local 3 Å shell contains 4–5 atoms.

The verified Cr set contains 50 atoms. Its maximum force norm at step 462200 is
5.73 eV Å^-1 (from-scratch), 8.81 eV Å^-1 (naive-FT), 7.74 eV Å^-1 (multi-T
FT), and 5.71 eV Å^-1 (Foundation-OMAT). At step 462000 Foundation-OMAT has a
14.28 eV Å^-1 maximum Cr force, while the other three models range from 4.61 to
7.66 eV Å^-1.

## Evidence-bounded conclusion

The four models agree moderately on the first two common configurations but
diverge sharply as the shared from-scratch trajectory enters a collapsed local
geometry. The result supports a model-disagreement/trajectory-integrity warning,
not a ranking of model accuracy, transfer gain, or DFT fidelity. The Foundation
energy offset further prevents direct absolute-energy ranking in this setup.

## Recommendation for Phase 2/3

Phase 2 should first audit the onset of the minimum-distance collapse using the
retained momenta, per-step geometry, and source continuation lineage; reject any
diffusion or barrier claim that includes the 0.629 Å regime. Phase 3 should use
DFT-gated, identity-matched snapshots or endpoints and report force/stress error
against DFT, with energy-zero alignment documented before any cross-model energy
claim. A second static panel around the last geometry passing a minimum-distance
gate is preferable to extending this collapsed sequence.
