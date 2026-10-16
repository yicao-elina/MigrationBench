# Rockfish Cr NEB audit — 2026-10-16

## Result

The three existing Rockfish jobs were reconciled without submission,
cancellation, modification, or duplication. All three are terminal
`COMPLETED` with `ExitCode=0:0`, absent from `squeue`, and their live raw files
match the retained local raw snapshot byte-for-byte for the audited inputs,
path tables, outputs, and job logs.

## Scientific gates

All three paths have ten images and 81 atoms with `Cr1 Sb32 Te48`, migrating Cr
at atom index 81, and the same QE cell from `pw_1.in` and `pw_2.in`
(6007.626539 Å³). The final complete tables contain finite energies and finite
gradient residuals. `neb.out` contains both directional activation lines and
`JOB DONE`; no fatal, NaN, Inf, MPI-abort, or FIFO marker was found.

| Path | Job | Forward (eV) | Reverse (eV) | Last complete iteration | Max gradient residual (eV/Å) | SCF warning |
|---|---:|---:|---:|---:|---:|---:|
| within-QL | 31916908 | 1.844900 | 2.346822 | 160 | 3.200354 | image 8 |
| gap → QL | 31916909 | 0.000000 | 0.095947 | 68 | 0.383903 | image 2 |
| in-gap diffusion | 31916910 | 1.524299 | 0.675386 | 81 | 0.684822 | image 4 |

The SCF warning is the controlling publication gate. These are finite,
completed path-table diagnostics, not converged DFT barriers. They remain
`COMPLETED_BUT_SCF_WARNING` and are excluded from final converged-barrier
claims. A clean rerun requires an explicit user decision; none was submitted.

## Cr versus Cu/Li/Na and MLIP diagnostic

The existing comparison package covers all four dopants, all three path
classes, 120 model-barrier rows, and 12 path-class summaries. The Cr rows are
kept distinct from the existing `SCREENED_EXPLORATORY` Cu/Li/Na rows. The
fixed-path Cr diagnostic contains 10 models × 3 paths × 10 images = 300 finite
model-image rows. It is a same-geometry single-point evaluation and is not a
self-relaxed MLIP NEB.

The provisional comparison supports these bounded observations: Cr within-QL
is the largest model/QE disagreement case (QE forward 1.844900 eV versus
model barrier-like range 21.176812–29.735416 eV); Cr in-gap diffusion has a
model range 0.899060–1.549396 eV around the warning-contaminated QE value
1.524299 eV; and Cr gap → QL is direction-dependent (forward 0.000000 eV,
reverse 0.095947 eV). These values are diagnostic comparisons only.

No combined paper figure was regenerated or promoted by this audit because the
Rockfish QE warning policy is unresolved.

## Evidence files

- `ROCKFISH_AUDIT.tsv`: scheduler, geometry, finiteness, directional, and publication-status gates.
- `RAW_HASH_RECONCILIATION.tsv`: live Rockfish versus local raw SHA256 checks.
- `REMOTE_SCHEDULER_20261016.txt`: scheduler evidence and the required Slurm configuration.
- `EXCLUSION_LEDGER.tsv`: preserved warning and publication boundaries.
- `COMPARISON_MANIFEST.json`: hashes and row counts for the provisional Cr versus Cu/Li/Na and fixed-path MLIP package.
