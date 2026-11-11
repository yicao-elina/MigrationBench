# DFT-RECOVERY-RECONSTRUCTION

## Article target
Results/SI: DFT recovery and MACE force-validation boundary

## Caption
A, reconstructed DFT-versus-MACE force RMSE by recovery case and model; B, maximum-force relation for the same reconstructed rows.

## Results
The reconstruction table contains normal, bounded, and runaway cases with finite-looking DFT/MACE comparison fields, but every row is explicitly labeled RECONSTRUCTION / REPRODUCED_SCOPE_LIMITED in the source table. Model force errors increase strongly in the runaway cases.

## Discussion
This package is useful for auditing the DFT-recovery boundary and for deciding whether a later accepted QE release can replace the reconstruction. It must not be presented as a converged DFT validation benchmark.

## Scientific boundary
- The source scope is RECONSTRUCTION / REPRODUCED_SCOPE_LIMITED.
- A finite reconstructed row is not equivalent to a normal, accepted QE target.
- Re-run or replace this package after raw QE outputs and gate manifests are independently closed.
