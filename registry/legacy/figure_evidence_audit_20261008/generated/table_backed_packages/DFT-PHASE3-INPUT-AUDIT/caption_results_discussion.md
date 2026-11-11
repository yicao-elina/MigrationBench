# DFT-PHASE3-INPUT-AUDIT

## Article target
Results/SI: reduced-cell DFT validation boundary

## Caption
A, maximum all-atom force; B, Cr-local maximum force for the full 2,050-atom source snapshot and two reduced ordered-block representations across four source steps.

## Results
The full source snapshot rises from 2.83 eV Å⁻¹ at step 460000 to 229.97 eV Å⁻¹ at step 462200. The reduced 82- and 81-atom representations retain the same source-step progression but are different validation units.

## Discussion
This is a pre-DFT input-integrity and stress-test figure. It documents why steps 462000 and 462200 are OOD/stress geometries and why a successful reduced-cell QE single point would still be a local comparison, not a converged migration barrier.

## Scientific boundary
- These are MACE forces stored in the source extxyz audits, not QE forces.
- The 81-atom model removes one Cr and is explicitly a same-site composition approximation.
- No plot panel may be used to claim DFT convergence or a barrier.
