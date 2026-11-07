# Recommendation for the manager

1. Do not spend effort forcing a 2050-atom QE single point. It is not a useful production validation unit here.
2. Use the exact 82-atom ordered periodic block as the primary reduced DFT test. It preserves the actual local environment, periodic cell, both Cr atoms, original atom ordering, and the force-outlier atom in the selected tile.
3. Use the 81-atom same-site model as a lower-cost sensitivity run only. It matches the existing 81-atom QE workflow and retains the selected Cr site, but removing the neighboring Cr changes the chemistry. Label it explicitly as `same_site_81_one_Cr_removed`.
4. Do not use the 61-atom processed endpoints to support this same-site comparison: they have a different cell and composition.
5. Treat the 462000/462200 frames as stress tests. The MACE force explosion means a successful DFT single point would be a local force/energy comparison at an OOD geometry, while an SCF failure would only establish missing coverage.

Acceptance requires a finite QE total energy, converged SCF, finite atomic forces, normal termination, input/source hashes, and the scheduler/job ledger. Slurm completion alone is not acceptance.
