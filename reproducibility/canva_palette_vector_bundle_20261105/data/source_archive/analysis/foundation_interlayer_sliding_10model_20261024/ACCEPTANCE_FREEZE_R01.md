# R01 acceptance contract

This freeze defines the complete canonical matrix: 10 models × 2 systems × 2
operators = 40 rows, with 10 images per row. It is an operator-comparison
diagnostic package, not a universal MLIP ranking.

## Fixed-path gate

`FIXED_PATH_SINGLE_POINT` evaluates the ten retained b-path2 geometries exactly
as supplied. The evaluator must record finite energy and finite forces for all
images, preserve image order, match the expected atom count, match the model
SHA256 in `MODEL_REGISTRY.tsv`, and capture runtime provenance. Energies are
reported relative to image 0. This gate does not make the retained QE values
converged DFT truth; the QE reference has a documented SCF warning.

## Self-relaxed gate

`SELF_RELAXED_ENDPOINT_ONLY_NEB` uses only the two endpoints, ASE IDPP
interpolation, ten images, FIRE for at most 500 steps, `fmax=0.05 eV/A`,
spring constant 0.1, seed 42, no climbing image, and fixed endpoints. It passes
only when the optimizer converges, the profile is finite, both NEB residual
force gates are ≤0.05 eV/A, and the final endpoints remain within 1e-8 Å of
their inputs. This is an MLIP PES path; it is not a DFT MEP.

## Completeness and review

The collection must contain exactly 20 fixed and 20 self rows, exactly ten
profile rows per accepted row, both systems, all ten model IDs, and a separate
QE reference table. Failed or duplicate attempts are preserved in the failure
and attempt registries. The final package is complete only after independent
data/provenance and visual checks pass. Fixed-path figures use the established
MACE-MP/MACE-OMAT/MACE-MH colors, boxed spines, no grid, normalized image
coordinate, and QE as a black dashed open-diamond reference labelled with its
SCF-warning status.
