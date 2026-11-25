# Corrected same-geometry Foundation-model comparison

## Figure caption

**Figure X | Same-geometry comparison of the complete local MACE Foundation-model set with the retained QE path table.** Each panel shows ten images from one retained Cu/Li/Na path. The black trace is the QE path-table energy, shown with explicit points and line segments; each colored trace is a single-point energy profile from one of the nine local Foundation models evaluated on exactly the same geometries. Models are colored with related Morandi shades within each model family. QE `.path` energies were converted from Hartree to eV; Foundation energies were normalized within each path to image 1. The NEB image-table ``error`` column is a force/path residual in eV Å⁻¹, not an energy standard error, and is reported separately in the convergence-audit companion figure.

## Results and discussion

The corrected comparison removes two confounders in the earlier exploratory figure: the QE `.path` energy unit conversion and the mismatch between the retained QE geometries and fresh endpoint-only linear interpolation geometries. The resulting model spread is strongly path dependent. Some paths show close model/QE agreement, whereas the Cu QLs path retains a large Foundation-model elevation despite exact geometry alignment. This indicates a model-response or extrapolation sensitivity along that path rather than a plotting-scale artifact. The barrier-like error heatmap reports these differences quantitatively without interpreting them as converged activation barriers.

The QE traces remain screened path-table references, not converged DFT barriers: the NEB audit found at least one electronic SCF warning in every path, and the formal DFT gate is 0/10. Therefore the figure supports a controlled model-disagreement and alignment analysis, while physical barrier ranking requires converged QE/NEB reruns.

## Reproducibility

Nine models were evaluated successfully on 8 paths × 10 images (720/720 evaluations). MH-0 and MH-1 used the explicit `omat_pbe` head. Input path identity is preserved through the processed path hashes and the path/image join. The previous endpoint-only linear matrix is retained separately and is not used as the QE comparison in this figure.
