# NEB residual audit

**Figure Sx | Image-wise residuals of the screened QE-NEB paths.** Each line and point shows the `error (eV/A)` reported by QE for the final complete image table. The dashed horizontal line is the `path_thr` requested in the corresponding `neb.dat` input. The residual is a force/path convergence diagnostic, not an energy uncertainty or standard error. A path passes this image-wise gate only when all movable images lie below the dashed threshold. The energy-profile figures therefore do not convert this quantity into a shaded eV interval.

The accompanying CSV preserves the final residual, threshold, last parsed outer iteration, and number of complete image tables for every path. Because electronic SCF warnings remain present in the current outputs, this audit is a quality-control record and does not promote any path to a formally converged DFT barrier.
