# Direct Versus Geometry-Transformed QE Relaxation

Speedup is reported only when both relaxations pass the force gate, converge to the same basin, every lineage segment passes input/job/pw.x runtime provenance, calculator identities match, and that identity is accepted by N24. Prefix ratios, current-geometry basin diagnostics, and threshold-crossing steps are not final speedups.

| Path | Image | Shift (A) | Direct steps / fmax | Transformed steps / fmax | Prefix step ratio | Basin | Comparison |
|---|---:|---:|---|---|---:|---|---|
| 1-3 | 2 | 0.423 | 36 / 0.061 | 50 / 0.055 | 0.720 | pending | pending_not_converged |
| 1-3 | 3 | 0.391 | 48 / 0.081 | 50 / 0.091 | 0.960 | pending | pending_not_converged |
