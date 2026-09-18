# Direct Versus Geometry-Transformed QE Relaxation

Speedup is reported only when both relaxations pass the force gate, converge to the same basin, every lineage segment passes input/job/pw.x runtime provenance, calculator identities match, and that identity is accepted by N24. Prefix ratios, current-geometry basin diagnostics, and threshold-crossing steps are not final speedups.

| Path | Image | Shift (A) | Direct steps / fmax | Transformed steps / fmax | Prefix step ratio | Basin | Comparison |
|---|---:|---:|---|---|---:|---|---|
| 1-5 | 2 | 0.392 | 45 / 0.080 | 50 / 0.066 | 0.900 | pending | pending_not_converged |
