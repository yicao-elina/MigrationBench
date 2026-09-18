# Direct Versus Geometry-Transformed QE Relaxation

Speedup is reported only when both relaxations pass the force gate, converge to the same basin, every lineage segment passes input/job/pw.x runtime provenance, calculator identities match, and that identity is accepted by N24. Prefix ratios, current-geometry basin diagnostics, and first or sustained threshold crossings from a running prefix are not final speedups.

| Path | Image | Shift (A) | Direct steps / fmax | Transformed steps / fmax | Prefix step ratio | Basin | Comparison |
|---|---:|---:|---|---|---:|---|---|
| 1-7 | 2 | 0.250 | 163 / 0.007 | 139 / 0.001 | 1.173 | pending | pending_not_converged |
| 1-7 | 5 | 0.132 | 92 / 0.003 | 109 / 0.003 | 0.844 | pending | same_basin_runtime_provenance_pending |
