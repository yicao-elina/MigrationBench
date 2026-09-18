# Direct Versus Geometry-Transformed QE Relaxation

Speedup is reported only when both relaxations pass the force gate, converge to the same basin, every lineage segment passes input/job/pw.x runtime provenance, calculator identities match, and that identity is accepted by N24. Prefix ratios, current-geometry basin diagnostics, and first or sustained threshold crossings from a running prefix are not final speedups.

| Path | Image | Shift (A) | Direct steps / fmax | Transformed steps / fmax | Prefix step ratio | Basin | Comparison |
|---|---:|---:|---|---|---:|---|---|
| 1-6 | 2 | 0.250 | 157 / 0.005 | 109 / 0.003 | 1.440 | pending | pending_not_converged |
| 1-6 | 5 | 0.143 | 149 / 0.004 | 163 / 0.003 | 0.914 | pending | same_basin_runtime_provenance_pending |
