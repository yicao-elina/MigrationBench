# Direct Versus Geometry-Transformed QE Relaxation

Speedup is reported only when both relaxations pass the force gate, converge to the same basin, every lineage segment passes input/job/pw.x runtime provenance, calculator identities match, and that identity is accepted by N24. Prefix ratios, current-geometry basin diagnostics, and first or sustained threshold crossings from a running prefix are not final speedups.

| Path | Image | Shift (A) | Direct steps / fmax | Transformed steps / fmax | Prefix step ratio | Basin | Comparison |
|---|---:|---:|---|---|---:|---|---|
| sb2te3_4x2x1::joshitha::unique_088 | 1 | 0.227 | 0 / pending | 0 / pending | pending | pending | pending_not_converged |
| sb2te3_4x2x1::joshitha::unique_093 | 1 | 0.048 | 0 / pending | 0 / pending | pending | pending | pending_not_converged |
| sb2te3_4x2x1::joshitha::unique_098 | 1 | 0.235 | 0 / pending | 0 / pending | pending | pending | pending_not_converged |
| sb2te3_4x2x1::joshitha::unique_166 | 1 | 0.092 | 0 / pending | 0 / pending | pending | pending | pending_not_converged |
| sb2te3_4x2x1::victor::unique_119 | 1 | 0.006 | 0 / pending | 0 / pending | pending | pending | pending_not_converged |
| sb2te3_4x2x1::victor::unique_200 | 1 | 0.090 | 0 / pending | 0 / pending | pending | pending | pending_not_converged |
