# NEB protocol matrix audit (2026-10-18)

This additive audit keeps the two NEB operators separate:

- **fixed-path:** each MLIP evaluates energies and forces on the same DFT-defined image sequence;
- **self-relaxed:** the MLIP evolves the image sequence under its own forces and reports an MLIP-native barrier proxy plus optimizer diagnostics.

The two operators are not pooled into a single model ranking. The DFT values used for the fixed-path comparison are pathway-specific candidate references: the 61-atom references are provisional/unconverged, and the 81-atom candidates have high residual forces. They are retained for diagnostic comparison, not presented as final DFT ground truth.

## Current matrix

The expected canonical matrix is:

| Protocol | Cell | Models/arms | Paths | Expected cells | Current status |
|---|---:|---:|---:|---:|---|
| fixed-path | 61 atoms | 10 model/seed arms | 5 | 50 | complete finite outputs; force-pathology flags retained |
| fixed-path | 81 atoms | 10 model/seed arms | 5 | 50 | complete finite outputs |
| self-relaxed | 61 atoms | 4 canonical seed-123 routes | 5 | 20 | 20 manifests present; 17 converged, 3 finite but non-converged |
| self-relaxed | 81 atoms | 4 canonical seed-123 routes | 5 | 20 | 20 rows present; 5 Foundation rows are legacy diagnostics, 1 new row converged, 14 finite but non-converged |

The machine-readable complete status table is [neb_matrix_status.tsv](derived/neb_matrix_status.tsv); the CSV version is [neb_matrix_status.csv](derived/neb_matrix_status.csv). Every row includes model/path identity, barrier, reference status, run status, publication status, source artifact, and failure flag.

## Figures

- [fixed_path_neb_matrix.pdf](figures/fixed_path_neb_matrix.pdf) / [PNG](figures/fixed_path_neb_matrix.png): barrier values assigned on common fixed geometries. The `x` marker identifies a finite result with a maximum image force above 10 eV A-1; this is retained as a model failure mode rather than discarded.
- [self_relaxed_neb_matrix.pdf](figures/self_relaxed_neb_matrix.pdf) / [PNG](figures/self_relaxed_neb_matrix.png): MLIP-native self-relaxed barrier proxies. `C` means the optimizer converged under the recorded MLIP protocol, `!` means finite but not converged, `D` is the legacy 81-atom Foundation diagnostic, and `P` is pending GPU retry.

The figures use the Plot Atlas publication defaults and the registered `jhu_mace_models` model-family palette. They are first-pass audit figures, not final manuscript panels: the grey/pending and diagnostic states are intentional.

## Rebuild

From the repository root:

```bash
/Users/alina/Documents/Codex/plot_library/.venv/bin/python \
  analysis/neb_protocol_matrix_20261018/build_neb_matrix_assets.py
```

The current snapshot includes the completed GPU retries `32199417` (fixed-path) and `32199418` (self-relaxed). Do not replace the old failed-attempt directories; the original CPU-loading failures are preserved as an explicit failure mode. The scheduler audit is [slurm_health_32199417_32199418.tsv](derived/slurm_health_32199417_32199418.tsv).

## Interpretation boundary

The self-relaxed barrier proxy is not a DFT barrier. To claim that relaxation improves physical fidelity, the final MLIP-relaxed images must be rescored with DFT. The present matrix therefore supports protocol comparison and stability diagnostics, but not a final self-relaxed accuracy ranking.
