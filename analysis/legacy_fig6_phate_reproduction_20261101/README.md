# Legacy NeurIPS Fig. 6 PHATE/t-SNE reproduction

This directory reproduces the four-panel figure in
`/Users/alina/Downloads/NeurIPS2025-AI4MAT-figure6-phate.pdf` from the
historical MigrationBench workshop release.

## Provenance

- Historical analysis commit: `61c29dd012bbf794213aa92238943afad9177ad7`
- Historical script: `results/t-SNE/phate-tsne/tsne.py`
- Historical feature matrix: `results/t-SNE/phate-tsne/analysis_Cr_Doped_600K_v2/Cr_Doped_Sb2Te3_600K_probe_features.csv`
- Feature matrix shape: `404 x 6192`; four contiguous blocks of 101 frames in
  the order `Foundation`, `Scratch`, `FT - 600K`, `FT - Multi-T`
- Historical projection settings: t-SNE perplexity 30 / random state 42 /
  1000 iterations; PHATE random state 42 / `n_jobs=-1`
- The attached PDF and the historical PDF in the canonical paper repository
  have SHA256
  `41197e7f6220a877e856c862c0e0ef85dce375eb4500eeee8a579726d4144990`.

The `source/` subtree is an extracted copy of the historical plotting script.
The feature matrix remains in the source Git object's immutable history rather
than being duplicated into this workspace. The wrapper can consume a temporary
materialization of that exact Git artifact.

## Rebuild

From the canonical paper repository:

```bash
tmpdir=$(mktemp -d /tmp/migrationbench_legacy_fig6.XXXXXX)
git archive 61c29dd -- \
  results/t-SNE/phate-tsne/analysis_Cr_Doped_600K_v2/Cr_Doped_Sb2Te3_600K_probe_features.csv \
  | tar -x -C "$tmpdir"

python analysis/legacy_fig6_phate_reproduction_20261101/reproduce_legacy_fig6.py \
  --features "$tmpdir/results/t-SNE/phate-tsne/analysis_Cr_Doped_600K_v2/Cr_Doped_Sb2Te3_600K_probe_features.csv" \
  --output-dir analysis/legacy_fig6_phate_reproduction_20261101/reproduced
```

The outputs are a separate PDF/PNG for panels a-d and an attachment-style 1x4
PDF/PNG composite. Numerical parity is reported as `RECONSTRUCTION` because
the original cluster Python/scikit-learn/PHATE environment is not recreated.
