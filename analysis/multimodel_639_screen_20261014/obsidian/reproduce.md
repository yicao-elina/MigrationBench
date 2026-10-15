# Reproduce

## Local audit and merge

From the repository root:

```bash
python analysis/multimodel_639_screen_20261014/merge_and_audit.py \
  --manifest analysis/multimodel_639_screen_20261014/candidate_manifest_639.csv \
  --new-results analysis/multimodel_639_screen_20261014/inputs/new_analysis/mlip_candidate_results_all.csv \
  --from-scratch-results analysis/multimodel_639_screen_20261014/inputs/from_scratch_results_reconciled.csv \
  --frame-features analysis/multimodel_639_screen_20261014/inputs/frame_features.csv \
  --reference-embedding analysis/multimodel_639_screen_20261014/inputs/reference_embedding.csv \
  --output analysis/multimodel_639_screen_20261014/analysis
```

Render the Plot Atlas figure with the atlas source path available:

```bash
PYTHONPATH=/Users/alina/Documents/Codex/plot_library/src \
python analysis/multimodel_639_screen_20261014/plot_multimodel_overview.py \
  --analysis analysis/multimodel_639_screen_20261014/analysis \
  --frame-features analysis/multimodel_639_screen_20261014/inputs/frame_features.csv \
  --reference analysis/multimodel_639_screen_20261014/inputs/reference_embedding.csv \
  --manifest analysis/multimodel_639_screen_20261014/candidate_manifest_639.csv \
  --output analysis/multimodel_639_screen_20261014/figures
```

## Rockfish provenance

- production array: `31981441`, 32/32 tasks, exit code 0;
- initial collector: `31981595`, failed after writing the raw concatenated result because of a metadata suffix collision;
- corrected collector: `31986653`, completed with remote `analysis_manifest.json` status `VERIFIED`;
- remote root: `/scratch16/pclancy3/yi/mace_multimodel_639_screen_20261014`.

The GPU calculation was not rerun after the collector fix.
