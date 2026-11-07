# Canva figure evidence package

This is the repository-facing entry point for the Canva figure catalog. Stable figure IDs are `CANVA-P01` through `CANVA-P10`. The page-level source, reproduction-script, data, manifest, and scientific-status mapping is in `CANVA_FIGURE_INDEX.tsv`.

The full provenance package remains under `docs/reproducibility/canva_figure_provenance/` because that directory contains the existing vendored derived data, manifests, review notes, and versioned figure builds. This `figures/publication/canva/` directory is intentionally a small canonical index rather than a second copy of 170 MB of figure inputs.

## Repository policy

- `VENDORED` means the figure output and the required derived data/script are in this repository.
- `VENDORED_PARTIAL` means a reproducible subset is vendored but the exact Canva export or a large raw input remains external.
- `POINTER_ONLY` means the catalog records the external source path, hash when available, script, and scientific boundary, but does not create a false local copy.
- Raw Rockfish trajectories, model checkpoints, and large external inputs remain pointer-only unless they are already committed compact evidence. This avoids creating a second source of truth.

## SQL entity relation

The registry database at `analysis/migrationbench_asset_registry_20261025/migrationbench_asset_registry.sqlite` now contains:

- `charts.chart_id` as the stable figure primary key;
- `canva_designs` for current and historical Canva designs;
- `canva_pages` for verified historical page mappings;
- `chart_links` for page/design/campaign/dataset/entity relations;
- `chart_sources` for figure exports, reproduction scripts, data roots, manifests, hashes, and vendoring status;
- `chart_data_files` and `chart_provenance` for source files, hashes, commands, and status.

Apply the additive migration with:

```bash
sqlite3 analysis/migrationbench_asset_registry_20261025/migrationbench_asset_registry.sqlite \
  < analysis/migrationbench_asset_registry_20261025/migrations/001_canva_design_and_figure_relations.sql
```

The current shortlink `https://canva.link/sjom0wludt7vh9u` resolves to design `DAHZXpLtN6U`, but the current design is marked `REAUTH_REQUIRED`; the ten page rows remain tied to the previously catalogued design `DAHZabm1B1g` until page-level verification is possible.
