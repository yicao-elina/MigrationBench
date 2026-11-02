# Paper figures/tables reproducibility release

This directory is the release entry point for the paper-facing figure, table,
Canva, plotting-script, data, and SQL provenance bundle. It is intentionally a
stacked review unit and does not target `main` directly.

## Branch topology

- Base: `codex/paper-figures-repro-base`, created from the latest `origin/main`.
- Release: `codex/paper-figures-repro-release`, reviewed against the base branch.
- Historical Canva catalog: PR #14 / `codex/canva-figure-provenance`.
- NEB seed-paired heatmap: PR #15; it is not included or modified here.

The release branch is a consolidated snapshot of the already-audited figure
and table work. It is not a claim that every plotted result is publication
accepted: `FIGURE_RELEASE_STATUS.tsv`, the Canva registries, manifests, and
independent reviews preserve `PROVISIONAL`, `RECONSTRUCTION`, `SCF_WARNING`,
`BLOCKED`, and `POINTER_ONLY` boundaries.

## Evidence chain

For every included figure/table, the intended chain is:

`output -> manifest/status -> reproduction script -> vendored derived data or external pointer -> SQL chart/table relation`

The canonical entry points are:

- [`FIGURE_RELEASE_STATUS.tsv`](../../../FIGURE_RELEASE_STATUS.tsv) — Figure 1–8 and Table 1–2 release state and blockers;
- [`figures/publication/canva/CANVA_FIGURE_INDEX.tsv`](../../../figures/publication/canva/CANVA_FIGURE_INDEX.tsv) — stable `CANVA-P01`–`CANVA-P10` page IDs and repository paths;
- [`docs/reproducibility/canva_figure_provenance/CANVA_FIGURE_REGISTRY.tsv`](../canva_figure_provenance/CANVA_FIGURE_REGISTRY.tsv) — full local/external path ledger;
- [`analysis/migrationbench_asset_registry_20261025/migrationbench_asset_registry.sqlite`](../../../analysis/migrationbench_asset_registry_20261025/migrationbench_asset_registry.sqlite) — chart/table entity registry;
- [`analysis/migrationbench_asset_registry_20261025/query_examples.sql`](../../../analysis/migrationbench_asset_registry_20261025/query_examples.sql) — read-only evidence queries;
- [`review/INDEPENDENT_REVIEW_FIG1_4.md`](../../../review/INDEPENDENT_REVIEW_FIG1_4.md) and [`review/INDEPENDENT_REVIEW_FIG5_8_TABLES.md`](../../../review/INDEPENDENT_REVIEW_FIG5_8_TABLES.md) — independent acceptance boundaries.

## Plotting contract

The bundle uses the recovered Plot Atlas contracts where the grammar matches:
`mlip_parity` for the R04 energy/force train/test parity panels and
`profile_ensemble` for ordered model/profile panels. The repository-local
builders remain the reproducibility entry points because they contain the
project-specific data joins, statuses, and manifests.

The release keeps editable SVG, archival PDF, and review PNG whenever the
figure is vendored. Large raw trajectories, model checkpoints, remote
Rockfish outputs, and unavailable Canva exports remain pointer-only with
recorded paths/hashes; they are not silently replaced by synthetic data.

## Rebuild and audit

Run from a checkout with the repository-relative bundle and any documented
external inputs materialized:

```bash
sqlite3 analysis/migrationbench_asset_registry_20261025/migrationbench_asset_registry.sqlite \
  < analysis/migrationbench_asset_registry_20261025/query_examples.sql > /dev/null
python3 docs/reproducibility/canva_figure_provenance/validate_catalog.py
git diff --check
```

The vendored Figure 1/2 and R04 compact-data rebuilders are clone-relative.
For pointer-only source families, provide the external audited repository
explicitly; no script depends on the maintainer's original absolute path:

```bash
export MIGRATIONBENCH_EXTERNAL_ROOT=/path/to/Wiley-MigrationBench
export PLOT_ATLAS_ROOT=/path/to/plot_library_recovered
```

`MIGRATIONBENCH_EXTERNAL_ROOT` is needed only for figures whose raw or
processed source remains external. `PLOT_ATLAS_ROOT` is needed for the
profile-ensemble builders; Figure 1A and the vendored parity builders do not
require that checkout. Missing external inputs fail closed and are not
replaced with synthetic observations.

The catalog validator can require external Downloads/Rockfish paths. An
`external_files_unavailable` result or a file-provider read failure is a
provenance gap, not evidence that a pointer-only figure is vendored.
