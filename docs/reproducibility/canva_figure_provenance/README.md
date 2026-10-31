# Canva figure provenance catalog

Audit date: 2026-10-31 (America/Los_Angeles)

This catalog maps the ten pages of the Canva design resolved from
`https://canva.link/r5g3b6fiw3kgn0f` to the local figure exports, raw inputs,
processed tables, plotting/editing scripts, and provenance manifests. The
Canva short link resolved to design `DAHZabm1B1g`.

The primary page-level index is
[`CANVA_FIGURE_REGISTRY.tsv`](CANVA_FIGURE_REGISTRY.tsv). Each row keeps the
full local path instead of replacing an unavailable external artifact with a
plausible-looking repository-relative path. `DOWNLOAD_EVIDENCE.tsv` records
the recent PDF copies found in `/Users/alina/Downloads` and their SHA256
comparison with the staging outputs.

## What is and is not committed

This PR commits the reproducibility index only: paths, hashes, rebuild
commands, source manifests, and explicit scientific/status boundaries. Large
raw trajectories, remote Rockfish outputs, model checkpoints, and 1 ns
trajectory collections remain at their existing local or remote locations.
They are pointer-only here because copying them into Git would create a second
authority and would not make the remote paths available to another machine.

The analysis packages referenced by this index were found under
`/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis` during the
audit. Several are in the dirty local working tree or staging area rather
than in `origin/main`; the catalog therefore distinguishes:

- `local_source_root`: the current local artifact root used for this audit;
- `target_repo_path`: the intended path inside
  `/Users/alina/Project/migrationbench-paper-repro` when the package is
  integrated;
- `external_raw_path`: a Rockfish or Downloads path that is not vendored by
  this PR.

## Rebuild entry points

The commands below are the audited entry points. Run them from a checkout that
contains the referenced analysis package and has the external raw inputs
materialized.

| Canva page | Rebuild entry point |
| --- | --- |
| 1, configuration universe | `python analysis/migrationbench_universe_publication_20261026/run_structure_universe.py`; then `compose_comprehensive_task_map.py` and `compose_publication_universe.py` |
| 2, workflow schematic | `python analysis/migrationbench_training_design_figure_20261024/make_editable_drawio.py`; compile the recorded `.tex` source for the PDF export |
| 3–4, parity and frame RMSE | `python analysis/rmse_train_test_20261026/make_figures.py`; use `--frame-only` for the frame panel and `make_audit.py` for the audit tables |
| 5, 1 ns MD | `python analysis/md_1ns_updated_publication_figure_20261025/render_md15_figure_R06G.py` |
| 6, Cr NEB | `python analysis/cr_interlayer_sliding_grouped_20261025/make_cr_three_path_2x3_R02.py` |
| 7, Cu/Li/Na 3×3 | `python analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/archive/legacy_figure_products_20261024/build_foundation_class_3x3_r01.py` |
| 8, Cr/Cu/Li/Na 4×3 | `python analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/archive/legacy_figure_products_20261024/build_foundation_class_4x3_r03.py` |
| 9, Cr interlayer 2×2 | `python analysis/cr_interlayer_sliding_grouped_20261025/scripts/build_cr_grouped_r04.py` |
| 10, foundation interlayer 2×2 | `python analysis/foundation_interlayer_sliding_10model_20261024/scripts/plot_interlayer_grouped_profile_atlas_r01.py` |

## Status interpretation

`SOURCE_MATCH` means the exported PDF and the audited local source have the
same bytes. `VISUAL_MATCH_DIFFERENT_PDF_EXPORT` means the page title/layout
and data lineage match but the downloaded PDF was exported with different PDF
metadata or serialization. `RECONSTRUCTION` means the Canva page is a
composition or a reviewed rebuild rather than a single immutable source file.
`POINTER_ONLY` means the raw input is intentionally external. These are
reproducibility statuses, not scientific acceptance claims.

Important scientific boundaries retained in the index include endpoint-only
MLIP NEB proxies, screened QE paths marked `COMPLETED_BUT_SCF_WARNING`, the
legacy Scratch line in the Cr triptych, and the provisional status of the MD
package. Scheduler completion and a rendered PDF are not promoted to
scientific acceptance.

## Integrity note for the supplied CR R04 example

The Downloads copy
`/Users/alina/Downloads/cr_interlayer_sliding_grouped_2x2_plot_atlas_r04.pdf`
has SHA256
`b47c847559b67afdb96fd8d89eb6f4716e13a6d364041615fd53b69a2d85133c`, which
matches the hash recorded by
`FIGURE_PROVENANCE_CR_INTERLAYER_GROUPED_R04.json`. The same-named staging
file under the analysis directory has the expected byte count but is a
zero-filled/empty PDF at audit time. The staging copy is therefore not used as
the integrity authority; the Download copy and the source tables/manifests
are retained as the recoverable evidence.

## Atomic Migration blueprint review package

The blueprint-derived build and review handoff is contained in:

- [`BLUEPRINT_FIGURE_TODO.md`](BLUEPRINT_FIGURE_TODO.md): figure-by-figure
  source paths, additional plots, blockers, and execution order;
- [`DATA_AND_VISUAL_ACCEPTANCE_CRITERIA.md`](DATA_AND_VISUAL_ACCEPTANCE_CRITERIA.md):
  data, provenance, numerical, visual, and independent-signoff gates;
- [`INDEPENDENT_REVIEWER_BRIEF.md`](../../../INDEPENDENT_REVIEWER_BRIEF.md)
  and [`review/`](../../../review/): independent reviewer protocol and
  current Figure 1–8/Table 1–2 findings;
- [`CANVA_PLACEMENT_MAP.md`](../../../CANVA_PLACEMENT_MAP.md): current Canva
  page correspondence, new blueprint slots, and placeholder rules;
- [`FIGURE_RELEASE_STATUS.tsv`](../../../FIGURE_RELEASE_STATUS.tsv):
  conservative release state for each figure/table;
- [`manifests/ATOMIC_MIGRATION_FIGURE_MANIFEST.json`](manifests/ATOMIC_MIGRATION_FIGURE_MANIFEST.json):
  absolute source paths and source hashes;
- [`figures/atomic_migration_blueprint_canvas_board.svg`](figures/atomic_migration_blueprint_canvas_board.svg):
  completed local composition board for Canva import/review, with explicit
  warning and blocked placeholders.

The builder is [`scripts/build_atomic_migration_figures.py`](scripts/build_atomic_migration_figures.py).
It is additive and writes only inside this worktree. The current package is a
review package, not scientific acceptance; no external Canva page was
overwritten because the connector required reauthentication at build time.
