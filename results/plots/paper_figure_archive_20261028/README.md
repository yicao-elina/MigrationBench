# Paper and certified figure archive

This directory is the organized local copy of the SQL-first reproducibility
archive staged on 2026-10-28. It preserves the paper-facing figure exports,
their derived data tables, chart manifests, and the reproduction index. The
archive contains the historical figure packages that were already selected or
audited; it is not a claim that every candidate is scientifically accepted.

## Layout

- `figures/`: PDF/PNG/SVG exports grouped by stable chart ID.
- `data/`: the exact tables copied into the archive for each chart.
- `provenance/`: chart-level manifests and evidence matrices.
- `scripts/REBUILD_SCRIPT_POINTERS.tsv`: exact original renderer paths or
  command-level pointers. R05 has a self-contained renderer in its own
  neighboring bundle; historical packages remain pointer-only where the
  source script was not safely materialized from the File Provider checkout.
- `index/FIGURE_REPRODUCTION_INDEX.tsv`: one row per chart, including status,
  source asset, data files, manifest, and original processing command.
- `index/FIGURE_CANDIDATES.tsv` and `index/release_selection.tsv`: candidate
  and selection boundaries; unselected candidates remain visible.
- `index/SOURCE_POINTERS.tsv`: pointers for large or external sources not
  copied into the repository.

The historical packages intentionally keep statuses such as
`NEEDS_AUDIT`, `FINITE_DIAGNOSTIC`, `PARTIAL`, and
`COMPLETE_WITH_CAVEATS`. These labels are part of the archive and must not be
flattened into a single accepted-publication state.

The exact current Scratch-600K R05 release is in the neighboring
`cr_three_path_fixed_vs_self_scratch600k_20261101_R05/` bundle and is linked
to the dated SQLite snapshot under `../registry/`.

Canva originals were not found as local source files in this archive. Any
future Canva export should be added as a new chart bundle with the design URL,
export hash, source-data link, and a rebuild status; an image-only Canva
export must remain `EXTERNAL_POINTER_ONLY` until its data/script lineage is
recorded.
