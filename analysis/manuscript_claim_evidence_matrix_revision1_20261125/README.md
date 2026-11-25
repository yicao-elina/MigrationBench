# MigrationBench Revision 1 claim evidence inventory

This package adds a deterministic, source-pinned claim-to-evidence inventory for the submitted Revision 1 main text and SI. It does not edit the manuscript, change scientific conclusions, or replace the existing reviewer-question ledger.

## Pinned sources

- Current GitHub `main` source tree: `412ba9c5276a07a13026184cf62e6ff36588e05f`.
- Submitted source baseline: `334260cdda3869d55999081cd610b34bfba53257` (`paper/revision1` blobs match the frozen 2026-11-09 Overleaf export recorded by the release README).
- Registry SHA-256: `13948ef0963418bcbd40cbfe6cb103fd28c0a3e62744c60c2af622e02a7d20dc` for `analysis/migrationbench_asset_registry_20261025/migrationbench_asset_registry.sqlite`.
- The exact manuscript, reviewer-ledger, and registry input hashes are pinned in `build_matrix.py` and listed in `manifest.json`. The generator refuses to run if any pinned input differs. It opens SQLite read-only and requires `quick_check=ok` and zero foreign-key violations.

The authoritative prose sources are `paper/revision1/sn-article-final.tex` and `paper/revision1/sn-article-SI.tex`. Table cells are also extracted from included table TeX files and response-to-reviewers tables. The 22 rows in `paper/revision1/evidence_index/claim_evidence_matrix.tsv` remain reviewer questions and are copied as the distinct `REVIEWER_QUESTION` row type; that file is not overwritten or repurposed. Flat `paper/*.tex` compatibility copies are excluded.

## Contents and coverage

- `MIGRATIONBENCH_R1_CLAIM_EVIDENCE_MATRIX_20261125.tsv`: generated matrix, 639 rows with unique IDs.
- `build_matrix.py`: Python 3 standard-library generator; stable ordering and field order.
- `schema.json`: TSV field definitions and enumerated claim/status values.
- `manifest.json`: pinned source commit, input hashes, row counts, and output hash.

Rows comprise 338 main/SI sentence or semicolon-separated assertion segments, 30 figure-panel rows, 22 table-object rows, 227 numeric table-cell rows, and 22 reviewer-question rows. The matrix contains 13 `TRACEABLE`, 313 `PARTIAL`, and 313 `MISSING` rows overall. For the main/SI assertion-segment denominator alone, the counts are 0 `TRACEABLE`, 32 `PARTIAL`, and 306 `MISSING`.

These are **artifact-lineage classifications, not a rate of scientifically validated claims**. The prose segmenter is rule-based: it splits at sentence punctuation and semicolons. Coordinated scientific assertions without punctuation may remain grouped in one row; human scientific atomization is still needed before treating each prose segment as an independently validated claim. Numeric table-cell rows are emitted for cells containing numeric tokens; the inventory does not claim to enumerate nonnumeric headers or labels as scientific assertions.

## Rebuild and byte check

From the repository root, Python 3 is the only runtime requirement:

```sh
python3 analysis/manuscript_claim_evidence_matrix_revision1_20261125/build_matrix.py \
  --output /tmp/MIGRATIONBENCH_R1_CLAIM_EVIDENCE_MATRIX_rebuilt.tsv
cmp analysis/manuscript_claim_evidence_matrix_revision1_20261125/MIGRATIONBENCH_R1_CLAIM_EVIDENCE_MATRIX_20261125.tsv \
  /tmp/MIGRATIONBENCH_R1_CLAIM_EVIDENCE_MATRIX_rebuilt.tsv
sha256sum /tmp/MIGRATIONBENCH_R1_CLAIM_EVIDENCE_MATRIX_rebuilt.tsv
```

The pinned rebuild was compared byte-for-byte with the audited TSV before this package was prepared. Expected SHA-256:

`8373598837d3525a0702c2be85954ad2335dadd29cb2878b3aff6bc0389072e0`

The default output path is the checked-in TSV beside the generator. Use `--output` to validate without overwriting it. The matrix's row ordering is stable: main prose, SI prose, SQL registry rows ordered by section/item/panel, table-cell extraction in fixed source order, then reviewer-ledger rows in source order. IDs are deterministic from source context and row order; duplicate IDs or missing required source/status fields stop generation.

## Row types and classification

- `NUMERIC_ASSERTION`: prose segment containing a numeric token.
- `METHOD_OR_SCOPE_ASSERTION`: nonnumeric prose segment matching method/scope vocabulary.
- `SCIENTIFIC_ASSERTION`: other extracted prose segment.
- `FIGURE_PANEL` / `TABLE_OBJECT`: one row per `rev1_results_index` entry, with exact item/panel and registered pointers/status. Captions are taken from the matching submitted TeX environment where available.
- `TABLE_CELL`: one row per numeric TeX table cell discovered in inline or included table sources; a table-level registry association does not prove a cell-to-source-row join.
- `REVIEWER_QUESTION`: copied from the existing 22-row reviewer ledger and counted separately from manuscript assertions.

The prose classifier is lexical and does not judge scientific meaning. Explicit LaTeX figure/table refs can connect a prose row to a registry item; citation-only rows are marked partial; no exact local pointer yields missing. No filename/scientific-similarity matching is performed.

## Lineage status semantics

- `TRACEABLE`: registry row has an asset identity, passing reproduction status, and generator pointer without a recorded manual-layout or known-issue exception. This says the output path is indexed; it does not establish scientific validity.
- `PARTIAL`: an explicit registry/artifact route exists, but the prose-to-result key, table-cell binding, source identity, reproduction, scope, exception, known issue, or manual layout remains unresolved. This includes explicit figure references without a claim-specific result row.
- `MISSING`: no exact evidence pointer was resolved in the inspected registry/source set, or the reviewer ledger explicitly records a provenance/validation gap.

The matrix preserves existing reproduction/registry statuses and states the gap reason per row. Known issue or manual presentation status remains visible; no result is upgraded from its current evidence. For interpretation of a scientific result, consult the referenced source files, registry, and paper-specific audit; do not treat this inventory as an acceptance decision.
