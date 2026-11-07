# Figure registry schema

Required manifest fields:

- `figure_id`: immutable `FIG-NNN` identifier.
- `logical_figure`, `group_id`, `parent_group`, `sibling_ids`: stable semantic identity.
- `article_targets`: proposed main-text/SI locations.
- `reviewer_links`, `reviewer_status`: reviewer question, completed evidence, and required closure.
- `scientific_status`: publication, historical, reconstruction, screening, or gate status.
- `lock_status`: `draft`, `reviewed`, `locked`, `promoted`, or `retired`.
- `asset`, `asset_sha256`: canonical rendered asset.
- `authored_hashes`: hashes of editable fragments used to protect locked records.
- `source_export`, `evidence_candidates`, `legacy_v1_page`: provenance pointers.
- `evidence_metadata_source`: audit matrix or table-package manifest that
  supplied generated evidence candidates.
- `evidence_auto.tex`, `evidence.md`: generated evidence surfaces; do not hand-edit.

The per-figure `.tex` fragments are the authored LaTeX surface. The registry
manifest is machine-readable metadata. The book wrapper is derived and may be
regenerated at any time.
