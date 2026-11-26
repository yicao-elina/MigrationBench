PRAGMA foreign_keys=ON;
CREATE TABLE snapshot_metadata (
  metadata_key TEXT PRIMARY KEY,
  metadata_value TEXT NOT NULL
);
CREATE TABLE claim_evidence_row (
  matrix_id TEXT PRIMARY KEY,
  claim_type TEXT NOT NULL,
  source_document TEXT NOT NULL,
  source_location TEXT NOT NULL,
  source_object_id TEXT NOT NULL,
  assertion_or_panel TEXT NOT NULL,
  exact_source_quote_or_value TEXT NOT NULL,
  numeric_tokens TEXT NOT NULL,
  referenced_labels TEXT NOT NULL,
  sql_item_id TEXT NOT NULL,
  sql_panel TEXT NOT NULL,
  registry_content TEXT NOT NULL,
  artifact_pointer TEXT NOT NULL,
  artifact_sha256 TEXT NOT NULL,
  generator_or_query TEXT NOT NULL,
  registry_status TEXT NOT NULL,
  reproduction_status TEXT NOT NULL,
  lineage_status TEXT NOT NULL,
  gap_reason TEXT NOT NULL,
  source_sha256 TEXT NOT NULL,
  CHECK (claim_type IN ('NUMERIC_ASSERTION','SCIENTIFIC_ASSERTION','METHOD_OR_SCOPE_ASSERTION','FIGURE_PANEL','TABLE_OBJECT','TABLE_CELL','REVIEWER_QUESTION')),
  CHECK (lineage_status IN ('TRACEABLE','PARTIAL','MISSING'))
);
CREATE INDEX idx_claim_evidence_status ON claim_evidence_row(lineage_status, claim_type);
CREATE INDEX idx_claim_evidence_source ON claim_evidence_row(source_document, source_location);
CREATE INDEX idx_claim_evidence_object ON claim_evidence_row(source_object_id, sql_item_id, sql_panel);
PRAGMA user_version=1;
