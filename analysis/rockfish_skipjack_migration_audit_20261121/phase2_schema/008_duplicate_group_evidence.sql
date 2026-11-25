-- Explicit many-to-many provenance links for exact/likely duplicate decisions.
CREATE TABLE duplicate_group_evidence (
  duplicate_group_id INTEGER NOT NULL REFERENCES duplicate_group(duplicate_group_id) ON DELETE CASCADE,
  source_id INTEGER NOT NULL REFERENCES registry_source(source_id),
  evidence_role TEXT NOT NULL CHECK (
    evidence_role IN ('HASH_COMPARISON','INODE_EVIDENCE','CLEANUP_MANIFEST','PROVENANCE_MANIFEST','REVIEW')
  ),
  notes TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (duplicate_group_id, source_id, evidence_role)
);

CREATE INDEX idx_duplicate_group_evidence_source
  ON duplicate_group_evidence(source_id, evidence_role);
