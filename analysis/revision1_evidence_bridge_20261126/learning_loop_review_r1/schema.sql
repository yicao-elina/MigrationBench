PRAGMA application_id = 1296183605;
PRAGMA user_version = 1;

CREATE TABLE snapshot_metadata (metadata_key TEXT PRIMARY KEY, metadata_value TEXT NOT NULL);

-- One row per audited evidence family. These are review summaries, not failure events.
CREATE TABLE failure_mode_review (
    failure_mode_id TEXT PRIMARY KEY,
    evidence_family TEXT NOT NULL,
    observed_signal TEXT NOT NULL,
    queryable_sources TEXT NOT NULL,
    lineage_status TEXT NOT NULL,
    repeatability_assessment TEXT NOT NULL,
    next_round_decision_and_gain TEXT NOT NULL,
    recommended_closure TEXT NOT NULL,
    grain TEXT NOT NULL CHECK (grain='AUDIT_SUMMARY_NOT_FAILURE_EVENT'),
    benefit_status TEXT NOT NULL CHECK (benefit_status='NOT_ESTABLISHED_IN_AUDIT'),
    raw_source_row_json TEXT NOT NULL
);

CREATE INDEX idx_failure_mode_review_family ON failure_mode_review(evidence_family);
