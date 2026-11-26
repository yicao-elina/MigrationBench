-- Additive manuscript-source registry. R1 claim rows and R2 Fig. 4 crosswalk remain unchanged.
PRAGMA foreign_keys = ON;

CREATE TABLE manuscript_source_version (
    source_version_id TEXT PRIMARY KEY,
    authority_status TEXT NOT NULL CHECK (authority_status IN (
        'SUBMITTED_RELEASE_CANONICAL',
        'CLOSED_UNMERGED_CANDIDATE',
        'LIVE_OVERLEAF_NOT_VERIFIED'
    )),
    git_commit TEXT,
    pull_request INTEGER,
    pull_request_state TEXT,
    source_capture_state TEXT NOT NULL,
    source_note TEXT NOT NULL
);

CREATE TABLE manuscript_source_file (
    source_version_id TEXT NOT NULL,
    repository_path TEXT NOT NULL,
    file_role TEXT NOT NULL,
    byte_count INTEGER NOT NULL CHECK (byte_count >= 0),
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
    file_note TEXT NOT NULL,
    PRIMARY KEY (source_version_id, repository_path),
    FOREIGN KEY (source_version_id) REFERENCES manuscript_source_version(source_version_id)
);

CREATE TABLE manuscript_ledger_delta (
    claim_id TEXT PRIMARY KEY,
    delta_type TEXT NOT NULL CHECK (delta_type IN ('CANDIDATE_ONLY','SUBMITTED_ONLY','ALTERED')),
    submitted_status TEXT,
    candidate_status TEXT,
    changed_fields TEXT NOT NULL,
    submitted_row_json TEXT,
    candidate_row_json TEXT,
    review_boundary TEXT NOT NULL
);

-- Full PR #10 file reconciliation: distinguish the closed PR lifecycle from
-- the fact that its changes later became ancestors of current main.
CREATE TABLE pr_file_reconciliation (
    pull_request INTEGER NOT NULL,
    pr_head_commit TEXT NOT NULL,
    source_path TEXT NOT NULL,
    pr_blob_oid TEXT NOT NULL,
    pr_byte_count INTEGER NOT NULL CHECK (pr_byte_count >= 0),
    pr_sha256 TEXT NOT NULL CHECK (length(pr_sha256) = 64),
    current_path TEXT,
    current_blob_oid TEXT,
    current_byte_count INTEGER CHECK (current_byte_count >= 0),
    current_sha256 TEXT,
    reconciliation_status TEXT NOT NULL CHECK (reconciliation_status IN (
        'SAME_PATH_IDENTICAL',
        'PRESERVED_CANDIDATE_IDENTICAL',
        'MISSING_UNRESOLVED',
        'CONTENT_CHANGED_UNRESOLVED'
    )),
    reconciliation_note TEXT NOT NULL,
    PRIMARY KEY (pull_request, source_path)
);

CREATE INDEX manuscript_source_file_by_version ON manuscript_source_file(source_version_id, file_role);
CREATE INDEX manuscript_ledger_delta_by_type ON manuscript_ledger_delta(delta_type, claim_id);
CREATE INDEX pr_file_reconciliation_by_status ON pr_file_reconciliation(reconciliation_status, source_path);

PRAGMA user_version = 3;
