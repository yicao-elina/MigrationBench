PRAGMA application_id = 1296183605;
PRAGMA user_version = 1;

CREATE TABLE snapshot_metadata (
    metadata_key TEXT PRIMARY KEY,
    metadata_value TEXT NOT NULL
);

-- One row per submitted R02 model row. This is not an attempt table.
CREATE TABLE model_attempt_audit (
    model_id TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    display_label TEXT NOT NULL,
    training_seed_or_variant TEXT NOT NULL,
    last_step INTEGER,
    last_time_ps REAL,
    target_step INTEGER,
    r02_status TEXT NOT NULL,
    r02_reason TEXT,
    raw_frames INTEGER,
    eligible_frames INTEGER,
    invalid_frames INTEGER,
    model_sha256 TEXT NOT NULL CHECK (length(model_sha256)=64 AND lower(model_sha256)=model_sha256 AND model_sha256 NOT GLOB '*[^0-9a-f]*'),
    input_sha256 TEXT NOT NULL CHECK (length(input_sha256)=64 AND lower(input_sha256)=input_sha256 AND input_sha256 NOT GLOB '*[^0-9a-f]*'),
    candidate_run_dir TEXT,
    run_dir_exists TEXT NOT NULL,
    smoke_dir TEXT,
    missing_links_raw TEXT NOT NULL,
    raw_source_row_json TEXT NOT NULL
);

CREATE TABLE evidence_artifact (
    artifact_id INTEGER PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES model_attempt_audit(model_id),
    evidence_role TEXT NOT NULL,
    evidence_path TEXT NOT NULL,
    observed_state TEXT NOT NULL CHECK (observed_state IN ('PRESENT','ABSENT','NOT_CHECKED','NOT_APPLICABLE')),
    reported_bytes INTEGER,
    observed_sha256 TEXT CHECK (observed_sha256 IS NULL OR (length(observed_sha256)=64 AND lower(observed_sha256)=observed_sha256 AND observed_sha256 NOT GLOB '*[^0-9a-f]*')),
    hash_match TEXT NOT NULL,
    source_note TEXT NOT NULL,
    UNIQUE(model_id, evidence_role, evidence_path)
);

CREATE TABLE scheduler_job_candidate (
    candidate_id INTEGER PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES model_attempt_audit(model_id),
    job_id TEXT NOT NULL,
    role_note TEXT NOT NULL,
    linkage_status TEXT NOT NULL CHECK (linkage_status='CANDIDATE_FROM_SCOPED_LOG'),
    source_evidence TEXT NOT NULL,
    UNIQUE(model_id, job_id, role_note)
);

CREATE TABLE trajectory_segment (
    segment_id INTEGER PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES model_attempt_audit(model_id),
    evidence_path TEXT NOT NULL,
    parsed_start_step INTEGER,
    reported_bytes INTEGER,
    sha256_status TEXT NOT NULL CHECK (sha256_status='NOT_RECORDED_IN_PRODUCER_MANIFEST'),
    UNIQUE(model_id, evidence_path)
);

CREATE TABLE missing_link (
    missing_link_id INTEGER PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES model_attempt_audit(model_id),
    gap_text TEXT NOT NULL,
    UNIQUE(model_id, gap_text)
);

CREATE INDEX idx_model_attempt_status ON model_attempt_audit(r02_status, family);
CREATE INDEX idx_evidence_role_state ON evidence_artifact(evidence_role, observed_state);
CREATE INDEX idx_candidate_job_model ON scheduler_job_candidate(model_id, job_id);
CREATE INDEX idx_missing_model ON missing_link(model_id);
