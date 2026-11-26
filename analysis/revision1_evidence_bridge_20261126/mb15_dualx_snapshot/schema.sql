PRAGMA application_id = 1296183605;
PRAGMA user_version = 1;

CREATE TABLE snapshot_metadata (
    metadata_key TEXT PRIMARY KEY,
    metadata_value TEXT NOT NULL
);

CREATE TABLE mb15_model_coverage (
    mb_model_id TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    training_seed_or_variant TEXT NOT NULL,
    md_status_r02 TEXT NOT NULL CHECK (md_status_r02 IN ('TARGET_REACHED_PROVISIONAL', 'TARGET_REACHED', 'SCIENTIFIC_FAILURE')),
    reason TEXT,
    last_step INTEGER CHECK (last_step IS NULL OR last_step >= 0),
    last_time_ps REAL CHECK (last_time_ps IS NULL OR last_time_ps >= 0),
    target_step INTEGER CHECK (target_step IS NULL OR target_step >= 0),
    raw_frames INTEGER CHECK (raw_frames IS NULL OR raw_frames >= 0),
    eligible_frames INTEGER CHECK (eligible_frames IS NULL OR eligible_frames >= 0),
    invalid_frames INTEGER CHECK (invalid_frames IS NULL OR invalid_frames >= 0),
    model_sha256 TEXT NOT NULL CHECK (length(model_sha256) = 64 AND lower(model_sha256) = model_sha256 AND model_sha256 NOT GLOB '*[^0-9a-f]*'),
    input_sha256 TEXT NOT NULL CHECK (length(input_sha256) = 64 AND lower(input_sha256) = input_sha256 AND input_sha256 NOT GLOB '*[^0-9a-f]*'),
    evidence_root TEXT NOT NULL,
    figure_row_present TEXT NOT NULL CHECK (figure_row_present IN ('YES', 'NO')),
    dualx_model_id_exact_sha TEXT,
    dualx_registry_exact_model_ids TEXT,
    dualx_r4_model_exact_ids TEXT,
    dualx_external_link_pass_rows INTEGER CHECK (dualx_external_link_pass_rows IS NULL OR dualx_external_link_pass_rows >= 0),
    dualx_primary_inference_runs INTEGER CHECK (dualx_primary_inference_runs IS NULL OR dualx_primary_inference_runs >= 0),
    dualx_primary_gradcam_runs INTEGER CHECK (dualx_primary_gradcam_runs IS NULL OR dualx_primary_gradcam_runs >= 0),
    mb_exact_static_candidate_rows INTEGER CHECK (mb_exact_static_candidate_rows IS NULL OR mb_exact_static_candidate_rows >= 0),
    mb_exact_neb_rows INTEGER CHECK (mb_exact_neb_rows IS NULL OR mb_exact_neb_rows >= 0),
    reuse_class TEXT NOT NULL,
    CHECK (figure_row_present != 'YES' OR mb_model_id != '')
);

CREATE INDEX idx_mb15_coverage_status ON mb15_model_coverage(md_status_r02, family);
CREATE INDEX idx_mb15_coverage_model_sha ON mb15_model_coverage(model_sha256);
