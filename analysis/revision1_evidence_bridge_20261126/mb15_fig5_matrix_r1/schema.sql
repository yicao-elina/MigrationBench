PRAGMA application_id = 1296183605;
PRAGMA user_version = 1;

CREATE TABLE snapshot_metadata (metadata_key TEXT PRIMARY KEY, metadata_value TEXT NOT NULL);

-- One row per submitted Figure 5 model × task metric record (15 × 8 = 120).
CREATE TABLE fig5_matrix_cell (
    row_id TEXT NOT NULL,
    row_label TEXT NOT NULL,
    model_kind TEXT NOT NULL,
    training_seed TEXT,
    column_id TEXT NOT NULL,
    column_label TEXT NOT NULL,
    task_block TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    raw_value_eV REAL,
    predicted_value_eV REAL,
    error_eV REAL,
    absolute_error_eV REAL,
    n_available INTEGER,
    n_expected INTEGER,
    coverage_fraction REAL,
    aggregation TEXT,
    status_summary TEXT NOT NULL,
    source_artifact TEXT NOT NULL,
    model_sha256 TEXT CHECK (model_sha256 IS NULL OR (length(model_sha256)=64 AND lower(model_sha256)=model_sha256 AND model_sha256 NOT GLOB '*[^0-9a-f]*')),
    scientific_status TEXT NOT NULL,
    publication_status TEXT NOT NULL,
    reference_value_eV REAL,
    error_definition TEXT,
    color_value_eV REAL,
    color_vmin_eV REAL,
    color_vmax_eV REAL,
    color_transform TEXT,
    color_scale TEXT,
    family TEXT NOT NULL,
    strategy TEXT NOT NULL,
    dft_reference_eV REAL,
    source_status TEXT,
    matrix_status TEXT NOT NULL,
    dft_reference_display_eV REAL,
    correction_record_id TEXT,
    raw_source_row_json TEXT NOT NULL,
    PRIMARY KEY (row_id, column_id)
);

CREATE INDEX idx_fig5_matrix_status ON fig5_matrix_cell(publication_status, scientific_status);
CREATE INDEX idx_fig5_matrix_hash ON fig5_matrix_cell(model_sha256);
CREATE INDEX idx_fig5_matrix_task ON fig5_matrix_cell(column_id, family);
