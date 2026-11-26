PRAGMA foreign_keys = ON;

CREATE TABLE release_snapshot (
    snapshot_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    campaign TEXT NOT NULL,
    source_commit TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    scientific_scope TEXT NOT NULL,
    created_utc TEXT NOT NULL
);

CREATE TABLE source_file (
    source_file_id INTEGER PRIMARY KEY,
    snapshot_id TEXT NOT NULL REFERENCES release_snapshot(snapshot_id),
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL CHECK(length(sha256)=64),
    byte_size INTEGER NOT NULL CHECK(byte_size >= 0),
    file_role TEXT NOT NULL,
    media_type TEXT,
    git_blob_oid TEXT,
    source_status TEXT NOT NULL DEFAULT 'CHECKED_IN_SOURCE',
    UNIQUE(snapshot_id, relative_path)
);

CREATE TABLE source_row (
    source_file_id INTEGER NOT NULL REFERENCES source_file(source_file_id),
    row_number INTEGER NOT NULL CHECK(row_number >= 2),
    row_json TEXT NOT NULL,
    PRIMARY KEY(source_file_id, row_number)
);

CREATE TABLE artifact_asset (
    artifact_id INTEGER PRIMARY KEY,
    snapshot_id TEXT NOT NULL REFERENCES release_snapshot(snapshot_id),
    logical_name TEXT NOT NULL,
    uri TEXT NOT NULL,
    sha256 TEXT CHECK(sha256 IS NULL OR length(sha256)=64),
    byte_size INTEGER CHECK(byte_size IS NULL OR byte_size >= 0),
    artifact_kind TEXT NOT NULL,
    scientific_status TEXT NOT NULL,
    source_file_id INTEGER REFERENCES source_file(source_file_id),
    UNIQUE(snapshot_id, logical_name, uri)
);

CREATE TABLE model_checkpoint (
    checkpoint_sha256 TEXT PRIMARY KEY CHECK(length(checkpoint_sha256)=64),
    canonical_model_id TEXT NOT NULL UNIQUE,
    model_label TEXT NOT NULL,
    model_family TEXT NOT NULL,
    training_seed INTEGER,
    model_uri TEXT NOT NULL,
    identity_status TEXT NOT NULL,
    source_md_seed_note TEXT,
    source_row_file_id INTEGER,
    source_row_number INTEGER,
    FOREIGN KEY(source_row_file_id, source_row_number)
      REFERENCES source_row(source_file_id, row_number)
);

CREATE TABLE evaluation_protocol (
    protocol_id TEXT PRIMARY KEY,
    protocol_name TEXT NOT NULL,
    calculation_type TEXT NOT NULL,
    protocol_status TEXT NOT NULL,
    analysis_manifest_sha256 TEXT,
    description TEXT NOT NULL,
    dft_validation_status TEXT NOT NULL
);

CREATE TABLE software_environment (
    environment_sha256 TEXT PRIMARY KEY CHECK(length(environment_sha256)=64),
    software_json TEXT NOT NULL,
    environment_status TEXT NOT NULL
);

CREATE TABLE md_run (
    run_id INTEGER PRIMARY KEY,
    source_label TEXT NOT NULL,
    training_seed INTEGER NOT NULL,
    md_velocity_seed INTEGER NOT NULL,
    run_key TEXT NOT NULL,
    source_model_sha256 TEXT NOT NULL REFERENCES model_checkpoint(checkpoint_sha256),
    input_sha256 TEXT NOT NULL CHECK(length(input_sha256)=64),
    source_root TEXT NOT NULL,
    host TEXT,
    slurm_job_id TEXT,
    slurm_array_task_id TEXT,
    terminal INTEGER,
    execution_status TEXT NOT NULL,
    stop_reason TEXT,
    onset_step INTEGER,
    onset_time_ps REAL,
    first_trigger TEXT,
    last_step INTEGER,
    last_time_ps REAL,
    max_force_eV_A REAL,
    max_temperature_K REAL,
    min_distance_A REAL,
    source_row_file_id INTEGER NOT NULL,
    source_row_number INTEGER NOT NULL,
    UNIQUE(source_label, training_seed, md_velocity_seed, run_key),
    FOREIGN KEY(source_row_file_id, source_row_number)
      REFERENCES source_row(source_file_id, row_number)
);

CREATE TABLE trajectory_frame (
    frame_id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES md_run(run_id),
    step INTEGER NOT NULL,
    time_ps REAL NOT NULL,
    source_window TEXT,
    candidate_group TEXT,
    source_snapshot_uri TEXT,
    source_snapshot_sha256 TEXT,
    frame_sha256 TEXT,
    source_row_file_id INTEGER NOT NULL,
    source_row_number INTEGER NOT NULL,
    UNIQUE(run_id, step, source_window),
    FOREIGN KEY(source_row_file_id, source_row_number)
      REFERENCES source_row(source_file_id, row_number)
);

CREATE TABLE candidate (
    candidate_id TEXT PRIMARY KEY,
    canonical_index INTEGER NOT NULL UNIQUE,
    frame_id INTEGER NOT NULL REFERENCES trajectory_frame(frame_id),
    candidate_group TEXT NOT NULL,
    evaluation_scope TEXT NOT NULL,
    source_md_checkpoint_sha256 TEXT NOT NULL REFERENCES model_checkpoint(checkpoint_sha256),
    prepared_structure_uri TEXT,
    prepared_structure_sha256 TEXT,
    prepared_file_sha256_status TEXT NOT NULL,
    candidate_status TEXT NOT NULL,
    dft_preparation_status TEXT,
    selection_role TEXT,
    selection_reason TEXT,
    min_distance_A REAL,
    source_geometry_gate INTEGER,
    source_row_file_id INTEGER NOT NULL,
    source_row_number INTEGER NOT NULL,
    FOREIGN KEY(source_row_file_id, source_row_number)
      REFERENCES source_row(source_file_id, row_number)
);

CREATE TABLE candidate_alias (
    namespace TEXT NOT NULL,
    alias TEXT NOT NULL,
    candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
    source_row_file_id INTEGER NOT NULL,
    source_row_number INTEGER NOT NULL,
    PRIMARY KEY(namespace, alias),
    FOREIGN KEY(source_row_file_id, source_row_number)
      REFERENCES source_row(source_file_id, row_number)
);

CREATE TABLE model_evaluation (
    evaluation_id INTEGER PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
    checkpoint_sha256 TEXT NOT NULL REFERENCES model_checkpoint(checkpoint_sha256),
    protocol_id TEXT NOT NULL REFERENCES evaluation_protocol(protocol_id),
    environment_sha256 TEXT REFERENCES software_environment(environment_sha256),
    execution_candidate_alias TEXT,
    input_uri TEXT,
    input_sha256 TEXT CHECK(input_sha256 IS NULL OR length(input_sha256)=64),
    natoms INTEGER,
    species_composition TEXT,
    pbc_json TEXT,
    cell_volume_A3 REAL,
    geometry_finite INTEGER,
    min_interatomic_distance_A REAL,
    evaluation_status TEXT NOT NULL,
    runtime_error TEXT,
    energy_eV REAL,
    energy_per_atom_eV REAL,
    forces_finite INTEGER,
    max_force_eV_A REAL,
    rms_force_eV_A REAL,
    p95_force_eV_A REAL,
    p99_force_eV_A REAL,
    top_force_species TEXT,
    top_force_atom_index INTEGER,
    scientific_status TEXT NOT NULL,
    source_row_file_id INTEGER NOT NULL,
    source_row_number INTEGER NOT NULL,
    UNIQUE(candidate_id, checkpoint_sha256, protocol_id),
    FOREIGN KEY(source_row_file_id, source_row_number)
      REFERENCES source_row(source_file_id, row_number)
);

CREATE TABLE selection_batch (
    batch_id TEXT PRIMARY KEY,
    selection_kind TEXT NOT NULL,
    batch_status TEXT NOT NULL,
    requested_count INTEGER,
    selected_count INTEGER,
    no_dft_submitted INTEGER,
    selection_manifest_file_id INTEGER REFERENCES source_file(source_file_id),
    selection_manifest_sha256 TEXT,
    boundary TEXT NOT NULL
);

CREATE TABLE selection_member (
    batch_id TEXT NOT NULL REFERENCES selection_batch(batch_id),
    candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
    rank_order INTEGER,
    selection_role TEXT,
    selection_basis TEXT,
    dft_status TEXT NOT NULL,
    source_row_file_id INTEGER NOT NULL,
    source_row_number INTEGER NOT NULL,
    PRIMARY KEY(batch_id, candidate_id),
    FOREIGN KEY(source_row_file_id, source_row_number)
      REFERENCES source_row(source_file_id, row_number)
);

CREATE TABLE campaign_status_evidence (
    status_evidence_id INTEGER PRIMARY KEY,
    scope TEXT NOT NULL,
    status_as_reported TEXT NOT NULL,
    time_scope TEXT NOT NULL,
    interpretation_boundary TEXT NOT NULL,
    source_file_id INTEGER NOT NULL REFERENCES source_file(source_file_id),
    UNIQUE(scope, status_as_reported, source_file_id)
);

CREATE TABLE figure (
    figure_id TEXT PRIMARY KEY,
    figure_file_id INTEGER REFERENCES source_file(source_file_id),
    manifest_file_id INTEGER REFERENCES source_file(source_file_id),
    figure_status TEXT NOT NULL,
    n_candidates INTEGER,
    n_model_rows INTEGER,
    n_models INTEGER,
    interpretation_boundary TEXT NOT NULL
);

CREATE TABLE figure_panel (
    figure_id TEXT NOT NULL REFERENCES figure(figure_id),
    panel_id TEXT NOT NULL,
    panel_kind TEXT NOT NULL,
    description TEXT NOT NULL,
    evidence_table_file_id INTEGER REFERENCES source_file(source_file_id),
    evidence_role TEXT NOT NULL,
    PRIMARY KEY(figure_id, panel_id)
);

CREATE TABLE figure_output (
    figure_id TEXT NOT NULL REFERENCES figure(figure_id),
    format TEXT NOT NULL,
    artifact_id INTEGER NOT NULL REFERENCES artifact_asset(artifact_id),
    PRIMARY KEY(figure_id, format)
);

CREATE TABLE claim (
    claim_id TEXT PRIMARY KEY,
    claim_text TEXT NOT NULL,
    scientific_status TEXT NOT NULL,
    boundary_or_gap TEXT NOT NULL,
    manuscript_location TEXT NOT NULL DEFAULT 'NOT_SPECIFIED_IN_CAPSULE',
    source_row_file_id INTEGER NOT NULL,
    source_row_number INTEGER NOT NULL,
    FOREIGN KEY(source_row_file_id, source_row_number)
      REFERENCES source_row(source_file_id, row_number)
);

CREATE TABLE claim_evidence (
    claim_id TEXT NOT NULL REFERENCES claim(claim_id),
    evidence_ref_verbatim TEXT NOT NULL,
    resolved_source_file_id INTEGER REFERENCES source_file(source_file_id),
    evidence_role TEXT NOT NULL,
    resolution_status TEXT NOT NULL,
    PRIMARY KEY(claim_id, evidence_ref_verbatim),
    FOREIGN KEY(resolved_source_file_id) REFERENCES source_file(source_file_id)
);

CREATE TABLE claim_figure_panel (
    claim_id TEXT NOT NULL REFERENCES claim(claim_id),
    figure_id TEXT NOT NULL,
    panel_id TEXT NOT NULL,
    evidence_role TEXT NOT NULL,
    PRIMARY KEY(claim_id, figure_id, panel_id),
    FOREIGN KEY(figure_id, panel_id) REFERENCES figure_panel(figure_id, panel_id)
);

CREATE VIEW v_candidate_trace AS
SELECT c.candidate_id, ca.namespace, ca.alias AS execution_alias,
       c.candidate_group, c.evaluation_scope, c.prepared_structure_uri,
       c.prepared_structure_sha256, c.prepared_file_sha256_status,
       f.step, f.time_ps, f.source_window, f.source_snapshot_uri,
       f.source_snapshot_sha256, r.run_key, r.source_label,
       r.training_seed, r.md_velocity_seed, r.execution_status AS md_status,
       r.onset_time_ps, r.first_trigger,
       r.source_model_sha256 AS source_md_checkpoint_sha256,
       r.source_root
FROM candidate c
JOIN trajectory_frame f USING(frame_id)
JOIN md_run r USING(run_id)
LEFT JOIN candidate_alias ca ON ca.candidate_id=c.candidate_id AND ca.namespace='execution';

CREATE VIEW v_candidate_evaluation_trace AS
SELECT t.*, e.checkpoint_sha256 AS evaluated_checkpoint_sha256,
       cp.canonical_model_id AS evaluated_model_id,
       e.protocol_id, e.evaluation_status, e.scientific_status AS evaluation_scientific_status,
       e.input_uri, e.input_sha256, e.energy_eV, e.energy_per_atom_eV,
       e.max_force_eV_A, e.rms_force_eV_A, e.p95_force_eV_A, e.p99_force_eV_A,
       e.environment_sha256
FROM v_candidate_trace t
JOIN model_evaluation e USING(candidate_id)
JOIN model_checkpoint cp ON cp.checkpoint_sha256=e.checkpoint_sha256;

CREATE VIEW v_claim_evidence_trace AS
SELECT cl.claim_id, cl.claim_text, cl.scientific_status AS claim_status,
       cl.boundary_or_gap, ce.evidence_ref_verbatim,
       sf.relative_path AS resolved_source_path, ce.resolution_status
FROM claim cl
LEFT JOIN claim_evidence ce USING(claim_id)
LEFT JOIN source_file sf ON sf.source_file_id=ce.resolved_source_file_id;

CREATE VIEW v_claim_figure_panel_trace AS
SELECT cl.claim_id, cl.claim_text, cl.scientific_status AS claim_status,
       cl.boundary_or_gap, cfp.figure_id, cfp.panel_id,
       fp.panel_kind, fp.description AS panel_description, cfp.evidence_role
FROM claim cl
JOIN claim_figure_panel cfp USING(claim_id)
JOIN figure_panel fp USING(figure_id, panel_id);
