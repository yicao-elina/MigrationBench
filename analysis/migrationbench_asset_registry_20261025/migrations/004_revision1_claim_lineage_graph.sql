-- Additive claim/panel/cell -> result -> run -> artifact lineage graph.
-- Apply only to a copy of the registry; keep source snapshots immutable.
-- This migration creates schema only. It deliberately inserts no scientific
-- claims, run mappings, or evidence links.
PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

INSERT OR IGNORE INTO metadata (key, value)
VALUES ('revision1_claim_lineage_graph',
        '004 (2026-11-25, additive schema only; no claim/result mappings asserted)');

CREATE TABLE IF NOT EXISTS manuscript_sources (
  manuscript_source_id TEXT PRIMARY KEY,
  document_role TEXT NOT NULL CHECK (document_role IN
    ('MAIN', 'SUPPLEMENTARY', 'RESPONSE', 'APPENDIX', 'OTHER')),
  source_path TEXT NOT NULL,
  source_sha256 TEXT NOT NULL CHECK (
    length(source_sha256) = 64 AND source_sha256 NOT GLOB '*[^0-9a-f]*'),
  revision_label TEXT NOT NULL,
  authority_status TEXT NOT NULL CHECK (authority_status IN
    ('SUBMITTED_SOURCE', 'CURRENT_DRAFT', 'HISTORICAL', 'UNVERIFIED', 'SUPERSEDED')),
  line_numbering TEXT NOT NULL DEFAULT '1-based source lines',
  captured_at TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT '',
  UNIQUE (source_path, source_sha256, revision_label)
);

CREATE TABLE IF NOT EXISTS manuscript_claims (
  claim_id TEXT PRIMARY KEY,
  manuscript_source_id TEXT NOT NULL REFERENCES manuscript_sources(manuscript_source_id),
  claim_kind TEXT NOT NULL CHECK (claim_kind IN
    ('NARRATIVE', 'NUMERIC', 'METHOD', 'INTERPRETATION', 'LIMITATION', 'REVIEWER_RESPONSE')),
  section_label TEXT NOT NULL DEFAULT '',
  exact_text TEXT NOT NULL,
  line_start INTEGER NOT NULL CHECK (line_start > 0),
  line_end INTEGER NOT NULL CHECK (line_end >= line_start),
  claim_review_status TEXT NOT NULL DEFAULT 'UNREVIEWED' CHECK (claim_review_status IN
    ('UNREVIEWED', 'AUTHOR_REVIEW', 'HUMAN_REVIEWED', 'SUPERSEDED')),
  evidence_status TEXT NOT NULL DEFAULT 'NOT_LINKED' CHECK (evidence_status IN
    ('TRACEABLE', 'PARTIAL', 'NOT_LINKED', 'NOT_RUN', 'MISSING', 'HOLD', 'CONFLICT')),
  scientific_status TEXT NOT NULL DEFAULT 'PROVENANCE_PARTIAL' CHECK (scientific_status IN
    ('PROVENANCE_PARTIAL', 'PROVISIONAL', 'SUPPORTED', 'CONTRADICTED', 'NOT_ASSESSED', 'EXCLUDED')),
  reviewer_note TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  UNIQUE (manuscript_source_id, line_start, line_end, claim_id)
);

CREATE TABLE IF NOT EXISTS manuscript_objects (
  manuscript_object_id TEXT PRIMARY KEY,
  manuscript_source_id TEXT NOT NULL REFERENCES manuscript_sources(manuscript_source_id),
  parent_object_id TEXT REFERENCES manuscript_objects(manuscript_object_id),
  object_kind TEXT NOT NULL CHECK (object_kind IN
    ('FIGURE', 'TABLE', 'PANEL', 'CELL', 'CAPTION', 'EQUATION', 'TEXT_SEGMENT', 'OTHER')),
  object_key TEXT NOT NULL,
  label TEXT NOT NULL DEFAULT '',
  registry_item_id TEXT,
  registry_panel_id TEXT,
  cell_key TEXT NOT NULL DEFAULT '',
  source_path TEXT NOT NULL DEFAULT '',
  source_sha256 TEXT NOT NULL DEFAULT '' CHECK (source_sha256 = '' OR
    (length(source_sha256) = 64 AND source_sha256 NOT GLOB '*[^0-9a-f]*')),
  evidence_status TEXT NOT NULL DEFAULT 'NOT_LINKED' CHECK (evidence_status IN
    ('TRACEABLE', 'PARTIAL', 'NOT_LINKED', 'NOT_RUN', 'MISSING', 'HOLD', 'CONFLICT', 'CAPTION_MISMATCH')),
  scientific_status TEXT NOT NULL DEFAULT 'PROVENANCE_PARTIAL' CHECK (scientific_status IN
    ('PROVENANCE_PARTIAL', 'PROVISIONAL', 'SUPPORTED', 'CONTRADICTED', 'NOT_ASSESSED', 'EXCLUDED')),
  notes TEXT NOT NULL DEFAULT '',
  CHECK ((registry_item_id IS NULL AND registry_panel_id IS NULL) OR
         (registry_item_id IS NOT NULL AND registry_panel_id IS NOT NULL)),
  FOREIGN KEY (registry_item_id, registry_panel_id)
    REFERENCES rev1_results_index(item_id, panel),
  UNIQUE (manuscript_source_id, object_key)
);

CREATE TABLE IF NOT EXISTS claim_object_links (
  claim_id TEXT NOT NULL REFERENCES manuscript_claims(claim_id),
  manuscript_object_id TEXT NOT NULL REFERENCES manuscript_objects(manuscript_object_id),
  relation TEXT NOT NULL CHECK (relation IN
    ('ASSERTED_BY', 'QUANTIFIED_IN', 'ILLUSTRATED_IN', 'QUALIFIED_BY', 'RESPONDED_IN')),
  evidence_status TEXT NOT NULL CHECK (evidence_status IN
    ('TRACEABLE', 'PARTIAL', 'NOT_LINKED', 'NOT_RUN', 'MISSING', 'HOLD', 'CONFLICT')),
  note TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (claim_id, manuscript_object_id, relation)
);

CREATE TABLE IF NOT EXISTS lineage_processes (
  process_id TEXT PRIMARY KEY,
  process_key TEXT NOT NULL UNIQUE,
  process_kind TEXT NOT NULL CHECK (process_kind IN
    ('TRAINING', 'SIMULATION', 'SCORING', 'POSTPROCESSING', 'PLOTTING', 'LAYOUT', 'IMPORT', 'OTHER')),
  repository_commit TEXT NOT NULL DEFAULT '',
  script_path TEXT NOT NULL DEFAULT '',
  script_sha256 TEXT NOT NULL DEFAULT '' CHECK (script_sha256 = '' OR
    (length(script_sha256) = 64 AND script_sha256 NOT GLOB '*[^0-9a-f]*')),
  command_text TEXT NOT NULL DEFAULT '',
  environment_id TEXT NOT NULL DEFAULT '',
  container_digest TEXT NOT NULL DEFAULT '',
  software_versions TEXT NOT NULL DEFAULT '',
  reproducibility_status TEXT NOT NULL DEFAULT 'NOT_ASSESSED' CHECK (reproducibility_status IN
    ('PASS', 'PASS_KNOWN_ISSUE', 'PARTIAL', 'FAIL', 'NOT_RUN', 'NOT_ASSESSED')),
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS lineage_runs (
  run_id TEXT PRIMARY KEY,
  run_key TEXT NOT NULL UNIQUE,
  campaign_id TEXT REFERENCES campaigns(campaign_id),
  run_kind TEXT NOT NULL CHECK (run_kind IN
    ('TRAINING', 'STATIC_EVALUATION', 'MD', 'NEB', 'DFT', 'SHAP', 'POSTPROCESSING', 'OTHER')),
  protocol_key TEXT NOT NULL DEFAULT '',
  model_id TEXT REFERENCES models(model_id),
  model_checkpoint_sha256 TEXT NOT NULL DEFAULT '' CHECK (model_checkpoint_sha256 = '' OR
    (length(model_checkpoint_sha256) = 64 AND model_checkpoint_sha256 NOT GLOB '*[^0-9a-f]*')),
  seed TEXT NOT NULL DEFAULT '',
  sample_key TEXT NOT NULL DEFAULT '',
  attempt_number INTEGER NOT NULL DEFAULT 0 CHECK (attempt_number >= 0),
  parent_run_id TEXT REFERENCES lineage_runs(run_id),
  scheduler_job_id TEXT NOT NULL DEFAULT '',
  scheduler_status TEXT NOT NULL DEFAULT 'UNKNOWN' CHECK (scheduler_status IN
    ('UNKNOWN', 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', 'NOT_SUBMITTED')),
  run_status TEXT NOT NULL CHECK (run_status IN
    ('PLANNED', 'NOT_RUN', 'RUNNING', 'COMPLETED', 'FAILED', 'PARTIAL', 'HELD', 'CANCELLED', 'UNKNOWN')),
  scientific_status TEXT NOT NULL DEFAULT 'PROVENANCE_PARTIAL' CHECK (scientific_status IN
    ('PROVENANCE_PARTIAL', 'PROVISIONAL', 'SUPPORTED', 'REJECTED', 'NOT_ASSESSED', 'EXCLUDED')),
  started_at TEXT,
  finished_at TEXT,
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS lineage_results (
  result_id TEXT PRIMARY KEY,
  result_key TEXT NOT NULL UNIQUE,
  result_kind TEXT NOT NULL CHECK (result_kind IN
    ('SCALAR', 'CELL', 'SERIES', 'PROFILE', 'SUMMARY', 'CLASSIFICATION', 'FIGURE_DATA', 'OTHER')),
  source_row_key TEXT NOT NULL DEFAULT '',
  result_grain TEXT NOT NULL DEFAULT '',
  task_id TEXT REFERENCES tasks(task_id),
  system_id TEXT REFERENCES systems(system_id),
  model_id TEXT REFERENCES models(model_id),
  model_checkpoint_sha256 TEXT NOT NULL DEFAULT '' CHECK (model_checkpoint_sha256 = '' OR
    (length(model_checkpoint_sha256) = 64 AND model_checkpoint_sha256 NOT GLOB '*[^0-9a-f]*')),
  protocol_key TEXT NOT NULL DEFAULT '',
  value_json TEXT NOT NULL DEFAULT '',
  unit TEXT NOT NULL DEFAULT '',
  evidence_status TEXT NOT NULL DEFAULT 'PARTIAL' CHECK (evidence_status IN
    ('TRACEABLE', 'PARTIAL', 'NOT_LINKED', 'NOT_RUN', 'MISSING', 'HOLD', 'CONFLICT')),
  scientific_status TEXT NOT NULL DEFAULT 'PROVENANCE_PARTIAL' CHECK (scientific_status IN
    ('PROVENANCE_PARTIAL', 'PROVISIONAL', 'SUPPORTED', 'REJECTED', 'NOT_ASSESSED', 'EXCLUDED')),
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS claim_result_links (
  claim_id TEXT NOT NULL REFERENCES manuscript_claims(claim_id),
  result_id TEXT NOT NULL REFERENCES lineage_results(result_id),
  relation TEXT NOT NULL CHECK (relation IN
    ('SUPPORTED_BY', 'QUALIFIED_BY', 'CONTRADICTED_BY', 'DERIVED_FROM', 'CONTEXT_FOR')),
  evidence_status TEXT NOT NULL CHECK (evidence_status IN
    ('TRACEABLE', 'PARTIAL', 'NOT_LINKED', 'NOT_RUN', 'MISSING', 'HOLD', 'CONFLICT')),
  note TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (claim_id, result_id, relation)
);

CREATE TABLE IF NOT EXISTS object_result_links (
  manuscript_object_id TEXT NOT NULL REFERENCES manuscript_objects(manuscript_object_id),
  result_id TEXT NOT NULL REFERENCES lineage_results(result_id),
  relation TEXT NOT NULL CHECK (relation IN
    ('PLOTTED_FROM', 'TABULATED_FROM', 'AGGREGATED_FROM', 'DERIVED_FROM', 'COMPARES_WITH')),
  evidence_status TEXT NOT NULL CHECK (evidence_status IN
    ('TRACEABLE', 'PARTIAL', 'NOT_LINKED', 'NOT_RUN', 'MISSING', 'HOLD', 'CONFLICT')),
  note TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (manuscript_object_id, result_id, relation)
);

CREATE TABLE IF NOT EXISTS result_run_links (
  result_id TEXT NOT NULL REFERENCES lineage_results(result_id),
  run_id TEXT NOT NULL REFERENCES lineage_runs(run_id),
  relation TEXT NOT NULL CHECK (relation IN
    ('PRODUCED_BY', 'AGGREGATED_FROM', 'VALIDATED_AGAINST', 'REPRODUCED_BY')),
  evidence_status TEXT NOT NULL CHECK (evidence_status IN
    ('TRACEABLE', 'PARTIAL', 'NOT_LINKED', 'NOT_RUN', 'MISSING', 'HOLD', 'CONFLICT')),
  note TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (result_id, run_id, relation)
);

CREATE TABLE IF NOT EXISTS lineage_artifacts (
  artifact_id TEXT PRIMARY KEY,
  artifact_key TEXT NOT NULL UNIQUE,
  artifact_role TEXT NOT NULL CHECK (artifact_role IN
    ('MANUSCRIPT_SOURCE', 'INPUT', 'OUTPUT', 'RESTART', 'CHECKPOINT', 'LOG', 'CONFIG', 'METRIC', 'DERIVED_DATA', 'FIGURE_EXPORT', 'TABLE_EXPORT', 'MANIFEST', 'OTHER')),
  source_path TEXT NOT NULL,
  path_scope TEXT NOT NULL CHECK (path_scope IN
    ('REPOSITORY_RELATIVE', 'SCRATCH16', 'DATA', 'SCRATCH4', 'EXTERNAL_URI', 'GIT_BLOB', 'OTHER')),
  sha256 TEXT NOT NULL DEFAULT '' CHECK (sha256 = '' OR
    (length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*')),
  byte_size INTEGER CHECK (byte_size IS NULL OR byte_size >= 0),
  mime_type TEXT NOT NULL DEFAULT '',
  availability_status TEXT NOT NULL CHECK (availability_status IN
    ('PRESENT', 'ARCHIVED', 'MISSING', 'INACCESSIBLE', 'POINTER_ONLY', 'NOT_CHECKED')),
  scientific_status TEXT NOT NULL DEFAULT 'PROVENANCE_PARTIAL' CHECK (scientific_status IN
    ('PROVENANCE_PARTIAL', 'PROVISIONAL', 'SUPPORTED', 'NOT_ASSESSED', 'EXCLUDED')),
  captured_at TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT '',
  CHECK (availability_status NOT IN ('PRESENT', 'ARCHIVED') OR length(sha256) = 64)
);

CREATE TABLE IF NOT EXISTS result_artifact_links (
  result_id TEXT NOT NULL REFERENCES lineage_results(result_id),
  artifact_id TEXT NOT NULL REFERENCES lineage_artifacts(artifact_id),
  artifact_role TEXT NOT NULL CHECK (artifact_role IN
    ('SOURCE_ROW', 'INPUT', 'DERIVED_OUTPUT', 'PLOTTED_DATA', 'TABULATED_DATA', 'MANIFEST', 'OTHER')),
  relation TEXT NOT NULL DEFAULT 'DERIVED_FROM' CHECK (relation IN
    ('DERIVED_FROM', 'MATERIALIZED_AS', 'CITES', 'VALIDATED_AGAINST')),
  evidence_status TEXT NOT NULL CHECK (evidence_status IN
    ('TRACEABLE', 'PARTIAL', 'NOT_LINKED', 'NOT_RUN', 'MISSING', 'HOLD', 'CONFLICT')),
  note TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (result_id, artifact_id, artifact_role)
);

CREATE TABLE IF NOT EXISTS run_artifact_links (
  run_id TEXT NOT NULL REFERENCES lineage_runs(run_id),
  artifact_id TEXT NOT NULL REFERENCES lineage_artifacts(artifact_id),
  artifact_role TEXT NOT NULL CHECK (artifact_role IN
    ('INPUT', 'OUTPUT', 'RESTART', 'CHECKPOINT', 'LOG', 'CONFIG', 'METRIC', 'MANIFEST', 'OTHER')),
  evidence_status TEXT NOT NULL CHECK (evidence_status IN
    ('TRACEABLE', 'PARTIAL', 'NOT_LINKED', 'NOT_RUN', 'MISSING', 'HOLD', 'CONFLICT')),
  note TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (run_id, artifact_id, artifact_role)
);

CREATE TABLE IF NOT EXISTS run_process_links (
  run_id TEXT NOT NULL REFERENCES lineage_runs(run_id),
  process_id TEXT NOT NULL REFERENCES lineage_processes(process_id),
  process_role TEXT NOT NULL CHECK (process_role IN
    ('EXECUTED_WITH', 'GENERATED_BY', 'ANALYZED_BY', 'REPRODUCED_BY')),
  evidence_status TEXT NOT NULL CHECK (evidence_status IN
    ('TRACEABLE', 'PARTIAL', 'NOT_LINKED', 'NOT_RUN', 'MISSING', 'HOLD', 'CONFLICT')),
  note TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (run_id, process_id, process_role)
);

CREATE TABLE IF NOT EXISTS manuscript_object_issues (
  issue_id TEXT PRIMARY KEY,
  manuscript_object_id TEXT NOT NULL REFERENCES manuscript_objects(manuscript_object_id),
  issue_kind TEXT NOT NULL CHECK (issue_kind IN
    ('CAPTION_MISMATCH', 'SOURCE_RELEASE_MISMATCH', 'MISSING_EXPORT', 'SCIENTIFIC_STATUS_CONFLICT', 'OTHER')),
  issue_status TEXT NOT NULL CHECK (issue_status IN ('OPEN', 'IN_REVIEW', 'RESOLVED', 'WONT_FIX')),
  description TEXT NOT NULL,
  resolution_note TEXT NOT NULL DEFAULT '',
  opened_at TEXT NOT NULL,
  closed_at TEXT
);

-- Explicit event records for prospective active-learning/failure-mode audits.
-- These tables store hypotheses and observed outcomes separately; they do not
-- encode causal benefit as a default interpretation.
CREATE TABLE IF NOT EXISTS diagnostic_events (
  diagnostic_id TEXT PRIMARY KEY,
  diagnostic_key TEXT NOT NULL UNIQUE,
  run_id TEXT REFERENCES lineage_runs(run_id),
  failure_mode_key TEXT NOT NULL DEFAULT '',
  signal_key TEXT NOT NULL,
  signal_value_json TEXT NOT NULL DEFAULT '',
  detection_threshold_json TEXT NOT NULL DEFAULT '',
  detection_status TEXT NOT NULL CHECK (detection_status IN
    ('DETECTED', 'NOT_DETECTED', 'INDETERMINATE', 'NOT_TESTED')),
  replication_scope TEXT NOT NULL DEFAULT '',
  replication_status TEXT NOT NULL DEFAULT 'NOT_TESTED' CHECK (replication_status IN
    ('NOT_TESTED', 'STABLE', 'UNSTABLE', 'CONFLICT', 'INSUFFICIENT_DATA')),
  evidence_status TEXT NOT NULL DEFAULT 'PARTIAL' CHECK (evidence_status IN
    ('TRACEABLE', 'PARTIAL', 'NOT_LINKED', 'NOT_RUN', 'MISSING', 'HOLD', 'CONFLICT')),
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS active_learning_decisions (
  decision_id TEXT PRIMARY KEY,
  decision_key TEXT NOT NULL UNIQUE,
  diagnostic_id TEXT NOT NULL REFERENCES diagnostic_events(diagnostic_id),
  proposed_action TEXT NOT NULL,
  action_status TEXT NOT NULL CHECK (action_status IN
    ('PROPOSED', 'EXECUTED', 'REJECTED', 'NOT_RUN', 'HELD')),
  next_run_id TEXT REFERENCES lineage_runs(run_id),
  baseline_run_id TEXT REFERENCES lineage_runs(run_id),
  outcome_result_id TEXT REFERENCES lineage_results(result_id),
  evaluation_protocol TEXT NOT NULL DEFAULT '',
  acceptance_gate TEXT NOT NULL DEFAULT '',
  measured_gain_value REAL,
  measured_gain_unit TEXT NOT NULL DEFAULT '',
  measured_gain_status TEXT NOT NULL DEFAULT 'NOT_EVALUATED' CHECK (measured_gain_status IN
    ('NOT_EVALUATED', 'PASS', 'FAIL', 'INCONCLUSIVE', 'NOT_COMPARABLE')),
  evidence_status TEXT NOT NULL DEFAULT 'PARTIAL' CHECK (evidence_status IN
    ('TRACEABLE', 'PARTIAL', 'NOT_LINKED', 'NOT_RUN', 'MISSING', 'HOLD', 'CONFLICT')),
  notes TEXT NOT NULL DEFAULT '',
  CHECK (measured_gain_status NOT IN ('PASS', 'FAIL') OR
         (baseline_run_id IS NOT NULL AND next_run_id IS NOT NULL AND outcome_result_id IS NOT NULL
          AND evaluation_protocol <> '' AND acceptance_gate <> ''))
);

CREATE INDEX IF NOT EXISTS idx_manuscript_claims_source_lines
  ON manuscript_claims(manuscript_source_id, line_start, line_end);
CREATE INDEX IF NOT EXISTS idx_manuscript_claims_status
  ON manuscript_claims(evidence_status, scientific_status, claim_review_status);
CREATE INDEX IF NOT EXISTS idx_manuscript_objects_registry
  ON manuscript_objects(registry_item_id, registry_panel_id);
CREATE INDEX IF NOT EXISTS idx_lineage_runs_checkpoint
  ON lineage_runs(model_checkpoint_sha256, run_kind, protocol_key);
CREATE INDEX IF NOT EXISTS idx_lineage_runs_status
  ON lineage_runs(run_status, scientific_status);
CREATE INDEX IF NOT EXISTS idx_lineage_results_row_key
  ON lineage_results(source_row_key, result_kind, protocol_key);
CREATE INDEX IF NOT EXISTS idx_lineage_results_checkpoint
  ON lineage_results(model_checkpoint_sha256, task_id, protocol_key);
CREATE INDEX IF NOT EXISTS idx_lineage_artifacts_hash
  ON lineage_artifacts(sha256);
CREATE INDEX IF NOT EXISTS idx_lineage_artifacts_path
  ON lineage_artifacts(path_scope, source_path);
CREATE INDEX IF NOT EXISTS idx_object_result_links_result
  ON object_result_links(result_id, evidence_status);
CREATE INDEX IF NOT EXISTS idx_result_artifact_links_artifact
  ON result_artifact_links(artifact_id, evidence_status);
CREATE INDEX IF NOT EXISTS idx_run_artifact_links_artifact
  ON run_artifact_links(artifact_id, evidence_status);
CREATE INDEX IF NOT EXISTS idx_manuscript_object_issues_status
  ON manuscript_object_issues(issue_status, issue_kind);
CREATE INDEX IF NOT EXISTS idx_diagnostic_events_mode_status
  ON diagnostic_events(failure_mode_key, detection_status, replication_status);
CREATE INDEX IF NOT EXISTS idx_diagnostic_events_signal
  ON diagnostic_events(signal_key, detection_status);
CREATE INDEX IF NOT EXISTS idx_active_learning_decisions_status
  ON active_learning_decisions(action_status, measured_gain_status, evidence_status);
CREATE INDEX IF NOT EXISTS idx_active_learning_decisions_runs
  ON active_learning_decisions(baseline_run_id, next_run_id);

CREATE VIEW IF NOT EXISTS v_r1_claim_lineage_status AS
SELECT c.claim_id, c.claim_kind, c.section_label, c.exact_text,
       c.line_start, c.line_end, c.claim_review_status, c.evidence_status,
       c.scientific_status, ms.document_role, ms.revision_label,
       COUNT(DISTINCT col.manuscript_object_id) AS linked_objects,
       COUNT(DISTINCT crl.result_id) AS linked_results,
       COUNT(DISTINCT rrl.run_id) AS linked_runs,
       COUNT(DISTINCT ral.artifact_id) AS linked_artifacts,
       CASE
         WHEN c.evidence_status IN ('NOT_LINKED', 'NOT_RUN', 'MISSING', 'HOLD', 'CONFLICT')
           THEN c.evidence_status
         WHEN COUNT(DISTINCT col.manuscript_object_id) = 0
           OR COUNT(DISTINCT crl.result_id) = 0
           OR COUNT(DISTINCT rrl.run_id) = 0
           OR COUNT(DISTINCT ral.artifact_id) = 0 THEN 'PARTIAL'
         ELSE c.evidence_status
       END AS closure_status
FROM manuscript_claims c
JOIN manuscript_sources ms USING (manuscript_source_id)
LEFT JOIN claim_object_links col USING (claim_id)
LEFT JOIN claim_result_links crl USING (claim_id)
LEFT JOIN result_run_links rrl ON rrl.result_id=crl.result_id
LEFT JOIN run_artifact_links ral ON ral.run_id=rrl.run_id
GROUP BY c.claim_id;

CREATE VIEW IF NOT EXISTS v_r1_object_lineage_status AS
SELECT o.manuscript_object_id, o.object_kind, o.object_key, o.label,
       o.registry_item_id, o.registry_panel_id, o.cell_key,
       o.evidence_status, o.scientific_status,
       COUNT(DISTINCT orl.result_id) AS linked_results,
       COUNT(DISTINCT rrl.run_id) AS linked_runs,
       COUNT(DISTINCT ral.artifact_id) AS linked_artifacts,
       COUNT(DISTINCT moi.issue_id) AS open_issue_count
FROM manuscript_objects o
LEFT JOIN object_result_links orl USING (manuscript_object_id)
LEFT JOIN result_run_links rrl USING (result_id)
LEFT JOIN run_artifact_links ral USING (run_id)
LEFT JOIN manuscript_object_issues moi
  ON moi.manuscript_object_id=o.manuscript_object_id AND moi.issue_status IN ('OPEN','IN_REVIEW')
GROUP BY o.manuscript_object_id;

COMMIT;
