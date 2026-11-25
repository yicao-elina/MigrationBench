-- Typed figure registry and source/derived-file relationships for paper evidence.
CREATE TABLE paper_figure (
  figure_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES project(project_id),
  campaign_id TEXT REFERENCES campaign(campaign_id),
  source_campaign_key TEXT,
  title TEXT NOT NULL,
  chart_kind TEXT NOT NULL,
  chart_status TEXT NOT NULL,
  provenance_status TEXT NOT NULL,
  source_registry_source_id INTEGER REFERENCES registry_source(source_id),
  data_query TEXT NOT NULL DEFAULT '',
  processing_command TEXT NOT NULL DEFAULT '',
  scientific_boundary TEXT NOT NULL DEFAULT '',
  notes TEXT NOT NULL DEFAULT ''
);

CREATE INDEX idx_paper_figure_campaign ON paper_figure(campaign_id, figure_id);
CREATE INDEX idx_paper_figure_source ON paper_figure(source_registry_source_id, figure_id);

CREATE TABLE paper_figure_asset (
  figure_asset_id INTEGER PRIMARY KEY,
  figure_id TEXT NOT NULL REFERENCES paper_figure(figure_id),
  artifact_id INTEGER REFERENCES artifact(artifact_id),
  asset_role TEXT NOT NULL CHECK (
    asset_role IN (
      'FIGURE_EXPORT','DERIVED_DATA','PROVENANCE_MANIFEST','CLAIM_MATRIX',
      'SOURCE_ASSET','SOURCE_MANIFEST','REBUILD_METHOD','ARCHIVE_INDEX'
    )
  ),
  source_path TEXT NOT NULL,
  availability TEXT NOT NULL CHECK (
    availability IN ('PRESENT','MISSING','EXTERNAL_POINTER','NOT_MATERIALIZED')
  ),
  relationship TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT '',
  UNIQUE(figure_id, asset_role, source_path)
);

CREATE INDEX idx_paper_figure_asset_figure ON paper_figure_asset(figure_id, asset_role);
CREATE INDEX idx_paper_figure_asset_artifact ON paper_figure_asset(artifact_id);

CREATE TABLE paper_claim_source_observation (
  source_observation_id INTEGER PRIMARY KEY,
  source_id INTEGER NOT NULL REFERENCES registry_source(source_id),
  source_claim_key TEXT NOT NULL,
  claim_id TEXT NOT NULL REFERENCES paper_claim(claim_id),
  source_status TEXT NOT NULL,
  source_publication_action TEXT NOT NULL DEFAULT '',
  source_job_or_run TEXT NOT NULL DEFAULT '',
  source_target TEXT NOT NULL DEFAULT '',
  source_evidence_refs TEXT NOT NULL DEFAULT '[]',
  observed_utc TEXT NOT NULL,
  UNIQUE(source_id, source_claim_key)
);

CREATE INDEX idx_paper_claim_source_claim ON paper_claim_source_observation(claim_id, source_id);
