-- Explicit links from source-backed paper claims to figure-level entities.
CREATE TABLE paper_claim_figure (
  claim_id TEXT NOT NULL REFERENCES paper_claim(claim_id),
  figure_id TEXT NOT NULL REFERENCES paper_figure(figure_id),
  source_id INTEGER NOT NULL REFERENCES registry_source(source_id),
  source_claim_key TEXT NOT NULL,
  relationship TEXT NOT NULL CHECK (
    relationship IN ('ILLUSTRATES','RECONSTRUCTION_OF','QUALIFIES','PANEL_CONTEXT')
  ),
  notes TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(claim_id, figure_id, relationship),
  UNIQUE(source_id, source_claim_key, figure_id, relationship)
);

CREATE INDEX idx_paper_claim_figure_figure ON paper_claim_figure(figure_id, claim_id);
