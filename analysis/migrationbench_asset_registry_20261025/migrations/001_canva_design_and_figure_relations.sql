PRAGMA foreign_keys = ON;

INSERT OR REPLACE INTO metadata (key, value) VALUES
  ('registry_version', '2026-10-25+canva-2026-11-01'),
  ('canva_provenance_migration', '001_canva_design_and_figure_relations.sql'),
  ('canva_audit_checked_at', '2026-11-01'),
  ('canva_current_design_id', 'DAHZXpLtN6U');

CREATE TABLE IF NOT EXISTS canva_designs (
  design_id TEXT PRIMARY KEY,
  shortlink_url TEXT NOT NULL UNIQUE,
  full_url TEXT NOT NULL DEFAULT '',
  design_title TEXT NOT NULL DEFAULT '',
  page_count INTEGER,
  audit_status TEXT NOT NULL,
  checked_at TEXT NOT NULL,
  previous_design_id TEXT,
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS canva_pages (
  design_id TEXT NOT NULL REFERENCES canva_designs(design_id),
  page_number INTEGER NOT NULL,
  figure_id TEXT NOT NULL UNIQUE,
  canva_title TEXT NOT NULL,
  chart_id TEXT NOT NULL REFERENCES charts(chart_id),
  mapping_status TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (design_id, page_number)
);

CREATE TABLE IF NOT EXISTS chart_sources (
  chart_id TEXT NOT NULL REFERENCES charts(chart_id),
  source_role TEXT NOT NULL,
  source_path TEXT NOT NULL,
  source_sha256 TEXT,
  path_scope TEXT NOT NULL DEFAULT 'repository_relative',
  exists_at_build INTEGER NOT NULL DEFAULT 0,
  notes TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (chart_id, source_role, source_path)
);

CREATE INDEX IF NOT EXISTS idx_canva_pages_chart ON canva_pages(chart_id);
CREATE INDEX IF NOT EXISTS idx_chart_sources_chart ON chart_sources(chart_id);
CREATE INDEX IF NOT EXISTS idx_canva_designs_audit ON canva_designs(audit_status);

INSERT OR IGNORE INTO canva_designs
  (design_id, shortlink_url, full_url, design_title, page_count, audit_status, checked_at, previous_design_id, notes)
VALUES
  ('DAHZXpLtN6U', 'https://canva.link/sjom0wludt7vh9u',
   'https://www.canva.com/design/DAHZXpLtN6U/zo59T24lBORz8zehPGM3sg/edit?utm_content=DAHZXpLtN6U&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton',
   '', NULL, 'REAUTH_REQUIRED', '2026-11-01', 'DAHZabm1B1g',
   'Current shortlink resolved by HTTP redirect; page-level Canva API inspection was blocked by connector reauthentication.');

INSERT OR IGNORE INTO canva_designs
  (design_id, shortlink_url, full_url, design_title, page_count, audit_status, checked_at, previous_design_id, notes)
VALUES
  ('DAHZabm1B1g', 'https://canva.link/r5g3b6fiw3kgn0f',
   'https://www.canva.com/design/DAHZabm1B1g/',
   'MigrationBench Canva figure board', 10, 'CATALOGUED_BY_PR14', '2026-10-31', NULL,
   'Historical ten-page catalog with figure IDs CANVA-P01 through CANVA-P10.');

INSERT OR IGNORE INTO charts
  (chart_id, title, chart_kind, chart_status, asset_path, asset_mime, asset_exists, campaign_id, source_id, data_query, notes)
VALUES
  ('canva_p01_figure01', 'Canva P01 — MigrationBench configuration universe + model workflow', 'canva_page', 'PROVENANCE_SUPPORTED', 'docs/reproducibility/canva_figure_provenance/R06_clean_raw_pca/figures/figure01_task_conditioned_universe.pdf', 'application/pdf', 1, 'paper_figures_canva_20261031', 'src_paper_figures_pr14', 'SELECT * FROM canva_pages WHERE figure_id = ''CANVA-P01'';', 'Composite reconstruction; workflow schematic and universe panel have separate source builders.'),
  ('canva_p02_figure02', 'Canva P02 — Do foundation models help to improve MLIP?', 'canva_page', 'SCHEMATIC_ONLY', '', 'application/pdf', 0, 'paper_figures_canva_20261031', 'src_paper_figures_pr14', 'SELECT * FROM canva_pages WHERE figure_id = ''CANVA-P02'';', 'Schematic-only pointer; no numerical result or raw data claim.'),
  ('canva_p03_figure03', 'Canva P03 — RMSE', 'canva_page', 'PUBLICATION_PROVISIONAL', '', 'application/pdf', 0, 'paper_figures_canva_20261031', 'src_paper_figures_pr14', 'SELECT * FROM canva_pages WHERE figure_id = ''CANVA-P03'';', 'Derived parity data and builder are vendored; exact downloaded PDF remains pointer-only.'),
  ('canva_p04_figure04', 'Canva P04 — Frame-level error distributions', 'canva_page', 'PUBLICATION_PROVISIONAL', '', 'application/pdf', 0, 'paper_figures_canva_20261031', 'src_paper_figures_pr14', 'SELECT * FROM canva_pages WHERE figure_id = ''CANVA-P04'';', 'Per-frame data and builder are vendored; exact downloaded PDF remains pointer-only.'),
  ('canva_p05_figure05', 'Canva P05 — MD (1ns)', 'canva_page', 'PROVISIONAL_RAW_TRAJECTORIES_EXTERNAL', 'docs/reproducibility/canva_figure_provenance/R03_typography18pt_final/figures/figure06_matched_md_diagnostics.pdf', 'application/pdf', 1, 'paper_figures_canva_20261031', 'src_paper_figures_pr14', 'SELECT * FROM canva_pages WHERE figure_id = ''CANVA-P05'';', 'Compact MD tables and composite builder are vendored; raw trajectories remain external.'),
  ('canva_p06_figure06', 'Canva P06 — NEB Cr', 'canva_page', 'ENDPOINT_ONLY_MLIP_PROXY_RECONSTRUCTION', '', 'application/pdf', 0, 'paper_figures_canva_20261031', 'src_paper_figures_pr14', 'SELECT * FROM canva_pages WHERE figure_id = ''CANVA-P06'';', 'External source tables and script pointer; self-relaxed rows are endpoint-only MLIP proxies.'),
  ('canva_p07_figure07', 'Canva P07 — NEB Cu Li Na', 'canva_page', 'RECONSTRUCTION_QE_SCF_WARNING', '', 'application/pdf', 0, 'paper_figures_canva_20261031', 'src_paper_figures_pr14', 'SELECT * FROM canva_pages WHERE figure_id = ''CANVA-P07'';', 'Legacy foundation page; QE references carry SCF-warning boundaries.'),
  ('canva_p08_figure08', 'Canva P08 — NEB more foundation model', 'canva_page', 'RECONSTRUCTION_QE_SCF_WARNING', '', 'application/pdf', 0, 'paper_figures_canva_20261031', 'src_paper_figures_pr14', 'SELECT * FROM canva_pages WHERE figure_id = ''CANVA-P08'';', 'Foundation 4x3 page; QE references carry SCF-warning boundaries.'),
  ('canva_p09_figure09', 'Canva P09 — NEB interlayer sliding', 'canva_page', 'RECONSTRUCTION_DFT_SCF_WARNING_ENDPOINT_ONLY_PROXY', '', 'application/pdf', 0, 'paper_figures_canva_20261031', 'src_paper_figures_pr14', 'SELECT * FROM canva_pages WHERE figure_id = ''CANVA-P09'';', 'Download copy is intact; local staging integrity split remains recorded.'),
  ('canva_p10_figure10', 'Canva P10 — NEB interlayer sliding more foundation model', 'canva_page', 'RECONSTRUCTION_DFT_SCF_WARNING_CLASS_SD', '', 'application/pdf', 0, 'paper_figures_canva_20261031', 'src_paper_figures_pr14', 'SELECT * FROM canva_pages WHERE figure_id = ''CANVA-P10'';', 'Remote Rockfish inputs remain pointer-only; class SD is retained as a boundary.');

INSERT OR IGNORE INTO canva_pages
  (design_id, page_number, figure_id, canva_title, chart_id, mapping_status, notes)
VALUES
  ('DAHZabm1B1g', 1, 'CANVA-P01', 'MigrationBench configuration universe + model workflow', 'canva_p01_figure01', 'CATALOGUED_BY_PR14', 'Historical catalog row; current design page identity unverified.'),
  ('DAHZabm1B1g', 2, 'CANVA-P02', 'Do foundation models help to improve MLIP?', 'canva_p02_figure02', 'CATALOGUED_BY_PR14', 'Historical catalog row; current design page identity unverified.'),
  ('DAHZabm1B1g', 3, 'CANVA-P03', 'RMSE', 'canva_p03_figure03', 'CATALOGUED_BY_PR14', 'Historical catalog row; current design page identity unverified.'),
  ('DAHZabm1B1g', 4, 'CANVA-P04', 'Frame-level error distributions with all train/test AIMD frames', 'canva_p04_figure04', 'CATALOGUED_BY_PR14', 'Historical catalog row; current design page identity unverified.'),
  ('DAHZabm1B1g', 5, 'CANVA-P05', 'MD (1ns)', 'canva_p05_figure05', 'CATALOGUED_BY_PR14', 'Historical catalog row; current design page identity unverified.'),
  ('DAHZabm1B1g', 6, 'CANVA-P06', 'NEB - Cr', 'canva_p06_figure06', 'CATALOGUED_BY_PR14', 'Historical catalog row; current design page identity unverified.'),
  ('DAHZabm1B1g', 7, 'CANVA-P07', 'NEB - Cu Li Na', 'canva_p07_figure07', 'CATALOGUED_BY_PR14', 'Historical catalog row; current design page identity unverified.'),
  ('DAHZabm1B1g', 8, 'CANVA-P08', 'NEB more foundation model', 'canva_p08_figure08', 'CATALOGUED_BY_PR14', 'Historical catalog row; current design page identity unverified.'),
  ('DAHZabm1B1g', 9, 'CANVA-P09', 'NEB - Interlayer sliding', 'canva_p09_figure09', 'CATALOGUED_BY_PR14', 'Historical catalog row; current design page identity unverified.'),
  ('DAHZabm1B1g', 10, 'CANVA-P10', 'NEB interlayer sliding more foundation model', 'canva_p10_figure10', 'CATALOGUED_BY_PR14', 'Historical catalog row; current design page identity unverified.');

INSERT OR IGNORE INTO chart_links (chart_id, entity_type, entity_id, role, notes)
SELECT 'canva_p' || printf('%02d', n) || '_figure' || printf('%02d', n), 'canva_design', 'DAHZabm1B1g', 'historical_design', 'Historical ten-page Canva catalog from PR14.'
FROM (SELECT 1 AS n UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9 UNION ALL SELECT 10);

INSERT OR IGNORE INTO chart_links (chart_id, entity_type, entity_id, role, notes)
SELECT 'canva_p' || printf('%02d', n) || '_figure' || printf('%02d', n), 'canva_page', 'CANVA-P' || printf('%02d', n), 'page', 'Stable figure ID from the historical Canva catalog.'
FROM (SELECT 1 AS n UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9 UNION ALL SELECT 10);

INSERT OR IGNORE INTO chart_links (chart_id, entity_type, entity_id, role, notes)
SELECT 'canva_p' || printf('%02d', n) || '_figure' || printf('%02d', n), 'campaign', 'paper_figures_canva_20261031', 'catalog_campaign', 'Canva figure provenance catalog audit campaign.'
FROM (SELECT 1 AS n UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9 UNION ALL SELECT 10);

INSERT OR IGNORE INTO chart_data_files (chart_id, file_id, display_name, local_path, file_format, file_exists, notes)
VALUES
  ('canva_p01_figure01', 'figure_index', 'Canva figure index row', 'figures/publication/canva/CANVA_FIGURE_INDEX.tsv', 'tsv', 1, 'Canonical repository-facing figure index.'),
  ('canva_p02_figure02', 'figure_index', 'Canva figure index row', 'figures/publication/canva/CANVA_FIGURE_INDEX.tsv', 'tsv', 1, 'Schematic page; numerical data not claimed.'),
  ('canva_p03_figure03', 'parity_summary', 'Vendored parity summary', 'docs/reproducibility/canva_figure_provenance/data/figure02/parity_summary.csv', 'csv', 1, 'Derived parity table.'),
  ('canva_p04_figure04', 'parity_summary', 'Vendored per-frame parity data', 'docs/reproducibility/canva_figure_provenance/data/figure02', 'directory', 1, 'Per-frame CSV/NPZ inputs are vendored.'),
  ('canva_p05_figure05', 'md_figure_data', 'Vendored MD figure data', 'docs/reproducibility/canva_figure_provenance/data/figure06/MD15_FIGURE_DATA_R06G.tsv', 'tsv', 1, 'Compact MD analysis table; raw trajectories external.'),
  ('canva_p06_figure06', 'external_source_tables', 'External Cr NEB source tables', 'external:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/cr_interlayer_sliding_grouped_20261025/three_path_r02/source_tables_remote', 'directory', 0, 'Pointer-only.'),
  ('canva_p07_figure07', 'external_source_tables', 'External foundation 3x3 profiles', 'external:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/archive/legacy_figure_products_20261024/data/foundation_class_3x3_R03', 'directory', 0, 'Pointer-only.'),
  ('canva_p08_figure08', 'external_source_tables', 'External foundation 4x3 profiles', 'external:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/archive/legacy_figure_products_20261024/data/foundation_class_4x3_R03', 'directory', 0, 'Pointer-only.'),
  ('canva_p09_figure09', 'external_source_tables', 'External Cr sliding source tables', 'external:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/cr_interlayer_sliding_grouped_20261025/source_tables', 'directory', 0, 'Pointer-only; Downloads copy is the integrity authority.'),
  ('canva_p10_figure10', 'external_source_tables', 'External foundation sliding inputs', 'external:/scratch16/pclancy3/yi/foundation_interlayer_sliding_10model_20261024/inputs/dft_bpath2', 'directory', 0, 'Pointer-only Rockfish input.');

INSERT OR IGNORE INTO chart_sources
  (chart_id, source_role, source_path, source_sha256, path_scope, exists_at_build, notes)
VALUES
  ('canva_p01_figure01', 'figure_export', 'docs/reproducibility/canva_figure_provenance/R06_clean_raw_pca/figures/figure01_task_conditioned_universe.pdf', 'b1d8f6cd1ea6fbaa1b0d0ab00ccec321ff5b5fedfdeef362233ec829039d81d6', 'repository_relative', 1, 'Vendored publication export.'),
  ('canva_p01_figure01', 'reproduction_script', 'docs/reproducibility/canva_figure_provenance/scripts/build_atomic_migration_figures.py', NULL, 'repository_relative', 1, 'Composite builder.'),
  ('canva_p01_figure01', 'derived_data', 'docs/reproducibility/canva_figure_provenance/data/figure01', NULL, 'repository_relative', 1, 'Vendored derived points.'),
  ('canva_p01_figure01', 'provenance_manifest', 'docs/reproducibility/canva_figure_provenance/R06_clean_raw_pca/manifests/ATOMIC_MIGRATION_FIGURE_MANIFEST.json', NULL, 'repository_relative', 1, 'Manifest.'),
  ('canva_p02_figure02', 'figure_export', 'external:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/migrationbench_training_design_figure_20261024/migrationbench_full_pipeline_two_distributions_three_models_final.pdf', '68e2edc5834268705ede1bb510b46963fc63452cf333c318ef0198c7924be7db', 'external_local', 0, 'Pointer-only schematic export.'),
  ('canva_p02_figure02', 'reproduction_script', 'external:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/migrationbench_training_design_figure_20261024/make_editable_drawio.py', NULL, 'external_local', 0, 'Pointer-only builder.'),
  ('canva_p03_figure03', 'figure_export', 'external:/Users/alina/Downloads/fig_parity_train_test_common_2x4.pdf', 'b9f4cd35229705460b8a9710fc238181bc0bacf1bde3cc8dda27ba30e8111322', 'external_download', 0, 'Download export hash from PR14 audit.'),
  ('canva_p03_figure03', 'reproduction_script', 'docs/reproducibility/canva_figure_provenance/scripts/build_figure02_latest.py', NULL, 'repository_relative', 1, 'Vendored builder.'),
  ('canva_p03_figure03', 'derived_data', 'docs/reproducibility/canva_figure_provenance/data/figure02', NULL, 'repository_relative', 1, 'Vendored parity inputs.'),
  ('canva_p03_figure03', 'provenance_manifest', 'docs/reproducibility/canva_figure_provenance/gallery/boxplot_cluster_reference_v04/outputs/figure02_temperature_reference_layout_v16_manifest.json', NULL, 'repository_relative', 1, 'Manifest.'),
  ('canva_p04_figure04', 'figure_export', 'external:/Users/alina/Downloads/fig_rmse_frame_distributions.pdf', '545c23f5de52c017e1ccd2f33b7a361fb36dab57e2f912f6f8dc95a6ba0da4b8', 'external_download', 0, 'Separate Canva/download export.'),
  ('canva_p04_figure04', 'reproduction_script', 'docs/reproducibility/canva_figure_provenance/scripts/build_figure02_latest.py --frame-only', NULL, 'repository_relative', 1, 'Vendored frame builder.'),
  ('canva_p04_figure04', 'derived_data', 'docs/reproducibility/canva_figure_provenance/data/figure02', NULL, 'repository_relative', 1, 'Vendored per-frame inputs.'),
  ('canva_p05_figure05', 'figure_export', 'docs/reproducibility/canva_figure_provenance/R03_typography18pt_final/figures/figure06_matched_md_diagnostics.pdf', 'd60fad62316ad3764e95b1b129be1d05e6922cceb065407583d3c1034d958c89', 'repository_relative', 1, 'Vendored compact figure export.'),
  ('canva_p05_figure05', 'reproduction_script', 'docs/reproducibility/canva_figure_provenance/scripts/build_atomic_migration_figures.py', NULL, 'repository_relative', 1, 'Composite builder.'),
  ('canva_p05_figure05', 'derived_data', 'docs/reproducibility/canva_figure_provenance/data/figure06', NULL, 'repository_relative', 1, 'Vendored compact MD tables.'),
  ('canva_p05_figure05', 'provenance_manifest', 'docs/reproducibility/canva_figure_provenance/R03_typography18pt_final/manifests/ATOMIC_MIGRATION_FIGURE_MANIFEST.json', NULL, 'repository_relative', 1, 'Manifest.'),
  ('canva_p06_figure06', 'figure_export', 'external:/Users/alina/Downloads/cr_three_path_fixed_vs_self_2x3_R02.pdf', '8a04cfd9549432811262008cf728aac7758129a62e038c03f701b5c9fb24e313', 'external_download', 0, 'Pointer-only download export.'),
  ('canva_p06_figure06', 'reproduction_script', 'external:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/cr_interlayer_sliding_grouped_20261025/make_cr_three_path_2x3_R02.py', NULL, 'external_local', 0, 'Pointer-only builder.'),
  ('canva_p07_figure07', 'figure_export', 'external:/Users/alina/Downloads/foundation_model_class_3x3_R03.pdf', '59d6202451f9f94ff21f3fba9924e025001c0af6c406c2bdf2ca1fc947023f84', 'external_download', 0, 'Pointer-only legacy export.'),
  ('canva_p07_figure07', 'reproduction_script', 'external:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/archive/legacy_figure_products_20261024/build_foundation_class_3x3_r01.py', NULL, 'external_local', 0, 'Pointer-only builder.'),
  ('canva_p08_figure08', 'figure_export', 'external:/Users/alina/Downloads/foundation_self_relaxed_4x3_R03.pdf', '77c07914bcfa16cb7f7edf2166fd2fbdf05165a9e5be3dfc94cc51e5aaf6f6a8', 'external_download', 0, 'Pointer-only export.'),
  ('canva_p08_figure08', 'reproduction_script', 'external:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/archive/legacy_figure_products_20261024/build_foundation_class_4x3_r03.py', NULL, 'external_local', 0, 'Pointer-only builder.'),
  ('canva_p09_figure09', 'figure_export', 'external:/Users/alina/Downloads/cr_interlayer_sliding_grouped_2x2_plot_atlas_r04.pdf', 'b47c847559b67afdb96fd8d89eb6f4716e13a6d364041615fd53b69a2d85133c', 'external_download', 0, 'Downloads copy is the integrity authority.'),
  ('canva_p09_figure09', 'reproduction_script', 'external:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/cr_interlayer_sliding_grouped_20261025/scripts/build_cr_grouped_r04.py', NULL, 'external_local', 0, 'Pointer-only builder.'),
  ('canva_p10_figure10', 'figure_export', 'external:/Users/alina/Downloads/interlayer_sliding_10model_grouped_2x2_plot_atlas_r04.pdf', 'cab018579c293179fe5e1c5269c8709a6cc49d44287861a842373e881e79f610', 'external_download', 0, 'Pointer-only export.'),
  ('canva_p10_figure10', 'reproduction_script', 'external:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/foundation_interlayer_sliding_10model_20261024/scripts/plot_interlayer_grouped_profile_atlas_r01.py', NULL, 'external_local', 0, 'Pointer-only builder.'),
  ('canva_p10_figure10', 'derived_data', 'external:/scratch16/pclancy3/yi/foundation_interlayer_sliding_10model_20261024/inputs/dft_bpath2', NULL, 'external_rockfish', 0, 'Pointer-only Rockfish input.');

INSERT OR IGNORE INTO chart_provenance
  (chart_id, manifest_path, manifest_exists, output_sha256, input_file_hashes, processing_command, provenance_status, notes)
VALUES
  ('canva_p01_figure01', 'docs/reproducibility/canva_figure_provenance/R06_clean_raw_pca/manifests/ATOMIC_MIGRATION_FIGURE_MANIFEST.json', 1, NULL, '{}', 'python docs/reproducibility/canva_figure_provenance/scripts/build_atomic_migration_figures.py', 'PROVENANCE_SUPPORTED', 'Current page-level Canva identity is unverified; historical catalog mapping retained.'),
  ('canva_p02_figure02', 'external:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/migrationbench_training_design_figure_20261024/literature_and_design_notes.md', 0, NULL, '{}', 'external pointer', 'SCHEMATIC_ONLY', 'No numerical claim.'),
  ('canva_p03_figure03', 'docs/reproducibility/canva_figure_provenance/gallery/boxplot_cluster_reference_v04/outputs/figure02_temperature_reference_layout_v16_manifest.json', 1, NULL, '{}', 'python docs/reproducibility/canva_figure_provenance/scripts/build_figure02_latest.py', 'PUBLICATION_PROVISIONAL', 'Exact Canva export is pointer-only.'),
  ('canva_p04_figure04', 'docs/reproducibility/canva_figure_provenance/gallery/boxplot_cluster_reference_v04/outputs/figure02_temperature_reference_layout_v16_manifest.json', 1, NULL, '{}', 'python docs/reproducibility/canva_figure_provenance/scripts/build_figure02_latest.py --frame-only', 'PUBLICATION_PROVISIONAL', 'Exact Canva export is pointer-only.'),
  ('canva_p05_figure05', 'docs/reproducibility/canva_figure_provenance/R03_typography18pt_final/manifests/ATOMIC_MIGRATION_FIGURE_MANIFEST.json', 1, NULL, '{}', 'python docs/reproducibility/canva_figure_provenance/scripts/build_atomic_migration_figures.py', 'PROVISIONAL_RAW_TRAJECTORIES_EXTERNAL', 'Raw trajectories remain external.'),
  ('canva_p06_figure06', 'external:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/cr_interlayer_sliding_grouped_20261025/results/cr_three_path_fixed_vs_self_2x3_R02.manifest.json', 0, NULL, '{}', 'external pointer', 'ENDPOINT_ONLY_MLIP_PROXY_RECONSTRUCTION', 'Self-relaxed rows are endpoint-only proxies.'),
  ('canva_p07_figure07', 'external:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/archive/legacy_figure_products_20261024/figure_data_manifest_foundation_class_3x3_R03.json', 0, NULL, '{}', 'external pointer', 'RECONSTRUCTION_QE_SCF_WARNING', 'Legacy/QE warning boundary retained.'),
  ('canva_p08_figure08', 'external:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/archive/legacy_figure_products_20261024/figure_data_manifest_foundation_class_4x3_R03.json', 0, NULL, '{}', 'external pointer', 'RECONSTRUCTION_QE_SCF_WARNING', 'QE warning boundary retained.'),
  ('canva_p09_figure09', 'external:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/cr_interlayer_sliding_grouped_20261025/manifests/FIGURE_PROVENANCE_CR_INTERLAYER_GROUPED_R04.json', 0, 'b47c847559b67afdb96fd8d89eb6f4716e13a6d364041615fd53b69a2d85133c', '{}', 'external pointer', 'RECONSTRUCTION_DFT_SCF_WARNING_ENDPOINT_ONLY_PROXY', 'Downloads PDF is integrity authority; local staging split retained.'),
  ('canva_p10_figure10', 'external:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/foundation_interlayer_sliding_10model_20261024/manifests/FIGURE_PROVENANCE_PLOT_ATLAS_GROUPED_R04.json', 0, 'cab018579c293179fe5e1c5269c8709a6cc49d44287861a842373e881e79f610', '{}', 'external pointer', 'RECONSTRUCTION_DFT_SCF_WARNING_CLASS_SD', 'Remote Rockfish inputs remain pointer-only.');
