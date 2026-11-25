-- Idempotent figure-registry migration for the user-certified Cr R05 bundle.
-- Base snapshot: migrationbench_asset_registry_20261101_base_20261028.sqlite
-- This migration adds chart/catalog lineage only; it does not alter legacy
-- scientific coverage rows or upgrade diagnostic rows to accepted barriers.

BEGIN;

INSERT OR REPLACE INTO sources
  (source_id, title, source_path, source_kind, evidence_level, as_of_date, notes)
VALUES
  ('src_scratch600k_r05',
   'Scratch-600K Cr three-path R05 figure bundle',
   'results/plots/cr_three_path_fixed_vs_self_scratch600k_20261101_R05',
   'figure bundle plus campaign registry',
   'USER_CERTIFIED_RELEASE',
   '2026-11-01',
   'Certified R05 figure; source tables, endpoint manifests, remote pointers, and render manifest are bundled.'
  );

INSERT OR REPLACE INTO campaigns
  (campaign_id, title, campaign_path, source_id, campaign_date, status, notes)
VALUES
  ('scratch_600k_migration_20261101',
   'Scratch-600K-Lmax1 Cr three-path migration',
   'results/plots/cr_three_path_fixed_vs_self_scratch600k_20261101_R05/source/scratch_600k_migration_20261101',
   'src_scratch600k_r05',
   '2026-11-01',
   'COMPLETE_WITH_CAVEATS',
   'Seeds 123/234/345; 81 atoms × 10 images; fixed 9/9 finite; self-relaxed 7/9 converged primary evidence and 2 diagnostic-only cases.'
  );

INSERT OR REPLACE INTO datasets
  (dataset_id, display_name, data_layer, task_id, system_id, campaign_id,
   record_count, unit_of_record, scientific_status, dft_reference_status, notes)
VALUES
  ('scratch600k_cr81_fixed_r05',
   'Scratch-600K Cr81 fixed-path image profiles R05',
   'MLIP prediction',
   'fixed_path_neb',
   'cr81_migration',
   'scratch_600k_migration_20261101',
   90,
   'seed × path × image profile',
   'FINITE_DIAGNOSTIC',
   'COMPLETED_BUT_SCF_WARNING',
   'All 9 seed×path combinations are finite; fixed-path single-point profiles remain diagnostic rather than validated barriers.'
  ),
  ('scratch600k_cr81_self_r05',
   'Scratch-600K Cr81 endpoint-only self-relaxed profiles R05',
   'MLIP prediction',
   'self_relaxed_neb',
   'cr81_migration',
   'scratch_600k_migration_20261101',
   70,
   'accepted seed × path × image profile',
   'COMPLETE_WITH_CAVEATS',
   'COMPLETED_BUT_SCF_WARNING',
   '7/9 converged primary-evidence seed×path cases are plotted; seed123×within-QL and seed234×in-gap are retained as diagnostics and excluded from the main values.'
  );

INSERT OR REPLACE INTO charts
  (chart_id, title, chart_kind, chart_status, asset_path, asset_mime,
   asset_exists, campaign_id, source_id, data_query, notes)
VALUES
  ('chart_cr_three_path_scratch600k_r05_20261101',
   'Cr three-path fixed vs self-relaxed — Scratch-600K R05',
   'composite_2x3',
   'USER_CERTIFIED_RELEASE',
   'results/plots/cr_three_path_fixed_vs_self_scratch600k_20261101_R05/figures/cr_three_path_fixed_vs_self_2x3_R05.png',
   'image/png',
   1,
   'scratch_600k_migration_20261101',
   'src_scratch600k_r05',
   'SELECT dataset_id, display_name, record_count, scientific_status, dft_reference_status, notes FROM datasets WHERE dataset_id IN (''scratch600k_cr81_fixed_r05'', ''scratch600k_cr81_self_r05'') ORDER BY dataset_id',
   'User-certified 2026-11-01. Legend matches page 13. Fixed and endpoint-only self-relaxed observables remain separate; self-relaxed curves are MLIP proxies, not DFT MEP barriers.'
  );

INSERT OR REPLACE INTO chart_panels
  (chart_id, panel_id, panel_label, panel_title, model_id, task_id,
   system_id, dataset_id, data_query, notes)
VALUES
  ('chart_cr_three_path_scratch600k_r05_20261101', 'fixed_in_gap', 'a', 'Fixed path — in-gap', NULL, 'fixed_path_neb', 'cr81_migration', 'scratch600k_cr81_fixed_r05', 'SELECT * FROM datasets WHERE dataset_id=''scratch600k_cr81_fixed_r05''', 'Scratch-600K complete 3-seed profile set.'),
  ('chart_cr_three_path_scratch600k_r05_20261101', 'fixed_gap_to_ql', 'b', 'Fixed path — gap→QL', NULL, 'fixed_path_neb', 'cr81_migration', 'scratch600k_cr81_fixed_r05', 'SELECT * FROM datasets WHERE dataset_id=''scratch600k_cr81_fixed_r05''', 'Scratch-600K complete 3-seed profile set.'),
  ('chart_cr_three_path_scratch600k_r05_20261101', 'fixed_within_ql', 'c', 'Fixed path — within-QL', NULL, 'fixed_path_neb', 'cr81_migration', 'scratch600k_cr81_fixed_r05', 'SELECT * FROM datasets WHERE dataset_id=''scratch600k_cr81_fixed_r05''', 'Scratch-600K complete 3-seed profile set.'),
  ('chart_cr_three_path_scratch600k_r05_20261101', 'self_in_gap', 'd', 'Self-relaxed — in-gap', NULL, 'self_relaxed_neb', 'cr81_migration', 'scratch600k_cr81_self_r05', 'SELECT * FROM datasets WHERE dataset_id=''scratch600k_cr81_self_r05''', 'Endpoint-only MLIP proxy; nonconverged seed234 diagnostic retained outside primary evidence.'),
  ('chart_cr_three_path_scratch600k_r05_20261101', 'self_gap_to_ql', 'e', 'Self-relaxed — gap→QL', NULL, 'self_relaxed_neb', 'cr81_migration', 'scratch600k_cr81_self_r05', 'SELECT * FROM datasets WHERE dataset_id=''scratch600k_cr81_self_r05''', 'Endpoint-only MLIP proxy.'),
  ('chart_cr_three_path_scratch600k_r05_20261101', 'self_within_ql', 'f', 'Self-relaxed — within-QL', NULL, 'self_relaxed_neb', 'cr81_migration', 'scratch600k_cr81_self_r05', 'SELECT * FROM datasets WHERE dataset_id=''scratch600k_cr81_self_r05''', 'Endpoint-only MLIP proxy; nonconverged seed123 diagnostic retained outside primary evidence.');

INSERT OR REPLACE INTO chart_links (chart_id, entity_type, entity_id, role, notes)
VALUES
  ('chart_cr_three_path_scratch600k_r05_20261101', 'campaign', 'scratch_600k_migration_20261101', 'campaign', 'Canonical Scratch-600K-Lmax1 migration campaign.'),
  ('chart_cr_three_path_scratch600k_r05_20261101', 'dataset', 'scratch600k_cr81_fixed_r05', 'fixed_path_dataset', '90 finite fixed-path image rows.'),
  ('chart_cr_three_path_scratch600k_r05_20261101', 'dataset', 'scratch600k_cr81_self_r05', 'self_relaxed_dataset', '70 primary self-relaxed image rows; 2 seed×path cases excluded from main values.'),
  ('chart_cr_three_path_scratch600k_r05_20261101', 'task', 'fixed_path_neb', 'fixed_observable', 'Keep separate from endpoint-only self-relaxed values.'),
  ('chart_cr_three_path_scratch600k_r05_20261101', 'task', 'self_relaxed_neb', 'self_observable', 'Endpoint-only MLIP proxy, not DFT MEP barrier.');

INSERT OR REPLACE INTO chart_data_files
  (chart_id, file_id, display_name, local_path, file_format, file_exists, notes)
VALUES
  ('chart_cr_three_path_scratch600k_r05_20261101', 'plotted_profiles', 'R05 plotted image profiles', 'results/plots/cr_three_path_fixed_vs_self_scratch600k_20261101_R05/data/cr_three_path_plotted_profiles_R05.tsv', 'TSV', 0, 'Exact rows used by the renderer; File Provider placeholder in this checkout, with source hash/path in POINTER_ONLY_INPUTS.tsv and the original R05 analysis commit.'),
  ('chart_cr_three_path_scratch600k_r05_20261101', 'coverage_audit', 'R05 coverage audit', 'results/plots/cr_three_path_fixed_vs_self_scratch600k_20261101_R05/data/cr_three_path_coverage_audit_R05.tsv', 'TSV', 1, 'Finite/convergence inclusion and exclusion audit.'),
  ('chart_cr_three_path_scratch600k_r05_20261101', 'result_matrix', 'Scratch-600K migration result matrix', 'results/plots/cr_three_path_fixed_vs_self_scratch600k_20261101_R05/source/scratch_600k_migration_20261101/processed/SCRATCH600K_MIGRATION_RESULT_MATRIX.tsv', 'TSV', 0, 'Campaign result matrix from Rockfish; pointer-only in this checkout, hash/path retained.'),
  ('chart_cr_three_path_scratch600k_r05_20261101', 'fixed_profiles', 'Scratch-600K fixed image profiles', 'results/plots/cr_three_path_fixed_vs_self_scratch600k_20261101_R05/source/scratch_600k_migration_20261101/processed/SCRATCH600K_FIXED_IMAGE_PROFILES.tsv', 'TSV', 0, '90 finite fixed-path image rows; pointer-only in this checkout, hash/path retained.');

INSERT OR REPLACE INTO chart_provenance
  (chart_id, manifest_path, manifest_exists, output_sha256, input_file_hashes,
   processing_command, provenance_status, notes)
VALUES
  ('chart_cr_three_path_scratch600k_r05_20261101',
   'results/plots/cr_three_path_fixed_vs_self_scratch600k_20261101_R05/provenance/cr_three_path_fixed_vs_self_2x3_R05.manifest.json',
   1,
   '{"pdf":"6cee335fb2e9783d554f46bd5e6b970d9772cb6c744e7a515afd37534476e726","png":"38ff03f8eec5362e93f71dfe9448e0fcb2ae99d304aa5827ee8133497e1f3cdb","svg":"415bc5c2f3fd060b1e81a11f21b30a6639e33bea40d24a06d56074d5b5638205"}',
   '{"manifest":"1eb45bae7d83061a035cf21ce9132ed071e9c2739a253ed36a1b3c2617f39c03","source_count":11,"scratch_endpoint_manifest_count":9}',
   'python results/plots/cr_three_path_fixed_vs_self_scratch600k_20261101_R05/scripts/make_cr_three_path_2x3_R05.py',
   'COMPLETE_WITH_BOUNDARY',
   'Rebuild and hash verification passed. Large remote trajectories/iteration extxyz are pointer-only and the DFT reference PDF is an external hash pointer.');

INSERT OR REPLACE INTO metadata (key, value) VALUES
  ('figure_registry_last_update', '2026-11-01'),
  ('figure_registry_update', 'chart_cr_three_path_scratch600k_r05_20261101 added from the user-certified R05 bundle'),
  ('figure_registry_base_snapshot', '2026-10-28 local SQL-first archive snapshot; current File Provider placeholder was not overwritten');

INSERT INTO audit_log (checked_at, check_name, result, details)
VALUES
  ('2026-11-01T00:00:00-07:00', 'r05_chart_registry_update', 'PASS', 'R05 chart, panels, datasets, data files, provenance, and campaign linkage inserted idempotently.'),
  ('2026-11-01T00:00:00-07:00', 'r05_scientific_boundary', 'PASS_WITH_CAVEAT', 'Fixed-path and endpoint-only self-relaxed observables remain separate; two nonconverged self cases remain diagnostic-only.');

COMMIT;
