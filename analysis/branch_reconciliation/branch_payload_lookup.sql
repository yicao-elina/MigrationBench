-- Look up historical release-branch payloads by path and see their current destination.
SELECT repo_path, stored_at, bytes, sha256, original_source, migration
FROM rev1_external_store
WHERE original_source LIKE 'GitHub branch release/revision1-submitted-20261110%'
ORDER BY repo_path;

-- Retrieve the campaign's distinct DFT b-path2 NEB outputs.
SELECT repo_path, stored_at, bytes, sha256
FROM rev1_external_store
WHERE repo_path LIKE 'analysis/foundation_interlayer_sliding_2x2_20261024/inputs/dft_bpath2/%/neb.out'
ORDER BY repo_path;
