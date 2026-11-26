-- QCE06: exact manuscript source versions and file identities.
-- This is source authority metadata, not scientific validation.
SELECT v.source_version_id, v.authority_status, v.git_commit,
       v.pull_request, v.pull_request_state, v.source_capture_state,
       v.source_note, f.repository_path, f.file_role, f.byte_count,
       f.sha256, f.file_note
FROM manuscript_source_version AS v
LEFT JOIN manuscript_source_file AS f
  ON f.source_version_id = v.source_version_id
ORDER BY v.source_version_id, f.repository_path;

-- QCE07: row-level submitted-ledger ↔ PR #10 candidate-ledger deltas.
-- Text/ledger differences do not validate cited artifacts.
SELECT claim_id, delta_type, submitted_status, candidate_status,
       changed_fields, submitted_row_json, candidate_row_json,
       review_boundary
FROM manuscript_ledger_delta
ORDER BY delta_type, claim_id;

-- QCE08: compact count of candidate-vs-submitted ledger changes.
SELECT delta_type, COUNT(*) AS claim_count
FROM manuscript_ledger_delta
GROUP BY delta_type
ORDER BY delta_type;

-- QCE09: full PR #10 changed-file reconciliation against current main.
-- Mainline ancestry and byte identity establish integration, not scientific validation.
SELECT pull_request, pr_head_commit, source_path, pr_blob_oid,
       pr_byte_count, pr_sha256, current_path, current_blob_oid,
       current_byte_count, current_sha256, reconciliation_status,
       reconciliation_note
FROM pr_file_reconciliation
WHERE pull_request = 10
ORDER BY reconciliation_status, source_path;
