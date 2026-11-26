-- QCE01: full pinned claim-to-evidence index, retaining source-grain rows.
SELECT matrix_id, claim_type, source_document, source_location, source_object_id,
       assertion_or_panel, exact_source_quote_or_value, numeric_tokens,
       referenced_labels, sql_item_id, sql_panel, registry_content,
       artifact_pointer, artifact_sha256, generator_or_query,
       registry_status, reproduction_status, lineage_status, gap_reason,
       source_sha256
FROM claim_evidence_row
ORDER BY source_document, source_location, matrix_id;

-- QCE02: counts by row type and source-lineage status; not scientific pass rates.
SELECT claim_type, lineage_status, COUNT(*) AS row_count
FROM claim_evidence_row
GROUP BY claim_type, lineage_status
ORDER BY claim_type, lineage_status;

-- QCE03: unresolved manuscript segments and cells, grouped by source location.
SELECT source_document, source_object_id, claim_type, lineage_status,
       COUNT(*) AS row_count,
       GROUP_CONCAT(DISTINCT gap_reason) AS gap_reasons
FROM claim_evidence_row
WHERE lineage_status IN ('PARTIAL','MISSING')
GROUP BY source_document, source_object_id, claim_type, lineage_status
ORDER BY source_document, source_object_id, claim_type, lineage_status;

-- QCE04: immutable source, generator, and coverage pins for this inventory.
SELECT metadata_key, metadata_value
FROM snapshot_metadata
ORDER BY metadata_key;
