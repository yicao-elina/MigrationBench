-- QLR01: all audit-summary rows; no row is an event-level failure label.
SELECT failure_mode_id,evidence_family,observed_signal,queryable_sources,lineage_status,
       repeatability_assessment,next_round_decision_and_gain,recommended_closure,
       grain,benefit_status
FROM failure_mode_review ORDER BY failure_mode_id;

-- QLR02: the gap between observed diagnostics and a proven learning-loop effect.
SELECT failure_mode_id,evidence_family,repeatability_assessment,next_round_decision_and_gain,
       recommended_closure,benefit_status,
       'REQUIRES_VERIFIED_DIAGNOSTIC_TO_DECISION_TO_TRAINING_TO_HELD_OUT_EVALUATION_CHAIN' AS evidence_gate
FROM failure_mode_review ORDER BY failure_mode_id;

-- QLR03: source pins and grain boundaries.
SELECT metadata_key,metadata_value FROM snapshot_metadata ORDER BY metadata_key;
