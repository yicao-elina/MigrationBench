# Status reconciliation — 2026-11-23

The inherited tables/CLAIM_EVIDENCE_MATRIX.tsv marked the 12-member source-geometry ensemble PENDING_COMPUTE. That status is stale relative to the package completed-run evidence: SOURCE_GEOMETRY_ENSEMBLE_AUDIT_20261023.md records all 12 Slurm array members completed with exit code 0 and 47 successful evaluations each, and tables/MACE12_SOURCE_ENSEMBLE_QUALITY_REPORT.tsv records 564 verified static predictions.

The matrix now marks only the claim that the 12-member static source-geometry evaluation was performed as VERIFIED. This does not establish MACE-versus-DFT error: matched final DFT geometries are unavailable. The result remains a model-disagreement diagnostic and is not a migration barrier or a stability/transport claim.
