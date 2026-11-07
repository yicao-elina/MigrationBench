# Cr migration-path registry (reader-facing names)

Machine-readable: `paper/revision1/data_processed/path_registry.csv` (copy in `data/processed/`), built by
`paper/revision1/scripts/build_path_registry.py`. SI Table S7 shows the same mapping.

| Name | Class | Atoms | Audit ID | Basis for the class |
|---|---|---:|---|---|
| G1 | within-gap | 61 | 1-4 | `in_gap_by_prior_site_definition` in `configs/experiments/migrationbench_path_queue.yaml`; barrier reference path |
| G2 | within-gap | 61 | 1-7 | `in_gap_by_prior_site_definition` (path queue) |
| Q1 | gap-to-QL/into-QL | 61 | 1-2 | `deep_penetration_by_prior_site_definition` (path queue) |
| Q2 | gap-to-QL/into-QL | 61 | 1-3 | `deep_penetration_by_prior_site_definition` (path queue) |
| Q3 | gap-to-QL/into-QL | 61 | 1-5 | `deep_penetration_by_prior_site_definition`; queue requests a geometry recheck |
| M1 | mixed | 61 | 1-6 | exact historical coordinates: `mixed_transition` (`figures/si_path_audit/AUDIT_REPORT.md`) |
| L1 | within-QL | 61 | 1-8 | exact historical coordinates: `probable_deep_penetration` |
| C1–C5 | within-QL | 81 | neb_1–neb_5 | continuous geometry: `deep_penetration`, penetration score 0.691–0.741, gap score 0.210–0.350 (`data/published/path_classification/81_atom_paths_v2_pbc.csv`) |

Notes
- 1-4 and 1-7 are both in-gap by the source site definition. The legacy figure-export label "gap→QL" on the
  1-4 profile page was superseded by this classification; the profile pages now carry the registry names.
- No 61-atom path is assigned to the endpoint-defined `gap_to_QL` category of the Li/Na/Cu reference set; the
  manuscript class "gap-to-QL/into-QL" is the deep-penetration geometry class (Cr moves from the vdW gap into the QL).
- Text in the main manuscript, SI, and response letter uses the names only; the audit ID appears in the SI registry
  table as the key to the released data.
