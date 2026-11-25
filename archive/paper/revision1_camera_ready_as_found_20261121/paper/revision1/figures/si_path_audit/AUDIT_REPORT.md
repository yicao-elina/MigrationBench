# Cr–Sb₂Te₃ migration-path match audit

Date: 2026-10-03

## Bottom line

The corrected 81-atom source is:

`/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/2.2-layer-diffusion/2D_421_Diffusion_traj`

It contains five completed paths, `neb_1`–`neb_5`; each has 10 images, 81 atoms, and composition `Cr1 Sb32 Te48`. The five paths form one continuous Cr migration chain: all four adjacent links pass the 0.10 Å continuity criterion.

They do not form a one-to-one copy of the three Li/Na/Cu reference categories. Under the endpoint rule “mobile-atom PBC distance ≤0.05 Å and host RMSD ≤0.10 Å”, only one distinct Cr endpoint matches: `neb_1` start matches the common `in_gap_diffusion/pw_1` endpoint for Li and Cu. There is no exact match to `within_QL`, `gap_to_QL`, or the Na endpoint. The two passing rows are the same Cr endpoint matched to two chemically equivalent reference inputs, not two Cr paths.

The independent geometry audit classifies all five 81-atom paths as `deep_penetration`, with penetration scores 0.691–0.741 and gap scores 0.210–0.350. Thus `within_QL_candidate` is only a cross-system proxy; it is not an endpoint-identity match. `gap_to_QL` is not established for this 81-atom chain.

## 61-atom coverage

The source-of-truth path queue contains the seven paths `1-2` through `1-8` (excluding no path in that range). The existing five-path Foundation matrix is retained in its original namespace, and exact historical DFT coordinates for `1-6` and `1-8` were evaluated separately.

Current evidence-bounded mapping:

| paths | geometry/source label | three-bin interpretation |
|---|---|---|
| 1-2, 1-3 | deep penetration | `within_QL_candidate` proxy |
| 1-4, 1-7 | in-gap by prior site definition | `in_gap_diffusion` candidate |
| 1-5 | deep penetration, geometry recheck requested | `within_QL_candidate` proxy only |
| 1-6 | exact historical coordinates: `mixed_transition` | `ambiguous` |
| 1-8 | exact historical coordinates: `probable_deep_penetration` | `ambiguous` |

No 61-atom path is assigned to `gap_to_QL` without an explicit topology/site match. The classification table keeps the continuous geometry label separate from the canonical three-bin proxy.

## Foundation-model calculations

All 50 81-atom evaluations and all 20 exact-coordinate 61-atom evaluations completed with finite energies and forces. The direct profile ranges are:

| scope | paths | model barrier range (eV) | mean across models/paths |
|---|---:|---:|---:|
| 81-atom historical `neb_1`–`neb_5` | 5 × 10 | 0.455–2.100 | path means 0.723–1.877 |
| exact 61-atom historical `1-6` | 10 | 0.412–0.952 | 0.611 |
| exact 61-atom historical `1-8` | 10 | 0–0.231 | 0.072 |

These are single-point Foundation evaluations on retained images, not newly converged NEB barriers. The largest per-image forces remain approximately 1.2–3.4 eV/Å, so the values are useful for model-profile comparison only and must not be reported as converged model NEB barriers.

The complete 61-path table combines the existing five-path audit with the new exact-coordinate `1-6/1-8` rerun. Its historical DFT status is explicitly retained and the old `pass` label is downgraded to conditional because the source audit does not establish the final production convergence gate.

## DFT/NEB quality boundary

The five 81-atom `neb.out` files contain SCF-convergence warnings. The retained historical maximum image errors are 0.232–0.636 eV/Å, while the source audit’s production gate is 0.03 eV/Å. `JOB DONE` and the historical `CONVERGED` labels therefore do not establish manuscript-grade DFT barriers. The 61-atom source similarly contains auxiliary/unconverged paths; `1-8` retains its historical `CONVERGED` table label but is not silently promoted beyond the independent audit boundary.

## Reproducibility and isolation

- Rockfish raw NEB trees were read-only.
- New Foundation outputs are under `/scratch16/pclancy3/yi/cr_81atom_foundation_neb_audit_20261003` and were copied into this audit namespace.
- Foundation jobs: 81-atom array `31543231`; candidate 61-atom `1-6/1-8` array `31543425`; exact historical 61-atom array `31543543`.
- The audit manifest records 90 finite JSON results and `raw_calculations_modified: false`.

## Main artifacts

- [audit manifest](<./foundation_results_manifest.json>)
- [81-atom inventory](<./tables/remote_81atom_path_inventory.csv>)
- [81-to-Li/Na/Cu endpoint matches](<./tables/category_endpoint_matches.csv>)
- [81-chain continuity](<./tables/cr_sequential_chain_continuity.csv>)
- [81 Foundation matrix](<./tables/81_atom_historical_2D_421.csv>)
- [exact 61-atom `1-6/1-8` Foundation matrix](<./tables/61_atom_historical_DFT_1-6_1-8.csv>)
- [complete 61-atom seven-path matrix](<./tables/foundation_matrix_61_all_paths.csv>)
- [61/81 path classification](<./tables/path_classification_61_81.csv>)
- [exact 61-atom geometry descriptors](<./tables/classification_61_exact.csv>)
- [result hashes](<./tables/foundation_result_sha256.csv>)
