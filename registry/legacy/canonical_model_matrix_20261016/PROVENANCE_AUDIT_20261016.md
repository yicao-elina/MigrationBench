# Canonical campaign provenance audit

This is the current evidence ledger for `canonical_model_matrix_20261016`. It separates computational state, scientific acceptance, provenance status, and publication use.

## Datasets

- The canonical from-scratch Multi-T grouped release has frozen train/valid/test hashes, trajectory-level split assignments, and zero audited group overlaps.
- The 600 K grouped release has the same verified group-disjointness result, but its upstream manifest remains explicitly `PROVISIONAL_UPSTREAM_MANIFEST`.
- The 5% releases preserve their family-specific validation and test splits and use deterministic stratified selections with all isolated-atom anchors retained.
- The 600 K and Multi-T sources are not concatenated; the 600 K source universe is contained in the broader Multi-T universe, so direct union would duplicate source trajectories and risk split collisions.

Authoritative evidence: `EXPERIMENT_MANIFEST.json` files, `DATASET_GROUP_AUDIT_20261016.json`, and the 5% submission record.

## Models

The three foundation checkpoints are registered as zero-shot comparators, not seed replicas. Six existing grouped full-data FT models have remote files whose hashes match the canonical registry and whose grouped test metrics are recorded in `MODEL_FILE_HASH_AUDIT_20261016.tsv`. Their compiled-model acceptance remains held until job `32075596` passes the loadability, finite-parameter, and finite Cr/Sb/Te probe gate.

The newly submitted scratch and 5% model families have no scientific acceptance until their array tasks terminate, compiled models and hashes are observed, test metrics are recorded, and the same model gate passes.

## Jobs and failure evidence

Current monitored jobs are `32074582` (canonical Multi-T scratch), `32075474` (additive grouped 600 K scratch), `32075144` (six 5% FT models), and `32075596` (existing-model gate). The failed bootstrap jobs remain in `BOOTSTRAP_FAILURE_LEDGER.tsv`; they are not overwritten or treated as successful predecessors.

## Static screen and publication boundary

The historical 639 × 3 table is a verified diagnostic with 1,917 unique finite rows, but it is not the canonical accepted model matrix. The new canonical screen is prepared and held until accepted model IDs are finalized. Its planned row gate is exactly `639 × accepted_model_count`, with one finite row per candidate/model pair.

The current revision-1 figure book passed validation for 218 registries and has baseline SHA256 recorded in `FIGURE_BOOK_BASELINE.json`. The canonical figure, SI/main-text handoff, and final release manifest remain pending the accepted screen.

## DFT boundary

The local periodic-shell builder passed a tooling-only 120-atom reconstruction smoke test. It is not a DFT result and is not a canonical candidate selection. Candidate selection and local-fragment preparation must follow a verified accepted canonical screen; the 2050-atom full-cell feasibility boundary remains explicit.
