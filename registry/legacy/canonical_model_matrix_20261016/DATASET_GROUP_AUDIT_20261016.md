# Canonical grouped dataset audit

Status: `VERIFIED` for the group-disjointness check. The audit used ASE on Rockfish and compared trajectory/group identifiers across train, validation, and test for every canonical full-data and 5% release.

The four releases have zero cross-split group overlaps. Isolated atomic-reference frames are retained only in training. The 5% selections preserve the full validation and test sets; therefore the full-data and 5% rows intentionally share those validation/test hashes within each source family.

The FT-600K upstream manifest remains labelled `PROVISIONAL_UPSTREAM_MANIFEST` in its experiment manifest. That provenance status is independent of the verified group-disjointness result and must not be silently upgraded.

The machine-readable counts and hashes are in `DATASET_GROUP_AUDIT_20261016.json`.
