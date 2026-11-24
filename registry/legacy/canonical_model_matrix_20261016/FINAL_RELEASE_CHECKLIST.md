# Canonical release checklist

This checklist is the release gate for `canonical_model_matrix_20261016`. It is intentionally additive: historical campaigns and failed jobs remain preserved and are not silently relabelled.

## Computational gates

- [x] Frozen grouped dataset manifests and exact split hashes.
- [x] Group-disjointness audit for full-data and 5% releases.
- [x] Environment-bootstrap repairs recorded without deleting failed evidence.
- [ ] Every submitted seed reaches a terminal state with a compiled model.
- [ ] Every included compiled model passes the explicit model gate.
- [ ] Canonical registry contains exact model hashes and resolved job IDs.
- [ ] Static screen contains exactly `639 × accepted_model_count` unique candidate/model rows.
- [ ] Static screen has finite energy and force values for every required pair.

## Scientific gates

- [ ] Test energy/force metrics are recorded separately for each family and seed.
- [ ] Within-family seed dispersion excludes foundation comparators and historical heterogeneous models.
- [ ] Stage-wise medians, q95 tails, maxima, and paired force disagreement are reproducible from the accepted screen.
- [ ] Any failure or nonfinite row remains in a failure ledger and blocks VERIFIED status.
- [ ] MLIP-only evidence is labelled separately from DFT-validated evidence.

## DFT and publication gates

- [ ] DFT-ready candidates are selected only from the accepted canonical screen.
- [ ] Local fragments remain `RECONSTRUCTION`/`PREPARED_ONLY` until QE convergence and parity checks pass.
- [ ] The 2050-atom full-cell infeasibility evidence remains explicit; no local fragment is described as exact parity.
- [ ] Plot Atlas figure panels and machine-readable figure manifest are rebuilt from accepted tables.
- [ ] The existing figure book at `/Users/alina/Project/migrationbench-paper-repro/paper/revision1/figure_book.pdf` is updated in place through `figurectl.py`; no second figure book is created.
- [ ] SI/main-text handoff records claim IDs, source hashes, computational status, and interpretation boundaries.
- [ ] Final release manifest reports computational, scientific, provenance, and publication statuses independently.

Until every unchecked required gate is closed, the release status is `INCOMPLETE` and no publication claim is upgraded.
