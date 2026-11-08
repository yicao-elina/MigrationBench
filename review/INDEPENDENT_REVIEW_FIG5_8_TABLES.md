# Independent review: Figures 5–8 and Tables 1–2

Review mode: read-only source/table audit plus rendered-output inspection.
Review date: 2026-10-31.

## Figure 5

Result: `BLOCKED`.

The current model-generated NEB package is endpoint-only/precanonical and was
stopped for endpoint repair. It lacks a complete per-run convergence record,
ten-image finite geometry/force evidence, endpoint identity, and final maximum
force contract. A polished path cannot establish a barrier or success rate.

Required repair: rerun from the corrected endpoint/checkpoint lineage, retain
all ten images and convergence logs, and classify failures separately from
reconstruction controls.

## Figure 6

Result: `PROVISIONAL / OBSERVED-HORIZON ONLY`.

The local package contains 15 model rows and derived series/RDF/VACF/MSD files.
Only four rows currently reach the nominal 1,000 ps; many failed rows stop near
200–485 ps. The generated figure correctly uses observed horizons and does not
extend failures or infer a diffusion ranking. Temperature anomalies and invalid
frames still require a data-level audit.

Required repair: reconcile every status with raw termination evidence, count
valid frames, audit anomalous temperatures, and define any future transport
acceptance gate before reporting transport quantities.

## Figure 7

Result: `SCF_WARNING / PROXY_DIAGNOSTIC`.

The tables contain 400 image-profile rows, 20 QE reference rows, and 10 model
identities. Fixed and generated/self-relaxed paths are separable in the source
and in the output. QE references carry `COMPLETED_BUT_SCF_WARNING`; generated
self paths are endpoint-only proxies. Class bands are sample SD across model
identities, not seed uncertainty.

Required repair: add checkpoint counts, preserve the warning labels, and rerun
full self-relaxed paths before using generated-path conclusions.

## Figure 8

Result: `CONDITIONAL / SUPPLEMENTAL MAP ONLY`.

The descriptor/PCA map is data-backed from the structure-universe manifest. The
current evidence does not contain a validated geometry-keyed join to the QE
error table, a frozen high-error threshold, held-out validation, or disagreement
calibration. The B–D boxes are correctly blocked placeholders.

Required repair: create the geometry-keyed table, freeze threshold and selection
rule prospectively, validate on held-out data, and report calibration/recognition
before making an acquisition claim.

## Tables 1–2

Result: `CONSERVATIVE AUDIT ONLY`.

The model/reference audit and task decision matrix must keep Foundation,
current/legacy Scratch, endpoint-only proxies, and QE-warning references as
distinct lineages. Every cell needs a source row and figure gate. A unified
publication matrix is blocked until Figures 2–4 and the relevant campaign gates
are frozen.
