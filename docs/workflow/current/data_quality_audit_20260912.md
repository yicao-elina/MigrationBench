# Data Quality Audit: Endpoint And Path Selection

## Dataset And Grain

Endpoint source tables contain one screened site per row. Candidate-pair rows must join two sites within one declared physical system. NEB topology rows describe one path, one candidate basin image, or one directed basin-to-basin segment.

## Findings

### High: Old endpoint graph mixed unresolved source systems

The superseded selector pooled Victor and Joshitha rows before pair generation. Of `12052` old candidate pairs, `5696` joined sites from different source files without proof that their cells, coordinate frames, and structure identities were compatible. This can create nonphysical endpoints and biased coverage.

The current extracts have zero overlapping raw `unique_id` values, but IDs were not source-namespaced. Uniqueness therefore depended on an accidental property of those files rather than the schema.

Remediation: `select_migration_endpoints.py` now creates `system_id::source_namespace::raw_site_id`, fails on duplicate global keys, and pairs only within a system. Bare CSV inputs are isolated by source. Cross-source pairing requires an explicit shared system ID after structure compatibility is established.

Interim evidence: `endpoint_selection_v2` contains `191` sites, `6356` pairs, and `48` representatives; checks report unique site/path keys, `0` cross-system pairs, and `0` missing screening energies.

Resolution after raw-data recovery: `pipeline_meta.json` establishes one host structure and QE template for both assignment files, all `245/245` QE inputs have composition `Sb32Te48Cr1`, and the reference cell is shared. Canonical `endpoint_selection_v3` therefore assigns both namespaces to explicit system `sb2te3_4x2x1`, restoring the `5696` cross-namespace edges without weakening the cross-system guard. It has `12052` same-system pairs and `24` representatives.

### High: Existing site-search DFT outputs are not linked to current inputs

Only two of the 191 selected site inputs have a colocated `relax.out`. Both outputs predate their current `relax.in`, which was rewritten later, so no input hash can be assigned retrospectively. They are retained as historical diagnostics but rejected as DFT training labels. The site proposer reports zero accepted independent DFT minima and keeps automatic QE submission closed.

### Critical For Manuscript Use: Frozen endpoints are not validated minima

All `28` endpoints in the 14-path topology audit are frozen NEB endpoints. Their NEB tables do not establish local stability. Current `1-6` and `1-7` each contain a substantially lower image-2 basin candidate, and their current segments are endpoint-dominated.

Impact: reported forward quantities can be dominated by endpoint energy differences rather than a resolved interior saddle. They cannot yet support a manuscript claim about a single migration barrier.

Remediation: standalone fixed-cell QE relaxations for images `1`, `2`, and `5` of both paths are running. Basin acceptance requires BFGS completion, maximum force `<=0.05 eV/A`, repeat-relax stability, valid geometry, and complete provenance.

### Medium: Periodic cell metadata is absent for five 81-atom XYZ paths

The five synchronized 81-atom history files contain positions but no cell metadata. Their energy topology is usable, but minimum-distance and curve-length descriptors are nonperiodic approximations.

Remediation: attach the authoritative QE cell/input before using those geometry descriptors for cross-path distance or mechanism clustering.

### Checkpoint-Scoped Pass With Ongoing Gate: Leakage

The policy excludes benchmark endpoints, NEB images, transition states, and post-outcome DFT images from training for the same mechanism. Splits are grouped by system, endpoint basins, and mechanism cluster. This is a design control, not proof that historical training files obey it.

The checkpoint-scoped historical audit is now complete for the current
`1-6/1-7` evaluation histories. Rockfish job `30845544` compared 1110 QE frames
with the declared fine-tuning train/valid/test files and found no exact global
or Cr-local fingerprints, no species-matched local RMSD <= `0.05 A`, and no
SOAP cosine distance <= `0.0001`. The valid statement is therefore: `no
training overlap detected for the declared fine-tuning checkpoint files under
the registered metrics and thresholds`. This is not evidence about unknown
foundation-model pretraining data, and every future benchmark/training-data
revision must rerun the audit.

The SI-ready empirical-distribution figure and machine-readable summary are
stored beside the audit report as `leakage_distance_distributions.svg` and
`leakage_distance_summary.json`. RMSD is reported only where local environments
have compatible species cardinality; SOAP remains defined for all 1,110 frames
against every declared split.

## Automated Tests

- Site and path primary-key uniqueness.
- Zero cross-system endpoint pairs.
- Composition/order consistency and matching energy/geometry image counts.
- Minimum pair-distance gate.
- Complete energies and atomic forces for every exported optimization iteration.
- Benchmark mechanism IDs absent from train/validation splits.
- Near-duplicate structure audit across train/evaluation boundaries.
