# QE Calculator Identity Audit

## Critical Finding

The current SI states that all DFT calculations use 100/400 Ry cutoffs, a 4x4x1 k-point grid, and spin-orbit coupling. The 377 locally traceable QE inputs do not support that statement.

No audited input explicitly enables both `noncolin=.true.` and `lspinorb=.true.`. Fully relativistic pseudopotential filenames do not activate SOC by themselves.

A direct Rockfish header/hash audit confirms that the referenced Cr, Sb, and Te
SG15 files are all readable, fully relativistic norm-conserving
pseudopotentials with `has_so="T"`. An explicit SOC sensitivity branch is
therefore technically compatible with these files. This compatibility does not
retroactively enable SOC in any historical input.

## Main Production Profiles

| Role | Composition | Cutoffs (Ry) | k points | Smearing (Ry) | Spin/SOC | Count in audited mirrors |
|---|---|---|---|---:|---|---:|
| Current 61-atom `1-6/1-7` family | `CrSb24Te36` | 50/200 | Gamma | 0.0147 | non-spin, no SOC | 82 |
| Current 81-atom NEB family | `CrSb32Te48` | 70/280 | 2x2x2 | 0.005 | non-spin, no SOC | 26 |
| Site-search candidate inputs | `CrSb32Te48` | 70/280 | 2x2x2 | 0.0147 | `nspin=2`, no SOC | 245 |
| Victor V path family | `VSb32Te48` | 100/400 | 1x1x1 | 0.0147 | non-spin, no SOC | 8 |

The common site-search cell reference itself uses 70/400 Ry and no spin, so it supplies cell metadata only; it is not a calculator template for new energies.

## Consequences

- Absolute energies and barriers must not be pooled across calculator identities.
- Site-search MACE energies can rank candidates but cannot calibrate the current non-spin 81-atom DFT NEB without independent validation.
- The 61-atom and 81-atom barrier families require separate method records unless a convergence study demonstrates equivalence.
- The SI must describe the settings actually used for each accepted result, or the accepted results must be recomputed under one validated canonical protocol.
- Current active NEB chains should not have settings changed mid-restart; that would break the energy surface within a path.

## Minimal Convergence Study

Use one endpoint and one near-saddle image from each supercell. Run independent SCF calculations under a preregistered matrix:

1. historical production identity;
2. increased cutoff at fixed k grid;
3. denser in-plane k grid with one k point along the vacuum direction;
4. spin-polarized collinear calculation;
5. explicit noncollinear SOC calculation if the pseudopotentials and scientific claim require it.

For each protocol, compare `Delta E = E_saddle - E_endpoint`, not unrelated absolute energies. A protocol is numerically adequate only if the barrier difference changes by no more than the declared tolerance, proposed as 0.02 eV, and both SCFs are fully converged. Magnetic and SOC variants are physical model changes and should be reported separately rather than treated as ordinary numerical convergence knobs.

## Current Decision

The calculator identity audit is complete, but the canonical protocol is not
yet fully resolved. The cutoff stage on exact `1-6/path108` images 2 and 4
passed: `50/200 Ry` and `100/400 Ry` give fixed-geometry contrasts of
`0.830367 eV` and `0.829871 eV`, a difference of `0.000497 eV` against the
predeclared `0.02 eV` tolerance. Canonical cutoff jobs were `30850191`,
`30850192`, `30850193`, and `30850195`; duplicate submissions `30850194` and
`30850197` remain excluded by the correction ledger.

The Gamma-versus-`2x2x1` stage completed and failed the gate. The exact
image-4 minus image-2 contrast is `+0.830367 eV` at Gamma and `-0.615659 eV`
at `2x2x1`, a `-1.446026 eV` change with an ordering reversal. Gamma-only
historical barriers cannot be final references. Jobs `30864939`, `30864944`,
`30864946`, and `30864947` completed the `3x3x1`/`4x4x1` comparison. It also
failed: the contrast is `-0.807902 eV` versus `-0.904860 eV`, a `0.096958 eV`
difference. Jobs `30871998` and `30871999` completed the same pair at
`5x5x1`: the `4x4x1` and `5x5x1` contrasts are `-0.904860` and
`-0.894609 eV`, differing by `0.010251 eV`. This passes the `0.02 eV` gate and
accepts `4x4x1` as the numerical mesh. Cutoff-at-`4x4x1` jobs
`30877380-30877383` also passed: the `50/200` and `100/400 Ry` contrasts are
`-0.904860` and `-0.905798 eV`, differing by `0.000938 eV`. Thus the accepted
numerical protocol is `50/200 Ry`, `4x4x1`. Nonspin jobs `30879759/61` and
collinear-Cr jobs `30879762/64` completed and passed SCF. The collinear
fixed-geometry contrast is `-0.692803 eV`, which differs from the nonspin
reference by `0.212056 eV`; this is a separate physical-model sensitivity, not
a numerical convergence pass. Explicit SOC jobs `30879766/68` were cancelled on
2026-09-15 UTC per user instruction to pause SOC unification. The current
partial protocol analysis is therefore intentionally marked invalid for the
combined spin/SOC gate, while retaining the completed collinear comparison.

## Artifacts

- Per-input table: `data_processed/qe_calculator_identity_audit/v1/qe_input_identities.csv`
- Identity manifest: `data_processed/qe_calculator_identity_audit/v1/calculator_identity_manifest.json`
- Reproducible auditor: `scripts/migrationbench/audit_qe_calculator_identity.py`
- Stage-1 inputs and submission ledger: `data_processed/qe_protocol_sensitivity/1-6_stage1_s42/`
- K-point-stage inputs and submission ledger: `data_processed/qe_protocol_sensitivity/1-6_kpoint_stage_s42/`
- K-point-stage result: `data_processed/qe_protocol_sensitivity/1-6_kpoint_stage_s42/kpoint_analysis.json`
- Dense-k inputs and ledger: `data_processed/qe_protocol_sensitivity/1-6_kpoint_dense_s42/`
- 5x5x1 inputs and ledger: `data_processed/qe_protocol_sensitivity/1-6_kpoint_5x5_s42/`
- Combined 4x4x1/5x5x1 decision: `data_processed/qe_protocol_sensitivity/1-6_kpoint_4x4_5x5_s42/kpoint_analysis.json`
- Cutoff-at-4x4x1 ledger: `data_processed/qe_protocol_sensitivity/1-6_cutoff_at_4x4_s42/`
- Spin/SOC ledger: `data_processed/qe_protocol_sensitivity/1-6_spin_soc_at_4x4_s42/`
- Active jobs: `configs/active_qe_protocol_sensitivity_jobs.json`
- Pair analyzer: `scripts/migrationbench/analyze_qe_protocol_sensitivity.py`
- Rockfish pseudopotential compatibility evidence:
  `data_processed/qe_pseudopotential_compatibility/sg15_fr_pbe_rockfish.json`
