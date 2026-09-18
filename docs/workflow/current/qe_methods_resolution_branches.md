# QE Methods Resolution Branches

## Why This Is A Hard Gate

The current SI describes one universal QE setup: 100/400 Ry cutoffs, a 4x4x1
k-point mesh, and spin-orbit coupling. The audited inputs instead contain 12
calculator identities. In particular, the current 61-atom NEB lineage uses
50/200 Ry, Gamma-only sampling, non-spin-polarized collinear DFT, and no
explicit SOC. No audited input enables both `noncolin=.true.` and
`lspinorb=.true.`.

Therefore a barrier, endpoint energy, or relaxation-speed comparison is
manuscript-eligible only when its complete calculator identity is recorded and
the relevant numerical and physical-model choices have been resolved. A fully
relativistic pseudopotential filename is not evidence that SOC was enabled.

## Stage 1: Cutoff Decision

The preregistered fixed-geometry comparison uses exact `1-6/path108` images 2
and 4. For each calculator profile, define

`Delta E(profile) = E_profile(image 4) - E_profile(image 2)`.

The cutoff sensitivity is

`delta_cutoff = abs(Delta E(100/400 Ry) - Delta E(50/200 Ry))`.

Both image SCFs in both profiles must converge. The historical cutoff passes
this stage only if `delta_cutoff <= 0.02 eV`. Absolute total energies from
different profiles are never compared directly.

### Branch C1: Cutoff Passes

- Retain the historical 50/200 Ry lineage as numerically adequate for this
  fixed-geometry 61-atom contrast only.
- Continue k-point sensitivity at fixed accepted cutoff.
- Do not yet promote any barrier or A/B speedup to manuscript grade: spin/SOC
  and full NEB convergence remain open gates.

### Branch C2: Cutoff Fails

- Quarantine historical 61-atom barriers as protocol-sensitive candidates.
- Select the converged cutoff profile and recompute endpoint relaxations and
  manuscript-grade NEB paths from structures, not mixed-profile restart data.
- Historical paths may remain initial geometries, but their energies cannot be
  combined with the new profile.

### Branch C3: An SCF Fails

- Report the stage as indeterminate.
- Repair or restart the failed SCF with the same profile and immutable parent
  state. Do not relax the tolerance or substitute a partial energy.

The analyzer emits four exact states: `pass`, `fail`, `pending`, and `invalid`.
`pending` means execution is still live or incomplete; `invalid` means at least
one terminal calculation lacks an accepted energy. These states must not be
collapsed because they imply different next actions.

## Stage 2: K-Point Decision

Repeat the same image-pair contrast using the accepted cutoff and a
preregistered denser in-plane mesh suitable for the cell. The numerical gate is
again `0.02 eV` on the energy contrast. The actual accepted mesh, not a desired
or historical template mesh, must be stated in the SI.

The first comparison failed: Gamma gives `+0.830367 eV`, whereas `2x2x1`
gives `-0.615659 eV`. The `-1.446026 eV` difference reverses the ordering of
the two fixed geometries, so Gamma-only historical barriers are quarantined.
The `3x3x1`/`4x4x1` branch also failed: image-4 minus image-2 is
`-0.807902 eV` and `-0.904860 eV`, respectively, a `0.096958 eV` difference.
The preregistered `5x5x1` extension completed as jobs `30871998/30871999`.
The `4x4x1` and `5x5x1` contrasts are `-0.904860` and `-0.894609 eV`, a
`0.010251 eV` difference that passes the `0.02 eV` gate and accepts `4x4x1`.
Cutoff convergence at that mesh passed: `50/200` and `100/400 Ry` differ by
`0.000938 eV` on the fixed-geometry contrast. The accepted numerical protocol
is therefore `50/200 Ry`, `4x4x1`. Nonspin, collinear-Cr, and explicit-SOC
pairs are running as `30879759/61`, `30879762/64`, and `30879766/68`.

## Stage 3: Spin And SOC Decision

Collinear spin and explicit noncollinear SOC are physical Hamiltonian choices,
not ordinary convergence knobs. Compare the same fixed geometries and report
their effects on the energy contrast, local Cr moment, and SCF stability. The
production identity must then be selected on physical grounds and applied
consistently to all manuscript-grade values in the same comparison family.

`monitor_qe_scf_warmups.py` preserves the final total and absolute
magnetization, noncollinear total-magnetization components, and the latest
per-atom magnetic moment components mapped to input species. The protocol
analyzer carries the low- and high-image magnetic records beside each energy
contrast; magnetic evidence must not be reconstructed from rounded console
summaries.

The Rockfish SG15 Cr/Sb/Te files pass the technical compatibility prerequisite:
all three are fully relativistic NC UPFs with spin-orbit projectors. Their exact
SHA-256 values are recorded in
`data_processed/qe_pseudopotential_compatibility/sg15_fr_pbe_rockfish.json`.
Explicit `noncolin` and `lspinorb` flags are still required in every SOC input.

## Geometry-Transform A/B Consequence

The direct and transformed members of each relaxation pair already use matched
settings, so their within-pair optimizer work is diagnostically comparable.
The formal acceleration claim has two additional gates:

1. both variants converge to the same basin under the same calculator;
2. that calculator passes the protocol gate above.

If the accepted calculator changes, rerun both A/B members. Never compare a
direct relaxation under one identity with a transformed relaxation under
another. Report ionic-step and SCF-iteration ratios as primary metrics; walltime
is secondary.

## Manuscript Text Templates

### Allowed Now

"An audit of the calculation provenance identified multiple QE calculator
identities across the historical datasets. We therefore treat all current
barriers as protocol-specific candidates until cutoff, k-point, spin, and SOC
tests are complete; each accepted result will be reported with its actual
calculator identity."

### Allowed Only After All Gates Pass

"For the accepted 61-atom protocol, increasing the cutoff changed the
fixed-geometry image-4 minus image-2 energy contrast by [CUTOFF-DELTA] eV, and
the k-point test changed it by [KPOINT-DELTA] eV. The selected spin/SOC treatment
changed the contrast by [PHYSICAL-MODEL-DELTA] eV. The final NEB barriers below
were recomputed or retained consistently under this validated identity."

### Forbidden Until Resolved

- "All DFT calculations used 100/400 Ry, 4x4x1, and SOC."
- Any pooled comparison of absolute energies across calculator identities.
- Any claim that the geometric transform accelerates DFT based on unconverged
  prefixes, different final basins, or an unaccepted calculator identity.

## Traceable Inputs

- Audit: `data_processed/qe_calculator_identity_audit/v1/`
- Stage-1 batch: `data_processed/qe_protocol_sensitivity/1-6_stage1_s42/`
- Canonical jobs: `30850191`, `30850192`, `30850193`, `30850195`
- Cancelled duplicates, never analyzed: `30850194`, `30850197`
- Stage-1 cutoff result: pass; `0.830367 eV` at `50/200 Ry` versus
  `0.829871 eV` at `100/400 Ry`, absolute difference `0.000497 eV`.
- Completed Gamma/2x2x1 jobs: `30851205`, `30851219`, `30851225`, `30851228`
- K-point-stage ledger: `data_processed/qe_protocol_sensitivity/1-6_kpoint_stage_s42/`
- K-point result: fail; `+0.830367 eV` at Gamma versus `-0.615659 eV` at
  `2x2x1`, difference `-1.446026 eV`.
- Completed 3x3x1/4x4x1 jobs: `30864939`, `30864944`, `30864946`, `30864947`
- Dense-k result: fail; `-0.807902 eV` versus `-0.904860 eV`, absolute
  difference `0.096958 eV`.
- Active 5x5x1 jobs: `30871998`, `30871999`
- Dense-k ledger: `data_processed/qe_protocol_sensitivity/1-6_kpoint_dense_s42/`
- 5x5x1 ledger: `data_processed/qe_protocol_sensitivity/1-6_kpoint_5x5_s42/`
- Analyzer: `scripts/migrationbench/analyze_qe_protocol_sensitivity.py`
- Fail-closed next-stage planner:
  `scripts/migrationbench/plan_next_qe_protocol_stage.py`
