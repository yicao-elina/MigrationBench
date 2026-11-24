# 639-candidate selection and provenance protocol

Status: `MLIP static screen VERIFIED`; full-cell DFT `BLOCKED_RESOURCE_GATED`;
reduced-cell DFT `SUBMITTED_PENDING`; no DFT mechanism conclusion is released.

This note specifies exactly how the 639 prepared-only configurations were
created, classified, and prioritized. It is the repository-facing companion
to the SI method fragment at
[`paper/revision1/appendix/from_scratch_mlip_candidate_selection_methods.tex`](../../../paper/revision1/appendix/from_scratch_mlip_candidate_selection_methods.tex).

## Scope and immutable input

The source inventory is the `dft_candidates_prepared_only/manifest.csv` layer
from the all-replicate failure-attribution analysis. Each row points to one
saved 2050-atom fully periodic snapshot. The candidate inventory was joined to
frame-level diagnostics by the exact key
`(source, training_seed, window, step)` and to run-level status by
`(source, training_seed)`. No candidate was removed because it looked
unstable; all 639 rows remain in the inventory and their source extxyz SHA256
is recorded.

Canonical IDs were assigned after deterministic sorting by source order
(`original`, `reproduce-1`, `reproduce-2`), training seed, step, window order,
and prepared path. The canonical ID is therefore `dftcand_0001`–
`dftcand_0639`. The production MLIP execution IDs (`cand_####`) are not
treated as canonical; they are reconciled to the canonical IDs through the
one-to-one immutable `prepared_extxyz` path mapping.

The committed compact evidence is in
`data/raw_compact/revision1/from_scratch_failure_attribution_20261013/`:

- `candidate_categorization.csv`: all 639 classified rows and source hashes;
- `candidate_category_summary.csv`: grouped count audit;
- `candidate_manifest_639.csv`: execution input manifest;
- `MODEL_REGISTRY.tsv`: the three exact MLIP checkpoints;
- `candidate_execution_mapping.csv`: execution-to-canonical reconciliation;
- `wave_01_manifest.csv`: the original mechanism-coverage wave;
- `mlip_dft_selection.csv` and `dft_batch_selection_remote.csv`: the later
  MLIP-informed reduced-cell DFT selection;
- `mlip_analysis_manifest.json`, `CLAIM_EVIDENCE_MATRIX.tsv`, and the resource
  and submission records.

The extxyz snapshots, model checkpoints, QE input/output trees, and Slurm log
trees are not copied into Git. Their approved external locations and SHA256
values remain in the compact manifests. This keeps the repository auditable
without pretending that a Git checkout contains the 2050-atom raw data or
model weights.

## Stage classification

The classification is a semantic relabeling of the source trajectory window;
it does not rerun MD or redefine the instability threshold.

| source window | canonical group | rule |
|---|---|---|
| `stable_early` | `stable_early_control` | early stable control window |
| `stable_pre` | `stable_pre_onset` | stable sampled frame before the source onset |
| `precursor` | `precursor_near_onset` | latest stable/precursor window before the source boundary |
| `threshold` with an observed trigger | `threshold_or_terminal` | operational threshold/terminal frame |
| `threshold` with `trigger == not_observed` | `terminal_stable_target` | 1000 ps stable endpoint used as a negative control |

The resulting counts are:

| group | count |
|---|---:|
| `stable_early_control` | 207 |
| `stable_pre_onset` | 216 |
| `precursor_near_onset` | 208 |
| `threshold_or_terminal` | 7 |
| `terminal_stable_target` | 1 |
| **total** | **639** |

## Original `wave_01` method

The first wave was selected before the complete three-model static MLIP screen.
It was designed to cover the physical stages and every observed run boundary,
not to maximize a post-hoc disagreement score:

1. Retain all seven `threshold_or_terminal` rows and the one
   `terminal_stable_target` row.
2. For each of the nine source/training-seed runs, retain the latest sampled
   `precursor` frame.
3. Retain one approximately 10 ps `stable_early` control for each of the three
   training-model hashes from the original source trajectory.

This gives exactly 20 structures: 7 threshold, 1 terminal stable target, 9
precursor, and 3 early stable controls. The exact row-level ledger is
`wave_01_manifest.csv` (SHA256
`292f57c9bc95cee2253aedfe74f1f953839aad54031c6c6998702f1554925b2a`).

## MLIP-informed DFT prioritization

After all 639 candidates were evaluated with all three registered MLIP
checkpoints, a separate selector generated the reduced-cell DFT candidate
list. The selection was fail-closed in this order:

1. Keep only candidates with `n_models_success == 3`.
2. Exclude source geometries with `min_distance_A < 0.8 Å`; this retained
   geometry failures as an exclusion ledger rather than deleting them.
3. Within each source group, sort by descending
   `ensemble_force_std_rms_eV_A`, then descending
   `pairwise_force_max_max_eV_A`, then ascending canonical candidate ID.
4. Apply the requested first-pass quotas: 4 threshold/terminal, 8
   precursor, 4 stable-pre, 2 early-stable, and 1 terminal-stable target.
5. If a quota cannot be filled after the geometry gate, fill the remaining
   slots from unused gate-passing candidates using the same deterministic
   ranking until the batch has 20 rows.

The geometry gate excluded 11 extreme rows (six threshold/terminal and five
precursor). Consequently, the realized batch is 1 threshold/terminal, 12
precursor, 4 stable-pre, 2 early-stable controls, and 1 terminal-stable target.
The exact selection ledger has local SHA256
`6b68429b5da5440feb2fab3081b4afe7cc6a7b0f7ea87203ce441537dffa060d`; the
remote-submission copy has SHA256
`78c1b5038761a14960836dccdd7899b77c3e4cfe9cc9464ce4f253315d207c6d1`.

The force-spread ranking is a prioritization heuristic. It does not mean that
the model with the largest force is wrong, nor that the selected structure is
the unique physical onset. The 120-atom PBC local-periodic-shell DFT inputs
are explicitly `RECONSTRUCTION` and do not have exact full-cell parity with
the 2050-atom snapshots.

## Model and seed provenance

The three model identities are canonicalized independently of filenames:

| canonical model ID | family | training seed | SHA256 |
|---|---|---:|---|
| `ablation_lmax_2_seed123` | `ablation_lmax_2` | 123 | `a26db3575de8484ca3c7fcf7621165e38ce0162a580cde022bccd16135c61351` |
| `ablation_lmax_2_seed234` | `ablation_lmax_2` | 234 | `52fccd0994664618256ec5e986dc5e962050c845d3f05f3d70862ff064009af5` |
| `baseline_seed345` | `baseline` | 345 | `9cadf6fa3c3212c512bc06ad1bb9392bd9b664c40949157ee6184e9c0b05811d` |

The source MD velocity seed is `20261005` and is not the same field as the
training seed. The static MLIP evaluation introduces no new MD seed.

## Reproducibility and acceptance gates

Use the execution instructions in
[`workflow/from_scratch_mlip_candidates/20261013/README.md`](../../../workflow/from_scratch_mlip_candidates/20261013/README.md).
The clean-checkout validation command is:

```bash
python scripts/validate_from_scratch_mlip_candidate_selection.py
```

The static screen is accepted only when the result contains 639 canonical
candidates, 1,917 candidate-model rows, no duplicate candidate-model pairs,
finite successful outputs, and model/input hashes matching the registry. DFT
is a separate gate: the full-cell QE attempt is resource-gated because the
2050-atom cell estimates approximately 31,900 electrons, 19,140 Kohn–Sham
states, and over 32 TB aggregate memory under the attempted setup. The later
120-atom reduced-cell run is recorded as `SUBMITTED_PENDING`; no QE result is
used in a mechanism claim until scheduler, QE termination, finite-output,
input identity, and reconstruction-parity checks pass.

## Claim boundary

This package supports the following reproducible statement: the same 639
source configurations were classified into five source-stage groups and
evaluated pairwise across three hash-identified MLIP checkpoints; the
resulting cross-model force disagreement was used to prioritize an explicitly
gated reduced-cell DFT follow-up. It does not yet support the stronger
statement that the MLIP force is physically incorrect or that force is the
causal trigger of the MD instability.
