# Cross-element checkpoint repair status

Date: 2026-10-02

## Executed isolated deployment

- Remote namespace: `/scratch16/pclancy3/yi/cross_element_generalization_repair_20261002`
- GPU smoke job: Rockfish `31510906`
- Device: A100/CUDA
- Original checkpoints were read-only; no historical model or campaign was overwritten.

## Evidence

| checkpoint | load environment | embedded atomic numbers | Cu/Li/Na | result |
|---|---|---|---|---|
| FT-600K | local CPU metadata audit | `[24, 51, 52]` | unsupported | species roster is Cr/Sb/Te only |
| FT-MultiT | A100/CUDA smoke | `[24, 51, 52]` | unsupported | CUDA deployment fixed, but checkpoint is also Cr/Sb/Te only |

FT-MultiT completed successfully on GPU and produced finite Cr/Sb/Te probe energies/forces. Therefore its previous `0/65` result was partly an environment/runtime blocker, but after that blocker was removed the checkpoint still failed the cross-element gate for Cu/Li/Na.

## What is and is not a small fix

### Small fix: deployment

The FT-MultiT CPU failure was fixed operationally by running the unchanged checkpoint in the existing Rockfish MACE CUDA environment on an A100. This is a deployment fix only.

### Not a safe small fix: adding elements

Appending Cu/Li/Na to `atomic_numbers` is not a valid repair. The checkpoint does not contain the corresponding element embeddings/parameters. A new checkpoint must be trained or structurally rebuilt from a compatible broad-element model, with explicit validation.

### Required new model

The repair target is a new six-element checkpoint, not an edited copy:

1. Initialize from the broad Foundation checkpoint.
2. Fine-tune on the Cr/Sb/Te AMD trajectory.
3. Add replay/anchor data containing Cr, Cu, Li, Na, Sb, and Te so the broad-element channels remain trained.
4. Use explicit energy and force losses, held-out element-wise validation, and a comparison against the unmodified Foundation model.
5. Save under `outputs/ft600k_broad_roster_repaired.model` and require all six element probes before path tests.

This is a real retraining/fine-tuning repair, not an in-place checkpoint patch.

## Broad-roster deployment started

The current Victor/Joshitha element union is recorded in `target_element_manifest.json`:

`Li, Na, V, Cr, Mn, Cu, Br, Mo, Sb, Te, Cs, Pb`

The isolated pilot `31511373` completed successfully. Its checkpoint retained the full
Foundation element roster and contained all 12 target elements; the GPU validation job
`31511395` recorded `roster_contains_target=true`. The formal 1000-epoch fine-tuning
has now been submitted as Rockfish job `31511398` under
`broad_roster_ft_seed123`.

This first formal run uses the original grouped Cr/Sb/Te AIMD data and
`foundation_model_elements=True`, so the 12-element coverage gate is satisfied by
Foundation initialization plus explicit roster validation. It does not, by itself,
claim new DFT supervision for Li/Na/V/Mn/Cu/Br/Mo/Cs/Pb; those elements require the
subsequent element-wise energy/force and system-level validation gate.
