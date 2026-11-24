# Task 2 — three-seed broad-roster final package

This directory is both a reproducibility package and an Obsidian-compatible
vault root for the completed Task 2 diagnostic study.

Start in Obsidian with [`obsidian/index.md`](obsidian/index.md). The notes link
to the audited tables, provenance manifests, input NEB paths, and the PNG/SVG/PDF
figures in this directory.

## Scope

- Three independent Cr-trained broad-roster fine-tunes: seeds 123, 456, and 789.
- All three reached epoch 999 and passed exact checkpoint-to-exported-model
  state-dict comparison: 90 tensors, maximum absolute difference 0.
- A100 validation loaded Foundation plus all three final models and produced
  1,200 finite profile rows, 120 summaries, and zero calculation failures.
- The ten archived QE/NEB paths are retained as provisional path references.

## Scientific boundary

The archived QE/NEB outputs retain SCF-convergence warnings and do not pass the
strict DFT scientific-success gate. The `<0.1 eV` last-two-table screen is a
sampling/triage criterion only. The model profiles are non-equilibrium snapshot
diagnostics, not DFT ground truth, barrier accuracy, DFT error, or transfer-gain
evidence.

## Reproduction entry points

- `scripts/verify_final_model_remote.py`: verify a unique epoch-999 checkpoint,
  exported model, compiled model, and exact tensor equality on the Rockfish MACE
  environment.
- `scripts/run_a100_task2_profiles.py`: run the finite element probes and the
  ten-path, three-variant diagnostic profile calculation on A100.
- `scripts/package_three_seed_task2.py`: build the audited tables and manifests.
- `scripts/render_three_seed_task2_figures.py`: regenerate the Scientific Plot
  Atlas PNG/SVG/PDF figures.
- `scripts/validate_three_seed_task2.py`: fail-closed package validation.

The model binaries are intentionally excluded from Git because they total about
556 MB. Their expected SHA256 values and the original Rockfish paths are retained
in `manifests/model_manifest.json`, `results/provenance/`, and
`obsidian/model-provenance.md`.
